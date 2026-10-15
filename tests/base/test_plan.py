"""Tests for `libertem_holo.base.plan` against the previous implementation."""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest
from libertem.io.dataset.memory import MemoryDataSet
from libertem.utils.devices import detect

from libertem_holo.base.plan import (
    PartialSpectrum,
    ReconstructionPlan,
    gather_idx_1d,
    half_spectrum,
)
from libertem_holo.base.utils import HoloParams, get_slice_fft
from libertem_holo.udf.reconstr import HoloReconstructUDF

if TYPE_CHECKING:
    from libertem.api import Context

try:
    import cupy as cp
except ImportError:
    cp = None


def legacy_wave(
    frame: np.ndarray,
    sb_pos: tuple[float, float],
    aperture: np.ndarray,
    slice_fft: tuple[slice, slice],
) -> np.ndarray:
    """Frozen copy of the previous `reconstruct_frame`, as the reference."""
    fft_frame = np.fft.fft2(frame) / np.prod(frame.shape)
    fft_frame = np.roll(fft_frame, np.array(sb_pos).astype(np.int64), axis=(0, 1))
    fft_frame = np.fft.fftshift(np.fft.fftshift(fft_frame)[slice_fft])
    fft_frame = fft_frame * aperture
    return np.fft.ifft2(fft_frame) * np.prod(frame.shape)


def legacy_bf(
    frame: np.ndarray,
    aperture: np.ndarray,
    slice_fft: tuple[slice, slice],
) -> np.ndarray:
    """Frozen copy of the previous `reconstruct_bf`, as the reference."""
    fft_frame = np.fft.fft2(frame)
    fft_frame = np.fft.fftshift(np.fft.fftshift(fft_frame)[slice_fft])
    fft_frame = fft_frame * aperture
    return np.abs(np.fft.ifft2(fft_frame))


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """Largest difference relative to the largest magnitude of the reference `b`."""
    return float(np.abs(a - b).max() / np.abs(b).max())


def sb_positions(sy: int, sx: int) -> list[tuple[float, float]]:
    """Sideband positions with windows straddling kx=0, Nyquist and the row wrap."""
    return [
        (0, 0), (0, sx // 2), (0, sx // 2 - 1), (0, sx // 2 + 1), (1, sx - 1),
        (sy - 1, sx - 1), (sy // 2 + 1, sx // 2 - 1), (sy // 3 + 0.7, sx // 3 + 0.2),
    ]


@pytest.mark.parametrize("n", [16, 17, 31, 64])
@pytest.mark.parametrize("o", [7, 8, 9])
@pytest.mark.parametrize("s", [0, 3, -2, 15])
def test_gather_idx_1d_matches_legacy_ops(n: int, o: int, s: int) -> None:
    """The 1D index formula replicates roll + fftshift + slice + fftshift."""
    a = np.arange(n)
    sl = get_slice_fft((o, o), (n, n))[0]
    ref = np.fft.fftshift(np.fft.fftshift(np.roll(a, s))[sl])
    assert np.array_equal(a[gather_idx_1d(n, o, s)], ref)


@pytest.mark.parametrize("kind", ["half", "pruned"])
@pytest.mark.parametrize("reuse_buffers", [True, False])
@pytest.mark.parametrize(
    "sig_shape", [(16, 16), (17, 17), (16, 17), (31, 30), (64, 60), (65, 61)],
)
def test_equivalence_with_legacy(
    kind: str, reuse_buffers: bool, sig_shape: tuple[int, int],  # noqa: FBT001
) -> None:
    """Both spectrum kinds reproduce the legacy path, odd-size layout included."""
    rng = np.random.default_rng(42)
    sy, sx = sig_shape
    frame = rng.random(sig_shape)
    out_shapes = [(8, 8), (7, 7), (8, 7), (7, 8), (6, 9), sig_shape]
    for out_shape, sb_pos in itertools.product(out_shapes, sb_positions(sy, sx)):
        aperture = rng.random(out_shape)
        aperture_bf = rng.random(out_shape)
        slice_fft = get_slice_fft(out_shape, sig_shape)
        plan = ReconstructionPlan(
            sig_shape=sig_shape,
            out_shape=out_shape,
            sb_pos=sb_pos,
            aperture=aperture,
            aperture_bf=aperture_bf,
            reuse_buffers=reuse_buffers,
        )
        # two frames in a row, to cover buffer reuse:
        for _ in range(2):
            spec = half_spectrum(frame) if kind == "half" else plan.spectrum(frame)
            wave, bf = plan.wave(spec), plan.bf(spec)
            assert wave.dtype == np.complex128
            assert bf.dtype == np.float64
            wave_ref = legacy_wave(frame, sb_pos, aperture, slice_fft)
            assert rel_err(wave, wave_ref) < 1e-12
            assert rel_err(bf, legacy_bf(frame, aperture_bf, slice_fft)) < 1e-12


def test_pruned_column_count() -> None:
    """A window straddling kx=0 maps onto fewer stored columns than it has."""
    out_shape = (8, 8)
    aperture = np.ones(out_shape)
    plan = ReconstructionPlan(
        sig_shape=(32, 32), out_shape=out_shape, sb_pos=(3, 2),
        aperture=aperture, aperture_bf=aperture,
    )
    # sideband window: columns -6..1 -> stored 0..6; bf window: -4..3 -> stored 0..4
    assert plan.n_columns == 7
    # sideband window: columns -28..-21 -> stored 4..11, nothing mirrored
    plan_sb_only = ReconstructionPlan(
        sig_shape=(32, 32), out_shape=out_shape, sb_pos=(3, 24), aperture=aperture,
    )
    assert plan_sb_only.n_columns == 8
    # sideband window straddling the Nyquist column: -20..-13 -> stored 12..16
    plan_nyquist = ReconstructionPlan(
        sig_shape=(32, 32), out_shape=out_shape, sb_pos=(3, 16), aperture=aperture,
    )
    assert plan_nyquist.n_columns == 5


def test_float32() -> None:
    """With `precision=False` the computation is in complex64."""
    rng = np.random.default_rng(1)
    sig_shape, out_shape, sb_pos = (64, 60), (8, 8), (3, 50)
    frame = rng.random(sig_shape).astype(np.float32)
    aperture = rng.random(out_shape).astype(np.float32)
    plan = ReconstructionPlan(
        sig_shape=sig_shape, out_shape=out_shape, sb_pos=sb_pos,
        aperture=aperture, aperture_bf=aperture, precision=False,
    )
    wave, bf = plan.wave_and_bf(frame)
    assert wave.dtype == np.complex64
    assert bf.dtype == np.float32
    slice_fft = get_slice_fft(out_shape, sig_shape)
    assert rel_err(wave, legacy_wave(frame, sb_pos, aperture, slice_fft)) < 1e-5
    assert rel_err(bf, legacy_bf(frame, aperture, slice_fft)) < 1e-5
    # integer input is cast according to `precision`, too:
    wave_int, _ = plan.wave_and_bf((frame * 1000).astype(np.uint16))
    assert wave_int.dtype == np.complex64
    spec = half_spectrum(frame, precision=False)
    assert spec.data.dtype == np.complex64
    assert plan.wave(spec).dtype == np.complex64


def test_plan_argument_errors() -> None:
    aperture = np.ones((8, 8))
    with pytest.raises(ValueError, match="aperture shape"):
        ReconstructionPlan(
            sig_shape=(32, 32), out_shape=(8, 9), sb_pos=(1, 1), aperture=aperture,
        )
    plan = ReconstructionPlan(
        sig_shape=(32, 32), out_shape=(8, 8), sb_pos=(1, 1), aperture=aperture,
    )
    with pytest.raises(ValueError, match="frame shape"):
        plan.reconstruct(np.ones((32, 33)))
    with pytest.raises(ValueError, match="aperture_bf"):
        plan.reconstruct_bf(np.ones((32, 32)))
    with pytest.raises(ValueError, match="frames of shape"):
        plan.wave(half_spectrum(np.ones((32, 33))))
    with pytest.raises(ValueError, match="unknown spectrum kind"):
        plan.wave(PartialSpectrum(kind="full", data=None, sig_shape=(32, 32)))  # type: ignore[arg-type]


def test_cached_windows_are_shared_read_only() -> None:
    """Plans with the same geometry share the cached index arrays."""
    aperture = np.ones((8, 8))
    geometry = {"sig_shape": (32, 32), "out_shape": (8, 8), "aperture": aperture}
    a = ReconstructionPlan(sb_pos=(1, 1), **geometry)
    b = ReconstructionPlan(sb_pos=(1.9, 1), **geometry)
    assert a._sb.rows is b._sb.rows  # noqa: SLF001
    assert not a._sb.rows.flags.writeable  # noqa: SLF001


@pytest.mark.need_cupy
def test_mirrored_columns_cupy() -> None:
    """The conjugation of mirrored columns works on cupy arrays, too."""
    d = detect()
    if not d["cudas"] or not d["has_cupy"]:
        pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
    rng = np.random.default_rng(7)
    sig_shape, out_shape, sb_pos = (64, 60), (16, 16), (5, 57)  # window straddles kx=0
    frame = rng.random(sig_shape)
    aperture = rng.random(out_shape)
    plan = ReconstructionPlan(
        sig_shape=sig_shape, out_shape=out_shape, sb_pos=sb_pos,
        aperture=aperture, aperture_bf=aperture, xp=cp,
    )
    wave, bf = plan.wave_and_bf(cp.asarray(frame))
    slice_fft = get_slice_fft(out_shape, sig_shape)
    assert rel_err(wave.get(), legacy_wave(frame, sb_pos, aperture, slice_fft)) < 1e-10
    assert rel_err(bf.get(), legacy_bf(frame, aperture, slice_fft)) < 1e-10


@pytest.mark.parametrize("precision", [True, False])
def test_udf_matches_legacy(
    lt_ctx: Context, holo_data: tuple, precision: bool,  # noqa: FBT001
) -> None:
    """The UDF reconstructs each frame like the previous implementation."""
    holo, _, _, _ = holo_data
    frame = holo[0, 0].astype(np.float32)
    params = HoloParams.from_hologram(
        frame, out_shape=(32, 32), central_band_mask_radius=4,
    )
    ds = MemoryDataSet(data=holo.astype(np.float32), num_partitions=2)
    udf = HoloReconstructUDF(
        out_shape=params.out_shape,
        sb_position=params.sb_position,
        aperture=params.aperture,
        aperture_bf=params.aperture_bf,
        precision=precision,
    )
    res = lt_ctx.run_udf(dataset=ds, udf=udf)
    waves, bfs = res["wave"].data, res["bf"].data
    assert waves.shape == (*holo.shape[:2], 32, 32)
    assert waves.dtype == (np.complex128 if precision else np.complex64)
    tol = 1e-10 if precision else 1e-5
    for idx in np.ndindex(holo.shape[:2]):
        # the dataset is float32; with `precision` the UDF transforms it in float64:
        f = holo[idx].astype(np.float32)
        if precision:
            f = f.astype(np.float64)
        wave_ref = legacy_wave(f, params.sb_position, params.aperture, params.slice_fft)
        bf_ref = legacy_bf(f, params.aperture_bf, params.slice_fft)
        assert rel_err(waves[idx], wave_ref) < tol
        # the brightfield result buffer of the UDF is float32:
        assert rel_err(bfs[idx], bf_ref) < 1e-5
