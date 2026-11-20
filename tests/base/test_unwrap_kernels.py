"""The numba kernels of the IRLS unwrapping, bit for bit against numpy."""
from __future__ import annotations

import numpy as np
import pytest

from libertem_holo.base.unwrap import _irls_numba as kernels
from libertem_holo.base.unwrap.irls import TAU, _negative_divergence, _wrap

SHAPES = [(1, 1), (1, 6), (6, 1), (7, 9)]
DTYPES = [np.float32, np.float64]


def _random(
    rng: np.random.Generator, shape: tuple[int, int], dtype: type,
) -> np.ndarray:
    return (rng.normal(size=shape) * 3).astype(dtype)


def _edges(
    rng: np.random.Generator, shape: tuple[int, int], dtype: type,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = shape
    return (
        rng.uniform(0.1, 1, size=(rows, cols - 1)).astype(dtype),
        rng.uniform(0.1, 1, size=(rows - 1, cols)).astype(dtype),
    )


@pytest.mark.with_numba
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_operator_and_right_hand_side(shape, dtype):
    rng = np.random.default_rng(0)
    p = _random(rng, shape, dtype)
    weights_x, weights_y = _edges(rng, shape, dtype)
    target_x, target_y = _edges(rng, shape, dtype)

    out = np.empty_like(p)
    row_sums = np.empty(shape[0])
    kernels.apply_operator(p, weights_x, weights_y, out, row_sums)
    flux_x = np.diff(p, axis=1)
    flux_x *= weights_x
    flux_y = np.diff(p, axis=0)
    flux_y *= weights_y
    np.testing.assert_array_equal(
        out, _negative_divergence(flux_x, flux_y, np.empty_like(p)),
    )
    np.testing.assert_allclose(
        row_sums, (p.astype(np.float64) * out.astype(np.float64)).sum(axis=1),
        rtol=1e-12,
    )

    kernels.negative_divergence_of_weighted(
        weights_x, target_x, weights_y, target_y, out,
    )
    np.testing.assert_array_equal(
        out,
        _negative_divergence(
            weights_x * target_x, weights_y * target_y, np.empty_like(p),
        ),
    )


@pytest.mark.with_numba
@pytest.mark.parametrize("dtype", DTYPES)
def test_cg_updates(dtype):
    rng = np.random.default_rng(1)
    phi, direction, residual, operated, z = (
        _random(rng, (7, 9), dtype) for _ in range(5)
    )
    alpha, beta = dtype(0.37), dtype(1.3)

    new_phi, new_residual = phi.copy(), residual.copy()
    row_sums = np.empty(7)
    kernels.cg_update(new_phi, direction, new_residual, operated, alpha, row_sums)
    np.testing.assert_array_equal(new_phi, phi + alpha * direction)
    np.testing.assert_array_equal(new_residual, residual - alpha * operated)
    np.testing.assert_allclose(
        row_sums, (new_residual.astype(np.float64) ** 2).sum(axis=1), rtol=1e-12,
    )

    kernels.dot_rows(direction, z, row_sums)
    np.testing.assert_allclose(
        row_sums,
        (direction.astype(np.float64) * z.astype(np.float64)).sum(axis=1),
        rtol=1e-12,
    )

    new_direction = direction.copy()
    kernels.next_direction(new_direction, z, beta)
    expected = direction * beta
    expected += z
    np.testing.assert_array_equal(new_direction, expected)


@pytest.mark.with_numba
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_charbonnier_weights(shape, dtype):
    rng = np.random.default_rng(2)
    phi = _random(rng, shape, dtype)
    target_x, target_y = _edges(rng, shape, dtype)
    edge_x, edge_y = _edges(rng, shape, dtype)
    squared = dtype(1e-2**2)
    weights_x, weights_y = np.empty_like(edge_x), np.empty_like(edge_y)

    kernels.charbonnier_weights(
        phi, target_x, target_y, edge_x, edge_y, squared, weights_x, weights_y,
    )
    for weights, axis, target, edge in (
        (weights_x, 1, target_x, edge_x),
        (weights_y, 0, target_y, edge_y),
    ):
        residual = np.diff(phi, axis=axis)
        residual -= target
        expected = squared / np.sqrt(np.square(residual) + squared)
        expected *= edge
        np.testing.assert_array_equal(weights, expected)


@pytest.mark.with_numba
@pytest.mark.parametrize("dtype", DTYPES)
def test_rounding_onto_branches(dtype):
    rng = np.random.default_rng(3)
    wrapped = _wrap(_random(rng, (7, 9), dtype))
    phi = _random(rng, (7, 9), dtype) * 4
    tau = dtype(TAU)

    difference = np.empty_like(phi)
    kernels.wrapped_difference(phi, wrapped, tau, difference)
    np.testing.assert_array_equal(difference, _wrap(phi - wrapped))

    offset = np.median(difference)
    branches = np.empty(phi.shape, dtype=np.int32)
    phase = np.empty(phi.shape, dtype=np.float64)
    kernels.branches_and_phase(phi, wrapped, offset, tau, branches, phase)
    expected = np.rint((phi - offset - wrapped) / tau).astype(np.int32)
    np.testing.assert_array_equal(branches, expected)
    np.testing.assert_array_equal(phase, wrapped.astype(np.float64) + TAU * expected)


@pytest.mark.with_numba
@pytest.mark.parametrize("shape", SHAPES)
def test_cuts_per_row(shape):
    rng = np.random.default_rng(4)
    branches = rng.integers(-2, 3, size=shape).astype(np.int32)
    jumps_x = rng.integers(-1, 2, size=(shape[0], shape[1] - 1)).astype(np.int32)
    jumps_y = rng.integers(-1, 2, size=(shape[0] - 1, shape[1])).astype(np.int32)
    edge_x, edge_y = _edges(rng, shape, np.float32)
    turns = np.empty(shape[0], dtype=np.int64)
    cost = np.empty(shape[0], dtype=np.float64)

    kernels.cuts_per_row(branches, jumps_x, jumps_y, edge_x, edge_y, turns, cost)
    cuts_x = np.abs(jumps_x + np.diff(branches, axis=1))
    cuts_y = np.abs(jumps_y + np.diff(branches, axis=0))
    assert turns.sum() == cuts_x.sum() + cuts_y.sum()
    expected_cost = (cuts_x * edge_x.astype(np.float64)).sum()
    expected_cost += (cuts_y * edge_y.astype(np.float64)).sum()
    np.testing.assert_allclose(cost.sum(), expected_cost, rtol=1e-12)
