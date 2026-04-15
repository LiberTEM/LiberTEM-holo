"""Model-based iterative reconstruction (MBIR) for 2D projected magnetization.

Unit conventions
----------------
All functions in this module follow these naming and unit rules:

* ``pixel_size_nm`` — pixel side length in **nanometres** (nm).
* ``b0_tesla`` — saturation induction :math:`B_0 = \\mu_0 M_s` in **Tesla** (T).
* ``thickness`` — sample thickness along the beam direction in **nanometres** (nm).
  Converted internally to voxels via ``thickness / pixel_size_nm``.
* ``phase`` — measured holographic phase in **radians** (rad).
* ``ramp_coeffs`` — background phase-ramp parameters
  ``[offset, slope_y, slope_x]`` with units **[rad, rad/nm, rad/nm]**.
  The slopes are multiplied by pixel coordinates scaled by ``pixel_size_nm``.
* ``PHI_0_T_NM2`` — magnetic flux quantum :math:`h/(2e)` expressed in
  :math:`\\text{T} \\cdot \\text{nm}^2`.

How *b0_tesla* affects the output magnetization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The RDFC kernel is proportional to ``b0_tesla / (2 * PHI_0_T_NM2)``, so
``b0_tesla`` is **absorbed** into the kernel.  The solver therefore returns a
normalised magnetization :math:`M^*` such that

.. math::

   M^* \\approx M_{\\text{true}} \\times
   \\frac{B_{0,\\text{true}}}{B_{0,\\text{used}}}

When ``b0_tesla`` equals the true saturation induction, :math:`M^*` is the
dimensionless, normalised magnetization (range roughly [-1, 1]).

When ``b0_tesla = 1.0``, the raw output has units of **Tesla** and equals the
physical projected magnetic induction :math:`\\mu_0 M_{\\text{phys}}`.  This
is often the most convenient choice when the true :math:`M_s` is unknown.
"""

from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple, Union

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax
import numpy as np
import unxt as u

PHI_0 = u.Quantity(2067.83, "T nm2 / rad")  # magnetic flux quantum h/(2e)
PHI_0_T_NM2 = 2067.83  # backward-compatible plain-float alias


# ---------------------------------------------------------------------------
# RampCoeffs — typed container for background ramp parameters
# ---------------------------------------------------------------------------

class RampCoeffs(NamedTuple):
    """Background phase-ramp coefficients with explicit units.

    Attributes
    ----------
    offset : Quantity["angle"]
        Constant phase offset in radians.
    slope_y : Quantity["angle / length"]
        Phase gradient along the y-axis in rad/nm.
    slope_x : Quantity["angle / length"]
        Phase gradient along the x-axis in rad/nm.
    """

    offset: u.Quantity
    slope_y: u.Quantity
    slope_x: u.Quantity

    @classmethod
    def zeros(cls, dtype=jnp.float64):
        """Create a zero-valued RampCoeffs."""
        return cls(
            offset=u.Quantity(jnp.zeros((), dtype=dtype), "rad"),
            slope_y=u.Quantity(jnp.zeros((), dtype=dtype), "rad/nm"),
            slope_x=u.Quantity(jnp.zeros((), dtype=dtype), "rad/nm"),
        )


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

def _to_nm(q: u.Quantity) -> u.Quantity:
    """Convert a length Quantity to nanometres."""
    return u.uconvert("nm", q)


def _to_tesla(q: u.Quantity) -> u.Quantity:
    """Convert a magnetic flux density Quantity to Tesla."""
    return u.uconvert("T", q)


def _to_rad(q: u.Quantity) -> u.Quantity:
    """Convert an angle Quantity to radians."""
    return u.uconvert("rad", q)


# ---------------------------------------------------------------------------
# Runtime validation helpers
# ---------------------------------------------------------------------------

def _validate_positive(value, name):
    """Raise ValueError if *value* is not positive.

    Handles both plain scalars and ``unxt.Quantity`` instances.
    """
    if isinstance(value, u.Quantity):
        v = float(value.value) if np.ndim(value.value) == 0 else float(np.min(np.asarray(value.value)))
    else:
        v = float(value) if np.ndim(value) == 0 else np.min(value)
    if v <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


class SolverResult(NamedTuple):
    """Result returned by :func:`solve_mbir_2d`.

    Attributes
    ----------
    magnetization : jnp.ndarray
        Reconstructed in-plane magnetization of shape ``(N, M, 2)``.
        When *thickness* is ``None`` (the default), this is the
        **projected** (thickness-integrated) magnetization.  When
        *thickness* is provided (in nm), the result is divided by
        the thickness in voxels so that it represents the per-voxel
        value.

        The physical meaning of the values depends on *b0_tesla*:
        with ``b0_tesla`` equal to the true :math:`\\mu_0 M_s` the
        values are dimensionless (normalised by :math:`M_s`); with
        ``b0_tesla = 1.0`` the values have units of **Tesla**.
        See the module docstring for details.
    ramp_coeffs : jnp.ndarray
        Background ramp coefficients ``[offset, slope_y, slope_x]``
        in units of **[rad, rad/nm, rad/nm]**.
    loss_history : jnp.ndarray
        Per-step loss values.
    """
    magnetization: jnp.ndarray
    ramp_coeffs: jnp.ndarray
    loss_history: jnp.ndarray


class LCurveResult(NamedTuple):
    """Result returned by :func:`lcurve_sweep` and :func:`lcurve_sweep_vmap`.

    Attributes
    ----------
    lambdas : np.ndarray
        Regularization weights used in the sweep.
    data_misfits : np.ndarray
        Data-fidelity term for each lambda.
    reg_norms : np.ndarray
        Regularization norm for each lambda.
    magnetizations : jnp.ndarray
        Reconstructed magnetizations, shape ``(n_lambdas, N, M, 2)``.
        When *thickness* is provided, these are per-voxel values.
        See :class:`SolverResult` for how *b0_tesla* affects units.
    ramp_coeffs : jnp.ndarray
        Background ramp coefficients per lambda, in units of
        **[rad, rad/nm, rad/nm]**.
    corner_index : int
        Index of the detected L-curve corner.
    """
    lambdas: np.ndarray
    data_misfits: np.ndarray
    reg_norms: np.ndarray
    magnetizations: jnp.ndarray
    ramp_coeffs: jnp.ndarray
    corner_index: int


class BootstrapThresholdResult(NamedTuple):
    """Result returned by :func:`bootstrap_threshold_uncertainty_2d`."""
    threshold: float
    threshold_low: float
    threshold_high: float
    threshold_draws: np.ndarray
    magnetizations: jnp.ndarray
    mean_magnetization: jnp.ndarray
    mean_norm: np.ndarray
    norm_low: np.ndarray
    norm_high: np.ndarray
    norm_ci95: np.ndarray
    relative_ci95: np.ndarray
    mask_frequency: np.ndarray


@dataclasses.dataclass(frozen=True)
class NewtonCGConfig:
    """Configuration for the Newton-CG solver.

    Parameters
    ----------
    cg_maxiter : int
        Maximum number of conjugate-gradient iterations used to
        solve the Newton system ``H @ delta = -g``.
    cg_tol : float
        CG convergence tolerance (relative residual norm).
    """
    cg_maxiter: int = 10000
    cg_tol: float = 1e-16


@dataclasses.dataclass(frozen=True)
class AdamConfig:
    """Configuration for the Adam solver."""
    num_steps: int = 2000
    learning_rate: float = 1e-2
    patience: int = 50
    min_delta: float = 1e-6


@dataclasses.dataclass(frozen=True)
class LBFGSConfig:
    """Configuration for the L-BFGS solver."""
    num_steps: int = 500
    patience: int = 50
    min_delta: float = 1e-6


SolverConfig = Union[NewtonCGConfig, AdamConfig, LBFGSConfig]

_SOLVER_DEFAULTS = {
    "newton_cg": NewtonCGConfig,
    "adam": AdamConfig,
    "lbfgs": LBFGSConfig,
}


def exchange_loss_fn(
    mag: jax.Array,
    reg_mask: jax.Array,
) -> jax.Array:
    r"""First-order regularization via masked finite differences.

    Computes the squared L2 norm of the spatial gradient of the
    magnetization field inside a mask:

    .. math::

        E = \sum_{i,j \in \text{mask}} \left\lVert
            \frac{\partial \mathbf{m}}{\partial y}\bigg|_{i,j}
            \right\rVert^2
          + \left\lVert
            \frac{\partial \mathbf{m}}{\partial x}\bigg|_{i,j}
            \right\rVert^2

    where each norm is over both magnetization components (u, v).
    The result is a scalar smoothness penalty: large when
    neighboring magnetization values differ, zero when the field
    is spatially uniform.  It enters the total MBIR loss as
    ``lambda_exchange * E``.

    Algorithm
    ---------
    1. **Neighbor detection**.  For each masked pixel the four
       cardinal neighbors (up, down, left, right) are checked
       against the mask.  ``neighbor_count`` records how many
       valid neighbors each pixel has (0--4).

    2. **Adaptive difference stencil**.  For each spatial axis the
       best available finite-difference is selected per pixel:

       * Both neighbors present → **central difference**
         (e.g. ``m[i+1,j] − m[i−1,j]``), accuracy *O(h²)*.
       * Only one neighbor → **forward or backward difference**
         (e.g. ``m[i+1,j] − m[i,j]``), accuracy *O(h)*.
       * No neighbor in that direction → zero contribution.

    3. **Neighbor-count normalization**.  Each difference is divided
       by the total neighbor count at that pixel.  Interior pixels
       (4 neighbors) are scaled by 1/4, edge pixels (2 neighbors)
       by 1/2, corner pixels (1 neighbor) by 1.  This keeps the
       per-pixel regularization contribution roughly uniform
       regardless of connectivity.

    4. **Squared L2 summation**.  The final loss is
       ``sum(diff_y²) + sum(diff_x²)`` over all masked pixels and
       both magnetization components.

    Differences from Pyramid's ``FirstOrderRegularisator``
    ------------------------------------------------------
    * **Dimensionality**: 2D (y, x) only, matching the projected
      magnetization geometry.  Pyramid regularizes all 3 spatial
      axes of a 3D volume.
    * **Stencil**: Adaptive central/one-sided selection at mask
      boundaries.  Pyramid uses forward differences everywhere via
      a sparse operator ``D`` built by ``jutil.diff``.
    * **Normalization**: Division by neighbor count.  Pyramid's
      sparse ``D`` matrix applies unit-weight forward differences
      with no per-pixel normalization.
    * **Implementation**: Pure JAX array operations; all
      derivatives obtained via autodiff.  Pyramid provides
      analytic ``jac``, ``hess_dot``, and ``hess_diag`` methods
      through its ``Regularisator`` class hierarchy.
    * **Lambda scale**: Because of the neighbor-count
      normalization and 2D-vs-3D geometry, a ``lambda_exchange``
      value here is not numerically identical to Pyramid's ``lam``
      for the same problem, even though both weight the same
      physical quantity (spatial roughness).

    Parameters
    ----------
    mag
        Magnetization array of shape ``(N, M, 2)``.
    reg_mask
        Boolean or binary mask of shape ``(N, M)`` defining the
        regularization region.

    Returns
    -------
    jax.Array
        Scalar exchange loss (L2 norm squared of finite differences).
    """
    reg_mask = jnp.asarray(reg_mask)
    if reg_mask.shape != mag.shape[:2]:
        raise ValueError(
            f"reg_mask must have shape {mag.shape[:2]}; got {reg_mask.shape}."
        )

    # Note: Unlike the 3D 'FirstOrderRegularisator', this function only
    # regularizes the 2 in-plane dimensions (y, x) matching the
    # projection geometry.

    mask = reg_mask.astype(bool)

    # Neighbor presence masks
    has_up = jnp.pad(mask[:-1, :], ((1, 0), (0, 0)), constant_values=False)
    has_down = jnp.pad(mask[1:, :], ((0, 1), (0, 0)), constant_values=False)
    has_left = jnp.pad(mask[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    has_right = jnp.pad(mask[:, 1:], ((0, 0), (0, 1)), constant_values=False)

    neighbor_count = (
        has_up.astype(mag.dtype)
        + has_down.astype(mag.dtype)
        + has_left.astype(mag.dtype)
        + has_right.astype(mag.dtype)
    )
    neighbor_count = neighbor_count * mask.astype(mag.dtype)
    denom = jnp.where(neighbor_count > 0, neighbor_count, jnp.ones_like(neighbor_count))

    # Shifted fields
    mag_up = jnp.pad(mag[:-1, :, :], ((1, 0), (0, 0), (0, 0)), constant_values=0)
    mag_down = jnp.pad(mag[1:, :, :], ((0, 1), (0, 0), (0, 0)), constant_values=0)
    mag_left = jnp.pad(mag[:, :-1, :], ((0, 0), (1, 0), (0, 0)), constant_values=0)
    mag_right = jnp.pad(mag[:, 1:, :], ((0, 0), (0, 1), (0, 0)), constant_values=0)

    # Y-direction difference (central if both neighbors exist, else forward/backward)
    both_y = (has_up & has_down)[..., None]
    only_down = (has_down & ~has_up)[..., None]
    only_up = (has_up & ~has_down)[..., None]
    diff_y = jnp.where(
        both_y, mag_down - mag_up,
        jnp.where(only_down, mag_down - mag,
                  jnp.where(only_up, mag - mag_up, 0))
    )

    # X-direction difference (central if both neighbors exist, else forward/backward)
    both_x = (has_left & has_right)[..., None]
    only_right = (has_right & ~has_left)[..., None]
    only_left = (has_left & ~has_right)[..., None]
    diff_x = jnp.where(
        both_x, mag_right - mag_left,
        jnp.where(only_right, mag_right - mag,
                  jnp.where(only_left, mag - mag_left, 0))
    )

    diff_y = diff_y * mask[..., None] / denom[..., None]
    diff_x = diff_x * mask[..., None] / denom[..., None]

    # L2 Norm (p=2) squared
    loss = jnp.sum(diff_y * diff_y) + jnp.sum(diff_x * diff_x)

    return loss


def forward_diff_norm(
    mag: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    r"""Forward-difference regularization norm matching Pyramid's convention.

    Computes ``\|D x\|^2`` using forward differences, with no
    per-pixel neighbor-count normalization.  This matches the
    ``FirstOrderRegularisator`` from Pyramid (which uses
    ``WeightedL2Square(D)`` with a sparse forward-diff operator),
    but is implemented in pure JAX for autodiff compatibility.

    Use this when comparing L-curve values with Pyramid's
    ``LCurve`` output, since :func:`exchange_loss_fn` applies
    adaptive central/one-sided differences with neighbor-count
    normalization that produce numerically different values.

    Parameters
    ----------
    mag
        Magnetization array of shape ``(N, M, 2)``.
    mask
        Binary mask of shape ``(N, M)`` defining the active region.

    Returns
    -------
    jax.Array
        Scalar ``\|Dx\|^2`` (sum of squared forward differences
        over y and x, both magnetization components).
    """
    mask = jnp.asarray(mask, dtype=bool)
    # Y-direction: forward difference (i+1) - (i), valid where both are masked
    valid_y = mask[:-1, :] & mask[1:, :]
    dy = (mag[1:, :, :] - mag[:-1, :, :]) * valid_y[..., None]
    # X-direction: forward difference (j+1) - (j), valid where both are masked
    valid_x = mask[:, :-1] & mask[:, 1:]
    dx = (mag[:, 1:, :] - mag[:, :-1, :]) * valid_x[..., None]
    return jnp.sum(dy ** 2) + jnp.sum(dx ** 2)


def get_freq_grid(
    height: int,
    width: int,
    pixel_size=None,
    *,
    pixel_size_nm=None,
) -> tuple:
    """Build frequency grids for FFT-based phase propagation.

    Parameters
    ----------
    height
        Number of pixels along the y-axis.
    width
        Number of pixels along the x-axis.
    pixel_size
        Pixel size as ``Quantity["length"]`` or plain float (nm).
    pixel_size_nm
        Legacy pixel size in nanometres (float).

    Returns
    -------
    f_y : jax.Array
        2D array of y-frequency values.
    f_x : jax.Array
        2D array of x-frequency values (real-FFT half-spectrum).
    denom : jax.Array
        ``f_x**2 + f_y**2`` with the zero-frequency bin set to 1
        to avoid division by zero.
    """
    if pixel_size is None and pixel_size_nm is not None:
        pixel_size = pixel_size_nm
    pixel_size = _ensure_quantity_pixel_size(pixel_size)
    pixel_size_val = _to_nm(pixel_size).value
    fy = jnp.fft.fftfreq(height, d=pixel_size_val)
    fx = jnp.fft.rfftfreq(width, d=pixel_size_val)
    f_y, f_x = jnp.meshgrid(fy, fx, indexing="ij")
    denom = f_x**2 + f_y**2
    denom = jnp.where(denom == 0, 1.0, denom)
    return f_y, f_x, denom


def _rdfc_elementary_phase(
    geometry: str,
    n: jax.Array,
    m: jax.Array,
) -> jax.Array:
    """Compute the elementary kernel phase for the RDFC mapper.

    Parameters
    ----------
    geometry
        Voxel geometry, either ``'disc'`` or ``'slab'``.
    n
        Row coordinate array.
    m
        Column coordinate array.
    Returns
    -------
    jax.Array
        Elementary phase kernel evaluated on the coordinate grid.
    """
    if geometry == "disc":
        in_or_out = jnp.logical_not(jnp.logical_and(n == 0, m == 0))
        return m / (n**2 + m**2 + 1e-30) * in_or_out
    if geometry == "slab":
        def _F_a(n_val, m_val):
            radius2 = n_val**2 + m_val**2 + 1e-30
            A = jnp.log(radius2)
            B = jnp.arctan(n_val / (m_val + 1e-30))
            return n_val * A - 2 * n_val + 2 * m_val * B

        return 0.5 * (
            _F_a(n - 0.5, m - 0.5)
            - _F_a(n + 0.5, m - 0.5)
            - _F_a(n - 0.5, m + 0.5)
            + _F_a(n + 0.5, m + 0.5)
        )
    raise ValueError("Unknown geometry (use 'disc' or 'slab')")


def _ensure_quantity_b0(b0):
    """Convert b0 argument to Quantity if it's a plain float/int."""
    if isinstance(b0, u.Quantity):
        return b0
    return u.Quantity(float(b0), "T")


def _ensure_quantity_pixel_size(pixel_size):
    """Convert pixel_size argument to Quantity if it's a plain float/int."""
    if isinstance(pixel_size, u.Quantity):
        return pixel_size
    return u.Quantity(float(pixel_size), "nm")


def _ensure_ramp_coeffs(ramp_coeffs):
    """Convert ramp_coeffs to RampCoeffs if it's a flat array."""
    if isinstance(ramp_coeffs, RampCoeffs):
        return ramp_coeffs
    # Legacy flat array [offset, slope_y, slope_x]
    return RampCoeffs(
        offset=u.Quantity(ramp_coeffs[0], "rad"),
        slope_y=u.Quantity(ramp_coeffs[1], "rad/nm"),
        slope_x=u.Quantity(ramp_coeffs[2], "rad/nm"),
    )


def _resolve_pixel_size_nm(pixel_size, pixel_size_nm):
    """Return pixel size in nm as a plain float from either new or legacy arg."""
    if pixel_size is None and pixel_size_nm is not None:
        pixel_size = pixel_size_nm
    return _to_nm(_ensure_quantity_pixel_size(pixel_size)).value


def _resolve_b0_tesla(b0, b0_tesla):
    """Return B0 in Tesla as a plain float from either new or legacy arg."""
    if b0 is None and b0_tesla is not None:
        b0 = b0_tesla
    if b0 is None:
        return 1.0
    return _to_tesla(_ensure_quantity_b0(b0)).value


def _resolve_thickness_nm(thickness):
    """Return thickness in nm as a plain float/array."""
    if thickness is None:
        return None
    if isinstance(thickness, u.Quantity):
        return _to_nm(thickness).value
    return thickness


@jax.jit(static_argnames=["dim_uv", "geometry", "dtype"])
def _build_rdfc_kernel_impl(
    dim_uv: tuple[int, int],
    b0_val: float,
    geometry: str = "disc",
    prw_vec: jax.Array | None = None,
    dtype: type = jnp.float64,
) -> dict[str, Any]:
    """Build an RDFC phase-mapping kernel in Fourier space (JIT-compiled).

    Operates on plain arrays only — no Quantity inside the trace.

    Parameters
    ----------
    dim_uv : tuple[int, int]
        Image dimensions ``(height, width)``.
    b0_val : float
        Saturation induction in Tesla (plain float).
    geometry : str, optional
        ``"disc"`` or ``"slab"``, default ``"disc"``.
    prw_vec : jax.Array or None, optional
        Projected reference wave vector ``(v, u)``.
    dtype : type, optional
        JAX float dtype, default ``jnp.float64``.

    Returns
    -------
    dict
        Dictionary with keys ``"u_fft"``, ``"v_fft"`` (complex
        JAX arrays with coeff baked in), ``"dim_uv"``, and
        ``"dim_pad"``.
    """
    coeff_val = b0_val / (2 * PHI_0_T_NM2)

    height, width = dim_uv
    dim_kern = (2 * height - 1, 2 * width - 1)
    dim_pad = (2 * height, 2 * width)

    u_coords = jnp.linspace(-(width - 1), width - 1, num=dim_kern[1]).astype(dtype)
    v_coords = jnp.linspace(-(height - 1), height - 1, num=dim_kern[0]).astype(dtype)
    uu, vv = jnp.meshgrid(u_coords, v_coords, indexing="xy")

    elem_uv = _rdfc_elementary_phase(geometry, uu, vv)
    elem_vu = _rdfc_elementary_phase(geometry, vv, uu)

    u_kernel = coeff_val * elem_uv
    v_kernel = -coeff_val * elem_vu

    if prw_vec is not None:
        uu_prw = uu + prw_vec[1]
        vv_prw = vv + prw_vec[0]
        elem_prw_uv = _rdfc_elementary_phase(geometry, uu_prw, vv_prw)
        elem_prw_vu = _rdfc_elementary_phase(geometry, vv_prw, uu_prw)
        u_kernel = u_kernel - coeff_val * elem_prw_uv
        v_kernel = v_kernel + coeff_val * elem_prw_vu

    u_pad = jnp.zeros(dim_pad, dtype=dtype).at[:dim_kern[0], :dim_kern[1]].set(u_kernel)
    v_pad = jnp.zeros(dim_pad, dtype=dtype).at[:dim_kern[0], :dim_kern[1]].set(v_kernel)

    return {
        "u_fft": jnp.fft.rfft2(u_pad),
        "v_fft": jnp.fft.rfft2(v_pad),
        "dim_uv": dim_uv,
        "dim_pad": dim_pad,
    }


def build_rdfc_kernel(
    dim_uv,
    b0=None,
    geometry="disc",
    prw_vec=None,
    dtype=jnp.float64,
    *,
    b0_tesla=None,
):
    """Build an RDFC phase-mapping kernel in Fourier space.

    Accepts either ``b0`` (Quantity) or ``b0_tesla`` (float, legacy).
    See :func:`_build_rdfc_kernel_impl` for full documentation.
    """
    if b0 is None and b0_tesla is None:
        b0 = u.Quantity(1.0, "T")
    elif b0_tesla is not None:
        b0 = _ensure_quantity_b0(b0_tesla)
    else:
        b0 = _ensure_quantity_b0(b0)
    b0 = _to_tesla(b0)
    result = _build_rdfc_kernel_impl(
        dim_uv, b0_val=b0.value, geometry=geometry, prw_vec=prw_vec, dtype=dtype,
    )
    result["coeff"] = b0 / (2 * PHI_0)  # Quantity["rad / nm2"]
    return result


def phase_mapper_rdfc(
    u_field: jax.Array,
    v_field: jax.Array,
    rdfc_kernel: dict[str, Any],
) -> jax.Array:
    """Map (u, v) magnetization components to a phase image via RDFC.

    Uses a precomputed Fourier-space kernel from
    :func:`build_rdfc_kernel`.

    Parameters
    ----------
    u_field
        In-plane magnetization component along x, shape ``(H, W)``.
        Dimensionless (plain JAX array).
    v_field
        In-plane magnetization component along y, shape ``(H, W)``.
        Dimensionless (plain JAX array).
    rdfc_kernel
        Kernel dictionary as returned by :func:`build_rdfc_kernel`.
        The coefficient is baked into the FFT arrays.

    Returns
    -------
    jax.Array
        Magnetic phase-shift image of shape ``(H, W)``.  The
        result has implicit units of ``rad/nm²`` (coeff baked in);
        multiply by ``pixel_size**2`` for radians.
    """
    height, width = u_field.shape
    dim_pad = (2 * height, 2 * width)

    u_pad = jnp.zeros(dim_pad, dtype=u_field.dtype)
    v_pad = jnp.zeros(dim_pad, dtype=v_field.dtype)
    u_pad = jax.lax.dynamic_update_slice(u_pad, u_field, (0, 0))
    v_pad = jax.lax.dynamic_update_slice(v_pad, v_field, (0, 0))

    u_fft = jnp.fft.rfft2(u_pad)
    v_fft = jnp.fft.rfft2(v_pad)

    phase_fft = u_fft * rdfc_kernel["u_fft"] + v_fft * rdfc_kernel["v_fft"]
    phase_pad = jnp.fft.irfft2(phase_fft, s=dim_pad)

    return jax.lax.dynamic_slice(
        phase_pad, (height - 1, width - 1), (height, width)
    )


def _apply_ramp_plain(
    ramp_coeffs: jax.Array,
    height: int,
    width: int,
    pixel_size_nm: float,
) -> jax.Array:
    """Plain-array ramp for use inside JIT-traced forward models."""
    y, x = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    ramp = ramp_coeffs[0]
    if ramp_coeffs.shape[0] > 1:
        ramp = ramp + ramp_coeffs[1] * (y * pixel_size_nm)
        ramp = ramp + ramp_coeffs[2] * (x * pixel_size_nm)
    return ramp


def apply_ramp(
    ramp_coeffs,
    height: int,
    width: int,
    pixel_size=None,
    *,
    pixel_size_nm=None,
):
    """Generate a first-order 2D polynomial background ramp.

    Accepts either ``RampCoeffs`` + ``Quantity["length"]`` (new API)
    or a flat array + float nm (legacy).

    Parameters
    ----------
    ramp_coeffs
        :class:`RampCoeffs` or legacy flat array of shape ``(3,)``
        with ``[offset, slope_y, slope_x]``.
    height
        Number of pixels along the y-axis.
    width
        Number of pixels along the x-axis.
    pixel_size
        Pixel size as ``Quantity["length"]``, or ``None`` when using
        *pixel_size_nm*.
    pixel_size_nm
        Legacy pixel size in nanometres (float).

    Returns
    -------
    Quantity["angle"] or jax.Array
        Ramp image of shape ``(height, width)``.
    """
    if pixel_size is None and pixel_size_nm is not None:
        pixel_size = pixel_size_nm
    pixel_size = _ensure_quantity_pixel_size(pixel_size)
    ramp_coeffs = _ensure_ramp_coeffs(ramp_coeffs)
    ps_val = _to_nm(pixel_size).value
    rc_arr = jnp.array([
        ramp_coeffs.offset.value,
        ramp_coeffs.slope_y.value * 1.0,  # already in rad/nm
        ramp_coeffs.slope_x.value * 1.0,
    ])
    ramp = _apply_ramp_plain(rc_arr, height, width, ps_val)
    return u.Quantity(ramp, "rad")


def forward_model_single_rdfc_2d(
    magnetization,
    ramp_coeffs,
    rdfc_kernel,
    pixel_size_nm,
):
    """RDFC forward model mapping projected magnetization to phase.

    Operates on plain JAX arrays internally (no Quantity inside
    the traced computation graph).

    Parameters
    ----------
    magnetization
        In-plane magnetization of shape ``(N, M, 2)`` where the
        last axis holds the (u, v) components.  Plain JAX array.
    ramp_coeffs
        Background ramp coefficients ``[offset, slope_y, slope_x]``.
        Plain JAX array of shape ``(3,)``.
    rdfc_kernel
        Kernel dictionary as returned by :func:`build_rdfc_kernel`.
    pixel_size_nm
        Pixel size in nanometres.  Plain float or scalar.

    Returns
    -------
    jax.Array
        Predicted phase image of shape ``(N, M)``.
    """
    height, width = magnetization.shape[:2]

    u_field = magnetization[..., 0]
    v_field = magnetization[..., 1]

    # Kernel has coeff baked in (units: 1/nm² implicit).
    # Multiply by pixel_size² to get radians (implicit).
    phase = pixel_size_nm**2 * phase_mapper_rdfc(u_field, v_field, rdfc_kernel)
    ramp = _apply_ramp_plain(ramp_coeffs, height, width, pixel_size_nm)

    return phase + ramp


def mbir_loss_2d(
    params: tuple[jax.Array, jax.Array],
    mask: jax.Array,
    phase: jax.Array,
    rdfc_kernel: dict[str, Any],
    pixel_size_nm: float,
    reg_config: dict[str, Any],
    reg_mask: jax.Array | None = None,
) -> jax.Array:
    """Compute the MBIR loss for 2D projected magnetization.

    The total loss is the sum of a least-squares data-fidelity term
    and optional exchange-energy regularization.

    Parameters
    ----------
    params
        Tuple of ``(magnetization, ramp_coeffs)`` where
        *magnetization* has shape ``(N, M, 2)`` and *ramp_coeffs*
        has shape ``(3,)``.
    mask
        Binary mask of shape ``(N, M)`` applied to the
        magnetization before the forward model.
    phase
        Observed phase image of shape ``(H, W)``.
    rdfc_kernel
        Kernel dictionary as returned by :func:`build_rdfc_kernel`.
    pixel_size_nm
        Pixel size in nanometres.
    reg_config
        Regularization configuration dictionary.  Recognised key:
        ``'lambda_exchange'`` (float, default 0.0).
    reg_mask
        Optional regularization mask of shape ``(N, M)`` passed to
        :func:`exchange_loss_fn`.  Defaults to *mask* when not
        provided.

    Returns
    -------
    jax.Array
        Scalar loss value.
    """
    if reg_mask is None:
        reg_mask = mask
    magnetization, ramp_coeffs = params
    phase = jnp.asarray(phase)

    magnetization = jnp.stack([
        magnetization[..., 0] * mask,
        magnetization[..., 1] * mask,
    ], axis=-1)

    predictions = forward_model_single_rdfc_2d(
        magnetization,
        ramp_coeffs,
        rdfc_kernel,
        pixel_size_nm,
    )

    residuals = predictions - phase
    loss = 0.5 * jnp.sum(residuals ** 2)

    lambda_exchange = jnp.asarray(
        reg_config.get("lambda_exchange", 0.0), dtype=loss.dtype
    )

    loss += lambda_exchange * exchange_loss_fn(magnetization, reg_mask)

    return loss


def _run_newton_cg_solver_2d(
    phase: jax.Array,
    init_mag: jax.Array,
    mask: jax.Array,
    pixel_size_nm: float,
    reg_config: dict[str, Any] | None = None,
    rdfc_kernel: dict[str, Any] | None = None,
    cg_tol: float = 1e-8,
    cg_maxiter: int = 10000,
    init_ramp_coeffs: jax.Array | None = None,
    reg_mask: jax.Array | None = None,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Minimize :func:`mbir_loss_2d` using a single Newton-CG solve.

    The MBIR objective is quadratic in the reconstruction
    parameters, so Newton-CG reduces to a single linear solve of
    ``H @ delta = -g``.  The solver accuracy is therefore fully
    controlled by the inner CG tolerance and iteration budget.

    Parameters
    ----------
    phase
        Observed phase image of shape ``(H, W)``.
    init_mag
        Initial magnetization of shape ``(N, M, 2)``.
    mask
        Binary mask of shape ``(N, M)`` applied to the
        magnetization.
    pixel_size_nm
        Pixel size in nanometres.
    reg_config
        Regularization configuration dictionary (see
        :func:`mbir_loss_2d`), default ``{}``.
    rdfc_kernel
        Kernel dictionary as returned by :func:`build_rdfc_kernel`.
    cg_tol
        Tolerance for the CG solver, default 1e-8.
    cg_maxiter
        Maximum number of CG iterations for the Newton solve,
        default 10000.
    init_ramp_coeffs
        Initial ramp coefficients of shape ``(3,)``.  Defaults to
        zeros.
    reg_mask
        Optional regularization mask of shape ``(N, M)``.
        Defaults to *mask*.

    Returns
    -------
    (magnetization, ramp_coeffs) : tuple[jax.Array, jax.Array]
        Optimized magnetization ``(N, M, 2)`` and ramp ``(3,)``.
    loss_history : jax.Array
        Length-1 array containing the loss after the Newton update.
    """
    if init_ramp_coeffs is None:
        init_ramp_coeffs = jnp.zeros((3,), dtype=init_mag.dtype)
    if reg_config is None:
        reg_config = {}

    x0_tree = (init_mag, init_ramp_coeffs)
    x0_flat, unravel = ravel_pytree(x0_tree)

    def objective_flat(x_flat):
        mag, ramp = unravel(x_flat)
        return mbir_loss_2d(
            (mag, ramp),
            mask,
            phase,
            rdfc_kernel,
            pixel_size_nm,
            reg_config,
            reg_mask=reg_mask,
        )

    loss_grad = jax.grad(objective_flat)

    grad_at_x0 = loss_grad(x0_flat)

    def matvec_hvp(v):
        return jax.jvp(loss_grad, (x0_flat,), (v,))[1]

    delta, _info = jax.scipy.sparse.linalg.cg(
        matvec_hvp, -grad_at_x0, tol=cg_tol, maxiter=cg_maxiter,
    )
    final_flat = x0_flat + delta
    history = jnp.expand_dims(objective_flat(final_flat), axis=0)

    final_mag, final_ramp = unravel(final_flat)

    return (final_mag, final_ramp), history


def _run_adam_solver_2d(
    phase: jax.Array,
    init_mag: jax.Array,
    mask: jax.Array,
    pixel_size_nm: float,
    reg_config: dict[str, Any] | None = None,
    num_steps: int = 2000,
    learning_rate: float = 1e-2,
    rdfc_kernel: dict[str, Any] | None = None,
    init_ramp_coeffs: jax.Array | None = None,
    patience: int = 50,
    min_delta: float = 1e-6,
    reg_mask: jax.Array | None = None,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Minimize :func:`mbir_loss_2d` using the Adam optimizer.

    Includes early stopping: optimisation halts when the loss has
    not improved by more than *min_delta* for *patience* consecutive
    steps.

    Parameters
    ----------
    phase
        Observed phase image of shape ``(N, M)``.
    init_mag
        Initial magnetization of shape ``(N, M, 2)``.
    mask
        Binary mask of shape ``(N, M)`` applied to the
        magnetization.
    pixel_size_nm
        Pixel size in nanometres.
    reg_config
        Regularization configuration dictionary (see
        :func:`mbir_loss_2d`), default ``{}``.
    num_steps
        Maximum number of optimisation steps, default 2000.
    learning_rate
        Adam learning rate, default 1e-2.
    rdfc_kernel
        Kernel dictionary as returned by :func:`build_rdfc_kernel`.
    init_ramp_coeffs
        Initial ramp coefficients of shape ``(3,)``.  Defaults to
        zeros.
    patience
        Number of steps without sufficient improvement before
        stopping, default 50.
    min_delta
        Minimum loss decrease to qualify as an improvement,
        default 1e-6.
    reg_mask
        Optional regularization mask of shape ``(N, M)``.
        Defaults to *mask*.

    Returns
    -------
    (magnetization, ramp_coeffs) : tuple[jax.Array, jax.Array]
        Optimized magnetization ``(N, M, 2)`` and ramp ``(3,)``.
    loss_history : jax.Array
        Per-step loss values (truncated at the step where early
        stopping triggered, if applicable).
    """
    if init_ramp_coeffs is None:
        init_ramp_coeffs = jnp.zeros((3,), dtype=init_mag.dtype)
    if reg_config is None:
        reg_config = {}

    params = (init_mag, init_ramp_coeffs)

    # Initialize optax optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # initial_loss needed for state initialization
    init_loss = mbir_loss_2d(
        params, mask, phase, rdfc_kernel, pixel_size_nm, reg_config,
        reg_mask=reg_mask,
    )

    # State: (params, opt_state, step_idx, loss_history_arr, best_loss, patience_counter, stop_flag)
    # We allocate a fixed-size array for history because shapes must be static, but handle early exit via while_loop
    loss_history = jnp.zeros(num_steps, dtype=init_loss.dtype)
    loss_history = loss_history.at[0].set(init_loss)

    init_state = (
        params,
        opt_state,
        0,  # step index
        loss_history,
        init_loss,  # best_loss
        0,  # patience counter
        False,  # stop flag
    )

    def cond_fn(state):
        _, _, step_i, _, _, _, stop_flag = state
        return jnp.logical_and(step_i < num_steps, jnp.logical_not(stop_flag))

    def body_fn(state):
        curr_params, curr_opt, i, history, best_loss, pat_count, _ = state

        loss_val, grads = jax.value_and_grad(mbir_loss_2d)(
            curr_params,
            mask,
            phase,
            rdfc_kernel,
            pixel_size_nm,
            reg_config,
            reg_mask=reg_mask,
        )

        updates, next_opt = optimizer.update(grads, curr_opt, curr_params)
        next_params = optax.apply_updates(curr_params, updates)

        # Check early stopping
        # Using a simple check: if current loss is not improving best_loss by min_delta
        improved = loss_val < (best_loss - min_delta)
        next_best_loss = jnp.where(improved, loss_val, best_loss)
        next_pat_count = jnp.where(improved, 0, pat_count + 1)
        next_stop = next_pat_count >= patience

        next_history = history.at[i].set(loss_val)

        return (
            next_params,
            next_opt,
            i + 1,
            next_history,
            next_best_loss,
            next_pat_count,
            next_stop,
        )

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)

    final_params, _, steps_taken, final_history, _, _, _ = final_state

    # Return full history (may contain trailing zeros after early stopping).
    return final_params, final_history


@jax.jit(static_argnames=("num_steps", "patience", "min_delta"))
def _run_lbfgs_solver_2d(
    phase: jax.Array,
    init_mag: jax.Array,
    mask: jax.Array,
    pixel_size_nm: float,
    reg_config: dict[str, Any] | None = None,
    num_steps: int = 500,
    rdfc_kernel: dict[str, Any] | None = None,
    init_ramp_coeffs: jax.Array | None = None,
    patience: int = 50,
    min_delta: float = 1e-6,
    reg_mask: jax.Array | None = None,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Minimize :func:`mbir_loss_2d` using L-BFGS with zoom line-search.

    Uses the optax L-BFGS implementation with early stopping.
    This function is JIT-compiled; *num_steps*, *patience*, and
    *min_delta* are static arguments.

    Parameters
    ----------
    phase
        Observed phase image of shape ``(N, M)``.
    init_mag
        Initial magnetization of shape ``(N, M, 2)``.
    mask
        Binary mask of shape ``(N, M)`` applied to the
        magnetization.
    pixel_size_nm
        Pixel size in nanometres.
    reg_config
        Regularization configuration dictionary (see
        :func:`mbir_loss_2d`), default ``{}``.
    num_steps
        Maximum number of optimisation steps, default 500.
    rdfc_kernel
        Kernel dictionary as returned by :func:`build_rdfc_kernel`.
    init_ramp_coeffs
        Initial ramp coefficients of shape ``(3,)``.  Defaults to
        zeros.
    patience
        Number of steps without sufficient improvement before
        stopping, default 50.
    min_delta
        Minimum loss decrease to qualify as an improvement,
        default 1e-6.
    reg_mask
        Optional regularization mask of shape ``(N, M)``.
        Defaults to *mask*.

    Returns
    -------
    (magnetization, ramp_coeffs) : tuple[jax.Array, jax.Array]
        Optimized magnetization ``(N, M, 2)`` and ramp ``(3,)``.
    loss_history : jax.Array
        Per-step loss values (zero-filled after early stopping,
        if applicable).
    """
    if init_ramp_coeffs is None:
        init_ramp_coeffs = jnp.zeros((3,), dtype=init_mag.dtype)
    if reg_config is None:
        reg_config = {}

    params = (init_mag, init_ramp_coeffs)

    def value_fn(p):
        return mbir_loss_2d(
            p,
            mask,
            phase,
            rdfc_kernel,
            pixel_size_nm,
            reg_config,
            reg_mask=reg_mask,
        )

    ls_inst = optax.scale_by_zoom_linesearch(max_linesearch_steps=15)
    solver = optax.lbfgs(linesearch=ls_inst)
    opt_state = solver.init(params)

    # Optax helper: lets us reuse value/grad stored in state (esp. with line-search)
    value_and_grad = optax.value_and_grad_from_state(value_fn)

    # Initial value for history dtype
    init_value, init_grad = value_and_grad(params, state=opt_state)
    loss_history = jnp.zeros((num_steps,), dtype=init_value.dtype)

    # State: (params, opt_state, step_idx, history, best_loss, patience_counter, stop_flag)
    init_state = (
        params,
        opt_state,
        0,
        loss_history,
        init_value,
        0,
        False,
    )

    def cond_fn(state):
        _, _, i, _, _, _, stop_flag = state
        return jnp.logical_and(i < num_steps, jnp.logical_not(stop_flag))

    def body_fn(state):
        curr_params, curr_state, i, history, best_loss, pat_count, _ = state

        value, grad = value_and_grad(curr_params, state=curr_state)

        updates, next_state = solver.update(
            grad,                 # positional grads
            curr_state,
            curr_params,
            value=value,          # current value at curr_params
            grad=grad,            # current grad at curr_params
            value_fn=value_fn,    # scalar objective for line-search
        )

        next_params = optax.apply_updates(curr_params, updates)
        history = history.at[i].set(value)

        improved = value < (best_loss - min_delta)
        next_best_loss = jnp.where(improved, value, best_loss)
        next_pat_count = jnp.where(improved, 0, pat_count + 1)
        next_stop = next_pat_count >= patience

        return (
            next_params,
            next_state,
            i + 1,
            history,
            next_best_loss,
            next_pat_count,
            next_stop,
        )

    final_params, final_opt_state, _, final_history, _, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, init_state
    )

    return final_params, final_history


def solve_mbir_2d(
    phase,
    init_mag,
    mask,
    pixel_size=None,
    solver="newton_cg",
    reg_config=None,
    rdfc_kernel=None,
    init_ramp_coeffs=None,
    reg_mask=None,
    *,
    pixel_size_nm=None,
):
    """
    Unified MBIR solver for 2D projected magnetization reconstruction.

    Parameters
    ----------
    phase : array_like
        Measured phase image in **radians**.
    init_mag : array_like
        Initial magnetization estimate, shape ``(N, M, 2)``.
    mask : array_like
        Binary mask of shape ``(N, M)`` applied to the
        magnetization.
    pixel_size_nm : float
        Voxel size in **nanometres** (nm).
    solver : str or SolverConfig, optional
        Which solver to use.  Pass a string (``"newton_cg"``, ``"adam"``,
        ``"lbfgs"``) for default parameters, or a config object
        (:class:`NewtonCGConfig`, :class:`AdamConfig`, :class:`LBFGSConfig`)
        for full control.  Default is ``"newton_cg"``.
    reg_config : dict, optional
        Regularization configuration (e.g. ``{"lambda_exchange": 1.0}``).
    rdfc_kernel : dict, optional
        Pre-built RDFC kernel from :func:`build_rdfc_kernel`.
    init_ramp_coeffs : array_like, optional
        Initial ramp coefficients ``[offset, slope_y, slope_x]``
        in units of **[rad, rad/nm, rad/nm]**.
    reg_mask : array_like, optional
        Regularization mask of shape ``(N, M)``.  Defaults to *mask*.

    Returns
    -------
    SolverResult
        Named tuple with fields ``magnetization``, ``ramp_coeffs``, and
        ``loss_history``.

        .. note::

           For iterative solvers (Adam, L-BFGS) the ``loss_history`` array
           may contain trailing zeros if the solver stopped early.
    """
    pixel_size_nm = _resolve_pixel_size_nm(pixel_size, pixel_size_nm)

    if isinstance(solver, str):
        solver_name = solver.lower()
        if solver_name not in _SOLVER_DEFAULTS:
            raise ValueError(
                f"Unknown solver {solver!r}. "
                f"Choose from {list(_SOLVER_DEFAULTS)}"
            )
        config = _SOLVER_DEFAULTS[solver_name]()
    elif isinstance(solver, (NewtonCGConfig, AdamConfig, LBFGSConfig)):
        config = solver
    else:
        raise TypeError(
            f"solver must be a string or a SolverConfig instance, got {type(solver)}"
        )

    shared = dict(
        phase=phase,
        init_mag=init_mag,
        mask=mask,
        pixel_size_nm=pixel_size_nm,
        reg_config=reg_config,
        rdfc_kernel=rdfc_kernel,
        init_ramp_coeffs=init_ramp_coeffs,
        reg_mask=reg_mask,
    )

    if isinstance(config, NewtonCGConfig):
        (mag, ramp), loss_history = _run_newton_cg_solver_2d(
            **shared,
            cg_tol=config.cg_tol,
            cg_maxiter=config.cg_maxiter,
        )
    elif isinstance(config, AdamConfig):
        (mag, ramp), loss_history = _run_adam_solver_2d(
            **shared,
            num_steps=config.num_steps,
            learning_rate=config.learning_rate,
            patience=config.patience,
            min_delta=config.min_delta,
        )
    elif isinstance(config, LBFGSConfig):
        (mag, ramp), loss_history = _run_lbfgs_solver_2d(
            **shared,
            num_steps=config.num_steps,
            patience=config.patience,
            min_delta=config.min_delta,
        )

    return SolverResult(
        magnetization=mag,
        ramp_coeffs=ramp,
        loss_history=loss_history,
    )


def reconstruct_2d(
    phase,
    pixel_size=None,
    b0=None,
    thickness=None,
    mask=None,
    lam=1e-3,
    solver="newton_cg",
    reg_mask=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
    solver_config=None,
    *,
    pixel_size_nm=None,
    b0_tesla=None,
):
    """Convenience wrapper for 2D MBIR magnetization reconstruction.

    Provides a simple interface similar to pyramid's
    ``reconstruction_2d_from_phasemap``.  Builds the RDFC kernel,
    initial magnetization guess, and mask automatically.

    Parameters
    ----------
    phase : array_like
        Measured phase image of shape ``(N, M)`` in **radians**.
    pixel_size_nm : float
        Pixel size in **nanometres** (nm).  Must be positive.
    b0_tesla : float
        Saturation induction :math:`B_0 = \\mu_0 M_s` in **Tesla**.
        Determines the scale of the returned magnetization — see
        *Notes* below.
    thickness : float or array_like
        Sample thickness in **nanometres** (nm) along the beam
        direction.  Converted to voxels internally via
        ``thickness / pixel_size_nm``.  The returned magnetization
        is divided by the thickness in voxels so that it represents
        the per-voxel value rather than the projected
        (thickness-integrated) value.  May be a scalar or a 2-D
        map of shape ``(N, M)``.
    mask : array_like, optional
        Binary mask of shape ``(N, M)``.  Defaults to all ones.
    lam : float, optional
        Regularization weight (``lambda_exchange``), default 1e-3.
    solver : str or SolverConfig, optional
        Solver selection string (``"newton_cg"``, ``"adam"``,
        ``"lbfgs"``) or a :class:`SolverConfig` instance.
        Ignored when *solver_config* is provided.
        Default is ``"newton_cg"``.
    reg_mask : array_like, optional
        Separate regularization mask of shape ``(N, M)``.
        Defaults to *mask*.
    geometry : str, optional
        Voxel geometry for the RDFC kernel (``"disc"`` or
        ``"slab"``), default ``"disc"``.
    prw_vec : array_like, optional
        Projected reference wave vector ``(v, u)``.
    rdfc_kernel : dict, optional
        Pre-built RDFC kernel from :func:`build_rdfc_kernel`.
        Built automatically when not provided.
    solver_config : SolverConfig, optional
        Explicit solver configuration object.  When provided,
        the *solver* string argument is ignored.

    Returns
    -------
    SolverResult
        Named tuple with fields ``magnetization``, ``ramp_coeffs``,
        and ``loss_history``.  See :class:`SolverResult` for how
        *b0_tesla* affects the magnetization units.

    Notes
    -----
    The RDFC kernel is proportional to ``b0_tesla``, so the solver
    absorbs it into the reconstructed magnetization.  With ``b0_tesla``
    equal to the true saturation induction the output is a
    dimensionless, normalised :math:`M / M_s`.  With ``b0_tesla = 1.0``
    the output has units of **Tesla**.
    """
    pixel_size_nm = _resolve_pixel_size_nm(pixel_size, pixel_size_nm)
    b0_tesla = _resolve_b0_tesla(b0, b0_tesla)
    thickness = _resolve_thickness_nm(thickness)
    _validate_positive(pixel_size_nm, "pixel_size_nm")
    _validate_positive(thickness, "thickness")

    phase = jnp.asarray(phase)
    if mask is None:
        mask = jnp.ones(phase.shape, dtype=bool)
    else:
        mask = jnp.asarray(mask, dtype=bool)

    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(
            phase.shape,
            b0_tesla=b0_tesla,
            geometry=geometry,
            prw_vec=prw_vec,
        )

    init_mag = jnp.zeros((*phase.shape, 2), dtype=jnp.float64)
    reg_config = {"lambda_exchange": float(lam)}

    if solver_config is not None:
        solver = solver_config

    result = solve_mbir_2d(
        phase=phase,
        init_mag=init_mag,
        mask=mask,
        pixel_size_nm=pixel_size_nm,
        solver=solver,
        reg_config=reg_config,
        rdfc_kernel=rdfc_kernel,
        reg_mask=reg_mask,
    )
    thickness = jnp.asarray(thickness, dtype=jnp.float64)
    thickness_vox = thickness / pixel_size_nm
    if thickness_vox.ndim >= 2:
        thickness_vox = thickness_vox[..., jnp.newaxis]

    mag = result.magnetization / thickness_vox

    return SolverResult(
        magnetization=mag,
        ramp_coeffs=result.ramp_coeffs,
        loss_history=result.loss_history,
    )


def forward_model_2d(
    magnetization,
    pixel_size=None,
    b0=None,
    ramp_coeffs=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
    thickness=None,
    *,
    pixel_size_nm=None,
    b0_tesla=None,
):
    """Convenience forward model for 2D projected magnetization.

    Computes the magnetic phase shift from a magnetization field,
    automatically building the RDFC kernel when not provided.

    Parameters
    ----------
    magnetization : array_like
        In-plane magnetization of shape ``(N, M, 2)`` where the
        last axis holds the (u, v) components.  If *thickness*
        is provided, the magnetization is assumed to be per-voxel
        and is multiplied by *thickness* before forward modeling.
    pixel_size_nm : float
        Pixel size in **nanometres** (nm).  Must be positive.
    b0_tesla : float, optional
        Saturation induction :math:`B_0 = \\mu_0 M_s` in **Tesla**,
        default 1.0.  Must match the value used during
        reconstruction for consistent round-tripping.
    ramp_coeffs : array_like, optional
        Background ramp coefficients ``[offset, slope_y, slope_x]``
        in units of **[rad, rad/nm, rad/nm]**.
        Defaults to zeros (no ramp).
    geometry : str, optional
        Voxel geometry for the RDFC kernel (``"disc"`` or
        ``"slab"``), default ``"disc"``.
    prw_vec : array_like, optional
        Projected reference wave vector ``(v, u)``.
    rdfc_kernel : dict, optional
        Pre-built RDFC kernel from :func:`build_rdfc_kernel`.
        Built automatically when not provided.
    thickness : float, optional
        Sample thickness in **nanometres** (nm) along the beam
        direction.  When provided, the magnetization is multiplied
        by the thickness in voxels (``thickness / pixel_size_nm``)
        before computing the phase (converting per-voxel to
        projected magnetization).  Default ``None`` assumes the
        input is already projected.

    Returns
    -------
    jax.Array
        Predicted phase image of shape ``(N, M)`` in **radians**.
    """
    pixel_size_nm = _resolve_pixel_size_nm(pixel_size, pixel_size_nm)
    b0_tesla = _resolve_b0_tesla(b0, b0_tesla)
    thickness = _resolve_thickness_nm(thickness)
    _validate_positive(pixel_size_nm, "pixel_size_nm")
    if thickness is not None:
        _validate_positive(thickness, "thickness")

    magnetization = jnp.asarray(magnetization)
    if thickness is not None:
        magnetization = magnetization * (thickness / pixel_size_nm)
    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(
            magnetization.shape[:2],
            b0_tesla=b0_tesla,
            geometry=geometry,
            prw_vec=prw_vec,
        )
    if ramp_coeffs is None:
        ramp_coeffs = jnp.zeros(3, dtype=magnetization.dtype)
    else:
        ramp_coeffs = jnp.asarray(ramp_coeffs)

    return forward_model_single_rdfc_2d(
        magnetization, ramp_coeffs, rdfc_kernel, pixel_size_nm,
    )


def reconstruct_2d_ensemble(
    phase,
    masks,
    pixel_size=None,
    b0=None,
    lam=1e-3,
    solver="newton_cg",
    reg_masks=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
    solver_config=None,
    *,
    pixel_size_nm=None,
    b0_tesla=None,
):
    """Batched MBIR reconstruction over an ensemble of bootstrap masks.

    Runs :func:`reconstruct_2d` for each mask in the ensemble using
    ``jax.vmap`` for efficient parallel execution on GPU.

    Parameters
    ----------
    phase : array_like
        Measured phase image of shape ``(H, W)`` in **radians**.
    masks : array_like
        Bootstrap mask ensemble of shape ``(N_boot, H, W)``.
    pixel_size_nm : float
        Pixel size in **nanometres** (nm).  Must be positive.
    b0_tesla : float, optional
        Saturation induction in **Tesla**, default 1.0.
        See :class:`SolverResult` for how this affects units.
    lam : float, optional
        Regularization weight (``lambda_exchange``), default 1e-3.
    solver : str or SolverConfig, optional
        Solver selection string (``"newton_cg"``, ``"adam"``,
        ``"lbfgs"``) or a :class:`SolverConfig` instance.
        Ignored when *solver_config* is provided.
        Default is ``"newton_cg"``.
    reg_masks : array_like, optional
        Separate regularization masks of shape ``(N_boot, H, W)``.
        Defaults to *masks*.
    geometry : str, optional
        Voxel geometry for the RDFC kernel (``"disc"`` or
        ``"slab"``), default ``"disc"``.
    prw_vec : array_like, optional
        Projected reference wave vector ``(v, u)``.
    rdfc_kernel : dict, optional
        Pre-built RDFC kernel from :func:`build_rdfc_kernel`.
        Built automatically when not provided.
    solver_config : SolverConfig, optional
        Explicit solver configuration object.  When provided,
        the *solver* string argument is ignored.

    Returns
    -------
    jax.Array
        Reconstructed magnetization ensemble of shape
        ``(N_boot, H, W, 2)``.

    Notes
    -----
    For iterative solvers (Adam, L-BFGS) early stopping is
    effectively disabled under ``vmap``; all bootstrap samples
    run for the maximum number of steps.
    """
    pixel_size_nm = _resolve_pixel_size_nm(pixel_size, pixel_size_nm)
    b0_tesla = _resolve_b0_tesla(b0, b0_tesla)
    _validate_positive(pixel_size_nm, "pixel_size_nm")

    phase = jnp.asarray(phase)
    masks = jnp.asarray(masks)
    if reg_masks is None:
        reg_masks = masks
    else:
        reg_masks = jnp.asarray(reg_masks)

    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(
            phase.shape,
            b0_tesla=b0_tesla,
            geometry=geometry,
            prw_vec=prw_vec,
        )

    # Resolve solver config once (Python-level dispatch, outside vmap)
    if solver_config is not None:
        config = solver_config
    elif isinstance(solver, str):
        solver_name = solver.lower()
        if solver_name not in _SOLVER_DEFAULTS:
            raise ValueError(
                f"Unknown solver {solver!r}. "
                f"Choose from {list(_SOLVER_DEFAULTS)}"
            )
        config = _SOLVER_DEFAULTS[solver_name]()
    elif isinstance(solver, (NewtonCGConfig, AdamConfig, LBFGSConfig)):
        config = solver
    else:
        raise TypeError(
            f"solver must be a string or a SolverConfig instance, "
            f"got {type(solver)}"
        )

    init_mag = jnp.zeros((*phase.shape, 2), dtype=jnp.float64)
    reg_config = {"lambda_exchange": float(lam)}

    # Build a vmappable function for the chosen solver
    if isinstance(config, NewtonCGConfig):
        def _solve_single(mask, reg_mask):
            (mag, _ramp), _loss = _run_newton_cg_solver_2d(
                phase=phase,
                init_mag=init_mag,
                mask=mask,
                pixel_size_nm=pixel_size_nm,
                reg_config=reg_config,
                rdfc_kernel=rdfc_kernel,
                cg_tol=config.cg_tol,
                cg_maxiter=config.cg_maxiter,
                reg_mask=reg_mask,
            )
            return mag
    elif isinstance(config, AdamConfig):
        def _solve_single(mask, reg_mask):
            (mag, _ramp), _loss = _run_adam_solver_2d(
                phase=phase,
                init_mag=init_mag,
                mask=mask,
                pixel_size_nm=pixel_size_nm,
                reg_config=reg_config,
                num_steps=config.num_steps,
                learning_rate=config.learning_rate,
                rdfc_kernel=rdfc_kernel,
                patience=config.patience,
                min_delta=config.min_delta,
                reg_mask=reg_mask,
            )
            return mag
    elif isinstance(config, LBFGSConfig):
        def _solve_single(mask, reg_mask):
            (mag, _ramp), _loss = _run_lbfgs_solver_2d(
                phase=phase,
                init_mag=init_mag,
                mask=mask,
                pixel_size_nm=pixel_size_nm,
                reg_config=reg_config,
                num_steps=config.num_steps,
                rdfc_kernel=rdfc_kernel,
                patience=config.patience,
                min_delta=config.min_delta,
                reg_mask=reg_mask,
            )
            return mag

    solve_batch = jax.jit(jax.vmap(_solve_single, in_axes=(0, 0)))
    return solve_batch(masks, reg_masks)


# Mapping from projection axis to (sum_axis, coeff_matrix, need_transpose).
# coeff maps (mx, my, mz) -> (u, v) following pyramid's SimpleProjector.
_SIMPLE_PROJ = {
    "z": {
        "sum_axis": 0,
        "coeff": [[1, 0, 0], [0, 1, 0]],   # u=mx, v=my
        "transpose": False,                  # (Y, X) is already (V, U)
    },
    "y": {
        "sum_axis": 1,
        "coeff": [[1, 0, 0], [0, 0, 1]],   # u=mx, v=mz
        "transpose": False,                  # (Z, X) is already (V, U)
    },
    "x": {
        "sum_axis": 2,
        "coeff": [[0, 0, 1], [0, 1, 0]],   # u=mz, v=my
        "transpose": True,                   # (Z, Y) -> (Y, Z) = (V, U)
    },
}


def project_3d(
    magnetization_3d,
    axis="z",
):
    """Project a 3D magnetization field along a major axis.

    Implements the simple-projector case from pyramid: sum along
    the projection axis and mix (mx, my, mz) into (u, v) components.

    Parameters
    ----------
    magnetization_3d : array_like
        3D magnetization of shape ``(Z, Y, X, 3)`` where the last
        axis holds ``(mx, my, mz)`` components.
    axis : {'z', 'y', 'x'}, optional
        Projection axis, default ``'z'``.

    Returns
    -------
    jax.Array
        Projected 2D magnetization of shape ``(V, U, 2)`` where
        the last axis holds ``(u, v)`` components suitable for
        :func:`phase_mapper_rdfc`.
    """
    axis = axis.lower()
    if axis not in _SIMPLE_PROJ:
        raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}")

    cfg = _SIMPLE_PROJ[axis]
    magnetization_3d = jnp.asarray(magnetization_3d)

    # Sum along projection direction: (Z, Y, X, 3) -> (*, *, 3)
    summed = jnp.sum(magnetization_3d, axis=cfg["sum_axis"])

    # Mix (mx, my, mz) -> (u, v) via coefficient matrix
    coeff = jnp.array(cfg["coeff"], dtype=summed.dtype)  # (2, 3)
    projected = jnp.einsum("...c,oc->...o", summed, coeff)  # (*, *, 2)

    if cfg["transpose"]:
        projected = jnp.transpose(projected, (1, 0, 2))

    return projected


def forward_model_3d(
    magnetization_3d,
    pixel_size=None,
    b0=None,
    axis="z",
    ramp_coeffs=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
    *,
    pixel_size_nm=None,
    b0_tesla=None,
):
    """Convenience forward model for a 3D magnetization volume.

    Projects a 3D magnetization field along a major axis using
    :func:`project_3d` (simple projector), then computes the
    magnetic phase shift via RDFC.

    Parameters
    ----------
    magnetization_3d : array_like
        3D magnetization of shape ``(Z, Y, X, 3)`` where the last
        axis holds ``(mx, my, mz)`` components.
    pixel_size_nm : float
        Voxel size in nanometres.
    b0_tesla : float, optional
        Magnetic induction in Tesla, default 1.0.
    axis : {'z', 'y', 'x'}, optional
        Projection axis, default ``'z'``.
    ramp_coeffs : array_like, optional
        Background ramp coefficients ``[offset, slope_y, slope_x]``.
        Defaults to zeros (no ramp).
    geometry : str, optional
        Voxel geometry for the RDFC kernel (``"disc"`` or
        ``"slab"``), default ``"disc"``.
    prw_vec : array_like, optional
        Projected reference wave vector ``(v, u)``.
    rdfc_kernel : dict, optional
        Pre-built RDFC kernel from :func:`build_rdfc_kernel`.
        Built automatically when not provided.

    Returns
    -------
    jax.Array
        Predicted phase image of shape ``(V, U)``.
    """
    projected = project_3d(magnetization_3d, axis=axis)

    return forward_model_2d(
        projected,
        pixel_size_nm=_resolve_pixel_size_nm(pixel_size, pixel_size_nm),
        b0_tesla=_resolve_b0_tesla(b0, b0_tesla),
        ramp_coeffs=ramp_coeffs,
        geometry=geometry,
        prw_vec=prw_vec,
        rdfc_kernel=rdfc_kernel,
    )


def decompose_loss(
    magnetization,
    ramp_coeffs,
    phase,
    mask,
    reg_mask,
    rdfc_kernel,
    pixel_size=None,
    *,
    pixel_size_nm=None,
    pyramid_compat=False,
):
    """Decompose the MBIR loss into data-fidelity and regularization terms.

    Evaluates the two components of the loss **without** the
    ``lambda_exchange`` multiplier on the regularization term, so
    that they can be compared on an L-curve plot.

    Parameters
    ----------
    magnetization : array_like
        Reconstructed magnetization of shape ``(N, M, 2)``.
    ramp_coeffs : array_like
        Background ramp coefficients ``[offset, slope_y, slope_x]``.
    phase : array_like
        Observed phase image of shape ``(N, M)``.
    mask : array_like
        Binary mask of shape ``(N, M)`` applied to the
        magnetization before the forward model.
    reg_mask : array_like
        Regularization mask of shape ``(N, M)`` passed to
        :func:`exchange_loss_fn`.
    rdfc_kernel : dict
        Pre-built RDFC kernel from :func:`build_rdfc_kernel`.
    pixel_size_nm : float
        Pixel size in nanometres.
    pyramid_compat : bool, optional
        If *True*, compute the regularization norm using
        :func:`forward_diff_norm` (simple forward differences, no
        per-pixel normalization), which matches Pyramid's
        ``FirstOrderRegularisator`` convention.  Default *False*
        uses :func:`exchange_loss_fn` (adaptive stencil with
        neighbor-count normalization).

    Returns
    -------
    data_misfit : float
        ``sum((predicted - observed)**2)`` — the squared-residual
        norm, matching Pyramid's ``chisq_m`` convention (no 1/2
        factor).
    exchange_norm : float
        Unweighted exchange regularization norm (no lambda
        multiplier).  When *pyramid_compat=True* this is
        ``||Dx||²`` (forward differences); otherwise it uses the
        adaptive stencil from :func:`exchange_loss_fn`.
    """
    pixel_size_nm = _resolve_pixel_size_nm(pixel_size, pixel_size_nm)
    magnetization = jnp.asarray(magnetization)
    ramp_coeffs = jnp.asarray(ramp_coeffs)
    phase = jnp.asarray(phase)
    mask = jnp.asarray(mask)
    reg_mask = jnp.asarray(reg_mask)

    masked_mag = jnp.stack([
        magnetization[..., 0] * mask,
        magnetization[..., 1] * mask,
    ], axis=-1)

    predicted = forward_model_single_rdfc_2d(
        masked_mag, ramp_coeffs, rdfc_kernel, pixel_size_nm,
    )
    residuals = predicted - phase
    data_misfit = float(jnp.sum(residuals ** 2))
    if pyramid_compat:
        exchange_norm = float(forward_diff_norm(masked_mag, reg_mask))
    else:
        exchange_norm = float(exchange_loss_fn(masked_mag, reg_mask))

    return data_misfit, exchange_norm

def bootstrap_threshold_uncertainty_2d(
    phase,
    mip_phase,
    threshold,
    pixel_size=None,
    b0=None,
    lam=1e-3,
    solver="newton_cg",
    n_boot=50,
    threshold_low=None,
    threshold_high=None,
    rng_seed=0,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
    solver_config=None,
    *,
    pixel_size_nm=None,
    b0_tesla=None,
):
    """Bootstrap a thresholded mask ensemble and summarize the uncertainty.

    Threshold draws are sampled uniformly from ``[threshold_low, threshold_high]``.
    Each draw defines a binary mask from ``abs(mip_phase) > threshold_draw``.
    The masks are passed through :func:`reconstruct_2d_ensemble` and the
    resulting magnetization ensemble is summarized with percentile maps for
    ``|M|``.

    Parameters
    ----------
    phase
        Observed phase image of shape ``(H, W)``.
    mip_phase
        MIP phase image used for thresholding, shape ``(H, W)``.
    threshold
        Central threshold value around which the bootstrap draws are sampled.
    pixel_size_nm
        Pixel size in nanometres.
    b0_tesla : float, optional
        Magnetic induction in Tesla, default 1.0.
    lam : float, optional
        Regularization weight (``lambda_exchange``), default 1e-3.
    solver : str or SolverConfig, optional
        Solver selection string or config object.  Ignored when
        *solver_config* is provided.  Default ``"newton_cg"``.
    n_boot : int, optional
        Number of threshold draws, default 50.
    threshold_low : float, optional
        Lower bound for the threshold draws.  Defaults to
        ``threshold - 0.25``.
    threshold_high : float, optional
        Upper bound for the threshold draws.  Defaults to
        ``threshold + 0.25``.
    rng_seed : int, optional
        Seed for the pseudo-random number generator, default 0.
    geometry : str, optional
        Voxel geometry for the RDFC kernel, default ``"disc"``.
    prw_vec : array_like, optional
        Projected reference wave vector ``(v, u)``.
    rdfc_kernel : dict, optional
        Pre-built RDFC kernel from :func:`build_rdfc_kernel`.
        Built automatically when ``None``.
    solver_config : SolverConfig, optional
        Explicit solver configuration object.

    Returns
    -------
    BootstrapThresholdResult
        Summary object containing the threshold draws, reconstructed
        magnetizations, 2.5th and 97.5th percentile maps, their 95% width,
        the relative 95% width, and the mask inclusion frequency.
    """
    pixel_size_nm = _resolve_pixel_size_nm(pixel_size, pixel_size_nm)
    b0_tesla = _resolve_b0_tesla(b0, b0_tesla)
    phase = jnp.asarray(phase)
    mip_phase = jnp.asarray(mip_phase)
    if phase.shape != mip_phase.shape:
        raise ValueError(
            f"phase and mip_phase must have the same shape; got {phase.shape} and {mip_phase.shape}."
        )

    if threshold_low is None:
        threshold_low = float(threshold) - 0.25
    if threshold_high is None:
        threshold_high = float(threshold) + 0.25
    if threshold_high <= threshold_low:
        raise ValueError(
            f"threshold_high must be greater than threshold_low; got {threshold_low} and {threshold_high}."
        )

    rng = np.random.default_rng(rng_seed)
    threshold_draws = rng.uniform(low=threshold_low, high=threshold_high, size=n_boot)

    mip_abs = np.abs(np.asarray(mip_phase))
    reference_mask = (mip_abs > threshold).astype(np.float64)
    bootstrap_masks = (
        mip_abs[None, ...] > threshold_draws[:, None, None]
    ).astype(np.float64)

    for draw_index in range(n_boot):
        if bootstrap_masks[draw_index].sum() == 0:
            bootstrap_masks[draw_index] = reference_mask

    bootstrap_mag = reconstruct_2d_ensemble(
        phase=phase,
        masks=bootstrap_masks,
        pixel_size_nm=pixel_size_nm,
        b0_tesla=b0_tesla,
        lam=lam,
        solver=solver,
        reg_masks=bootstrap_masks,
        geometry=geometry,
        prw_vec=prw_vec,
        rdfc_kernel=rdfc_kernel,
        solver_config=solver_config,
    )

    bootstrap_mag = jnp.asarray(bootstrap_mag)
    mean_magnetization = jnp.mean(bootstrap_mag, axis=0)
    mean_norm = np.linalg.norm(np.asarray(mean_magnetization), axis=-1)

    norm_samples = np.linalg.norm(np.asarray(bootstrap_mag), axis=-1)
    norm_low = np.percentile(norm_samples, 2.5, axis=0)
    norm_high = np.percentile(norm_samples, 97.5, axis=0)
    norm_ci95 = norm_high - norm_low
    relative_ci95 = norm_ci95 / (mean_norm + 1e-12)
    mask_frequency = bootstrap_masks.mean(axis=0)

    return BootstrapThresholdResult(
        threshold=threshold,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
        threshold_draws=threshold_draws,
        magnetizations=bootstrap_mag,
        mean_magnetization=mean_magnetization,
        mean_norm=mean_norm,
        norm_low=norm_low,
        norm_high=norm_high,
        norm_ci95=norm_ci95,
        relative_ci95=relative_ci95,
        mask_frequency=mask_frequency,
    )


def plot_bootstrap_threshold_uncertainty(result: BootstrapThresholdResult):
    """Plot the summary produced by :func:`bootstrap_threshold_uncertainty_2d`."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(16, 11), constrained_layout=True)

    def set_panel_title(ax, title, subtitle):
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)

    display_mask = result.mask_frequency > 0.5
    if not np.any(display_mask):
        display_mask = np.ones_like(result.mean_norm, dtype=bool)

    mean_region = result.mean_norm[display_mask]
    low_region = result.norm_low[display_mask]
    high_region = result.norm_high[display_mask]
    ci_region = result.norm_ci95[display_mask]
    rel_region = 100.0 * result.relative_ci95[display_mask]

    vmax_mean = float(np.percentile(mean_region, 99))
    vmax_low = float(np.percentile(low_region, 99))
    vmax_high = float(np.percentile(high_region, 99))
    vmax_ci = float(np.percentile(ci_region, 99))
    vmax_rel = float(np.percentile(rel_region, 99))

    im = axs[0, 0].imshow(
        result.mean_norm, cmap="viridis", origin="lower", vmin=0, vmax=vmax_mean
    )
    set_panel_title(
        axs[0, 0],
        "Bootstrap mean |M|",
        "Typical magnitude across draws.",
    )
    plt.colorbar(im, ax=axs[0, 0], fraction=0.046)

    im = axs[0, 1].imshow(
        result.norm_low, cmap="viridis", origin="lower", vmin=0, vmax=vmax_low
    )
    set_panel_title(
        axs[0, 1],
        "2.5% percentile of |M|",
        "Lower 95% interval bound.",
    )
    plt.colorbar(im, ax=axs[0, 1], fraction=0.046)

    im = axs[0, 2].imshow(
        result.norm_high, cmap="viridis", origin="lower", vmin=0, vmax=vmax_high
    )
    set_panel_title(
        axs[0, 2],
        "97.5% percentile of |M|",
        "Upper 95% interval bound.",
    )
    plt.colorbar(im, ax=axs[0, 2], fraction=0.046)

    im = axs[1, 0].imshow(
        result.norm_ci95, cmap="magma", origin="lower", vmin=0, vmax=vmax_ci
    )
    set_panel_title(
        axs[1, 0],
        "95% CI width of |M|",
        "Larger values mean more spread.",
    )
    plt.colorbar(im, ax=axs[1, 0], fraction=0.046)

    im = axs[1, 1].imshow(
        100.0 * result.relative_ci95,
        cmap="cividis",
        origin="lower",
        vmin=0,
        vmax=vmax_rel,
    )
    set_panel_title(
        axs[1, 1],
        "Relative 95% CI width of |M| (%)",
        "Width divided by the local mean.",
    )
    plt.colorbar(im, ax=axs[1, 1], fraction=0.046)

    im = axs[1, 2].imshow(
        result.mask_frequency, cmap="gray", origin="lower", vmin=0, vmax=1
    )
    set_panel_title(
        axs[1, 2],
        "Threshold inclusion frequency",
        "Near 1 is stable; near 0 is rare.",
    )
    plt.colorbar(im, ax=axs[1, 2], fraction=0.046)

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig, axs


def kneedle_corner(data_misfits, reg_norms):
    """Find the L-curve corner using the Kneedle algorithm.

    Projects points in log-log space onto the line connecting the
    two endpoints and returns the index of the point with the
    largest perpendicular distance (the "elbow").

    Parameters
    ----------
    data_misfits : array_like
        1D array of data-fidelity values, one per lambda.
    reg_norms : array_like
        1D array of regularization norms (unweighted), one per
        lambda.

    Returns
    -------
    corner_index : int
        Index into the input arrays of the detected corner point.
        Returns ``-1`` when fewer than 3 points are provided.
    score : float
        Maximum normalized perpendicular distance.  Larger values
        indicate a more pronounced corner.
    """
    x = np.log10(np.asarray(reg_norms, dtype=np.float64))
    y = np.log10(np.asarray(data_misfits, dtype=np.float64))

    if len(x) < 3:
        return -1, 0.0

    order = np.argsort(x)
    x, y = x[order], y[order]

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    if x_range == 0 or y_range == 0:
        return -1, 0.0

    x_n = (x - x.min()) / (x_range + 1e-30)
    y_n = (y - y.min()) / (y_range + 1e-30)

    # Perpendicular distance from the line joining first and last point
    x1, y1 = x_n[0], y_n[0]
    x2, y2 = x_n[-1], y_n[-1]
    line_len = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-30
    d = np.abs(
        (y2 - y1) * x_n - (x2 - x1) * y_n + x2 * y1 - y2 * x1
    ) / line_len

    best_sorted = int(np.argmax(d))
    return int(order[best_sorted]), float(d[best_sorted])


def lcurve_sweep(
    phase,
    mask,
    pixel_size=None,
    lambdas=None,
    b0=None,
    thickness=None,
    solver="newton_cg",
    reg_mask=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
    solver_config=None,
    pyramid_compat=False,
    *,
    pixel_size_nm=None,
    b0_tesla=None,
):
    """Sequential L-curve sweep over regularization weights.

    Runs :func:`solve_mbir_2d` for each value in *lambdas*,
    collects the data-fidelity and regularization norms, and
    detects the L-curve corner via :func:`kneedle_corner`.

    Parameters
    ----------
    phase : array_like
        Measured phase image of shape ``(N, M)`` in **radians**.
    mask : array_like
        Binary mask of shape ``(N, M)``.
    pixel_size_nm : float
        Pixel size in **nanometres** (nm).  Must be positive.
    lambdas : array_like
        1D array of ``lambda_exchange`` values to sweep.
    b0_tesla : float
        Saturation induction :math:`B_0 = \\mu_0 M_s` in **Tesla**.
        See :class:`SolverResult` for how this affects units.
    thickness : float or array_like
        Sample thickness in **nanometres** (nm).  The returned
        magnetizations are divided by the thickness in voxels
        (``thickness / pixel_size_nm``) to give per-voxel values.
        May be a scalar or a 2-D map of shape ``(N, M)``.
    solver : str or SolverConfig, optional
        Solver selection string or config object.  Ignored when
        *solver_config* is provided.  Default ``"newton_cg"``.
    reg_mask : array_like, optional
        Regularization mask of shape ``(N, M)``.  Defaults to
        *mask*.
    geometry : str, optional
        Voxel geometry for the RDFC kernel, default ``"disc"``.
    prw_vec : array_like, optional
        Projected reference wave vector ``(v, u)``.
    rdfc_kernel : dict, optional
        Pre-built RDFC kernel.  Built automatically when ``None``.
    solver_config : SolverConfig, optional
        Explicit solver configuration.
    pyramid_compat : bool, optional
        If *True*, compute the regularization norm using
        :func:`forward_diff_norm` (simple forward differences)
        instead of :func:`exchange_loss_fn`, to match Pyramid's
        ``FirstOrderRegularisator`` convention.  Default *False*.

    Returns
    -------
    LCurveResult
        Named tuple with ``lambdas``, ``data_misfits``,
        ``reg_norms``, ``magnetizations``, ``ramp_coeffs``, and
        ``corner_index``.
    """
    pixel_size_nm = _resolve_pixel_size_nm(pixel_size, pixel_size_nm)
    b0_tesla = _resolve_b0_tesla(b0, b0_tesla)
    thickness = _resolve_thickness_nm(thickness)
    _validate_positive(pixel_size_nm, "pixel_size_nm")
    _validate_positive(thickness, "thickness")

    phase = jnp.asarray(phase)
    mask = jnp.asarray(mask, dtype=bool)
    if reg_mask is None:
        reg_mask = mask
    else:
        reg_mask = jnp.asarray(reg_mask)

    lambdas = np.atleast_1d(np.asarray(lambdas, dtype=np.float64))

    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(
            phase.shape,
            b0_tesla=b0_tesla,
            geometry=geometry,
            prw_vec=prw_vec,
        )

    actual_solver = solver_config if solver_config is not None else solver

    data_misfits = []
    reg_norms = []
    mag_list = []
    ramp_list = []

    init_mag = jnp.zeros((*phase.shape, 2), dtype=jnp.float64)

    for lam in lambdas:
        reg_config = {"lambda_exchange": float(lam)}

        result = solve_mbir_2d(
            phase=phase,
            init_mag=init_mag,
            mask=mask,
            pixel_size_nm=pixel_size_nm,
            solver=actual_solver,
            reg_config=reg_config,
            rdfc_kernel=rdfc_kernel,
            reg_mask=reg_mask,
        )

        dm, rn = decompose_loss(
            result.magnetization, result.ramp_coeffs,
            phase, mask, reg_mask, rdfc_kernel, pixel_size_nm,
            pyramid_compat=pyramid_compat,
        )
        data_misfits.append(dm)
        reg_norms.append(rn)
        mag_list.append(result.magnetization)
        ramp_list.append(result.ramp_coeffs)

    data_misfits = np.array(data_misfits)
    reg_norms = np.array(reg_norms)
    corner_idx, _ = kneedle_corner(data_misfits, reg_norms)

    all_mag = jnp.stack(mag_list)
    thickness = jnp.asarray(thickness, dtype=jnp.float64)
    thickness_vox = thickness / pixel_size_nm
    if thickness_vox.ndim >= 2:
        thickness_vox = thickness_vox[..., jnp.newaxis]
    all_mag = all_mag / thickness_vox

    return LCurveResult(
        lambdas=lambdas,
        data_misfits=data_misfits,
        reg_norms=reg_norms,
        magnetizations=all_mag,
        ramp_coeffs=jnp.stack(ramp_list),
        corner_index=corner_idx,
    )


def lcurve_sweep_vmap(
    phase,
    mask,
    pixel_size=None,
    lambdas=None,
    b0=None,
    thickness=None,
    solver="newton_cg",
    reg_mask=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
    solver_config=None,
    pyramid_compat=False,
    *,
    pixel_size_nm=None,
    b0_tesla=None,
):
    """Parallel L-curve sweep using ``jax.vmap`` over lambda values.

    Runs all reconstructions in parallel (no warm-starting).
    This is faster on GPU when many lambda values are evaluated,
    but uses more memory than :func:`lcurve_sweep`.

    Parameters
    ----------
    phase : array_like
        Measured phase image of shape ``(N, M)`` in **radians**.
    mask : array_like
        Binary mask of shape ``(N, M)``.
    pixel_size_nm : float
        Pixel size in **nanometres** (nm).  Must be positive.
    lambdas : array_like
        1D array of ``lambda_exchange`` values to sweep.
    b0_tesla : float
        Saturation induction :math:`B_0 = \\mu_0 M_s` in **Tesla**.
        See :class:`SolverResult` for how this affects units.
    thickness : float or array_like
        Sample thickness in **nanometres** (nm).  The returned
        magnetizations are divided by the thickness in voxels
        (``thickness / pixel_size_nm``) to give per-voxel values.
        May be a scalar or a 2-D map of shape ``(N, M)``.
    solver : str or SolverConfig, optional
        Solver selection string or config object.  Ignored when
        *solver_config* is provided.  Default ``"newton_cg"``.
    reg_mask : array_like, optional
        Regularization mask of shape ``(N, M)``.  Defaults to
        *mask*.
    geometry : str, optional
        Voxel geometry for the RDFC kernel, default ``"disc"``.
    prw_vec : array_like, optional
        Projected reference wave vector ``(v, u)``.
    rdfc_kernel : dict, optional
        Pre-built RDFC kernel.  Built automatically when ``None``.
    solver_config : SolverConfig, optional
        Explicit solver configuration.
    pyramid_compat : bool, optional
        If *True*, compute the regularization norm using
        :func:`forward_diff_norm` (simple forward differences)
        instead of :func:`exchange_loss_fn`, to match Pyramid's
        ``FirstOrderRegularisator`` convention.  Default *False*.

    Returns
    -------
    LCurveResult
        Named tuple with ``lambdas``, ``data_misfits``,
        ``reg_norms``, ``magnetizations``, ``ramp_coeffs``, and
        ``corner_index``.

    Notes
    -----
    For iterative solvers (Adam, L-BFGS) early stopping is
    effectively disabled under ``vmap``; all lambda values run for
    the maximum number of steps.
    """
    pixel_size_nm = _resolve_pixel_size_nm(pixel_size, pixel_size_nm)
    b0_tesla = _resolve_b0_tesla(b0, b0_tesla)
    thickness = _resolve_thickness_nm(thickness)
    _validate_positive(pixel_size_nm, "pixel_size_nm")
    _validate_positive(thickness, "thickness")

    phase = jnp.asarray(phase)
    mask = jnp.asarray(mask, dtype=bool)
    if reg_mask is None:
        reg_mask = mask
    else:
        reg_mask = jnp.asarray(reg_mask)

    lambdas_np = np.atleast_1d(np.asarray(lambdas, dtype=np.float64))
    lambdas_jax = jnp.asarray(lambdas_np)

    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(
            phase.shape,
            b0_tesla=b0_tesla,
            geometry=geometry,
            prw_vec=prw_vec,
        )

    if solver_config is not None:
        config = solver_config
    elif isinstance(solver, str):
        solver_name = solver.lower()
        if solver_name not in _SOLVER_DEFAULTS:
            raise ValueError(
                f"Unknown solver {solver!r}. "
                f"Choose from {list(_SOLVER_DEFAULTS)}"
            )
        config = _SOLVER_DEFAULTS[solver_name]()
    elif isinstance(solver, (NewtonCGConfig, AdamConfig, LBFGSConfig)):
        config = solver
    else:
        raise TypeError(
            f"solver must be a string or a SolverConfig instance, "
            f"got {type(solver)}"
        )

    init_mag = jnp.zeros((*phase.shape, 2), dtype=jnp.float64)

    # Build a vmappable function for the chosen solver.
    if isinstance(config, NewtonCGConfig):
        def _solve_for_lam(lam):
            reg_config = {"lambda_exchange": lam}
            (mag, ramp), _loss = _run_newton_cg_solver_2d(
                phase=phase,
                init_mag=init_mag,
                mask=mask,
                pixel_size_nm=pixel_size_nm,
                reg_config=reg_config,
                rdfc_kernel=rdfc_kernel,
                cg_tol=config.cg_tol,
                cg_maxiter=config.cg_maxiter,
                reg_mask=reg_mask,
            )
            return mag, ramp
    elif isinstance(config, AdamConfig):
        def _solve_for_lam(lam):
            reg_config = {"lambda_exchange": lam}
            (mag, ramp), _loss = _run_adam_solver_2d(
                phase=phase,
                init_mag=init_mag,
                mask=mask,
                pixel_size_nm=pixel_size_nm,
                reg_config=reg_config,
                num_steps=config.num_steps,
                learning_rate=config.learning_rate,
                rdfc_kernel=rdfc_kernel,
                patience=config.patience,
                min_delta=config.min_delta,
                reg_mask=reg_mask,
            )
            return mag, ramp
    elif isinstance(config, LBFGSConfig):
        def _solve_for_lam(lam):
            reg_config = {"lambda_exchange": lam}
            (mag, ramp), _loss = _run_lbfgs_solver_2d(
                phase=phase,
                init_mag=init_mag,
                mask=mask,
                pixel_size_nm=pixel_size_nm,
                reg_config=reg_config,
                num_steps=config.num_steps,
                rdfc_kernel=rdfc_kernel,
                patience=config.patience,
                min_delta=config.min_delta,
                reg_mask=reg_mask,
            )
            return mag, ramp

    solve_batch = jax.jit(jax.vmap(_solve_for_lam))
    all_mag, all_ramp = solve_batch(lambdas_jax)

    # Decompose losses (vmapped)
    _norm_fn = forward_diff_norm if pyramid_compat else exchange_loss_fn

    def _decompose_single(mag, ramp):
        masked_mag = jnp.stack([
            mag[..., 0] * mask,
            mag[..., 1] * mask,
        ], axis=-1)
        predicted = forward_model_single_rdfc_2d(
            masked_mag, ramp, rdfc_kernel, pixel_size_nm,
        )
        residuals = predicted - phase
        dm = jnp.sum(residuals ** 2)
        rn = _norm_fn(masked_mag, reg_mask)
        return dm, rn

    decompose_batch = jax.jit(jax.vmap(_decompose_single))
    all_dm, all_rn = decompose_batch(all_mag, all_ramp)

    data_misfits = np.asarray(all_dm)
    reg_norms = np.asarray(all_rn)
    corner_idx, _ = kneedle_corner(data_misfits, reg_norms)

    thickness = jnp.asarray(thickness, dtype=jnp.float64)
    thickness_vox = thickness / pixel_size_nm
    if thickness_vox.ndim >= 2:
        thickness_vox = thickness_vox[..., jnp.newaxis]
    all_mag = all_mag / thickness_vox

    return LCurveResult(
        lambdas=lambdas_np,
        data_misfits=data_misfits,
        reg_norms=reg_norms,
        magnetizations=all_mag,
        ramp_coeffs=all_ramp,
        corner_index=corner_idx,
    )


def plot_lcurve(
    lcurve_result,
    pyramid_style=False,
    ax=None,
    cmap="nipy_spectral",
    colorbar=True,
    **kwargs,
):
    """
    Plot an L-curve from an LCurveResult, with optional Pyramid-style axes.

    Parameters
    ----------
    lcurve_result : LCurveResult
        Result from lcurve_sweep or lcurve_sweep_vmap.
    pyramid_style : bool, optional
        If True, plot y = reg_norm / lambda vs x = data_misfit (Pyramid style).
        If False (default), plot y = data_misfit vs x = reg_norm (standard style).
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure is created.
    cmap : str, optional
        Colormap for lambda values.
    colorbar : bool, optional
        Whether to show a colorbar. Default True.
    **kwargs :
        Additional arguments passed to scatter/plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the plot.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    lambdas = np.asarray(lcurve_result.lambdas)
    reg_norms = np.asarray(lcurve_result.reg_norms)
    data_misfits = np.asarray(lcurve_result.data_misfits)

    if pyramid_style:
        x = data_misfits
        y = reg_norms
        xlabel = r"$\Vert\mathbf{F}(\mathbf{x})-\mathbf{y}\Vert^2$"
        ylabel = r"$\Vert\mathbf{D}\mathbf{x}\Vert^2$"
    else:
        x = reg_norms
        y = data_misfits
        xlabel = "Regularisation norm (exchange)"
        ylabel = "Data misfit"

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(x, y, c=lambdas, cmap=cmap, norm=LogNorm(), s=80, zorder=2, **kwargs)
    ax.plot(x, y, "k-", lw=1.5, zorder=1)
    if hasattr(lcurve_result, "corner_index") and lcurve_result.corner_index >= 0:
        ax.plot(
            x[lcurve_result.corner_index],
            y[lcurve_result.corner_index],
            "r*", ms=18, zorder=3,
            label=f"corner λ={lambdas[lcurve_result.corner_index]:.2e}",
        )
        ax.legend(frameon=False, fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title("MBIR L-curve" + (" (Pyramid style)" if pyramid_style else ""))
    ax.grid(alpha=0.25)
    if colorbar:
        plt.colorbar(sc, ax=ax, label=r"$\lambda$")
    return ax
