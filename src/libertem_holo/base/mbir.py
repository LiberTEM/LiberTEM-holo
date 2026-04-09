"""Model-based iterative reconstruction (MBIR) for 2D projected magnetization."""

from __future__ import annotations

import dataclasses
from typing import NamedTuple, Union

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax
import numpy as np

PHI_0_T_NM2 = 2067.83  # flux quantum in T*nm^2


class SolverResult(NamedTuple):
    """Result returned by :func:`solve_mbir_2d`."""
    magnetization: jnp.ndarray
    ramp_coeffs: jnp.ndarray
    loss_history: jnp.ndarray


class LCurveResult(NamedTuple):
    """Result returned by :func:`lcurve_sweep` and :func:`lcurve_sweep_vmap`."""
    lambdas: np.ndarray
    data_misfits: np.ndarray
    reg_norms: np.ndarray
    magnetizations: jnp.ndarray
    ramp_coeffs: jnp.ndarray
    corner_index: int


@dataclasses.dataclass(frozen=True)
class NewtonCGConfig:
    """Configuration for the Newton-CG solver."""
    num_steps: int = 50
    cg_tol: float = 1e-5


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


def get_freq_grid(
    height: int,
    width: int,
    voxel_size_nm: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build frequency grids for FFT-based phase propagation.

    Parameters
    ----------
    height
        Number of pixels along the y-axis.
    width
        Number of pixels along the x-axis.
    voxel_size_nm
        Pixel size in nanometres, used as the FFT sampling interval.

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
    fy = jnp.fft.fftfreq(height, d=voxel_size_nm)
    fx = jnp.fft.rfftfreq(width, d=voxel_size_nm)
    f_y, f_x = jnp.meshgrid(fy, fx, indexing="ij")
    denom = f_x**2 + f_y**2
    denom = jnp.where(denom == 0, 1.0, denom)
    return f_y, f_x, denom


def _rdfc_elementary_phase(
    geometry: str,
    n: jax.Array,
    m: jax.Array,
    voxel_size_nm: float,
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
    voxel_size_nm
        Pixel size in nanometres.

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
            radius2 = voxel_size_nm**2 * (n_val**2 + m_val**2) + 1e-30
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


def build_rdfc_kernel(
    voxel_size_nm: float,
    dim_uv: tuple[int, int],
    b0_tesla: float = 1.0,
    geometry: Literal["disc", "slab"] = "disc",
    prw_vec: jax.Array | None = None,
    dtype: Any = jnp.float64,
) -> dict[str, Any]:
    """Build an RDFC phase-mapping kernel in Fourier space.

    JAX translation of ``pyramid.Kernel`` for ``PhaseMapperRDFC``.

    Parameters
    ----------
    voxel_size_nm
        Pixel size in nanometres.
    dim_uv
        ``(height, width)`` of the magnetization field.
    b0_tesla
        External magnetic field strength in Tesla, default 1.0.
    geometry
        Voxel geometry used for the elementary phase kernel.
    prw_vec
        Optional projected reference wave vector ``(v, u)``.
        When provided, the kernel accounts for the reference
        wave tilt.
    dtype
        JAX dtype for the kernel arrays, default ``jnp.float64``.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``'u_fft'``, ``'v_fft'`` (FFT'd kernel
        components), ``'dim_uv'``, and ``'dim_pad'``.
    """
    height, width = dim_uv
    dim_kern = (2 * height - 1, 2 * width - 1)
    dim_pad = (2 * height, 2 * width)

    u_coords = jnp.linspace(-(width - 1), width - 1, num=dim_kern[1]).astype(dtype)
    v_coords = jnp.linspace(-(height - 1), height - 1, num=dim_kern[0]).astype(dtype)
    uu, vv = jnp.meshgrid(u_coords, v_coords, indexing="xy")

    coeff = b0_tesla * voxel_size_nm**2 / (2 * PHI_0_T_NM2)
    u_kernel = coeff * _rdfc_elementary_phase(geometry, uu, vv, voxel_size_nm)
    v_kernel = -coeff * _rdfc_elementary_phase(geometry, vv, uu, voxel_size_nm)

    if prw_vec is not None:
        uu_prw = uu + prw_vec[1]
        vv_prw = vv + prw_vec[0]
        u_kernel = u_kernel - coeff * _rdfc_elementary_phase(
            geometry, uu_prw, vv_prw, voxel_size_nm
        )
        v_kernel = v_kernel + coeff * _rdfc_elementary_phase(
            geometry, vv_prw, uu_prw, voxel_size_nm
        )

    u_pad = jnp.zeros(dim_pad, dtype=dtype)
    v_pad = jnp.zeros(dim_pad, dtype=dtype)
    u_pad = u_pad.at[:dim_kern[0], :dim_kern[1]].set(u_kernel)
    v_pad = v_pad.at[:dim_kern[0], :dim_kern[1]].set(v_kernel)

    u_fft = jnp.fft.rfft2(u_pad)
    v_fft = jnp.fft.rfft2(v_pad)

    return {
        "u_fft": u_fft,
        "v_fft": v_fft,
        "dim_uv": dim_uv,
        "dim_pad": dim_pad,
    }


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
    v_field
        In-plane magnetization component along y, shape ``(H, W)``.
    rdfc_kernel
        Kernel dictionary as returned by :func:`build_rdfc_kernel`.

    Returns
    -------
    jax.Array
        Magnetic phase-shift image of shape ``(H, W)``.
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


def apply_ramp(
    ramp_coeffs: jax.Array,
    height: int,
    width: int,
    voxel_size_nm: float,
) -> jax.Array:
    """Generate a first-order 2D polynomial background ramp.

    Parameters
    ----------
    ramp_coeffs
        Array of ``[offset, slope_y, slope_x]``.
    height
        Number of pixels along the y-axis.
    width
        Number of pixels along the x-axis.
    voxel_size_nm
        Pixel size in nanometres.

    Returns
    -------
    jax.Array
        Ramp image of shape ``(height, width)``.
    """
    y, x = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    ramp = ramp_coeffs[0]
    if ramp_coeffs.shape[0] > 1:
        ramp = ramp + ramp_coeffs[1] * (y * voxel_size_nm)
        ramp = ramp + ramp_coeffs[2] * (x * voxel_size_nm)
    return ramp


def forward_model_single_rdfc_2d(
    magnetization: jax.Array,
    ramp_coeffs: jax.Array,
    rdfc_kernel: dict[str, Any],
    voxel_size_nm: float,
) -> jax.Array:
    """RDFC forward model mapping projected magnetization to phase.

    Computes the magnetic phase shift from a 2D projected
    magnetization field and adds a polynomial background ramp.

    Parameters
    ----------
    magnetization
        In-plane magnetization of shape ``(N, M, 2)`` where the
        last axis holds the (u, v) components.
    ramp_coeffs
        Background ramp coefficients ``[offset, slope_y, slope_x]``.
    rdfc_kernel
        Kernel dictionary as returned by :func:`build_rdfc_kernel`.
    voxel_size_nm
        Pixel size in nanometres.

    Returns
    -------
    jax.Array
        Predicted phase image of shape ``(N, M)``.
    """
    height, width = magnetization.shape[:2]

    u_field = magnetization[..., 0]
    v_field = magnetization[..., 1]

    phase = phase_mapper_rdfc(u_field, v_field, rdfc_kernel)
    ramp = apply_ramp(ramp_coeffs, height, width, voxel_size_nm)

    return phase + ramp


def mbir_loss_2d(
    params: tuple[jax.Array, jax.Array],
    mask: jax.Array,
    phase: jax.Array,
    rdfc_kernel: dict[str, Any],
    voxel_size_nm: float,
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
    voxel_size_nm
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
        voxel_size_nm,
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
    voxel_size_nm: float,
    reg_config: dict[str, Any] | None = None,
    num_steps: int = 50,
    rdfc_kernel: dict[str, Any] | None = None,
    cg_tol: float = 1e-5,
    init_ramp_coeffs: jax.Array | None = None,
    reg_mask: jax.Array | None = None,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Minimize :func:`mbir_loss_2d` using a Newton-CG step.

    Computes a single Newton update by solving ``H @ delta = -g``
    with JAX's conjugate-gradient solver, where *H* is the Hessian
    and *g* the gradient of the loss.

    Parameters
    ----------
    phase
        Observed phase image of shape ``(H, W)``.
    init_mag
        Initial magnetization of shape ``(N, M, 2)``.
    mask
        Binary mask of shape ``(N, M)`` applied to the
        magnetization.
    voxel_size_nm
        Pixel size in nanometres.
    reg_config
        Regularization configuration dictionary (see
        :func:`mbir_loss_2d`), default ``{}``.
    num_steps
        Maximum number of CG iterations, default 50.
    rdfc_kernel
        Kernel dictionary as returned by :func:`build_rdfc_kernel`.
    cg_tol
        Tolerance for the CG solver, default 1e-5.
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
        Array of length 1 containing the final loss value.
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
            voxel_size_nm,
            reg_config,
            reg_mask=reg_mask,
        )

    loss_grad = jax.grad(objective_flat)

    def matvec_hvp(v):
        # Hessian-vector product: H @ v
        return jax.jvp(loss_grad, (x0_flat,), (v,))[1]

    # Calculate initial gradient and solve H*delta = -g
    grad_at_x0 = loss_grad(x0_flat)
    delta, info = jax.scipy.sparse.linalg.cg(
        matvec_hvp, -grad_at_x0, tol=cg_tol, maxiter=num_steps
    )

    final_mag, final_ramp = unravel(x0_flat + delta)
    # Re-evaluate loss for compatibility (scalar)
    final_loss = objective_flat(x0_flat + delta)

    return (final_mag, final_ramp), jnp.array([final_loss])


def _run_adam_solver_2d(
    phase: jax.Array,
    init_mag: jax.Array,
    mask: jax.Array,
    voxel_size_nm: float,
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
    voxel_size_nm
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
        params, mask, phase, rdfc_kernel, voxel_size_nm, reg_config,
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
            voxel_size_nm,
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

    # Truncate history (technically zero-filled after stop) or return full array + valid count
    return final_params, final_history[:steps_taken]


@jax.jit(static_argnames=("num_steps", "patience", "min_delta"))
def _run_lbfgs_solver_2d(
    phase: jax.Array,
    init_mag: jax.Array,
    mask: jax.Array,
    voxel_size_nm: float,
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
    voxel_size_nm
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
            voxel_size_nm,
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
    voxel_size_nm,
    solver="newton_cg",
    reg_config=None,
    rdfc_kernel=None,
    init_ramp_coeffs=None,
    reg_mask=None,
):
    """
    Unified MBIR solver for 2D projected magnetization reconstruction.

    Parameters
    ----------
    phase : array_like
        Measured phase image.
    init_mag : array_like
        Initial magnetization estimate, shape ``(N, M, 2)``.
    mask : array_like
        Binary mask of shape ``(N, M)`` applied to the
        magnetization.
    voxel_size_nm : float
        Voxel size in nanometres.
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
        Initial ramp coefficients ``[offset, slope_y, slope_x]``.
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
        voxel_size_nm=voxel_size_nm,
        reg_config=reg_config,
        rdfc_kernel=rdfc_kernel,
        init_ramp_coeffs=init_ramp_coeffs,
        reg_mask=reg_mask,
    )

    if isinstance(config, NewtonCGConfig):
        (mag, ramp), loss_history = _run_newton_cg_solver_2d(
            **shared,
            num_steps=config.num_steps,
            cg_tol=config.cg_tol,
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
    voxel_size_nm,
    b0_tesla=1.0,
    mask=None,
    lam=1e-3,
    solver="newton_cg",
    reg_mask=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
    solver_config=None,
):
    """Convenience wrapper for 2D MBIR magnetization reconstruction.

    Provides a simple interface similar to pyramid's
    ``reconstruction_2d_from_phasemap``.  Builds the RDFC kernel,
    initial magnetization guess, and mask automatically.

    Parameters
    ----------
    phase : array_like
        Measured phase image of shape ``(N, M)``.
    voxel_size_nm : float
        Pixel size in nanometres.
    b0_tesla : float, optional
        Magnetic induction in Tesla, default 1.0.
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
        and ``loss_history``.
    """
    phase = jnp.asarray(phase)
    if mask is None:
        mask = jnp.ones(phase.shape, dtype=bool)
    else:
        mask = jnp.asarray(mask, dtype=bool)

    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(
            voxel_size_nm,
            phase.shape,
            b0_tesla=b0_tesla,
            geometry=geometry,
            prw_vec=prw_vec,
        )

    init_mag = jnp.zeros((*phase.shape, 2), dtype=jnp.float64)
    reg_config = {"lambda_exchange": float(lam)}

    if solver_config is not None:
        solver = solver_config

    return solve_mbir_2d(
        phase=phase,
        init_mag=init_mag,
        mask=mask,
        voxel_size_nm=voxel_size_nm,
        solver=solver,
        reg_config=reg_config,
        rdfc_kernel=rdfc_kernel,
        reg_mask=reg_mask,
    )


def forward_model_2d(
    magnetization,
    voxel_size_nm,
    b0_tesla=1.0,
    ramp_coeffs=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
):
    """Convenience forward model for 2D projected magnetization.

    Computes the magnetic phase shift from a magnetization field,
    automatically building the RDFC kernel when not provided.

    Parameters
    ----------
    magnetization : array_like
        In-plane magnetization of shape ``(N, M, 2)`` where the
        last axis holds the (u, v) components.
    voxel_size_nm : float
        Pixel size in nanometres.
    b0_tesla : float, optional
        Magnetic induction in Tesla, default 1.0.
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
        Predicted phase image of shape ``(N, M)``.
    """
    magnetization = jnp.asarray(magnetization)
    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(
            voxel_size_nm,
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
        magnetization, ramp_coeffs, rdfc_kernel, voxel_size_nm,
    )


def reconstruct_2d_ensemble(
    phase,
    masks,
    voxel_size_nm,
    b0_tesla=1.0,
    lam=1e-3,
    solver="newton_cg",
    reg_masks=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
    solver_config=None,
):
    """Batched MBIR reconstruction over an ensemble of bootstrap masks.

    Runs :func:`reconstruct_2d` for each mask in the ensemble using
    ``jax.vmap`` for efficient parallel execution on GPU.

    Parameters
    ----------
    phase : array_like
        Measured phase image of shape ``(H, W)``.
    masks : array_like
        Bootstrap mask ensemble of shape ``(N_boot, H, W)``.
    voxel_size_nm : float
        Pixel size in nanometres.
    b0_tesla : float, optional
        Magnetic induction in Tesla, default 1.0.
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
    phase = jnp.asarray(phase)
    masks = jnp.asarray(masks)
    if reg_masks is None:
        reg_masks = masks
    else:
        reg_masks = jnp.asarray(reg_masks)

    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(
            voxel_size_nm,
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
                voxel_size_nm=voxel_size_nm,
                reg_config=reg_config,
                num_steps=config.num_steps,
                rdfc_kernel=rdfc_kernel,
                cg_tol=config.cg_tol,
                reg_mask=reg_mask,
            )
            return mag
    elif isinstance(config, AdamConfig):
        def _solve_single(mask, reg_mask):
            (mag, _ramp), _loss = _run_adam_solver_2d(
                phase=phase,
                init_mag=init_mag,
                mask=mask,
                voxel_size_nm=voxel_size_nm,
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
                voxel_size_nm=voxel_size_nm,
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


# ---------------------------------------------------------------------------
# 3D projection & forward model
# ---------------------------------------------------------------------------

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
    voxel_size_nm,
    b0_tesla=1.0,
    axis="z",
    ramp_coeffs=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
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
    voxel_size_nm : float
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
        voxel_size_nm,
        b0_tesla=b0_tesla,
        ramp_coeffs=ramp_coeffs,
        geometry=geometry,
        prw_vec=prw_vec,
        rdfc_kernel=rdfc_kernel,
    )


# ---------------------------------------------------------------------------
# L-curve analysis
# ---------------------------------------------------------------------------


def decompose_loss(
    magnetization,
    ramp_coeffs,
    phase,
    mask,
    reg_mask,
    rdfc_kernel,
    voxel_size_nm,
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
    voxel_size_nm : float
        Pixel size in nanometres.

    Returns
    -------
    data_misfit : float
        ``0.5 * sum((predicted - observed)**2)``
    exchange_norm : float
        Unweighted exchange regularization norm (no lambda
        multiplier).
    """
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
        masked_mag, ramp_coeffs, rdfc_kernel, voxel_size_nm,
    )
    residuals = predicted - phase
    data_misfit = float(0.5 * jnp.sum(residuals ** 2))
    exchange_norm = float(exchange_loss_fn(masked_mag, reg_mask))

    return data_misfit, exchange_norm


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
    voxel_size_nm,
    lambdas,
    b0_tesla=1.0,
    solver="newton_cg",
    reg_mask=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
    solver_config=None,
    warm_start=True,
):
    """Sequential L-curve sweep over regularization weights.

    Runs :func:`solve_mbir_2d` for each value in *lambdas*,
    collects the data-fidelity and regularization norms, and
    detects the L-curve corner via :func:`kneedle_corner`.

    When *warm_start* is ``True`` (default), lambdas are sorted
    in ascending order and each reconstruction is initialized from
    the previous result.

    Parameters
    ----------
    phase : array_like
        Measured phase image of shape ``(N, M)``.
    mask : array_like
        Binary mask of shape ``(N, M)``.
    voxel_size_nm : float
        Pixel size in nanometres.
    lambdas : array_like
        1D array of ``lambda_exchange`` values to sweep.
    b0_tesla : float, optional
        Magnetic induction in Tesla, default 1.0.
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
    warm_start : bool, optional
        If ``True`` (default), sort lambdas and seed each run
        from the previous result.

    Returns
    -------
    LCurveResult
        Named tuple with ``lambdas``, ``data_misfits``,
        ``reg_norms``, ``magnetizations``, ``ramp_coeffs``, and
        ``corner_index``.
    """
    phase = jnp.asarray(phase)
    mask = jnp.asarray(mask, dtype=bool)
    if reg_mask is None:
        reg_mask = mask
    else:
        reg_mask = jnp.asarray(reg_mask)

    lambdas = np.atleast_1d(np.asarray(lambdas, dtype=np.float64))
    if warm_start:
        sort_idx = np.argsort(lambdas)
        lambdas = lambdas[sort_idx]

    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(
            voxel_size_nm,
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

    current_mag = jnp.zeros((*phase.shape, 2), dtype=jnp.float64)

    for lam in lambdas:
        reg_config = {"lambda_exchange": float(lam)}

        result = solve_mbir_2d(
            phase=phase,
            init_mag=current_mag,
            mask=mask,
            voxel_size_nm=voxel_size_nm,
            solver=actual_solver,
            reg_config=reg_config,
            rdfc_kernel=rdfc_kernel,
            reg_mask=reg_mask,
        )

        dm, rn = decompose_loss(
            result.magnetization, result.ramp_coeffs,
            phase, mask, reg_mask, rdfc_kernel, voxel_size_nm,
        )
        data_misfits.append(dm)
        reg_norms.append(rn)
        mag_list.append(result.magnetization)
        ramp_list.append(result.ramp_coeffs)

        if warm_start:
            current_mag = result.magnetization

    data_misfits = np.array(data_misfits)
    reg_norms = np.array(reg_norms)
    corner_idx, _ = kneedle_corner(data_misfits, reg_norms)

    return LCurveResult(
        lambdas=lambdas,
        data_misfits=data_misfits,
        reg_norms=reg_norms,
        magnetizations=jnp.stack(mag_list),
        ramp_coeffs=jnp.stack(ramp_list),
        corner_index=corner_idx,
    )


def lcurve_sweep_vmap(
    phase,
    mask,
    voxel_size_nm,
    lambdas,
    b0_tesla=1.0,
    solver="newton_cg",
    reg_mask=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
    solver_config=None,
):
    """Parallel L-curve sweep using ``jax.vmap`` over lambda values.

    Runs all reconstructions in parallel (no warm-starting).
    This is faster on GPU when many lambda values are evaluated,
    but uses more memory than :func:`lcurve_sweep`.

    Parameters
    ----------
    phase : array_like
        Measured phase image of shape ``(N, M)``.
    mask : array_like
        Binary mask of shape ``(N, M)``.
    voxel_size_nm : float
        Pixel size in nanometres.
    lambdas : array_like
        1D array of ``lambda_exchange`` values to sweep.
    b0_tesla : float, optional
        Magnetic induction in Tesla, default 1.0.
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
            voxel_size_nm,
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

    # Build a vmappable function for the chosen solver
    if isinstance(config, NewtonCGConfig):
        def _solve_for_lam(lam):
            reg_config = {"lambda_exchange": lam}
            (mag, ramp), _loss = _run_newton_cg_solver_2d(
                phase=phase,
                init_mag=init_mag,
                mask=mask,
                voxel_size_nm=voxel_size_nm,
                reg_config=reg_config,
                num_steps=config.num_steps,
                rdfc_kernel=rdfc_kernel,
                cg_tol=config.cg_tol,
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
                voxel_size_nm=voxel_size_nm,
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
                voxel_size_nm=voxel_size_nm,
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
    def _decompose_single(mag, ramp):
        masked_mag = jnp.stack([
            mag[..., 0] * mask,
            mag[..., 1] * mask,
        ], axis=-1)
        predicted = forward_model_single_rdfc_2d(
            masked_mag, ramp, rdfc_kernel, voxel_size_nm,
        )
        residuals = predicted - phase
        dm = 0.5 * jnp.sum(residuals ** 2)
        rn = exchange_loss_fn(masked_mag, reg_mask)
        return dm, rn

    decompose_batch = jax.jit(jax.vmap(_decompose_single))
    all_dm, all_rn = decompose_batch(all_mag, all_ramp)

    data_misfits = np.asarray(all_dm)
    reg_norms = np.asarray(all_rn)
    corner_idx, _ = kneedle_corner(data_misfits, reg_norms)

    return LCurveResult(
        lambdas=lambdas_np,
        data_misfits=data_misfits,
        reg_norms=reg_norms,
        magnetizations=all_mag,
        ramp_coeffs=all_ramp,
        corner_index=corner_idx,
    )
