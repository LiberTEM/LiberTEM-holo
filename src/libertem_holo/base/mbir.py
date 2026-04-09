import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax
import numpy as np

PHI_0_T_NM2 = 2067.83  # flux quantum in T*nm^2


def exchange_loss_fn(mag, reg_mask):
    """
    2D exchange: neighbor-difference energy inside reg_mask_2d with neighbor-count scaling.
    Expects magnetization as (2, H, W) and reg_mask_2d as (H, W).
    """
    reg_mask = jnp.asarray(reg_mask)
    if reg_mask.shape != mag.shape[1:]:
        raise ValueError(
            f"reg_mask must have shape {mag.shape[1:]}; got {reg_mask.shape}."
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
    mag_up = jnp.pad(mag[:, :-1, :], ((0, 0), (1, 0), (0, 0)), constant_values=0)
    mag_down = jnp.pad(mag[:, 1:, :], ((0, 0), (0, 1), (0, 0)), constant_values=0)
    mag_left = jnp.pad(mag[:, :, :-1], ((0, 0), (0, 0), (1, 0)), constant_values=0)
    mag_right = jnp.pad(mag[:, :, 1:], ((0, 0), (0, 0), (0, 1)), constant_values=0)

    # Y-direction difference (central if both neighbors exist, else forward/backward)
    both_y = has_up & has_down
    only_down = has_down & ~has_up
    only_up = has_up & ~has_down
    diff_y = jnp.where(
        both_y, mag_down - mag_up,
        jnp.where(only_down, mag_down - mag,
                  jnp.where(only_up, mag - mag_up, 0))
    )

    # X-direction difference (central if both neighbors exist, else forward/backward)
    both_x = has_left & has_right
    only_right = has_right & ~has_left
    only_left = has_left & ~has_right
    diff_x = jnp.where(
        both_x, mag_right - mag_left,
        jnp.where(only_right, mag_right - mag,
                  jnp.where(only_left, mag - mag_left, 0))
    )

    diff_y = diff_y * mask[None, ...] / denom[None, ...]
    diff_x = diff_x * mask[None, ...] / denom[None, ...]

    # L2 Norm (p=2) squared
    loss = jnp.sum(diff_y * diff_y) + jnp.sum(diff_x * diff_x)

    return loss


def get_freq_grid(height, width, voxel_size_nm):
    """Frequency grids for FFT-based phase propagation."""
    fy = jnp.fft.fftfreq(height, d=voxel_size_nm)
    fx = jnp.fft.rfftfreq(width, d=voxel_size_nm)
    f_y, f_x = jnp.meshgrid(fy, fx, indexing="ij")
    denom = f_x**2 + f_y**2
    denom = jnp.where(denom == 0, 1.0, denom)
    return f_y, f_x, denom


def _rdfc_elementary_phase(geometry, n, m, voxel_size_nm):
    """Elementary kernel phase for the RDFC mapper."""
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
    voxel_size_nm,
    dim_uv,
    b0_tesla=1.0,
    geometry="disc",
    prw_vec=None,
    dtype=jnp.float64,
):
    """
    JAX translation of pyramid.Kernel for PhaseMapperRDFC.

    Returns a dict with FFT'd kernel components and slicing metadata.
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


def phase_mapper_rdfc(u_field, v_field, rdfc_kernel):
    """
    JAX implementation of PhaseMapperRDFC using a precomputed kernel.

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


def apply_ramp(ramp_coeffs, height, width, voxel_size_nm):
    """
    2D polynomial background ramp.
    ramp_coeffs: [offset, slope_y, slope_x] (order 1).
    """
    y, x = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    ramp = ramp_coeffs[0]
    if ramp_coeffs.shape[0] > 1:
        ramp = ramp + ramp_coeffs[1] * (y * voxel_size_nm)
        ramp = ramp + ramp_coeffs[2] * (x * voxel_size_nm)
    return ramp


def forward_model_single_rdfc_2d(
    magnetization,
    ramp_coeffs,
    rdfc_kernel,
    voxel_size_nm,
):
    """
    RDFC forward model for a single 2D projected magnetization: (u, v) -> phase + ramp.
    Expects magnetization as (2, H, W).
    """
    height, width = magnetization.shape[1:]

    u_field = magnetization[0]
    v_field = magnetization[1]

    phase = phase_mapper_rdfc(u_field, v_field, rdfc_kernel)
    ramp = apply_ramp(ramp_coeffs, height, width, voxel_size_nm)

    return phase + ramp


def mbir_loss_2d(
    params,
    mag_mask,
    reg_mask,
    phase,
    rdfc_kernel,
    voxel_size_nm,
    reg_config,
):
    """Loss = data fidelity + regularization (2D projected magnetization)."""
    magnetization, ramp_coeffs = params
    phase = jnp.asarray(phase)

    magnetization = magnetization * mag_mask

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


def run_newton_cg_solver_2d(
    phase,
    init_mag,
    mag_mask,
    reg_mask,
    voxel_size_nm,
    reg_config=None,
    num_steps=50,
    rdfc_kernel=None,
    cg_tol=1e-5,
    init_ramp_coeffs=None,
):
    """
    Newton-CG optimizer: Minimizes mbir_loss_2d w.r.t (mag, ramp).
    Uses JAX's conjugate gradient solver (scipy.sparse.linalg.cg equivalent).
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
            mag_mask,
            reg_mask,
            phase,
            rdfc_kernel,
            voxel_size_nm,
            reg_config,
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


def run_adam_solver_2d(
    phase,
    init_mag,
    mag_mask,
    reg_mask,
    voxel_size_nm,
    reg_config=None,
    num_steps=2000,
    learning_rate=1e-2,
    rdfc_kernel=None,
    init_ramp_coeffs=None,
    patience=50,
    min_delta=1e-6,
):
    """
    Adam optimizer: Minimizes mbir_loss_2d w.r.t (mag, ramp).
    Expects magnetization as (2, H, W).
    Includes early stopping if loss improvement is smaller than min_delta for patience steps.
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
        params, mag_mask, reg_mask, phase, rdfc_kernel, voxel_size_nm, reg_config
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
            mag_mask,
            reg_mask,
            phase,
            rdfc_kernel,
            voxel_size_nm,
            reg_config,
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
def run_lbfgs_solver_2d(
    phase,
    init_mag,
    mag_mask,
    reg_mask,
    voxel_size_nm,
    reg_config=None,
    num_steps=500,
    rdfc_kernel=None,
    init_ramp_coeffs=None,
    patience=50,
    min_delta=1e-6,
):
    if init_ramp_coeffs is None:
        init_ramp_coeffs = jnp.zeros((3,), dtype=init_mag.dtype)
    if reg_config is None:
        reg_config = {}

    params = (init_mag, init_ramp_coeffs)

    def value_fn(p):
        return mbir_loss_2d(
            p,
            mag_mask,
            reg_mask,
            phase,
            rdfc_kernel,
            voxel_size_nm,
            reg_config,
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
