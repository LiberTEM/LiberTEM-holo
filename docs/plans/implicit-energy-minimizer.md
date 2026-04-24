# Implicit Differentiation Through `EnergyMinimizer` in the MBIR Notebook

## Summary

Replace the current main fit path in `notebooks/MBIR/mbir_and_neuralmag_real_data.ipynb` with an implicit-equilibrium fit that differentiates through the actual NeuralMag `EnergyMinimizer` solution.

The main fit should:

1. optimize a raw 3D magnetization seed and phase ramp,
2. relax the seed with the real JAX `EnergyMinimizer` solve,
3. compute the phase loss on the relaxed magnetization,
4. obtain gradients by implicit differentiation rather than reverse-mode through the minimizer loop.

The notebook will stay notebook-local. No shared library or vendored NeuralMag code should be modified for the first implementation.

## Key Changes

### Main fit objective

- Remove the current direct `phase_loss + weighted physics_energy` objective as the primary training path.
- Keep the outer trainable parameters as:
  - `params["m"]`: raw pre-relaxation magnetization seed
  - `params["ramp"]`: phase ramp coefficients
- Define the main loss on the relaxed state:
  - `seed -> support mask / normalization -> EnergyMinimizer solve -> forward phase -> masked phase loss`
- Do not include exchange/demag energy directly in the outer loss by default. Their role is enforced through the equilibrium solve.

### Differentiation strategy

- Do not attempt direct reverse-mode differentiation through `nm.EnergyMinimizer.minimize()` or the current JAX `_compiled_minimize` loop, because the backend uses a dynamic `jax.lax.while_loop`.
- Wrap the actual minimizer solve in `jax.lax.custom_root(...)` and differentiate it with the implicit function theorem.
- Use the real minimizer solve as the primal root solve:
  - instantiate `nm.EnergyMinimizer(state, ...)`
  - require the JAX backend
  - use the backend’s compiled minimize kernel as the `solve` callback
- Define a notebook-local augmented equilibrium residual to make the implicit system well posed:
  - on support: `cross(m, h) + norm_weight * (||m||^2 - 1) * m`
  - off support: `m`
- Use a matrix-free linear solve inside `tangent_solve`, with GMRES as the default.

### Notebook-local helpers and configs

- Add notebook-local helpers for:
  - building the support mask from `rho`
  - normalizing the seed on support and zeroing it off support
  - resolving a JAX effective-field closure from the NeuralMag state
  - wrapping the minimizer kernel for use in `custom_root`
  - evaluating the augmented equilibrium residual
  - solving for the relaxed magnetization from a seed
  - computing phase from the relaxed magnetization and ramp
- Keep the change notebook-only. Do not add a new shared inversion module.
- Add explicit config blocks in the notebook for:
  - primal minimizer settings
  - implicit residual settings
  - linear solver settings
  - outer optimizer settings

### Defaults

- Primal minimizer defaults:
  - `tol = 5e-3`
  - `max_iter = 50`
  - `method = "alternating"`
  - `update = "cayley"`
  - `tau_init = None`
  - `tau_min = 1e-18`
  - `tau_max = 1e-4`
- Residual defaults:
  - `norm_weight = 1.0`
- GMRES defaults:
  - `gmres_tol = 1e-3`
  - `gmres_atol = 1e-5`
  - `gmres_restart = 20`
  - `gmres_maxiter = 50`
- Outer fit defaults:
  - `m_learning_rate = 1e-3`
  - `ramp_learning_rate = 2e-4`
  - `cycles = 6`
  - `steps_per_cycle = 4`
  - `clip_grad_norm = 1.0`
  - `param_clip_value = 4.0`
  - `ramp_clip_value = 0.1`
  - `early_stop_patience = 3`

### Presets and runtime control

- Add a `fit_preset` switch near the fit setup.
- Default preset: `full`
  - keep the current cropped ROI
  - keep `max_nz_cap = 96`
  - use the default implicit solver settings
- Add a fallback preset: `debug`
  - centered `128 x 128` fit window inside the existing cropped ROI
  - `max_nz_cap = 32`
  - minimizer `max_iter = 15`
  - outer loop `cycles = 2`
  - outer loop `steps_per_cycle = 2`
- Do not silently fall back to the old direct-energy fit.

### Verification section

- Keep the bottom `nm.EnergyMinimizer` section, but turn it into an exact-solver consistency check.
- Start verification from the notebook’s implicitly relaxed magnetization.
- Run the public minimizer with:
  - `tol = fit_primal_tol`
  - `max_iter = 25`
- Report:
  - notebook equilibrium residual norm
  - exact minimizer initial `max_g`
  - exact minimizer final `max_g`
  - extra iterations required by the exact minimizer
  - phase RMS before and after exact minimization
  - relative state difference between notebook-relaxed and exact-relaxed states

## Interfaces and Outputs

- The notebook becomes the only implementation site for this first version.
- New notebook-local configs:
  - `implicit_minimizer_config`
  - `implicit_residual_config`
  - `implicit_linear_solver_config`
  - `implicit_fit_config`
  - `fit_preset`
- New notebook-local helpers should be named consistently with their role, for example:
  - `support_mask_from_rho`
  - `normalize_seed_on_support`
  - `make_effective_field_fn`
  - `make_minimizer_kernel`
  - `equilibrium_residual`
  - `solve_relaxed_m`
  - `phase_from_relaxed_params`
  - `run_implicit_minimizer_fit`

## Test Plan

- Tiny toy-state check:
  - verify `jax.grad` through the `custom_root`-wrapped minimizer path is finite
- Tiny consistency check:
  - compare the backend solve used in the notebook against the public `minimize()` result on the same small state
- Notebook debug preset:
  - complete at least one full outer cycle with finite loss and gradients
  - verify phase loss decreases from initialization
- Notebook full preset:
  - complete at least one outer cycle without NaNs or immediate OOM
- State constraints:
  - verify magnetization remains unit norm on support and zero off support after relaxation
- Verification section:
  - exact minimizer should improve or preserve equilibrium quality
  - exact minimizer should not materially worsen masked phase RMS

## Assumptions

- The notebook kernel remains the `.venv-notebook` environment with the NeuralMag JAX backend available.
- Differentiating through `EnergyMinimizer` means implicit differentiation through the converged equilibrium returned by the minimizer, not reverse-mode through every internal step.
- Using the backend compiled minimize kernel from the instantiated JAX minimizer is acceptable for the notebook prototype, even though it is a private attribute.
- This notebook-first implementation is intended to prove the method before any shared-library refactor.
