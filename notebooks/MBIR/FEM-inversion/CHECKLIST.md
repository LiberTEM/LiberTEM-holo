# Implementation Checklist — Differentiable Phase-Image Inversion

Derived from [PLAN.md](PLAN.md) v3.3 and the current state of `mag_inversion` after Phases 1 and 2. Items marked ✅ already exist in the repo; `[ ]` items are outstanding.

Conventions:
- Spatial arrays: `(Z, Y, X)`; magnetization: `(Z, Y, X, 3)`.
- Isotropic voxels `dx = dy = dz`. Phase-only, single-axis (z) projection unless stated.
- All new public APIs live under `src/libertem_holo/base/mbir/` and are unit-tested under `tests/`.

---

## Phase 0 — Environment ✅

- ✅ JAX + GPU available (verified in `phase1_phase2_validation.ipynb`).
- ✅ `unxt`, `diffrax`, `optax`, `equinox` installed.
- ✅ `neuralmag` importable (adapter tests pass).
- [ ] Document the exact conda env used for reproducibility (one-line in README or PLAN appendix; `conda env export` snippet).

---

## Phase 1 — Forward pipeline (structured grid, no physics) ✅

- ✅ `soft_disc_support` — differentiable sigmoid disc (`src/libertem_holo/base/mbir/synthetic.py`).
- ✅ `vortex_magnetization`, `uniform_magnetization`, `domain_wall_magnetization`.
- ✅ `forward_phase_from_density_and_magnetization` with `m_eff = ρ · m` (`forward.py`).
- ✅ Sanity tests: uniform `m_z ⇒ φ≈0`, vortex antisymmetry, gradient finiteness, FD spot check (`tests/test_mbir_phase1_forward_wrapper.py`, `tests/test_mbir_synthetic.py`).

Gaps to close before Phase 3 starts:
- [ ] Add a finite-difference gradient test vs analytic `jax.grad` for **both** `rho` and `m` arguments at a non-trivial point (not just finite/shape check). One consolidated test in `tests/base/` is enough.
- [ ] Document pixel-size / units contract at the top of `forward.py` (is pixel size `Quantity["length"]` or plain float nm? what does `.value` strip?).

---

## Phase 2 — NeuralMag integration (partial) 🟡

Implemented:
- ✅ `neuralmag_state_to_mbir_rho_m` adapter with 8-point nodal→cell averaging, `(X,Y,Z)→(Z,Y,X)` transpose, isotropic-voxel assertion (`neuralmag_adapter.py`).
- ✅ `NeuralMagEnergyBackend` scaffold with resolver-based term caching (`mbir_energy_backend.py`).
- ✅ Unit tests for adapter and backend scaffold (`tests/test_neuralmag_adapter.py`, `tests/test_mbir_energy_backend.py`, `tests/test_mbir_neuralmag_synthetic.py`).

Outstanding (2a / 2b / 2c / 2d in PLAN):
- [ ] **2a reverse adapter**: `mbir_rho_m_to_neuralmag(rho_3d, m_3d, mesh) -> VectorFunction` (or state setter) for round-trip tests.
- [ ] **2a round-trip test**: `neuralmag → mbir → neuralmag` RMSE < 1e-6 on a vortex disc.
- [ ] **2b energy backend – real NeuralMag path**: replace the mock-resolver-only tests with a real `state.resolve("E_exchange", ["m"])` and `state.resolve("E_demag", ["m"])` call on a tiny mesh (e.g. `16³`), confirming the returned callable is jit-safe.
- [ ] **2b jit safety**: assert `state.resolve()` is called **outside** `@jax.jit` boundary (document this in a docstring + add a test that jits the `energies` call).
- [ ] **2c vortex-disc ground truth script/fixture**:
  - [ ] Permalloy params: `Ms=8e5`, `A=1.3e-11`, `Ku=0`.
  - [ ] Mesh `(64,64,64)` @ `dx=5e-9 m`. Soft-disc `ρ` via `state.rho` (use `state.eps`, not 0).
  - [ ] Vortex ansatz init → `LLGSolver.relax()` → `m_true`.
  - [ ] Save `(rho_true, m_true, phi_true, pixel_size)` as a cached `.npz` fixture under `tests/test_mbir_data/` so Phase 3 does not re-run LLG on every test.
- [ ] **2d verification**:
  - [ ] Adapter round-trip RMSE < 1e-6.
  - [ ] `E_exchange(m_true)` finite positive scalar.
  - [ ] `φ_true(vortex)` qualitative check in the validation notebook.
  - [ ] **Analytic-vs-relaxed φ comparison**: report `‖φ_relaxed − φ_analytic‖ / ‖φ_analytic‖` — this number sets the bar for how much a physics prior could plausibly help.

Housekeeping:
- [ ] Move `mbir_energy_backend.py` under `src/libertem_holo/base/mbir/` (it lives one level up today; align with the rest of the mbir package).
- [ ] Promote `phase1_phase2_validation.ipynb` to run headlessly in CI via `jupyter nbconvert --execute` (smoke job).

---

## Phase 3 — Regime A: fixed shape, invert magnetization (L1/L2) 🔴 NOT STARTED

This is the next major code target.

### 3.1 `PhysicsBackend` protocol
- [ ] Create `src/libertem_holo/base/mbir/inversion/backends.py`.
- [ ] Define `PhysicsBackend` Protocol with `prepare(rho, m) -> FieldState` and `energies(field) -> dict[str, Array]`.
- [ ] Implement `IdentityBackend` (returns `{}`).
- [ ] Implement `SmoothnessBackend`:
  - [ ] 3D finite-difference exchange-like penalty: `Σ‖∇m‖²` over interior voxels.
  - [ ] Weight by `ρ` so vacuum does not contribute.
  - [ ] Tests: zero loss for uniform `m`, positive for vortex, gradient finite.
- [ ] Implement `NeuralMagCritic`:
  - [ ] Wraps `NeuralMagEnergyBackend` with the `PhysicsBackend` interface.
  - [ ] Transposes `(Z,Y,X,3) → (X,Y,Z,3)` and weights by `ρ` before calling `state.resolve()` callables.
  - [ ] Tests: matches direct `state.resolve` output within float tolerance.

### 3.2 Inversion loop
- [ ] Create `src/libertem_holo/base/mbir/inversion/solver.py` with `invert_magnetization(phi_meas, rho, backend, *, lambda_phys, max_iter, lr, init) -> InversionResult`.
- [ ] Use `optax.adam` (or `optax.lbfgs`); record loss history per backend.
- [ ] **Hard unit-norm projection** of `m` inside `ρ > 0.5` applied **every step**, not post-hoc. Unit test that `|m| = 1` within mask after each step.
- [ ] `init` options: zero, analytic ansatz, warm-start from another backend's result.
- [ ] Return dataclass with `m_recon`, `loss_history`, `phi_pred`, wall-clock time.

### 3.3 Metrics module
`src/libertem_holo/base/mbir/inversion/metrics.py`:

**Observable:**
- [ ] `projected_m_error(m_recon, m_true)` — in-plane `∫m dz` relative L2.
- [ ] `phase_residual(phi_pred, phi_true)` — relative L2.
- [ ] Iterations to a fixed data-loss threshold (simple helper).

**Not directly observed:**
- [ ] `mz_rmse(m_recon, m_true)` — voxel-wise `M_z` error.
- [ ] `depth_correlation(m_recon, m_true, yx)` — `M_x(z)` profile correlation at fixed `(y,x)` vs. the smeared-along-z baseline.
- [ ] `vortex_core_z_error(m_recon, y_c, x_c)` — `argmax_z M_z`.
- [ ] `equilibrium_residual(m_recon, backend_alt)` — `‖m × H_eff‖`, **using `backend_alt` ≠ the backend used as A.2 prior** (e.g. include anisotropy/DMI). Document this in the docstring.

**Shape-amplitude confound:**
- [ ] Helper `run_with_scaled_rho(pipeline, scale=1.5)` to re-run each prior with `1.5·ρ_true` and report `⟨|m_recon|⟩` and `|m|` histogram.

### 3.4 Experiments notebook
- [ ] Create `notebooks/MBIR/FEM-inversion/phase3_regimeA.ipynb`:
  - [ ] Load cached `(rho_true, m_true, phi_true)` fixture from Phase 2c.
  - [ ] Run A.0 (identity), A.1 (smoothness), A.2 (NeuralMag critic) with identical `max_iter=500` and zero init.
  - [ ] Plot: loss curves, projected-m error, `M_z` recovery, depth correlation, vortex-core z error, equilibrium residual.
  - [ ] Run shape-amplitude confound diagnostic (scale `ρ_true` by 1.5).
  - [ ] Write a short "Interpretation" cell mapping outcomes to the table in PLAN Phase 3.

### 3.5 Tests
- [ ] `tests/base/test_inversion_backends.py` — backend contracts.
- [ ] `tests/test_regime_a_smoke.py` — 16³ smoke test: each backend runs 10 iters and `loss_history[-1] < loss_history[0]`.

---

## Phase 3b — Material parameter inversion (L3) 🔴 NOT STARTED

Precondition from PLAN: **shape is known**. Do not start until Phase 3 produces its headline numbers.

### 3b.1 L3 forward operator
- [ ] Create `src/libertem_holo/base/mbir/inversion/l3_forward.py` with `forward_phi_of_theta(theta, rho_true, mesh, *, init_ansatz, n_inits=2) -> jnp.ndarray`:
  - [ ] Build NeuralMag state with `theta = (A_ex, Ms, Ku)` (plus fixed `Ku_axis`).
  - [ ] LLG relax from `init_ansatz`; optionally relax from `n_inits` different inits and keep the min-loss branch.
  - [ ] Adapter → `(Z,Y,X,3)` → `forward_phase_from_density_and_magnetization`.
- [ ] Enforce mesh `dx=dy=dz` **equal to** the ground-truth `dx` (assert at construction).
- [ ] Unit test: `forward_phi_of_theta(theta_true)` matches cached `phi_true` within tolerance.

### 3b.2 Ground-truth with nonzero `Ku`
- [ ] Extend Phase 2c fixture: second ground truth disc with `Ku ~ 1e4 J/m³` along `ŷ` (or Co-like params). Cache as separate `.npz`.

### 3b.3 Grid search (3b-i — priority)
- [ ] Pilot on `32³` mesh with `5×5×5 = 125` grid. Record wall-clock per relax + RDFC.
- [ ] Production sweep `64³`, `~10×10×10` grid over `A_ex ∈ [0.5,3]e-11`, `Ms ∈ [4,12]e5`, `Ku ∈ [0,5]e4`.
- [ ] **Metastability budget**: 2 inits per θ (vortex ansatz + uniform in-plane); record min-loss branch.
- [ ] Save landscape as `.npz` and plot 2D marginals `(A_ex,Ms)`, `(A_ex,Ku)`, `(Ms,Ku)`.
- [ ] Identifiability diagnostics: Hessian (or quadratic fit) at minimum; report flat directions.

### 3b.4 Notebook
- [ ] `notebooks/MBIR/FEM-inversion/phase3b_L3_gridsearch.ipynb`:
  - [ ] Pilot → production sweep → landscape plots → interpretation table from PLAN Phase 3b.

### 3b.5 Deferred paths (document but do not implement)
- [ ] Add a short README note that `3b-ii` (gradients) requires a standalone diffrax-through-LLG toy on a few voxels to succeed first.
- [ ] Add a short README note that `3b-iii` (BO) is the fallback if 3b-ii fails.

---

## Phase 4 — Regime B: parametric shape (optional) 🔴

- [ ] Extend `soft_disc_support` or add `parametric_disc(center, radius, z_range, sigma) -> (ρ, ∂ρ/∂params)`.
- [ ] B.0 notebook cell: recover shape with `m` fixed.
- [ ] B.1: recover `m` with slightly wrong shape.
- [ ] B.2: joint shape+`m` with the three priors.
- [ ] Metrics: IoU / radius error; reuse Phase 3 metrics for `m`.

Treat as conditional scope; only start if Phase 3/3b outcomes justify it.

---

## Phase 4b — Small multi-tilt diagnostic (optional, Phase-1-scale work) 🔴

- [ ] Build **tilt-aware differentiable forward model**: `forward_phase(ρ, m, pixel_size, tilt_angle)` via rotated resampling (differentiable interp). This is not a one-liner — budget accordingly.
- [ ] `multi_tilt_loss(ρ, m, phi_meas_list, tilt_angles, pixel_size)`.
- [ ] 4b-0 L1 diagnostic: 1 vs 2 vs 3 tilts on 3D-ambiguity metrics.
- [ ] 4b-1 L2 multi-tilt with fixed shape.
- [ ] 4b-2 (optional) multi-tilt with parametric shape.

Do not start until Phase 3 / 3b-i produce headline results.

---

## Phase 5 — Regime C: density-field shape (stretch) 🔴

- [ ] `rho = sigmoid(rho_raw)` parameterization with smoothness / binarization regularizers.
- [ ] Joint `(ρ_raw, m)` inversion under each of the three priors.
- [ ] Explicitly stretch scope; document expected failure modes.

---

## Phase 6 — Warm-starts / initialization study 🔴

- [ ] For each regime, compare zero init, analytic ansatz init, LLG-relaxed random init.
- [ ] Report convergence speed and final loss.

---

## Phase 7 — Experimental data 🔴 (far future)

Out of scope for this program. Listed for completeness:
- phase calibration, pixel-size matching, tilt alignment, support estimation, uncertainty weighting.

---

## Cross-cutting infrastructure

These are not phase-specific but are blockers/enablers for the above.

- [ ] **Caching**: persistent `.npz` fixtures for `(ρ_true, m_true, φ_true)` at `32³` and `64³`, two `K_u` variants. Loaded by notebooks and tests.
- [ ] **CI**: add a job that executes the validation notebook headlessly and, once Phase 3 lands, the Regime A smoke notebook.
- [ ] **API surface**: expose the new backends, solver, and metrics via `src/libertem_holo/base/mbir/__init__.py`.
- [ ] **Units audit**: decide once and for all whether public MBIR APIs take `unxt.Quantity` or plain floats; document in `units.py` docstring. Today it is mixed.
- [ ] **Metrics module in notebooks is a duplication risk** — implement in `src/` and import in notebooks, not the other way round.
- [ ] **Plotting helpers**: `plot_loss_landscape_2d`, `plot_depth_profile`, `plot_m_slices` collected in `src/libertem_holo/base/mbir/inversion/plotting.py`. Keep notebook cells thin.
- [ ] **Documentation**: short `notebooks/MBIR/FEM-inversion/README.md` pointing at PLAN, CHECKLIST, and the two validation notebooks, with "what exists today" vs "what is pending".

---

## Suggested execution order

1. Close Phase 2 gaps (reverse adapter, real `state.resolve` path, cached ground-truth fixtures, notebook in CI).
2. Implement Phase 3 (backends, solver, metrics, Regime A notebook, smoke tests).
3. Phase 3b-i grid search on known shape — the headline identifiability result.
4. Decide whether to pursue Phase 4 / 4b / 3b-ii based on those results.

Everything beyond step 3 is conditional scope.
