# Implementation Checklist — Differentiable Phase-Image Inversion

Derived from [PLAN.md](PLAN.md) v4.1 and the current state of `mag_inversion` after Phases 1 and 2. Items marked ✅ already exist in the repo; `[ ]` items are outstanding.

**Primary goal**: demonstrate that physics-coupled magnetization reconstruction (L2) yields better 3D magnetization from few images than per-voxel approaches (L1), and test whether electrostatic/magnetic phase separation becomes possible as a consequence.

Conventions:
- Spatial arrays: `(Z, Y, X)`; magnetization: `(Z, Y, X, 3)`.
- Isotropic voxels `dx = dy = dz`. Phase-only, single-axis (z) projection unless stated.
- All new public APIs live under `src/libertem_holo/base/mbir/` and are unit-tested under `tests/`.

## Status Summary

| Phase | Component | Status | Gate before Phase 3 impl? |
|---|---|---|---|
| 0 | Environment | ✅ | No |
| 1 | Forward pipeline + synthetic fields | ✅ | No |
| 2.1 | Adapter: NeuralMag → MBIR | ✅ | Yes |
| 2.2 | Adapter: MBIR → NeuralMag | ✅ | Yes |
| 2.3 | Real NeuralMag energy validation | ✅ | Yes |
| 2.4 | Cached ground-truth fixtures + loader API | ✅ | Yes |
| 2.5 | Package cleanup + notebook CI | 🟡 | No |
| 3 | Regime A: physics-coupled magnetization reconstruction ★ | 🟡 | n/a |
| 3b | L3 material sensitivity study (secondary) | ❌ | n/a |

## Phase 3 Entry Gates

Do **not** start Phase 3 implementation until all of these are true:

- ✅ Reverse adapter exists and passes a round-trip RMSE threshold on a known vortex state.
- ✅ Real `state.resolve()` energy evaluation is validated on a tiny NeuralMag state and is safe to call from a jitted loss path, with `resolve()` itself kept outside `jit`.
- ✅ Ground-truth fixtures exist at both `32^3` and `64^3`, with a stable loader API used by tests and notebooks instead of regenerating LLG states ad hoc.

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
- [ ] Add a finite-difference gradient test for **each** magnetization component `m[..., 0:3]`; finite/shape checks alone are not enough to catch sign or axis mistakes.
- [ ] Document pixel-size / units contract at the top of `forward.py` (is pixel size `Quantity["length"]` or plain float nm? what does `.value` strip?).
- [ ] Add a regression test that `pixel_size=1.0` and `pixel_size=Quantity(1.0, "nm")` produce identical forward output.

---

## Phase 2 — NeuralMag integration (partial) 🟡

Implemented:
- ✅ `neuralmag_state_to_mbir_rho_m` adapter with 8-point nodal→cell averaging, real NeuralMag `(z, y, x)` tensor-order handling, and isotropic-voxel assertion (`neuralmag_adapter.py`).
- ✅ Reverse `mbir_rho_m_to_neuralmag(...)` / `mbir_rho_m_to_neuralmag_state(...)` adapter with exact cell-state export and tested nodal projection.
- ✅ `NeuralMagEnergyBackend.from_state(...)` and `energy_terms(...)` for real `state.resolve(...)` usage outside `jit` and resolved callables inside `jit` (`mbir_energy_backend.py`).
- ✅ Cached fixture API in `src/libertem_holo/base/mbir/fixtures.py` with shared loader use in tests and notebook.
- ✅ Unit tests for adapter, backend, and synthetic fixture paths (`tests/test_neuralmag_adapter.py`, `tests/test_mbir_energy_backend.py`, `tests/test_mbir_neuralmag_synthetic.py`).

### 2.1 Adapter: NeuralMag → MBIR ✅

- ✅ `neuralmag_state_to_mbir_rho_m` exists and is tested.

### 2.2 Adapter: MBIR → NeuralMag ✅

- ✅ Implement `mbir_rho_m_to_neuralmag(...)` plus `mbir_rho_m_to_neuralmag_state(...)` with explicit cell-state and nodal-state targets.
- ✅ Define the reverse adapter contract: MBIR keeps `rho` and `m` separate; the NeuralMag-facing tensor carries the weighted magnetic field `rho * m` expected by the energy path.
- ✅ Acceptance criterion met on a deterministic cached vortex-disc case with nodal-projection round-trip RMSE `< 1e-2` (`~5.58e-3` measured on the cached `32^3` vortex). Exact cell-state export round-trips at machine precision.
- ✅ Regression tests cover support-region unit-norm preservation and cached-vortex round-trip behavior.

### 2.3 Real NeuralMag energy validation ✅

- ✅ Replace purely mock-resolver validation with real `state.resolve("E_exchange", ["m"])` and `state.resolve("E_demag", ["m"])` calls on a tiny mesh.
- ✅ Acceptance criterion met: real resolved energy callables return finite positive scalars.
- ✅ Document and encode the `jit` contract in code: `state.resolve()` happens outside `jit`; only the returned callables are used inside the jitted loss path.
- ✅ Add a test that jits the loss path using already-resolved callables.
- ✅ Fix the immediate Quantity/raw-array fixture mismatch so this path is genuinely runnable.

### 2.4 Ground-truth fixtures + loader API ✅

- ✅ Create `src/libertem_holo/base/mbir/fixtures.py` for a permalloy-like vortex disc with `Ms=8e5`, `A=1.3e-11`, `Ku=0`.
- ✅ Generate and cache both `32^3` and `64^3` cases with mesh `(N,N,N)`, `dx=5e-9 m`, soft-disc `ρ`, vortex ansatz init, and `LLGSolver.relax()`.
- ✅ Save stable `.npz` artifacts:
  - ✅ `tests/test_mbir_data/vortex_disc_32_ku0.npz`
  - ✅ `tests/test_mbir_data/vortex_disc_64_ku0.npz`
- ✅ Standardize `.npz` keys: `rho_true`, `m_true`, `phi_true`, `pixel_size_nm`.
- ✅ Add a shared loader API used by tests and the validation notebook.
- ✅ Acceptance criterion met: tests and notebook load cached fixtures instead of running LLG during normal execution.

### 2.5 Verification and housekeeping 🟡

- ✅ Adapter round-trip RMSE on the cached vortex fixture is characterized and regression-tested at the actual nodal-projection tolerance (`~5.58e-3`, asserted `< 1e-2`).
- ✅ `E_exchange(m_true)` is exercised through the real resolved-energy path on a tiny NeuralMag state.
- ✅ `φ_true(vortex)` qualitative check is shown in the validation notebook.
- ✅ Report `‖φ_relaxed − φ_analytic‖ / ‖φ_analytic‖`; current notebook output is `~0.5493` for the cached `32^3` fixture.
- ✅ Move `mbir_energy_backend.py` under `src/libertem_holo/base/mbir/` before Phase 3 starts so the package layout is coherent.
- ✅ Export the moved backend from `src/libertem_holo/base/mbir/__init__.py`.
- [ ] Promote `phase1_phase2_validation.ipynb` to a headless CI smoke run via `jupyter nbconvert --execute` or equivalent.

---

## Phase 3 — Regime A: physics-coupled magnetization reconstruction (L1/L2) ★ 🟡 PARTIAL

This is the **headline experiment**. The question: does coupling all voxels through micromagnetic energy (exchange + demag) yield a measurably better 3D magnetization from a single image than treating voxels independently?

### 3.1 `PhysicsBackend` protocol
- ✅ Create `src/libertem_holo/base/mbir/inversion/backends.py`.
- ✅ Define `PhysicsBackend` Protocol with `prepare(rho, m) -> FieldState` and `energies(field) -> dict[str, Array]`.
- ✅ Document the execution contract explicitly: `prepare()` may do non-jittable setup like `state.resolve()`, while the returned `energies()` path must be safe to call from inside the solver's jitted loss.
- ✅ Implement `IdentityBackend` (returns `{}`).
- ✅ Implement `SmoothnessBackend`:
  - ✅ 3D finite-difference exchange-like penalty: `Σ‖∇m‖²` over interior voxels.
  - ✅ Weight by `ρ` so vacuum does not contribute.
  - ✅ Tests: zero loss for uniform `m`, positive for vortex, gradient finite.
- ✅ Implement `NeuralMagCritic`:
  - ✅ Wraps `NeuralMagEnergyBackend` with the `PhysicsBackend` interface.
  - ✅ Transposes `(Z,Y,X,3) → (X,Y,Z,3)` and weights by `ρ` before calling `state.resolve()` callables.
  - ✅ Tests: matches direct `state.resolve` output within float tolerance.

### 3.2 Inversion loop
- ✅ Create `src/libertem_holo/base/mbir/inversion/solver.py` with `invert_magnetization(phi_meas, rho, backend, *, lambda_phys, max_iter, lr, init) -> InversionResult`.
- ✅ Use `optax.adam` (or `optax.lbfgs`); record loss history per backend.
- ✅ **Hard unit-norm projection** of `m` inside `ρ > 0.5` applied **every step**, not post-hoc. Unit test that `|m| = 1` within mask after each step.
- ✅ Implement normalization as a small utility function so it is testable independently of the optimizer loop.
- ✅ Acceptance criterion: after projection, `||m|| = 1.0 ± 1e-6` inside support and `m = 0` outside support.
- ✅ `init` options: zero, analytic ansatz, warm-start from another backend's result.
- ✅ Return dataclass with `m_recon`, `loss_history`, `phi_pred`, wall-clock time.

### 3.3 Metrics module
`src/libertem_holo/base/mbir/inversion/metrics.py`:

**Solver outputs / directly observable from reconstruction output:**
- ✅ `phase_residual(phi_pred, phi_true)` — relative L2.
- ✅ Iterations to a fixed data-loss threshold (simple helper).

**Ground-truth diagnostics for the synthetic study:**
- ✅ `projected_m_error(m_recon, m_true)` — in-plane `∫m dz` relative L2.
- ✅ `mz_rmse(m_recon, m_true)` — voxel-wise `M_z` error.
- ✅ `depth_correlation(m_recon, m_true, yx)` — `M_x(z)` profile correlation at fixed `(y,x)` vs. the smeared-along-z baseline.
- ✅ `vortex_core_z_error(m_recon, y_c, x_c)` — `argmax_z M_z`.
- ✅ `equilibrium_residual(m_recon, backend_alt)` — `‖m × H_eff‖`, **using `backend_alt` ≠ the backend used as A.2 prior** (e.g. include anisotropy/DMI). Document this in the docstring.

**Shape-amplitude confound:**
- ✅ Helper `run_with_scaled_rho(pipeline, scale=1.5)` to re-run each prior with `1.5·ρ_true` and report `⟨|m_recon|⟩` and `|m|` histogram.

### 3.4 Experiments notebook
- ✅ Create `notebooks/MBIR/FEM-inversion/phase3_regimeA.ipynb`:
  - ✅ Load cached `(rho_true, m_true, phi_true)` fixture from Phase 2c.
  - ✅ Run A.0 (identity), A.1 (smoothness), A.2 (NeuralMag critic) with identical `max_iter=500` and zero init.
  - ✅ Plot: loss curves, projected-m error, `M_z` recovery, depth correlation, vortex-core z error, equilibrium residual.
  - ✅ Run shape-amplitude confound diagnostic (scale `ρ_true` by 1.5).
  - ✅ Write a short "Interpretation" cell mapping outcomes to the table in PLAN Phase 3, including the current caveat that the hard unit-norm projection makes the present confound run non-discriminative.

### 3.5 Electrostatic separation extension (A.2e)
- [ ] Implement electrostatic forward: `phi_elec(rho, V_0) = C_E * V_0 * jnp.sum(rho, axis=0) * dx`.
- [ ] Generate ground truth with both `phi_mag` and `phi_elec` contributions using a known MIP value.
- [ ] Run A.2e: jointly optimize `m` and `V_0` with micromagnetic prior on `m`.
- [ ] Compare: does L2 recover both `m` and `V_0`? Does L1 absorb the electrostatic contribution into biased `m`?
- [ ] Report `V_0` recovery error and `m` quality metrics vs A.0/A.1/A.2.

### 3.6 Tests
- ✅ `tests/base/test_inversion_backends.py` — backend contracts.
- ✅ `tests/test_regime_a_smoke.py` — 16³ smoke test: each backend runs 10 iters, loads cached truth instead of running LLG, and satisfies `loss_history[-1] < loss_history[0]`.

### 3.7 Exit criteria for Phase 3

**Engineering exit (code works):**
- ✅ A.0, A.1, and A.2 all run end-to-end from the same cached truth fixture.
- [ ] Regime A smoke test passes in CI.

**Scientific exit (result interpreted):**
- ✅ At least one non-trivial diagnostic beyond phase residual is reported in the notebook (`M_z` RMSE, depth correlation, or vortex-core depth).
- ✅ The shape-amplitude confound diagnostic has been run once and interpreted, even if it is not favorable to NeuralMag.
- ✅ **Headline result**: a clear comparison showing whether A.2 (physics-coupled) yields better 3D magnetization than A.0/A.1 on the non-observable metrics. Current result is mixed: A.2 improves projected-m error and vortex-core depth, but not `M_z` RMSE or depth-profile correlation yet.
- [ ] *(Optional extension)* Electrostatic separation experiment (A.2e) has been run at least once and interpreted. Not blocking for Phase 3 closure.

---

## Phase 3b — Material parameter sensitivity study (L3, secondary) 🔴 NOT STARTED

> **Secondary to Phase 3.** The primary deliverable is better magnetization from physics coupling. This phase is an optional extension.

Precondition from PLAN: **shape is known** (the material-family-known assumption is global to the program). Do not start until Phase 3 produces its headline numbers.

### 3b.1 L3 forward operator
- [ ] Create `src/libertem_holo/base/mbir/inversion/l3_forward.py` with `forward_phi_of_theta(theta, rho_true, mesh, *, init_ansatz, n_inits=2) -> jnp.ndarray`:
  - [ ] Build NeuralMag state with `theta = (A_ex, Ms, Ku)` (plus fixed `Ku_axis`), centered on nominal known-material values for `A_ex` and `Ms`.
  - [ ] LLG relax from deterministic init strategies; keep the min-loss branch across the configured init set.
  - [ ] Adapter → `(Z,Y,X,3)` → `forward_phase_from_density_and_magnetization`.
- [ ] Enforce mesh `dx=dy=dz` **equal to** the ground-truth `dx` (assert at construction).
- [ ] Unit test: `forward_phi_of_theta(theta_true)` matches cached `phi_true` within tolerance.

### 3b.2 Ground-truth with nonzero `Ku`
- [ ] Extend Phase 2c fixture: second ground truth disc with `Ku ~ 1e4 J/m³` along `ŷ`, while keeping `A_ex` and `Ms` at nominal values for the chosen known material. Cache as separate `.npz`.

### 3b.3 Grid search
- [ ] Pilot on `32³` mesh with `5×5×5 = 125` grid. Record wall-clock per relax + RDFC.
- [ ] Production sweep `64³` over a **local** parameter window, for example `A_ex ∈ [0.8,1.2]A_ex,0`, `Ms ∈ [0.9,1.1]Ms,0`, `Ku ∈ [0,5]e4`, with widening only if the pilot shows sensitivity outside that band.
- [ ] **Metastability budget is mandatory, not optional**: at least 2 deterministic inits per `θ` (`vortex`, `uniform_in_plane`); record which branch won.
- [ ] Save landscape metadata including `theta`, chosen init strategy, and relaxed-state energy so the sweep is reproducible.
- [ ] Save landscape as `.npz` and plot 2D marginals `(A_ex,Ms)`, `(A_ex,Ku)`, `(Ms,Ku)`, marking the nominal known-material point.
- [ ] Identifiability diagnostics: Hessian (or quadratic fit) at minimum; report flat directions.

### 3b.4 Notebook
- [ ] `notebooks/MBIR/FEM-inversion/phase3b_L3_gridsearch.ipynb`:
  - [ ] Pilot → production sweep → landscape plots → interpretation table from PLAN Phase 3b.

### 3b.5 Deferred paths (document but do not implement)
- [ ] Add a short README note that the **gradient path** (differentiating through LLG relax) requires a standalone diffrax-through-LLG toy on a few voxels to succeed first.
- [ ] Add a short README note that **Bayesian optimization** is the fallback if the gradient path fails.

---
---

## Future scope (not part of active program)

Noted for continuity only. Do not start until Phase 3 produces its headline results.

- Parametric shape inversion (Regime B)
- Multi-tilt diagnostic (requires tilt-aware differentiable forward model)
- Spatially varying material (`L4`)
- Free-form shape `ρ = sigmoid(ρ_raw)` (Regime C)
- Warm-start / initialization study
- Experimental data pipeline

---

## Cross-cutting infrastructure

These are not phase-specific but are blockers/enablers for the above.

- [ ] **Caching**: persistent `.npz` fixtures for `(ρ_true, m_true, φ_true)` at `32³` and `64³` with `K_u = 0` for Phase 3. A second `K_u ≠ 0` fixture is only needed if Phase 3b is pursued. Loaded by notebooks and tests through one loader API, not by ad hoc `np.load` paths scattered across the repo.
- [ ] **CI**: add a job that executes the validation notebook headlessly and, once Phase 3 lands, the Regime A smoke notebook.
- ✅ **API surface**: expose the new backends, solver, and metrics via `src/libertem_holo/base/mbir/__init__.py`.
- [ ] **Units audit**: decide once and for all whether public MBIR APIs take `unxt.Quantity` or plain floats; document in `units.py` docstring. Today it is mixed.
- ✅ **Metrics module in notebooks is a duplication risk** — implement in `src/` and import in notebooks, not the other way round.
- ✅ **Plotting helpers**: `plot_loss_landscape_2d`, `plot_depth_profile`, `plot_m_slices` collected in `src/libertem_holo/base/mbir/inversion/plotting.py`. Keep notebook cells thin.
- [ ] **Documentation**: short `notebooks/MBIR/FEM-inversion/README.md` pointing at PLAN, CHECKLIST, and the two validation notebooks, with "what exists today" vs "what is pending".
- [ ] **Runtime budget**: record expected wall-clock for fixture generation, Regime A smoke tests, and the 3b pilot sweep so CI and notebook expectations stay realistic.

---

## Suggested execution order

1. Close the **Phase 3 entry gates**: reverse adapter, real `state.resolve` validation, cached ground-truth fixtures + loader API.
2. Clean package structure and reproducibility basics: move the energy backend into `mbir/`, lock down units tests, add notebook smoke execution.
3. **Implement Phase 3 Regime A** ★: backends, solver core, normalization utility, metrics, and smoke tests. This is the headline.
4. **Run and interpret the Regime A notebook**: A.0 vs A.1 vs A.2 comparison on 3D magnetization quality (especially $M_z$, depth, equilibrium consistency).
5. **Run the electrostatic separation experiment (A.2e)**: test whether physics-constrained magnetic phase allows MIP recovery.
6. *(Optional)* Phase 3b material sensitivity study on known shape — secondary identifiability result.
7. Decide whether to pursue future directions (multi-tilt, shape inversion, spatially varying material) based on the Phase 3 magnetization-quality results.
