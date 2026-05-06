"""Model-based iterative reconstruction (MBIR) for 2D projected magnetization.

Unit conventions
Public functions in this package generally require ``unxt.Quantity`` inputs
for physical parameters and return ``Quantity``-annotated outputs, unless
an API explicitly documents scalar convenience inputs.

* ``pixel_size`` — pixel side length as a ``unxt.Quantity`` with length
  units (e.g. ``Quantity(0.58, "nm")``).  Converted to nanometres internally.
* ``phase`` — measured holographic phase in **radians** (rad).
* ``ramp_coeffs`` — background phase-ramp parameters as a :class:`RampCoeffs`
  named tuple with ``unxt.Quantity`` fields (offset in rad, slopes in rad/nm).
* ``PHI_0`` — magnetic flux quantum :math:`h/(2e)` expressed as
  ``Quantity(2067.83, "T nm2")``.
* ``B_REF`` — conventional 1 T reference magnetic induction kept for
  reconstruction defaults and reference conversions.
* ``KERNEL_COEFF`` — :math:`B_{\\text{ref}} / (2 \\Phi_0)` with units
  :math:`1/\\text{nm}^2` for the 1 T reference case.

The reconstructed magnetization is **dimensionless** (normalised
:math:`M / M_s`). Public forward-model calls now also require an explicit
``reference_induction`` so phase amplitude is chosen at the call boundary
instead of being hidden inside the kernel. Phase outputs carry
``Quantity["rad"]``.

This is a package; the implementation is split across submodules:

* :mod:`.units` — constants, ``RampCoeffs``, unit validators/normalizers.
* :mod:`.types` — result types and solver configuration.
* :mod:`.regularization` — regularization terms.
* :mod:`.kernel` — RDFC kernel construction and phase mapping.
* :mod:`.forward` — forward models.
* :mod:`.solver` — MBIR loss and Newton-CG solver.
* :mod:`.physical` — physical-unit post-processing.
* :mod:`.bootstrap` — bootstrap threshold uncertainty.
* :mod:`.lcurve` — L-curve sweeps.
* :mod:`.plotting` — matplotlib helpers.
* :mod:`.synthetic` — synthetic support and magnetization primitives.

All public names are re-exported from this package so users can import
them directly from :mod:`libertem_holo.base.mbir`.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

_EXPORTS_BY_MODULE = {
    ".units": (
        "B_REF",
        "ELECTRON_INTERACTION_CONSTANT_300KV",
        "KERNEL_COEFF",
        "MU_0",
        "PHI_0",
        "RampCoeffs",
        "add_units_to_inputs",
        "make_quantity",
    ),
    ".types": (
        "BootstrapThresholdResult",
        "LCurveResult",
        "NewtonCGConfig",
        "RegConfig",
        "SolverConfig",
        "SolverResult",
    ),
    ".regularization": ("exchange_loss_fn",),
    ".kernel": (
        "get_freq_grid",
        "phase_mapper_rdfc",
        "_rdfc_elementary_phase",
    ),
    ".forward": (
        "apply_ramp",
        "phase_from_magnetisation",
        "phase_from_magnetization",
        "phase_from_density_and_magnetization",
        "forward_phase_from_density_and_magnetization",
        "forward_model_2d",
        "forward_model_3d",
        "forward_model_single_rdfc_2d",
        "project_3d",
    ),
    ".solver": (
        "mbir_loss_2d",
        "reconstruct_2d",
        "reconstruct_2d_ensemble",
        "solve_mbir_2d",
    ),
    ".physical": (
        "estimate_mip_phase_from_thickness",
        "estimate_thickness_from_mip_phase",
        "to_local_induction",
        "to_local_magnetization",
        "to_projected_induction_integral",
        "to_projected_magnetization_integral",
    ),
    ".lcurve": (
        "decompose_loss",
        "kneedle_corner",
        "lcurve_sweep",
        "lcurve_sweep_vmap",
    ),
    ".bootstrap": ("bootstrap_threshold_uncertainty_2d",),
    ".plotting": (
        "plot_bootstrap_mask_summary",
        "plot_lcurve",
        "plot_physical_bootstrap_uncertainty",
    ),
    ".synthetic": (
        "domain_wall_magnetization",
        "soft_disc_support",
        "uniform_magnetization",
        "vortex_magnetization",
    ),
    ".fixtures": (
        "generate_vortex_disc_fixture",
        "load_vortex_disc_fixture",
        "save_vortex_disc_fixture",
        "vortex_disc_fixture_path",
    ),
    ".energy_backend": ("NeuralMagEnergyBackend",),
    ".differentiable_anisotropy": (
        "DifferentiableAnisotropyConfig",
        "DifferentiableAnisotropyFitResult",
        "angle_params_to_anisotropy_axes",
        "axis_angles_to_unit_vector",
        "cubic_anisotropy_invariant",
        "joint_phase_anisotropy_loss",
        "joint_phase_anisotropy_loss_terms",
        "mean_cubic_anisotropy_loss",
        "optimize_joint_phase_anisotropy",
        "orthonormalize_anisotropy_axes_jax",
        "pad_phase_view_zyx_jax",
        "phase_data_loss",
        "smoothness_loss_3d",
        "support_norm_loss",
        "unit_vector_to_axis_angles",
    ),
    ".equilibrium_orientation_fit": (
        "EquilibriumOrientationFitConfig",
        "EquilibriumOrientationFitTarget",
        "EquilibriumOrientationProblem",
        "angular_distance_deg",
        "build_equilibrium_orientation_problem",
        "coarse_grain_volume_xyz",
        "ensure_neuralmag_jax_backend",
        "finite_difference_axis_gradient_check",
        "fit_axis_from_phase",
        "make_vmapped_multi_start",
        "make_vmapped_steepest_descent_multi_start",
        "one_step_match_check",
        "phase_from_relaxed_m",
        "phase_loss_after_native_relax",
        "phase_loss_after_relax",
        "phase_loss_and_axis_grad",
        "prepare_equilibrium_fit_target",
        "prepare_equilibrium_fit_target_from_npz",
        "project_m_to_support",
        "relax_magnetization_native",
        "relax_magnetization",
    ),
    ".neuralmag_phase_recovery": (
        "NeuralMagAnisotropyCandidateFit",
        "NeuralMagAnisotropySelectionResult",
        "NeuralMagPhaseRecoveryConfig",
        "NeuralMagPhaseRecoveryResult",
        "NeuralMagPhaseTarget",
        "calibrate_phase_energy_scale",
        "center_crop",
        "dilate_mask_2d",
        "make_initial_m_cell",
        "make_support_projection",
        "normalize_on_support",
        "orthonormalize_anisotropy_axes",
        "pad_for_phase_view",
        "prepare_neuralmag_phase_target_from_phase_image",
        "predict_neuralmag_phase",
        "prepare_neuralmag_phase_target",
        "run_neuralmag_phase_recovery",
        "select_anisotropy_orientation_from_phase",
        "xyz_to_zyx",
    ),
}

if find_spec(f"{__name__}.inversion") is not None:
    _EXPORTS_BY_MODULE[".inversion"] = (
        "CombinedBackend",
        "EquilibriumTorqueBackend",
        "FieldState",
        "IdentityBackend",
        "InversionResult",
        "NeuralMagCritic",
        "PhysicsBackend",
        "ScaledRhoExperimentResult",
        "SmoothnessBackend",
        "WeightedBackend",
        "analytic_vortex_init",
        "depth_correlation",
        "equilibrium_residual",
        "invert_magnetization",
        "iterations_to_threshold",
        "mz_rmse",
        "phase_residual",
        "plot_depth_profile",
        "plot_loss_history",
        "plot_loss_landscape_2d",
        "plot_m_slices",
        "project_unit_norm",
        "projected_m_error",
        "run_with_scaled_rho",
        "support_center_yx",
        "vortex_core_z_error",
    )

_NAME_TO_MODULE = {
    name: module_name
    for module_name, names in _EXPORTS_BY_MODULE.items()
    for name in names
}

__all__ = [
    name
    for module_name, names in _EXPORTS_BY_MODULE.items()
    for name in names
]

phase_from_magnetisation = import_module(
    ".forward", __name__
).phase_from_magnetisation
phase_from_magnetization = import_module(
    ".forward", __name__
).phase_from_magnetization
phase_from_density_and_magnetization = import_module(
    ".forward", __name__
).phase_from_density_and_magnetization


def __getattr__(name: str):
    module_name = _NAME_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
