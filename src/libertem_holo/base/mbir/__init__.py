"""Model-based iterative reconstruction (MBIR) for 2D projected magnetization.

Unit conventions
----------------
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
* ``B_REF`` — reference magnetic induction :math:`B_0 = 1\\,\\text{T}`,
  baked into ``KERNEL_COEFF``.
* ``KERNEL_COEFF`` — :math:`B_{\\text{ref}} / (2 \\Phi_0)` with units
  :math:`1/\\text{nm}^2`.

The reconstructed magnetization is **dimensionless** (normalised
:math:`M / M_s`).  Phase outputs carry ``Quantity["rad"]``.

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

from .units import (
    B_REF,
    ELECTRON_INTERACTION_CONSTANT_300KV,
    KERNEL_COEFF,
    MU_0,
    PHI_0,
    RampCoeffs,
    add_units_to_inputs,
    make_quantity,
)
from .types import (
    BootstrapThresholdResult,
    LCurveResult,
    NewtonCGConfig,
    RegConfig,
    SolverConfig,
    SolverResult,
)
from .regularization import exchange_loss_fn
from .kernel import (
    build_rdfc_kernel,
    get_freq_grid,
    phase_mapper_rdfc,
    _rdfc_elementary_phase,
)
from .forward import (
    apply_ramp,
    forward_phase_from_density_and_magnetization,
    forward_model_2d,
    forward_model_3d,
    forward_model_single_rdfc_2d,
    project_3d,
)
from .solver import (
    mbir_loss_2d,
    reconstruct_2d,
    reconstruct_2d_ensemble,
    solve_mbir_2d,
)
from .physical import (
    estimate_mip_phase_from_thickness,
    estimate_thickness_from_mip_phase,
    to_local_induction,
    to_local_magnetization,
    to_projected_induction_integral,
    to_projected_magnetization_integral,
)
from .lcurve import (
    decompose_loss,
    kneedle_corner,
    lcurve_sweep,
    lcurve_sweep_vmap,
)
from .bootstrap import bootstrap_threshold_uncertainty_2d
from .plotting import (
    plot_bootstrap_mask_summary,
    plot_lcurve,
    plot_physical_bootstrap_uncertainty,
)
from .synthetic import (
    domain_wall_magnetization,
    soft_disc_support,
    uniform_magnetization,
    vortex_magnetization,
)
from .fixtures import (
    generate_vortex_disc_fixture,
    load_vortex_disc_fixture,
    save_vortex_disc_fixture,
    vortex_disc_fixture_path,
)
from .energy_backend import NeuralMagEnergyBackend
from .inversion import (
    FieldState,
    IdentityBackend,
    InversionResult,
    NeuralMagCritic,
    PhysicsBackend,
    SmoothnessBackend,
    invert_magnetization,
    project_unit_norm,
)

__all__ = [
    "B_REF",
    "BootstrapThresholdResult",
    "ELECTRON_INTERACTION_CONSTANT_300KV",
    "FieldState",
    "IdentityBackend",
    "InversionResult",
    "KERNEL_COEFF",
    "LCurveResult",
    "MU_0",
    "NeuralMagCritic",
    "NeuralMagEnergyBackend",
    "NewtonCGConfig",
    "PHI_0",
    "PhysicsBackend",
    "RampCoeffs",
    "RegConfig",
    "SmoothnessBackend",
    "SolverConfig",
    "SolverResult",
    "add_units_to_inputs",
    "apply_ramp",
    "bootstrap_threshold_uncertainty_2d",
    "build_rdfc_kernel",
    "decompose_loss",
    "estimate_mip_phase_from_thickness",
    "estimate_thickness_from_mip_phase",
    "exchange_loss_fn",
    "forward_phase_from_density_and_magnetization",
    "forward_model_2d",
    "forward_model_3d",
    "forward_model_single_rdfc_2d",
    "generate_vortex_disc_fixture",
    "get_freq_grid",
    "invert_magnetization",
    "kneedle_corner",
    "lcurve_sweep",
    "lcurve_sweep_vmap",
    "load_vortex_disc_fixture",
    "make_quantity",
    "mbir_loss_2d",
    "phase_mapper_rdfc",
    "plot_bootstrap_mask_summary",
    "plot_lcurve",
    "plot_physical_bootstrap_uncertainty",
    "project_3d",
    "project_unit_norm",
    "reconstruct_2d",
    "reconstruct_2d_ensemble",
    "solve_mbir_2d",
    "to_local_induction",
    "to_local_magnetization",
    "to_projected_induction_integral",
    "to_projected_magnetization_integral",
    "domain_wall_magnetization",
    "soft_disc_support",
    "save_vortex_disc_fixture",
    "uniform_magnetization",
    "vortex_disc_fixture_path",
    "vortex_magnetization",
]
