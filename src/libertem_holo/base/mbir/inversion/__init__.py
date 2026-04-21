from .backends import (
    CombinedBackend,
    FieldState,
    IdentityBackend,
    NeuralMagCritic,
    PhysicsBackend,
    SmoothnessBackend,
    WeightedBackend,
)
from .experiments import (
    ScaledRhoExperimentResult,
    analytic_vortex_init,
    run_with_scaled_rho,
    support_center_yx,
)
from .metrics import (
    depth_correlation,
    equilibrium_residual,
    iterations_to_threshold,
    mz_rmse,
    phase_residual,
    projected_m_error,
    vortex_core_z_error,
)
from .plotting import (
    plot_depth_profile,
    plot_loss_history,
    plot_loss_landscape_2d,
    plot_m_slices,
)
from .solver import InversionResult, invert_magnetization, project_unit_norm

__all__ = [
    "CombinedBackend",
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
]