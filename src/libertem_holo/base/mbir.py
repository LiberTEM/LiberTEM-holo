"""Model-Based Iterative Reconstruction (MBIR) for magnetic electron holography.

This module provides functions for recovering in-plane magnetic fields from
electron holography phase images using model-based iterative reconstruction
with Tikhonov regularization. All physical parameters are expressed using
:mod:`unxt` quantities for clarity and consistency.

The relationship between the measured electron phase and the projected
in-plane magnetic field components (for a uniformly magnetized film of
thickness *t*) is given by the Aharonov-Bohm effect:

.. math::

    \\frac{\\partial \\phi}{\\partial x} = -\\frac{e \\, t}{\\hbar} B_y

    \\frac{\\partial \\phi}{\\partial y} = +\\frac{e \\, t}{\\hbar} B_x

where :math:`\\phi` is the magnetic phase shift (radians), *e* is the
elementary charge, :math:`\\hbar` is the reduced Planck constant, and
:math:`B_x`, :math:`B_y` are the in-plane magnetic field components.

References
----------
.. [1] Lichte, H., & Lehmann, M. (2008). Electron holography—basics and
       applications. *Reports on Progress in Physics*, 71, 016102.
       https://doi.org/10.1088/0034-4885/71/1/016102

.. [2] Mansuripur, M. (1991). Computation of electron diffraction patterns
       in Lorentz electron microscopy of thin magnetic films.
       *Journal of Applied Physics*, 69, 2455.
       https://doi.org/10.1063/1.348682
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import unxt

# Physical constants as unxt.Quantity objects
#: Elementary charge
ELEMENTARY_CHARGE: unxt.Quantity = unxt.Quantity(1.602176634e-19, "C")

#: Reduced Planck constant
HBAR: unxt.Quantity = unxt.Quantity(1.054571817e-34, "J s")

#: Magnetic flux quantum  Φ₀ = h / (2e)
FLUX_QUANTUM: unxt.Quantity = unxt.Quantity(2.067833848e-15, "T m2")


@dataclass
class MBIRResult:
    """Container for MBIR reconstruction results.

    Attributes
    ----------
    b_x
        In-plane magnetic field component along x (columns), with units.
    b_y
        In-plane magnetic field component along y (rows), with units.
    b_magnitude
        Magnitude of the in-plane magnetic field, with units.
    voxel_size
        Pixel size of the reconstructed B-field map, with units.
    regularization_parameter
        The regularization parameter λ used in the reconstruction.
    """

    b_x: unxt.Quantity
    b_y: unxt.Quantity
    b_magnitude: unxt.Quantity
    voxel_size: unxt.Quantity
    regularization_parameter: float


@dataclass
class LCurveResult:
    """Container for L-curve analysis results.

    Attributes
    ----------
    lambdas
        Array of regularization parameter values tested.
    residual_norms
        Residual norms ||A x - b|| for each regularization parameter.
    solution_norms
        Solution norms ||x|| for each regularization parameter.
    optimal_lambda
        The optimal regularization parameter selected from the L-curve corner.
    """

    lambdas: np.ndarray
    residual_norms: np.ndarray
    solution_norms: np.ndarray
    optimal_lambda: float = field(init=False)

    def __post_init__(self) -> None:
        self.optimal_lambda = float(
            self.lambdas[_find_lcurve_corner(self.residual_norms, self.solution_norms)]
        )


def _phase_conversion_factor(
    thickness: unxt.Quantity,
    voxel_size: unxt.Quantity,
) -> float:
    """Compute phase-to-B conversion factor in SI units.

    Parameters
    ----------
    thickness
        Sample thickness as a :class:`unxt.Quantity` with length units.
    voxel_size
        Pixel size as a :class:`unxt.Quantity` with length units.

    Returns
    -------
    float
        Conversion factor C = e * t / ħ  [rad / (T · m)]
        such that  ``phase_gradient [rad/pixel] = C * voxel_size [m] * B [T]``.
    """
    t_m = float(unxt.ustrip("m", thickness))
    pix_m = float(unxt.ustrip("m", voxel_size))
    e_C = float(unxt.ustrip("C", ELEMENTARY_CHARGE))
    hbar_Js = float(unxt.ustrip("J s", HBAR))
    # C = e * t / hbar [rad/(T*m)] such that:
    #   dphi/dx [rad/pixel] = C * pix_m [m/pixel] * B [T]
    return e_C * t_m * pix_m / hbar_Js


def phase_to_b_field(
    phase: unxt.Quantity,
    *,
    voxel_size: unxt.Quantity,
    thickness: unxt.Quantity,
    output_unit: str = "T",
) -> tuple[unxt.Quantity, unxt.Quantity]:
    """Convert a magnetic phase image to in-plane B-field components.

    Uses a simple finite-difference gradient to invert the phase-to-B
    relationship. This method is noise-sensitive; for regularized inversion
    see :func:`reconstruct_b_field_tikhonov`.

    Parameters
    ----------
    phase
        2-D magnetic phase image as a :class:`unxt.Quantity` with angle
        units (e.g. ``'rad'``). The electrostatic contribution should be
        removed before passing in this array.
    voxel_size
        Real-space pixel size of the phase image as a
        :class:`unxt.Quantity` with length units (e.g. ``'nm'``).
    thickness
        Projected sample thickness as a :class:`unxt.Quantity` with length
        units (e.g. ``'nm'``).
    output_unit
        SI unit string for the returned B-field quantities.
        Common choices: ``'T'``, ``'mT'``.

    Returns
    -------
    b_x, b_y
        In-plane field components :math:`B_x` and :math:`B_y` as
        :class:`unxt.Quantity` objects with *output_unit* units, where
        *x* is along columns and *y* is along rows.

    Examples
    --------
    >>> import numpy as np
    >>> import unxt
    >>> from libertem_holo.base.mbir import phase_to_b_field
    >>> rng = np.random.default_rng(0)
    >>> phase_arr = rng.standard_normal((32, 32))
    >>> phase = unxt.Quantity(phase_arr, 'rad')
    >>> voxel = unxt.Quantity(5.0, 'nm')
    >>> thickness = unxt.Quantity(50.0, 'nm')
    >>> bx, by = phase_to_b_field(phase, voxel_size=voxel, thickness=thickness)
    >>> str(bx.unit)
    'T'
    """
    phase_arr = np.asarray(phase.value)
    conv = _phase_conversion_factor(thickness, voxel_size)

    # grad_y = dphi/dy (row direction), grad_x = dphi/dx (col direction)
    grad_y, grad_x = np.gradient(phase_arr)

    # Bx = (hbar / (e*t*pix)) * dphi/dy  =>  Bx [T] = grad_y / conv
    # By = -(hbar / (e*t*pix)) * dphi/dx  =>  By [T] = -grad_x / conv
    b_x_arr = grad_y / conv
    b_y_arr = -grad_x / conv

    b_x = unxt.uconvert(output_unit, unxt.Quantity(b_x_arr, "T"))
    b_y = unxt.uconvert(output_unit, unxt.Quantity(b_y_arr, "T"))

    return b_x, b_y


def b_field_to_phase(
    b_x: unxt.Quantity,
    b_y: unxt.Quantity,
    *,
    voxel_size: unxt.Quantity,
    thickness: unxt.Quantity,
    output_unit: str = "rad",
) -> unxt.Quantity:
    """Compute the expected magnetic phase shift from in-plane B-field components.

    This is the *forward model* used in MBIR: given a magnetic field map it
    computes the phase that an electron would accumulate.

    Parameters
    ----------
    b_x
        In-plane field component along x (columns) as a
        :class:`unxt.Quantity` with magnetic flux density units.
    b_y
        In-plane field component along y (rows) as a
        :class:`unxt.Quantity` with magnetic flux density units.
    voxel_size
        Real-space pixel size as a :class:`unxt.Quantity` with length units.
    thickness
        Projected sample thickness as a :class:`unxt.Quantity` with length
        units.
    output_unit
        Unit string for the returned phase quantity.  Typically ``'rad'``.

    Returns
    -------
    phase
        Predicted phase image as a :class:`unxt.Quantity`.

    Examples
    --------
    >>> import numpy as np
    >>> import unxt
    >>> from libertem_holo.base.mbir import b_field_to_phase
    >>> b_x = unxt.Quantity(np.zeros((32, 32)), 'T')
    >>> b_y = unxt.Quantity(np.zeros((32, 32)), 'T')
    >>> voxel = unxt.Quantity(5.0, 'nm')
    >>> thickness = unxt.Quantity(50.0, 'nm')
    >>> phi = b_field_to_phase(b_x, b_y, voxel_size=voxel, thickness=thickness)
    >>> str(phi.unit)
    'rad'
    """
    bx_arr = np.asarray(unxt.ustrip("T", b_x))
    by_arr = np.asarray(unxt.ustrip("T", b_y))
    conv = _phase_conversion_factor(thickness, voxel_size)

    ny, nx = bx_arr.shape
    kx = np.fft.fftfreq(nx) * 2 * np.pi  # rad/pixel
    ky = np.fft.fftfreq(ny) * 2 * np.pi
    kx2d, ky2d = np.meshgrid(kx, ky)
    k2 = kx2d**2 + ky2d**2
    k2[0, 0] = 1.0  # avoid division by zero; DC component set to 0 below

    bx_hat = np.fft.fft2(bx_arr)
    by_hat = np.fft.fft2(by_arr)

    # phase_hat = -conv * i * (ky*Bx_hat - kx*By_hat) / k^2
    # (derived from ∂φ/∂y = C·Bx, ∂φ/∂x = -C·By via the Poisson equation)
    phase_hat = -conv * 1j * (ky2d * bx_hat - kx2d * by_hat) / k2
    phase_hat[0, 0] = 0.0  # zero mean

    phase_arr = np.real(np.fft.ifft2(phase_hat))
    phase = unxt.uconvert(output_unit, unxt.Quantity(phase_arr, "rad"))
    return phase


def reconstruct_b_field_tikhonov(
    phase: unxt.Quantity,
    *,
    voxel_size: unxt.Quantity,
    thickness: unxt.Quantity,
    regularization_parameter: float = 1e-3,
    output_unit: str = "T",
) -> MBIRResult:
    """Reconstruct in-plane magnetic field using Tikhonov regularization.

    Solves the linear inverse problem

    .. math::

        \\min_{B_x, B_y} \\| A(B_x, B_y) - \\phi \\|^2
        + \\lambda \\bigl( \\| B_x \\|^2 + \\| B_y \\|^2 \\bigr)

    in the Fourier domain, where *A* is the forward operator mapping
    magnetic field to phase (see :func:`b_field_to_phase`).

    Parameters
    ----------
    phase
        2-D magnetic phase image as a :class:`unxt.Quantity` with angle
        units (e.g. ``'rad'``). The electrostatic contribution should be
        removed before passing in.
    voxel_size
        Real-space pixel size of the phase image as a
        :class:`unxt.Quantity` with length units (e.g. ``'nm'``).
    thickness
        Projected sample thickness as a :class:`unxt.Quantity` with length
        units (e.g. ``'nm'``).
    regularization_parameter
        Tikhonov regularization parameter *λ*.  Larger values produce
        smoother (but less accurate) results.  Use :func:`compute_lcurve`
        to select an appropriate value.
    output_unit
        Unit string for the returned B-field components.

    Returns
    -------
    MBIRResult
        Dataclass containing the reconstructed field components
        :attr:`~MBIRResult.b_x` and :attr:`~MBIRResult.b_y`, their
        :attr:`~MBIRResult.b_magnitude`, the input
        :attr:`~MBIRResult.voxel_size`, and the
        :attr:`~MBIRResult.regularization_parameter` used.

    Examples
    --------
    >>> import numpy as np
    >>> import unxt
    >>> from libertem_holo.base.mbir import reconstruct_b_field_tikhonov
    >>> rng = np.random.default_rng(0)
    >>> phase_arr = rng.standard_normal((32, 32))
    >>> phase = unxt.Quantity(phase_arr, 'rad')
    >>> voxel = unxt.Quantity(5.0, 'nm')
    >>> thickness = unxt.Quantity(50.0, 'nm')
    >>> result = reconstruct_b_field_tikhonov(
    ...     phase, voxel_size=voxel, thickness=thickness,
    ...     regularization_parameter=1e-3
    ... )
    >>> str(result.b_x.unit)
    'T'
    """
    phase_arr = np.asarray(phase.value)
    conv = _phase_conversion_factor(thickness, voxel_size)

    ny, nx = phase_arr.shape
    kx = np.fft.fftfreq(nx) * 2 * np.pi
    ky = np.fft.fftfreq(ny) * 2 * np.pi
    kx2d, ky2d = np.meshgrid(kx, ky)
    k2 = kx2d**2 + ky2d**2
    k2[0, 0] = 1.0  # avoid division by zero

    phase_hat = np.fft.fft2(phase_arr)

    # Forward operator: Φ̃ = -i*C*(ky*B̃x - kx*B̃y)/k²
    # Adjoint: A†Φ̃ → B̃x = i*ky*C*Φ̃/k², B̃y = -i*kx*C*Φ̃/k²
    # |A|² per component = C²*(ky² or kx²)/k⁴
    # Normal equations in divergence-free parameterisation (q̃ = k*φ̃*i/C):
    #   B̃x = i*ky*C*Φ̃ / (C² + λ*k²)
    #   B̃y = -i*kx*C*Φ̃ / (C² + λ*k²)
    denominator = conv**2 + regularization_parameter * k2

    bx_hat = 1j * ky2d * phase_hat * conv / denominator
    by_hat = -1j * kx2d * phase_hat * conv / denominator

    # Force DC component to zero (undefined mean field)
    bx_hat[0, 0] = 0.0
    by_hat[0, 0] = 0.0

    bx_arr = np.real(np.fft.ifft2(bx_hat))
    by_arr = np.real(np.fft.ifft2(by_hat))

    b_x = unxt.uconvert(output_unit, unxt.Quantity(bx_arr, "T"))
    b_y = unxt.uconvert(output_unit, unxt.Quantity(by_arr, "T"))
    b_mag_arr = np.sqrt(bx_arr**2 + by_arr**2)
    b_mag = unxt.uconvert(output_unit, unxt.Quantity(b_mag_arr, "T"))

    return MBIRResult(
        b_x=b_x,
        b_y=b_y,
        b_magnitude=b_mag,
        voxel_size=voxel_size,
        regularization_parameter=regularization_parameter,
    )


def compute_lcurve(
    phase: unxt.Quantity,
    *,
    voxel_size: unxt.Quantity,
    thickness: unxt.Quantity,
    lambdas: np.ndarray | None = None,
    n_lambdas: int = 50,
    lambda_min: float = 1e-6,
    lambda_max: float = 1e2,
) -> LCurveResult:
    """Compute the L-curve for Tikhonov regularization parameter selection.

    The L-curve is a log-log plot of the residual norm
    :math:`\\|A x - b\\|` vs the solution norm :math:`\\|x\\|` for a range
    of regularization parameters.  The optimal parameter is near the
    *corner* of the L-curve, where both norms are balanced.

    Parameters
    ----------
    phase
        2-D magnetic phase image as a :class:`unxt.Quantity` with angle
        units (e.g. ``'rad'``).
    voxel_size
        Real-space pixel size as a :class:`unxt.Quantity` with length
        units.
    thickness
        Projected sample thickness as a :class:`unxt.Quantity` with length
        units.
    lambdas
        Array of regularization parameters to test.  If *None*, a
        log-spaced array of *n_lambdas* values between *lambda_min* and
        *lambda_max* is used.
    n_lambdas
        Number of regularization parameters to test (used when *lambdas*
        is *None*).
    lambda_min
        Minimum regularization parameter (used when *lambdas* is *None*).
    lambda_max
        Maximum regularization parameter (used when *lambdas* is *None*).

    Returns
    -------
    LCurveResult
        Dataclass with :attr:`~LCurveResult.lambdas`,
        :attr:`~LCurveResult.residual_norms`,
        :attr:`~LCurveResult.solution_norms`, and the automatically
        identified :attr:`~LCurveResult.optimal_lambda`.

    Examples
    --------
    >>> import numpy as np
    >>> import unxt
    >>> from libertem_holo.base.mbir import compute_lcurve
    >>> rng = np.random.default_rng(0)
    >>> phase = unxt.Quantity(rng.standard_normal((32, 32)), 'rad')
    >>> voxel = unxt.Quantity(5.0, 'nm')
    >>> thickness = unxt.Quantity(50.0, 'nm')
    >>> lcurve = compute_lcurve(phase, voxel_size=voxel, thickness=thickness,
    ...                         n_lambdas=10)
    >>> lcurve.lambdas.shape
    (10,)
    """
    if lambdas is None:
        lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), n_lambdas)

    phase_arr = np.asarray(phase.value)
    conv = _phase_conversion_factor(thickness, voxel_size)

    ny, nx = phase_arr.shape
    kx = np.fft.fftfreq(nx) * 2 * np.pi
    ky = np.fft.fftfreq(ny) * 2 * np.pi
    kx2d, ky2d = np.meshgrid(kx, ky)
    k2 = kx2d**2 + ky2d**2
    k2[0, 0] = 1.0

    phase_hat = np.fft.fft2(phase_arr)

    residual_norms = np.zeros(len(lambdas))
    solution_norms = np.zeros(len(lambdas))

    for i, lam in enumerate(lambdas):
        denominator = conv**2 + lam * k2

        bx_hat = 1j * ky2d * phase_hat * conv / denominator
        by_hat = -1j * kx2d * phase_hat * conv / denominator
        bx_hat[0, 0] = 0.0
        by_hat[0, 0] = 0.0

        bx_arr = np.real(np.fft.ifft2(bx_hat))
        by_arr = np.real(np.fft.ifft2(by_hat))

        # Residual: ||A(Bx,By) - phi||
        phase_hat_rec = -conv * 1j * (ky2d * bx_hat - kx2d * by_hat) / k2
        phase_hat_rec[0, 0] = 0.0
        phi_rec = np.real(np.fft.ifft2(phase_hat_rec))
        residual_norms[i] = np.linalg.norm(phi_rec - phase_arr)

        # Solution norm: sqrt(||Bx||^2 + ||By||^2)
        solution_norms[i] = np.sqrt(
            np.linalg.norm(bx_arr) ** 2 + np.linalg.norm(by_arr) ** 2
        )

    return LCurveResult(
        lambdas=lambdas,
        residual_norms=residual_norms,
        solution_norms=solution_norms,
    )


def _find_lcurve_corner(
    residual_norms: np.ndarray,
    solution_norms: np.ndarray,
) -> int:
    """Find the corner of the L-curve using maximum curvature.

    Computes the curvature of the L-curve in log-log space and returns
    the index of the maximum curvature (the corner).

    Parameters
    ----------
    residual_norms
        Array of residual norms.
    solution_norms
        Array of solution norms.

    Returns
    -------
    int
        Index of the L-curve corner.
    """
    log_r = np.log(np.maximum(residual_norms, 1e-300))
    log_s = np.log(np.maximum(solution_norms, 1e-300))

    # First and second derivatives (finite differences)
    dr = np.gradient(log_r)
    ds = np.gradient(log_s)
    ddr = np.gradient(dr)
    dds = np.gradient(ds)

    # Curvature: kappa = (r' * s'' - s' * r'') / (r'^2 + s'^2)^(3/2)
    denom = (dr**2 + ds**2) ** 1.5
    denom = np.where(denom > 1e-300, denom, 1e-300)
    curvature = (dr * dds - ds * ddr) / denom

    return int(np.argmax(curvature))


def phase_scale_factor(
    *,
    voxel_size: unxt.Quantity,
    thickness: unxt.Quantity,
) -> unxt.Quantity:
    """Return the multiplicative factor converting B-field to phase gradient.

    Returns the scalar *C* such that

    .. math::

        \\frac{\\partial \\phi}{\\partial x} \\,[\\text{rad/pixel}]
        = C \\cdot B_y \\,[\\text{T}]

    where *C = e · t · Δx / ħ*.

    Parameters
    ----------
    voxel_size
        Real-space pixel size as a :class:`unxt.Quantity` with length units.
    thickness
        Sample thickness as a :class:`unxt.Quantity` with length units.

    Returns
    -------
    unxt.Quantity
        Scale factor as a dimensionless :class:`unxt.Quantity` expressed in
        ``rad / T``.

    Examples
    --------
    >>> import unxt
    >>> from libertem_holo.base.mbir import phase_scale_factor
    >>> voxel = unxt.Quantity(5.0, 'nm')
    >>> thickness = unxt.Quantity(50.0, 'nm')
    >>> C = phase_scale_factor(voxel_size=voxel, thickness=thickness)
    >>> str(C.unit)
    'rad / T'
    """
    t_m = unxt.uconvert("m", thickness)
    pix_m = unxt.uconvert("m", voxel_size)
    # C = e * t * pix / hbar  [C * m * m / (J*s)] = [C*m²/(J*s)]
    # J = kg*m²/s², C/s = A, T = kg/(A*s²) = kg*s/(C*m²)
    # So [C*m²/(J*s)] = [C*m²/(kg*m²/s²*s)] = [C*s/(kg)] = [1/T] = [rad/T]
    # Return as an explicit 'rad/T' quantity
    c_val = float(unxt.ustrip("m", t_m)) * float(unxt.ustrip("m", pix_m))
    c_val *= float(unxt.ustrip("C", ELEMENTARY_CHARGE))
    c_val /= float(unxt.ustrip("J s", HBAR))
    return unxt.Quantity(c_val, "rad / T")


def plot_b_field(
    result: MBIRResult,
    *,
    ax_bx=None,
    ax_by=None,
    ax_bmag=None,
    cmap_component: str = "RdBu_r",
    cmap_magnitude: str = "viridis",
    vmax: float | None = None,
) -> None:
    """Plot reconstructed B-field components and magnitude.

    Parameters
    ----------
    result
        An :class:`MBIRResult` instance as returned by
        :func:`reconstruct_b_field_tikhonov`.
    ax_bx
        Matplotlib axes for the :math:`B_x` component.  If *None* a new
        figure is created.
    ax_by
        Matplotlib axes for the :math:`B_y` component.
    ax_bmag
        Matplotlib axes for the field magnitude.
    cmap_component
        Colormap name for the signed field components.
    cmap_magnitude
        Colormap name for the field magnitude.
    vmax
        Symmetric colour-scale maximum for the component plots.
        If *None*, the 99th percentile of the magnitude is used.
    """
    import matplotlib.pyplot as plt

    if ax_bx is None or ax_by is None or ax_bmag is None:
        fig, (ax_bx, ax_by, ax_bmag) = plt.subplots(1, 3, figsize=(12, 4))

    unit_label = str(result.b_x.unit)
    bx = np.asarray(result.b_x.value)
    by = np.asarray(result.b_y.value)
    bmag = np.asarray(result.b_magnitude.value)

    if vmax is None:
        vmax = float(np.percentile(bmag, 99))

    pix = float(unxt.ustrip("nm", result.voxel_size))

    im0 = ax_bx.imshow(bx, cmap=cmap_component, vmin=-vmax, vmax=vmax, origin="lower")
    ax_bx.set_title(f"$B_x$ [{unit_label}]")
    plt.colorbar(im0, ax=ax_bx, label=unit_label)

    im1 = ax_by.imshow(by, cmap=cmap_component, vmin=-vmax, vmax=vmax, origin="lower")
    ax_by.set_title(f"$B_y$ [{unit_label}]")
    plt.colorbar(im1, ax=ax_by, label=unit_label)

    im2 = ax_bmag.imshow(bmag, cmap=cmap_magnitude, vmin=0, vmax=vmax, origin="lower")
    ax_bmag.set_title(f"|B| [{unit_label}]")
    plt.colorbar(im2, ax=ax_bmag, label=unit_label)

    for ax in (ax_bx, ax_by, ax_bmag):
        ax.set_xlabel(f"x [pixel × {pix:.2g} nm]")
        ax.set_ylabel(f"y [pixel × {pix:.2g} nm]")


def plot_lcurve(
    lcurve: LCurveResult,
    *,
    ax=None,
    mark_optimal: bool = True,
) -> None:
    """Plot the L-curve for regularization parameter selection.

    Parameters
    ----------
    lcurve
        An :class:`LCurveResult` instance as returned by
        :func:`compute_lcurve`.
    ax
        Matplotlib axes to plot into.  If *None* a new figure is created.
    mark_optimal
        If *True*, mark the selected optimal regularization parameter.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.loglog(lcurve.residual_norms, lcurve.solution_norms, "b-o", markersize=3)

    if mark_optimal:
        opt_idx = _find_lcurve_corner(lcurve.residual_norms, lcurve.solution_norms)
        ax.loglog(
            lcurve.residual_norms[opt_idx],
            lcurve.solution_norms[opt_idx],
            "r*",
            markersize=12,
            label=f"λ = {lcurve.optimal_lambda:.2e}",
        )
        ax.legend()

    ax.set_xlabel("Residual norm $\\|A\\mathbf{B} - \\phi\\|$")
    ax.set_ylabel("Solution norm $\\|\\mathbf{B}\\|$")
    ax.set_title("L-curve")


def reconstruct_b_field_pyramid(
    phase: unxt.Quantity,
    *,
    voxel_size: unxt.Quantity,
    thickness: unxt.Quantity,
    regularization_parameter: float = 1e-3,
    n_levels: int = 3,
    output_unit: str = "T",
) -> MBIRResult:
    """Reconstruct in-plane magnetic field using the multi-scale pyramid method.

    The pyramid method performs Tikhonov-regularised reconstruction at
    multiple spatial frequency scales, progressively refining the estimate
    from coarse to fine resolution.  This reduces ringing artefacts
    compared to single-scale Tikhonov regularization.

    Parameters
    ----------
    phase
        2-D magnetic phase image as a :class:`unxt.Quantity` with angle
        units (e.g. ``'rad'``).
    voxel_size
        Real-space pixel size as a :class:`unxt.Quantity` with length
        units (e.g. ``'nm'``).
    thickness
        Projected sample thickness as a :class:`unxt.Quantity` with length
        units (e.g. ``'nm'``).
    regularization_parameter
        Tikhonov regularization parameter *λ* applied at each pyramid
        level.
    n_levels
        Number of pyramid levels (scales).  Each level halves the image
        size.  More levels capture more coarse-scale structure but require
        larger images.
    output_unit
        Unit string for the returned B-field components.

    Returns
    -------
    MBIRResult
        Reconstruction result at full resolution.

    Examples
    --------
    >>> import numpy as np
    >>> import unxt
    >>> from libertem_holo.base.mbir import reconstruct_b_field_pyramid
    >>> rng = np.random.default_rng(0)
    >>> phase = unxt.Quantity(rng.standard_normal((64, 64)), 'rad')
    >>> voxel = unxt.Quantity(5.0, 'nm')
    >>> thickness = unxt.Quantity(50.0, 'nm')
    >>> result = reconstruct_b_field_pyramid(
    ...     phase, voxel_size=voxel, thickness=thickness,
    ...     regularization_parameter=1e-3, n_levels=2
    ... )
    >>> str(result.b_x.unit)
    'T'
    """
    from scipy.ndimage import zoom

    phase_arr = np.asarray(phase.value)
    ny, nx = phase_arr.shape

    bx_full = np.zeros_like(phase_arr)
    by_full = np.zeros_like(phase_arr)

    for level in range(n_levels, 0, -1):
        scale = 1.0 / 2 ** (level - 1)
        # Down-sample phase
        if scale < 1.0:
            phase_scaled = zoom(phase_arr, scale, order=1)
        else:
            phase_scaled = phase_arr.copy()

        # Effective voxel size at this scale
        voxel_scaled = unxt.uconvert(
            str(voxel_size.unit),
            unxt.Quantity(
                float(unxt.ustrip(str(voxel_size.unit), voxel_size)) / scale,
                str(voxel_size.unit),
            ),
        )

        # Tikhonov reconstruction at this scale
        phase_q = unxt.Quantity(phase_scaled, "rad")
        result_l = reconstruct_b_field_tikhonov(
            phase_q,
            voxel_size=voxel_scaled,
            thickness=thickness,
            regularization_parameter=regularization_parameter,
            output_unit="T",
        )

        bx_l = np.asarray(result_l.b_x.value)
        by_l = np.asarray(result_l.b_y.value)

        # Up-sample to full resolution
        if bx_l.shape != (ny, nx):
            bx_up = zoom(bx_l, (ny / bx_l.shape[0], nx / bx_l.shape[1]), order=1)
            by_up = zoom(by_l, (ny / by_l.shape[0], nx / by_l.shape[1]), order=1)
        else:
            bx_up = bx_l
            by_up = by_l

        # Accumulate contributions (each level corrects the residual)
        if level == n_levels:
            bx_full = bx_up
            by_full = by_up
        else:
            # Residual phase after removing contribution from coarser levels
            phase_from_coarse = b_field_to_phase(
                unxt.Quantity(bx_full, "T"),
                unxt.Quantity(by_full, "T"),
                voxel_size=voxel_size,
                thickness=thickness,
            )
            residual_arr = phase_arr - np.asarray(phase_from_coarse.value)
            phase_res_q = unxt.Quantity(residual_arr, "rad")
            res_result = reconstruct_b_field_tikhonov(
                phase_res_q,
                voxel_size=voxel_size,
                thickness=thickness,
                regularization_parameter=regularization_parameter,
                output_unit="T",
            )
            bx_full = bx_full + np.asarray(res_result.b_x.value)
            by_full = by_full + np.asarray(res_result.b_y.value)

    b_x = unxt.uconvert(output_unit, unxt.Quantity(bx_full, "T"))
    b_y = unxt.uconvert(output_unit, unxt.Quantity(by_full, "T"))
    b_mag = unxt.uconvert(
        output_unit, unxt.Quantity(np.sqrt(bx_full**2 + by_full**2), "T")
    )

    return MBIRResult(
        b_x=b_x,
        b_y=b_y,
        b_magnitude=b_mag,
        voxel_size=voxel_size,
        regularization_parameter=regularization_parameter,
    )
