"""Simple 3D voxel shapes for synthetic magnetization distributions.

These are standalone replacements for ``pyramid.magcreator.shapes`` and
``pyramid.magcreator.magcreator.create_mag_dist_homog`` that correctly
produce shapes with the exact requested voxel count along each axis.
"""

from math import pi

import numpy as np


def disc(dim, center, radius, height, axis='z'):
    """Create the boolean shape of a cylindrical disc.

    Unlike pyramid's implementation, this function guarantees that the
    disc occupies exactly *height* voxels along the cylinder axis by
    using integer-range indexing instead of a half-integer threshold.

    Parameters
    ----------
    dim : tuple (N=3)
        Grid dimensions ``(z, y, x)``.
    center : tuple (N=3)
        Centre of the disc in pixel coordinates ``(z, y, x)``.
    radius : float
        Radius of the disc in pixel coordinates.
    height : int
        Height (thickness) of the disc in voxels along *axis*.
    axis : {'z', 'y', 'x'}, optional
        Orientation of the disc axis.  Default ``'z'``.

    Returns
    -------
    shape : ndarray of bool, shape *dim*
        Boolean mask that is ``True`` inside the disc.
    """
    dim = tuple(dim)
    center = tuple(center)
    if len(dim) != 3:
        raise ValueError("dim must have length 3")
    if len(center) != 3:
        raise ValueError("center must have length 3")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if height <= 0:
        raise ValueError("height must be positive")
    if axis not in ('z', 'y', 'x'):
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")

    indices = np.indices(dim)

    # --- Radial mask (half-integer pixel centres) ---
    # Use +0.5 so the centre of each voxel is measured.
    coords = [idx + 0.5 - c for idx, c in zip(indices, center)]
    zz, yy, xx = coords

    if axis == 'z':
        uu, vv = xx, yy
        axis_idx = indices[0]
        axis_center = center[0]
    elif axis == 'y':
        uu, vv = zz, xx
        axis_idx = indices[1]
        axis_center = center[1]
    else:  # 'x'
        uu, vv = yy, zz
        axis_idx = indices[2]
        axis_center = center[2]

    radial_mask = np.hypot(uu, vv) <= radius

    # --- Height mask (integer range → exact voxel count) ---
    start = int(axis_center - height // 2)
    height_mask = (axis_idx >= start) & (axis_idx < start + height)

    return radial_mask & height_mask


def create_mag_dist_homog(mag_shape, phi, theta=pi / 2):
    """Create a homogeneous 3D magnetization distribution.

    Parameters
    ----------
    mag_shape : ndarray of bool, shape ``(Nz, Ny, Nx)``
        Boolean volume indicating the magnetized region.
    phi : float
        Azimuthal angle (radians) of the magnetization direction.
    theta : float, optional
        Polar angle (radians), default ``pi/2`` (in-plane, no
        z-component).

    Returns
    -------
    field : ndarray, shape ``(3, Nz, Ny, Nx)``
        Magnetization components ``[Mx, My, Mz]`` with unit
        amplitude inside *mag_shape* and zero outside.
    """
    mag_shape = np.asarray(mag_shape)
    if mag_shape.ndim != 3:
        raise ValueError("mag_shape must be 3-dimensional")

    x_mag = np.sin(theta) * np.cos(phi) * mag_shape
    y_mag = np.sin(theta) * np.sin(phi) * mag_shape
    z_mag = np.cos(theta) * mag_shape
    return np.array([x_mag, y_mag, z_mag])
