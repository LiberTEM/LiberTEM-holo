import numpy as np

from libertem_holo.base.mbir.synthetic import (
    domain_wall_magnetization,
    soft_disc_support,
    uniform_magnetization,
    vortex_magnetization,
)

MAX_RADIAL_COMPONENT = 0.15
MIN_TANGENTIAL_COMPONENT = 0.1


def test_soft_disc_support_shape_dtype_and_profile():
    support = soft_disc_support((3, 9, 9), radius=3.0, edge_width=0.8)

    assert support.shape == (3, 9, 9)
    assert support.dtype == np.float32
    assert np.all(np.isfinite(np.asarray(support)))
    assert np.all(np.asarray(support) >= 0.0)
    assert np.all(np.asarray(support) <= 1.0)
    assert np.asarray(support[1, 4, 4]) > np.asarray(support[0, 0, 0])


def test_uniform_magnetization_shape_and_direction():
    support = soft_disc_support((7, 7, 2), radius=2.5, edge_width=0.5)
    mag = uniform_magnetization(
        (7, 7, 2),
        support_xyz=support,
        direction_xyz=(0.0, 1.0, 0.0),
        magnitude=2.0,
    )

    assert mag.shape == (7, 7, 2, 3)
    assert np.all(np.isfinite(np.asarray(mag)))
    assert np.allclose(np.asarray(mag[..., 0]), 0.0, atol=1e-7)
    assert np.allclose(np.asarray(mag[..., 2]), 0.0, atol=1e-7)
    assert np.all(np.asarray(mag[..., 1]) >= 0.0)


def test_vortex_magnetization_curling_and_core():
    mag = vortex_magnetization((11, 11, 1), core_radius=1.2)

    assert mag.shape == (11, 11, 1, 3)
    assert np.all(np.isfinite(np.asarray(mag)))

    center_mz = float(np.asarray(mag[5, 5, 0, 2]))
    edge_mz = float(np.asarray(mag[9, 5, 0, 2]))
    assert center_mz > edge_mz

    # For a vortex, in-plane magnetization is tangential: m · r_hat ~ 0.
    my = float(np.asarray(mag[8, 5, 0, 1]))
    mx = float(np.asarray(mag[8, 5, 0, 0]))
    assert abs(mx) < MAX_RADIAL_COMPONENT
    assert my > MIN_TANGENTIAL_COMPONENT


def test_domain_wall_magnetization_transition():
    mag = domain_wall_magnetization((13, 5, 2), wall_width=1.5)

    assert mag.shape == (13, 5, 2, 3)
    assert np.all(np.isfinite(np.asarray(mag)))
    assert np.allclose(np.asarray(mag[..., 1]), 0.0, atol=1e-7)

    mx_line = np.asarray(mag[:, 0, 0, 0])
    mz_line = np.asarray(mag[:, 0, 0, 2])
    assert mx_line[0] < -0.9
    assert mx_line[-1] > 0.9
    assert mz_line[6] > mz_line[0]
