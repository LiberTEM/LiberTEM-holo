import numpy as np

from libertem_holo.base.mbir.inversion import (
    plot_depth_profile,
    plot_loss_history,
    plot_loss_landscape_2d,
    plot_m_slices,
)


def test_plot_loss_history_returns_line_data():
    fig, ax, info = plot_loss_history(np.array([3.0, 2.0, 1.0], dtype=np.float32), label="A.0")
    assert info["iterations"].tolist() == [1, 2, 3]
    assert info["loss_history"].tolist() == [3.0, 2.0, 1.0]
    fig.clear()


def test_plot_depth_profile_returns_expected_profiles():
    m_true = np.zeros((4, 3, 3, 3), dtype=np.float32)
    m_recon = np.zeros_like(m_true)
    m_true[:, 1, 1, 0] = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    m_recon[:, 1, 1, 0] = np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32)

    fig, ax, info = plot_depth_profile(m_recon, m_true, (1, 1), component=0)
    assert info["yx"] == (1, 1)
    assert info["component"] == 0
    assert info["true_profile"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert info["recon_profile"].tolist() == [3.0, 2.0, 1.0, 0.0]
    fig.clear()


def test_plot_m_slices_returns_selected_plane():
    m_true = np.zeros((5, 4, 4, 3), dtype=np.float32)
    m_recon = np.zeros_like(m_true)
    m_true[2, ..., 2] = 1.0
    m_recon[2, ..., 2] = -1.0

    fig, axes, info = plot_m_slices(m_recon, m_true, z_index=2, component=2)
    assert info["z_index"] == 2
    assert info["component"] == 2
    assert info["true_slice"].shape == (4, 4)
    assert info["recon_slice"].shape == (4, 4)
    fig.clear()


def test_plot_loss_landscape_2d_returns_minimum():
    x_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y_values = np.array([10.0, 20.0], dtype=np.float32)
    loss_grid = np.array([[5.0, 1.0, 4.0], [6.0, 7.0, 8.0]], dtype=np.float32)

    fig, ax, info = plot_loss_landscape_2d(x_values, y_values, loss_grid, x_label="A_ex", y_label="Ms")
    assert info["minimum"] == (2.0, 10.0, 1.0)
    fig.clear()