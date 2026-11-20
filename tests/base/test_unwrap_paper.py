"""Tests of the reference implementation of the IRLS paper (`unwrap._irls_paper`)."""
from __future__ import annotations

import numpy as np
import pytest

from libertem_holo.base.unwrap import unwrap_phase_irls
from libertem_holo.base.unwrap._irls_paper import (
    PaperOptions,
    _PaperSystem,
    unwrap_phase_irls_paper,
)
from libertem_holo.base.unwrap.irls import _PoissonPreconditioner


def _wrap(value: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * value))


def _difference_matrix(n: int) -> np.ndarray:
    """Build the paper's S: (n - 1) x n forward differences."""
    s = np.zeros((n - 1, n))
    s[np.arange(n - 1), np.arange(n - 1)] = -1
    s[np.arange(n - 1), np.arange(1, n)] = 1
    return s


def test_cosine_transforms_solve_the_papers_sylvester_equation():
    """The U block of the preconditioner (eq. 26-31) by eigendecompositions."""
    rows, cols = 7, 5
    rng = np.random.default_rng(0)
    r = rng.normal(size=(rows, cols))
    r -= r.mean()  # in the range of the operator
    s = _difference_matrix(rows)
    t = _difference_matrix(cols).T  # the paper's T is M x (M - 1)
    lambda_s, p_s = np.linalg.eigh(s.T @ s)
    lambda_t, p_t = np.linalg.eigh(t @ t.T)
    denominator = lambda_s[:, None] + lambda_t[None, :]
    transformed = p_s.T @ r @ p_t
    z_prime = np.zeros_like(transformed)
    nonzero = denominator > 1e-12
    z_prime[nonzero] = transformed[nonzero] / denominator[nonzero]
    expected = p_s @ z_prime @ p_t.T

    poisson = _PoissonPreconditioner((rows, cols), np.dtype(np.float64), pad=False)
    np.testing.assert_allclose(poisson(r), expected, atol=1e-12)


def test_system_matrix_is_symmetric_positive_semidefinite():
    rng = np.random.default_rng(1)
    wrapped = _wrap(rng.normal(size=(9, 11)) * 3)
    weights = (rng.uniform(0.1, 1, size=(9, 10)), rng.uniform(0.1, 1, size=(8, 11)))
    system = _PaperSystem(wrapped, weights, PaperOptions(), workers=None)
    system.set_weights(
        (rng.uniform(1e-3, 1, size=(9, 10)), rng.uniform(1e-3, 1, size=(8, 11))),
    )

    def random_blocks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            rng.normal(size=(9, 11)),
            rng.normal(size=(9, 10)),
            rng.normal(size=(8, 11)),
        )

    def dot(a: tuple, b: tuple) -> float:
        return sum(float(np.vdot(x, y)) for x, y in zip(a, b, strict=True))

    a, b = random_blocks(), random_blocks()
    assert np.isclose(dot(system.apply(a), b), dot(a, system.apply(b)))
    assert dot(a, system.apply(a)) > 0


@pytest.mark.parametrize("shape", [(24, 31), (13, 29), (1, 7), (7, 1)])
def test_paper_recovers_a_ramp(shape):
    y, x = np.mgrid[:shape[0], :shape[1]]
    truth = 0.7 * y + 0.9 * x
    result = unwrap_phase_irls_paper(_wrap(truth))
    for phase in (result.phase, result.unwrapped):
        error = phase - truth
        np.testing.assert_allclose(error, error.mean(), atol=1e-6)
    assert result.cut_turns == 0
    assert result.stopped_by == "heuristic"


def test_paper_cg_budget_follows_the_heuristic():
    y, x = np.mgrid[:40, :48].astype(float)
    wrapped = _wrap(np.arctan2(y - 19.5, x - 23.5) - np.arctan2(y - 12, x - 30))
    options = PaperOptions()
    result = unwrap_phase_irls_paper(wrapped, options=options)
    budgets = [i.cg_budget for i in result.iterations]
    assert budgets[0] == options.cg_budget_start
    for before, after, iteration in zip(
        budgets, budgets[1:], result.iterations, strict=False,
    ):
        if iteration.relative_improvement > options.improvement_tolerance:
            assert after == before
        else:
            assert after == int(np.ceil(options.cg_budget_factor * before))
    # The solve stops after an iteration with an increased budget that did not
    # improve the objective significantly:
    assert result.stopped_by == "heuristic"
    assert budgets[-1] > budgets[-2]
    assert result.iterations[-1].relative_improvement <= options.improvement_tolerance


def test_paper_and_irls_agree_on_a_dipole():
    y, x = np.mgrid[:40, :48].astype(float)
    wrapped = _wrap(np.arctan2(y - 19.5, x - 23.5) - np.arctan2(y - 12, x - 30))
    paper = unwrap_phase_irls_paper(wrapped)
    ours = unwrap_phase_irls(wrapped)
    turns = (paper.phase - wrapped) / (2 * np.pi)
    np.testing.assert_allclose(turns, np.round(turns), atol=1e-9)
    assert paper.cut_turns == ours.cut_turns


def test_paper_uses_amplitude_weights_for_a_complex_wave():
    rng = np.random.default_rng(3)
    y, x = np.mgrid[:64, :64] / 64
    truth = 10 * np.exp(-((y - .5)**2 + (x - .5)**2) / .05)
    blocked = (y - 0.2)**2 + (x - 0.2)**2 < 0.01
    phase = np.where(blocked, rng.uniform(-np.pi, np.pi, size=truth.shape), truth)
    wave = np.where(blocked, 0.02, 1.0) * np.exp(1j * phase)
    result = unwrap_phase_irls_paper(wave)
    assert result.cut_cost < result.cut_turns
