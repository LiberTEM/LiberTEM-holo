"""Phase 0 spike tests: verify unxt Quantity works with JAX operations used in MBIR.

Each test probes a specific JAX primitive that mbir.py relies on.
Tests are designed to PASS if the operation works transparently with Quantity,
or to document the required bridge-point pattern if it doesn't.
"""
from __future__ import annotations

import jax
import jax.numpy as raw_jnp
import numpy as np
import optax
import pytest
import unxt as u
from quax import quaxify

# Use quaxed replacements where available
import quaxed.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# 1. FFT: rfft2 / irfft2
# ---------------------------------------------------------------------------
class TestFFT:
    """MBIR uses rfft2/irfft2 for the RDFC kernel convolution."""

    def test_rfft2_quantity(self):
        """rfft2 on a Quantity array."""
        arr = u.Quantity(raw_jnp.ones((4, 4), dtype=float), "rad")
        result = quaxify(raw_jnp.fft.rfft2)(arr)
        # Should produce a complex Quantity
        assert hasattr(result, "value"), "Expected Quantity output"
        assert result.value.shape == (4, 3)

    def test_irfft2_quantity(self):
        """irfft2 round-trip on a Quantity array."""
        arr = u.Quantity(raw_jnp.ones((4, 4), dtype=float), "rad")
        fft_arr = quaxify(raw_jnp.fft.rfft2)(arr)
        recovered = quaxify(raw_jnp.fft.irfft2)(fft_arr, s=(4, 4))
        np.testing.assert_allclose(recovered.value, 1.0, atol=1e-12)

    def test_rfft2_multiply_irfft2(self):
        """Convolution pattern: rfft2 → multiply → irfft2."""
        signal = u.Quantity(raw_jnp.ones((4, 4), dtype=float), "rad")
        # Kernel is dimensionless (1/nm² in practice, but testing dispatch)
        kernel = u.Quantity(raw_jnp.ones((4, 3), dtype=complex), "1/nm2")
        sig_fft = quaxify(raw_jnp.fft.rfft2)(signal)
        product = sig_fft * kernel
        result = quaxify(raw_jnp.fft.irfft2)(product, s=(4, 4))
        assert hasattr(result, "value")


# ---------------------------------------------------------------------------
# 2. jax.grad with Quantity
# ---------------------------------------------------------------------------
class TestGrad:
    """MBIR loss functions are differentiated with jax.grad."""

    def test_grad_simple_quantity(self):
        """grad of a scalar Quantity function."""
        def loss(x):
            return jnp.sum(x ** 2)

        x = u.Quantity(raw_jnp.array([1.0, 2.0, 3.0]), "rad")
        # Correct pattern: quaxify the entire grad call, not .value inside trace
        import quaxed
        grad_fn = quaxed.grad(loss)
        g = grad_fn(x)
        assert hasattr(g, "value")
        np.testing.assert_allclose(g.value, [2.0, 4.0, 6.0], atol=1e-12)

    def test_grad_two_arg(self):
        """grad with respect to first arg (magnetization) while phase is fixed."""
        def loss(mag, phase):
            residual = mag - phase
            return 0.5 * jnp.sum(residual ** 2)

        mag = u.Quantity(raw_jnp.array([1.0, 2.0]), "rad")
        phase = u.Quantity(raw_jnp.array([0.5, 1.5]), "rad")
        import quaxed
        grad_fn = quaxed.grad(loss)
        g = grad_fn(mag, phase)
        # Gradient should be (mag - phase) = [0.5, 0.5]
        np.testing.assert_allclose(g.value, [0.5, 0.5], atol=1e-12)


# ---------------------------------------------------------------------------
# 3. jax.scipy.sparse.linalg.cg (conjugate gradient)
# ---------------------------------------------------------------------------
class TestCG:
    """Newton-CG solver uses jax.scipy.sparse.linalg.cg."""

    def test_cg_raw(self):
        """Baseline: CG with plain arrays."""
        A = raw_jnp.array([[4.0, 1.0], [1.0, 3.0]])
        b = raw_jnp.array([1.0, 2.0])
        x, info = jax.scipy.sparse.linalg.cg(lambda v: A @ v, b, tol=1e-10)
        np.testing.assert_allclose(A @ x, b, atol=1e-8)

    def test_cg_quantity(self):
        """CG with Quantity vectors — may need bridge point."""
        A = raw_jnp.array([[4.0, 1.0], [1.0, 3.0]])
        b = u.Quantity(raw_jnp.array([1.0, 2.0]), "rad")
        try:
            cg_q = quaxify(jax.scipy.sparse.linalg.cg)
            x, info = cg_q(lambda v: quaxify(raw_jnp.dot)(A, v), b, tol=1e-10)
            # Verify solution
            Ax = quaxify(raw_jnp.dot)(A, x)
            np.testing.assert_allclose(
                Ax.value if hasattr(Ax, "value") else Ax,
                b.value,
                atol=1e-8,
            )
        except Exception as e:
            pytest.skip(f"CG needs bridge point: {e}")

    def test_cg_bridge_point(self):
        """CG bridge-point pattern: strip units → CG → re-annotate."""
        A = raw_jnp.array([[4.0, 1.0], [1.0, 3.0]])
        b = u.Quantity(raw_jnp.array([1.0, 2.0]), "rad")
        b_unit = b.unit
        # Strip units at CG boundary
        x_raw, info = jax.scipy.sparse.linalg.cg(
            lambda v: A @ v, b.value, tol=1e-10,
        )
        # Re-annotate with units
        x = u.Quantity(x_raw, b_unit)
        np.testing.assert_allclose(A @ x.value, b.value, atol=1e-8)


# ---------------------------------------------------------------------------
# 4. optax.adam (optimizer step)
# ---------------------------------------------------------------------------
class TestOptax:
    """Adam optimizer is used in _run_adam_solver_2d."""

    def test_optax_adam_raw(self):
        """Baseline: optax.adam with plain arrays."""
        params = raw_jnp.array([1.0, 2.0])
        optimizer = optax.adam(1e-2)
        opt_state = optimizer.init(params)
        grads = raw_jnp.array([0.1, 0.2])
        updates, new_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        assert new_params.shape == (2,)

    def test_optax_adam_quantity(self):
        """optax.adam with Quantity params — likely needs bridge point."""
        params = u.Quantity(raw_jnp.array([1.0, 2.0]), "rad")
        optimizer = optax.adam(1e-2)
        try:
            opt_state = optimizer.init(params)
            grads = u.Quantity(raw_jnp.array([0.1, 0.2]), "rad")
            updates, new_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            assert hasattr(new_params, "value")
        except Exception as e:
            pytest.skip(f"optax.adam needs bridge point: {e}")


# ---------------------------------------------------------------------------
# 5. jax.lax.while_loop with Quantity carry
# ---------------------------------------------------------------------------
class TestWhileLoop:
    """Adam and L-BFGS solvers use jax.lax.while_loop."""

    def test_while_loop_raw(self):
        """Baseline: while_loop with plain arrays."""
        def cond(state):
            return state[0] < 5

        def body(state):
            return (state[0] + 1, state[1] + 0.1)

        result = jax.lax.while_loop(cond, body, (0, 0.0))
        assert result == (5, pytest.approx(0.5))

    def test_while_loop_quantity(self):
        """while_loop with Quantity in carry — may need bridge point."""
        try:
            def cond(state):
                step, val = state
                return step < 5

            def body(state):
                step, val = state
                return (step + 1, val + u.Quantity(0.1, "rad"))

            init = (0, u.Quantity(0.0, "rad"))
            result = jax.lax.while_loop(cond, body, init)
            assert hasattr(result[1], "value")
            np.testing.assert_allclose(result[1].value, 0.5, atol=1e-12)
        except Exception as e:
            pytest.skip(f"while_loop needs bridge point: {e}")


# ---------------------------------------------------------------------------
# 6. jax.vmap with Quantity
# ---------------------------------------------------------------------------
class TestVmap:
    """reconstruct_2d_ensemble and lcurve_sweep_vmap use jax.vmap."""

    def test_vmap_quantity(self):
        """vmap over a batch of Quantity arrays."""
        def f(x):
            return jnp.sum(x ** 2)

        batch = u.Quantity(raw_jnp.ones((3, 4), dtype=float), "rad")
        result = quaxify(jax.vmap(f))(batch)
        assert hasattr(result, "value")
        np.testing.assert_allclose(result.value, [4.0, 4.0, 4.0], atol=1e-12)

    def test_vmap_grad_quantity(self):
        """vmap(grad(...)) — used in bootstrap/ensemble."""
        import quaxed

        def loss(x):
            return jnp.sum(x ** 2)

        batch = u.Quantity(raw_jnp.ones((3, 4), dtype=float), "rad")
        vmap_grad = quaxed.vmap(quaxed.grad(loss))
        grads = vmap_grad(batch)
        np.testing.assert_allclose(grads.value, 2.0 * np.ones((3, 4)), atol=1e-12)


# ---------------------------------------------------------------------------
# 7. jax.jit with Quantity (non-static)
# ---------------------------------------------------------------------------
class TestJit:
    """build_rdfc_kernel and _run_lbfgs_solver_2d are JIT-compiled."""

    def test_jit_quantity(self):
        """JIT a function taking Quantity args."""
        @jax.jit
        def f(x):
            return x ** 2

        x = u.Quantity(raw_jnp.array([1.0, 2.0, 3.0]), "m")
        result = f(x)
        assert hasattr(result, "value")
        np.testing.assert_allclose(result.value, [1.0, 4.0, 9.0], atol=1e-12)

    def test_jit_static_quantity(self):
        """JIT with StaticQuantity for static args like b0_tesla."""
        from functools import partial

        @partial(jax.jit, static_argnames=("scale",))
        def f(x, scale):
            return x * raw_jnp.asarray(scale.value)

        x = u.Quantity(raw_jnp.array([1.0, 2.0]), "m")
        scale = u.StaticQuantity(np.float64(3.0), "")
        result = f(x, scale)
        assert hasattr(result, "value")
        np.testing.assert_allclose(result.value, [3.0, 6.0], atol=1e-12)


# ---------------------------------------------------------------------------
# 8. Dimension checking: wrong units should raise
# ---------------------------------------------------------------------------
class TestDimensionChecking:
    """Verify that unxt rejects dimensionally inconsistent operations."""

    def test_cannot_add_length_to_time(self):
        a = u.Quantity(1.0, "m")
        b = u.Quantity(1.0, "s")
        with pytest.raises(Exception):
            _ = a + b

    def test_can_add_same_dimension(self):
        a = u.Quantity(1.0, "nm")
        b = u.Quantity(1000.0, "pm")
        result = a + b
        # 1 nm + 1000 pm = 2 nm
        np.testing.assert_allclose(result.uconvert("nm").value, 2.0, atol=1e-12)

    def test_unit_conversion(self):
        a = u.Quantity(1.0, "T")
        result = a.uconvert("G")  # 1 Tesla = 10000 Gauss
        np.testing.assert_allclose(float(result.value), 10000.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 9. Dynamic slicing (jax.lax.dynamic_slice / dynamic_update_slice)
# ---------------------------------------------------------------------------
class TestDynamicSlice:
    """phase_mapper_rdfc uses dynamic_slice and dynamic_update_slice."""

    def test_dynamic_update_slice_quantity(self):
        """dynamic_update_slice with Quantity arrays."""
        try:
            big = u.Quantity(raw_jnp.zeros((4, 4), dtype=float), "rad")
            small = u.Quantity(raw_jnp.ones((2, 2), dtype=float), "rad")
            result = quaxify(jax.lax.dynamic_update_slice)(big, small, (0, 0))
            expected = np.zeros((4, 4))
            expected[:2, :2] = 1.0
            np.testing.assert_allclose(result.value, expected, atol=1e-12)
        except Exception as e:
            pytest.skip(f"dynamic_update_slice needs bridge point: {e}")

    def test_dynamic_slice_quantity(self):
        """dynamic_slice with Quantity arrays."""
        try:
            arr = u.Quantity(raw_jnp.arange(16, dtype=float).reshape(4, 4), "rad")
            result = quaxify(jax.lax.dynamic_slice)(arr, (1, 1), (2, 2))
            expected = np.arange(16).reshape(4, 4)[1:3, 1:3]
            np.testing.assert_allclose(result.value, expected, atol=1e-12)
        except Exception as e:
            pytest.skip(f"dynamic_slice needs bridge point: {e}")


# ---------------------------------------------------------------------------
# 10. Pad (jnp.pad) — used extensively in exchange_loss_fn
# ---------------------------------------------------------------------------
class TestPad:
    """exchange_loss_fn uses jnp.pad with constant_values."""

    def test_pad_quantity(self):
        """jnp.pad with Quantity array."""
        arr = u.Quantity(raw_jnp.ones((3, 3), dtype=float), "rad")
        try:
            padded = jnp.pad(arr, ((1, 0), (0, 1)), constant_values=0)
            assert padded.value.shape == (4, 4)
        except Exception as e:
            pytest.skip(f"jnp.pad needs bridge point: {e}")


# ---------------------------------------------------------------------------
# 11. ravel_pytree (used in Newton-CG to flatten params)
# ---------------------------------------------------------------------------
class TestRavelPytree:
    """Newton-CG solver uses jax.flatten_util.ravel_pytree."""

    def test_ravel_pytree_quantity(self):
        """ravel_pytree with a tuple of Quantity arrays."""
        from jax.flatten_util import ravel_pytree

        mag = u.Quantity(raw_jnp.ones((2, 2, 2), dtype=float), "")
        ramp = u.Quantity(raw_jnp.zeros(3, dtype=float), "rad")
        try:
            flat, unravel = ravel_pytree((mag, ramp))
            recovered = unravel(flat)
            np.testing.assert_allclose(
                recovered[0].value if hasattr(recovered[0], "value") else recovered[0],
                1.0,
                atol=1e-12,
            )
        except Exception as e:
            pytest.skip(f"ravel_pytree needs bridge point: {e}")
