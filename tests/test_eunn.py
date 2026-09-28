"""Verify that EUNNCell's L-layer construction produces a valid unitary.

The tunable EUNN (Jing et al. 2017) parameterizes W = D * F_L * ... * F_1
where each F_k is a block-diagonal of H/2 independent 2x2 unitary mixing
units with alternating cyclic-shift permutations between layers. The paper
claims L=H gives full U(H) coverage; any L > 0 should still yield a valid
unitary (U Uᴴ = I). This test materializes U by feeding the identity
through ``EUNNCell._apply_unitary`` (staticmethod, no module init needed)
and checks unitarity at several (H, L) configurations, plus gradient
finiteness at the equal-superposition carry used as reset state.
"""
import jax
import jax.numpy as jnp
import pytest

from qurl.models.network import EUNNCell, initial_urnn_carry


jax.config.update('jax_platform_name', 'cpu')  # matches other test modules; harmless under JAX_PLATFORMS=cpu


# Thresholds: float32 accumulates ~1e-4 after a few composed unitaries, so
# we check correctness at float64 (tight) and do a looser float32 sanity
# check to confirm the production dtype path stays well-behaved.
UNITARITY_TOL_F64 = 1e-10
UNITARITY_TOL_F32 = 1e-3


def _materialize_unitary(angles, diag, complex_dtype):
    H = angles.shape[0]
    eye = jnp.eye(H, dtype=complex_dtype)
    return EUNNCell._apply_unitary(angles, diag, eye)


def _random_eunn_params(key, H, L, real_dtype):
    k1, k2 = jax.random.split(key)
    angles = jax.random.uniform(k1, (H, L), real_dtype, 0.0, 2 * jnp.pi)
    diag = jax.random.uniform(k2, (H,), real_dtype, 0.0, 2 * jnp.pi)
    return angles, diag


@pytest.mark.parametrize('H,L', [(4, 2), (8, 4), (16, 4), (16, 16), (32, 8)])
def test_unitarity_f64(H, L):
    """Correctness check at float64 — the math must be exactly unitary.

    x64 is scoped to this test via ``jax.experimental.enable_x64`` rather
    than a module-level ``jax.config.update`` — the latter runs at pytest
    collection time and silently promotes every other test in the session
    (see test_born_rule_actor.py::test_full_actor_critic_integration, which
    failed under a stray global x64 flag). The context manager is safe here
    because this path uses no jit/vmap/grad (see its docstring's caveat).
    """
    with jax.experimental.enable_x64():
        angles, diag = _random_eunn_params(jax.random.PRNGKey(0), H, L, jnp.float64)
        # Guard against silent float64->float32 downcast if x64 wasn't actually active.
        assert angles.dtype == jnp.float64 and diag.dtype == jnp.float64
        U = _materialize_unitary(angles, diag, jnp.complex128)
        assert U.dtype == jnp.complex128
        err = jnp.max(jnp.abs(U @ U.conj().T - jnp.eye(H, dtype=jnp.complex128)))
    assert err < UNITARITY_TOL_F64, f'(H={H}, L={L}): ‖U Uᴴ − I‖_∞ = {err:.2e}'


@pytest.mark.parametrize('H,L', [(4, 2), (8, 4), (16, 4), (16, 16), (32, 8)])
def test_unitarity_f32_numerics(H, L):
    """Sanity check at production dtype (float32): accumulated roundoff stays small."""
    angles, diag = _random_eunn_params(jax.random.PRNGKey(0), H, L, jnp.float32)
    U = _materialize_unitary(angles, diag, jnp.complex64)
    err = jnp.max(jnp.abs(U @ U.conj().T - jnp.eye(H, dtype=jnp.complex64)))
    assert err < UNITARITY_TOL_F32, f'(H={H}, L={L}): ‖U Uᴴ − I‖_∞ = {err:.2e}'


def test_unitarity_full_coverage_at_L_equals_H():
    """At L=H the paper claims W covers all of U(H); at minimum W must
    still be unitary. Checked at float64 at a few sizes."""
    with jax.experimental.enable_x64():
        for H in (4, 8, 16):
            angles, diag = _random_eunn_params(jax.random.PRNGKey(H), H, H, jnp.float64)
            assert angles.dtype == jnp.float64 and diag.dtype == jnp.float64
            U = _materialize_unitary(angles, diag, jnp.complex128)
            assert U.dtype == jnp.complex128
            err = jnp.max(jnp.abs(U @ U.conj().T - jnp.eye(H, dtype=jnp.complex128)))
            assert err < UNITARITY_TOL_F64, f'L=H={H}: ‖U Uᴴ − I‖_∞ = {err:.2e}'


def test_gradient_finite_at_reset_carry():
    """No NaN/Inf grads when h is at the equal-superposition reset state."""
    H, L = 8, 2
    angles, diag = _random_eunn_params(jax.random.PRNGKey(1), H, L, jnp.float32)
    h = initial_urnn_carry(batch_size=3, hidden_size=H, norm_scale=1.0)

    def loss(angles, diag, h):
        out = EUNNCell._apply_unitary(angles, diag, h)
        return (out.conj() * out).real.sum()

    grads = jax.grad(loss, argnums=(0, 1, 2))(angles, diag, h)
    for g in grads:
        assert jnp.all(jnp.isfinite(g)), f'non-finite grad: max |g| = {jnp.max(jnp.abs(g))}'


def test_cell_init_and_apply_shapes():
    """End-to-end sanity: EUNNCell.init + apply produces the right shapes."""
    H, L, B, D = 8, 4, 2, 5
    cell = EUNNCell(hidden_size=H, capacity=L)
    h0 = initial_urnn_carry(B, H, 1.0)
    ins = jnp.zeros((B, D), jnp.float32)
    params = cell.init(jax.random.PRNGKey(0), h0, ins)
    h1 = cell.apply(params, h0, ins)
    assert h1.shape == (B, H)
    assert h1.dtype == jnp.complex64


def test_odd_hidden_size_raises():
    cell = EUNNCell(hidden_size=7, capacity=2)
    h0 = jnp.zeros((1, 7), jnp.complex64)
    ins = jnp.zeros((1, 3), jnp.float32)
    with pytest.raises(ValueError, match='even hidden_size'):
        cell.init(jax.random.PRNGKey(0), h0, ins)
