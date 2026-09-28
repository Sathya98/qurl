"""Verify that the factored Householder application

    h - 2 v (vᴴ h)

produces the same output as the materialized form

    (I - 2 v vᴴ) h

for unit-norm v in ℂᴺ, across batched inputs, and that their gradients agree.
This is the correctness check behind the O(N²) → O(N) rewrite proposed for
URNNCell / LegacyURNNCell in qurl/models/network.py.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qurl.models.network import complex_unit_norm, householder_matrix


jax.config.update('jax_platform_name', 'cpu')


def _apply_materialized(v, h):
    """(I - 2 v vᴴ) h via the current qurl path: build R, then matvec."""
    R = householder_matrix(v)
    if v.ndim == 1:
        return R @ h
    return jnp.einsum('...ij,...j->...i', R, h)


def _apply_factored(v, h):
    """Factored left action: (I - 2 v vᴴ) h = h - 2 v (vᴴ h).

    URNNCell uses this (``einsum('bij,bj->bi', R, h)``).
    """
    vh_h = jnp.sum(jnp.conj(v) * h, axis=-1, keepdims=True)
    return h - 2.0 * v * vh_h


def _apply_factored_right(v, h):
    """Factored right action: h (I - 2 v vᴴ) = h - 2 (h · v) conj(v).

    LegacyURNNCell uses this (``h @ R``). The '·' here is the non-conjugated
    dot product Σ hᵢ vᵢ, because in ``(h R)ⱼ = Σᵢ hᵢ R_{ij}`` the summation
    hits ``Rᵢⱼ = δᵢⱼ − 2 vᵢ conj(vⱼ)`` — so vᵢ appears without a conj, and
    the leading factor on the resulting rank-1 correction is conj(vⱼ).
    """
    h_dot_v = jnp.sum(h * v, axis=-1, keepdims=True)
    return h - 2.0 * jnp.conj(v) * h_dot_v


def _random_complex(key, shape):
    k1, k2 = jax.random.split(key)
    return (jax.random.normal(k1, shape) + 1j * jax.random.normal(k2, shape)
            ).astype(jnp.complex64)


@pytest.mark.parametrize('N', [8, 32, 128])
def test_single_vector_matches(N):
    key = jax.random.PRNGKey(0)
    kv, kh = jax.random.split(key)
    v = complex_unit_norm(_random_complex(kv, (N,)))
    h = _random_complex(kh, (N,))

    y_mat = _apply_materialized(v, h)
    y_fac = _apply_factored(v, h)
    assert jnp.allclose(y_mat, y_fac, atol=1e-5), \
        f'max|diff|={jnp.max(jnp.abs(y_mat - y_fac))}'


@pytest.mark.parametrize('batch,N', [(4, 16), (32, 64), (7, 128)])
def test_batched_matches(batch, N):
    """Per-batch v's (URNNCell case) — each batch element gets its own R."""
    key = jax.random.PRNGKey(1)
    kv, kh = jax.random.split(key)
    v = complex_unit_norm(_random_complex(kv, (batch, N)))
    h = _random_complex(kh, (batch, N))

    y_mat = _apply_materialized(v, h)
    y_fac = _apply_factored(v, h)
    assert y_mat.shape == (batch, N) and y_fac.shape == (batch, N)
    assert jnp.allclose(y_mat, y_fac, atol=1e-5)


def test_broadcast_shared_v_per_batch_h_right_action():
    """LegacyURNNCell case: one v shared across all batch elements, ``h @ R``.

    The materialized path is a single (N, N) matrix applied from the right
    to every batched h. The factored form broadcasts v over the batch axis.
    Uses the right-action factored form because that's what ``h @ R`` is.
    """
    batch, N = 16, 64
    key = jax.random.PRNGKey(2)
    kv, kh = jax.random.split(key)
    v = complex_unit_norm(_random_complex(kv, (N,)))          # shared
    h = _random_complex(kh, (batch, N))

    R = householder_matrix(v)
    y_mat = h @ R                                             # (batch, N)
    y_fac = _apply_factored_right(v[None, :], h)
    assert jnp.allclose(y_mat, y_fac, atol=1e-5), \
        f'max|diff|={jnp.max(jnp.abs(y_mat - y_fac))}'


def test_left_vs_right_action_differ_on_complex():
    """Sanity: for complex Hermitian R, left and right actions are NOT equal.

    R is Hermitian (Rᴴ = R) but that means Rᵀ = conj(R) ≠ R in general, so
    h @ R ≠ R @ h when v has any imaginary part. This test guards against
    accidentally treating them as interchangeable.
    """
    N = 32
    key = jax.random.PRNGKey(10)
    kv, kh = jax.random.split(key)
    v = complex_unit_norm(_random_complex(kv, (N,)))
    h = _random_complex(kh, (N,))

    R = householder_matrix(v)
    left = R @ h
    right = h @ R
    assert not jnp.allclose(left, right, atol=1e-3), \
        'left and right actions matched — test premise broken'


def test_unitarity_preserved():
    """Both forms should preserve the L2 norm of h for unit-norm v."""
    batch, N = 8, 128
    key = jax.random.PRNGKey(3)
    kv, kh = jax.random.split(key)
    v = complex_unit_norm(_random_complex(kv, (batch, N)))
    h = _random_complex(kh, (batch, N))

    norm_h = jnp.linalg.norm(h, axis=-1)
    norm_mat = jnp.linalg.norm(_apply_materialized(v, h), axis=-1)
    norm_fac = jnp.linalg.norm(_apply_factored(v, h), axis=-1)
    assert jnp.allclose(norm_h, norm_mat, atol=1e-4)
    assert jnp.allclose(norm_h, norm_fac, atol=1e-4)


def test_involution():
    """Householder reflections are their own inverse: R(R h) == h."""
    batch, N = 4, 64
    key = jax.random.PRNGKey(4)
    kv, kh = jax.random.split(key)
    v = complex_unit_norm(_random_complex(kv, (batch, N)))
    h = _random_complex(kh, (batch, N))

    h2_mat = _apply_materialized(v, _apply_materialized(v, h))
    h2_fac = _apply_factored(v, _apply_factored(v, h))
    assert jnp.allclose(h, h2_mat, atol=1e-4)
    assert jnp.allclose(h, h2_fac, atol=1e-4)


def test_gradient_matches():
    """jax.grad through both forms must yield the same VJP.

    If this regresses, the factored rewrite would silently change training
    dynamics in URNNCell. Loss is a real scalar built from a real-valued
    reduction of the output — standard for complex-param optimization.
    """
    batch, N = 3, 32
    key = jax.random.PRNGKey(5)
    kv, kh = jax.random.split(key)
    v0 = complex_unit_norm(_random_complex(kv, (batch, N)))
    h0 = _random_complex(kh, (batch, N))

    def loss_mat(v, h):
        y = _apply_materialized(v, h)
        return (y.real ** 2 + y.imag ** 2).sum()

    def loss_fac(v, h):
        y = _apply_factored(v, h)
        return (y.real ** 2 + y.imag ** 2).sum()

    # argnums=(0, 1) → gradients w.r.t. both v and h.
    g_mat = jax.grad(loss_mat, argnums=(0, 1), holomorphic=False)(v0, h0)
    g_fac = jax.grad(loss_fac, argnums=(0, 1), holomorphic=False)(v0, h0)
    assert jnp.allclose(g_mat[0], g_fac[0], atol=1e-4), 'dL/dv differs'
    assert jnp.allclose(g_mat[1], g_fac[1], atol=1e-4), 'dL/dh differs'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
