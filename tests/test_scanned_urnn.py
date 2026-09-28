"""Unit tests for the uRNN port in qurl/models/network.py.

Covers the 7 design-plan invariants: param shapes/dtypes, scan forward shapes,
reset semantics, unitarity sanity, JIT+grad, mixed-dtype optax.global_norm,
and the legacy-variant param layout.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from qurl.models.network import (
    ScannedURNN, URNNCell, LegacyURNNCell, ModReLU,
    initial_urnn_carry, get_memory_initial_carry,
)


# Force CPU so the tests are deterministic and run on any host.
jax.config.update('jax_platform_name', 'cpu')


def _init(mod, T, B, D):
    """Init a ScannedURNN with random dummy inputs."""
    init_carry = initial_urnn_carry(B, mod.hidden_size, mod.norm_scale)
    ins = jax.random.normal(jax.random.PRNGKey(0), (T, B, D))
    resets = jnp.zeros((T, B), dtype=bool)
    return mod.init(jax.random.PRNGKey(1), init_carry, (ins, resets)), init_carry, ins, resets


def test_standard_param_dtypes():
    mod = ScannedURNN(hidden_size=16, variant='standard', add_input_dense=True)
    vars_, _, _, _ = _init(mod, T=4, B=2, D=16)
    # Only 'params' collection — permutation is an XLA constant, not a variable.
    assert list(vars_.keys()) == ['params']
    p = vars_['params']
    assert p['cell']['diag_embed']['kernel'].dtype == jnp.float32
    assert p['cell']['diag_embed']['bias'].dtype == jnp.float32
    assert p['cell']['rot_embed']['kernel'].dtype == jnp.complex64
    assert p['cell']['input_embed']['kernel'].dtype == jnp.complex64
    assert p['cell']['activation']['beta'].dtype == jnp.float32


def test_forward_shapes_and_dtypes():
    T, B, H, D = 8, 4, 32, 16
    mod = ScannedURNN(hidden_size=H, variant='standard')
    vars_, init_carry, ins, resets = _init(mod, T, B, D)
    new_c, y = mod.apply(vars_, init_carry, (ins, resets))
    assert new_c.shape == (B, H)
    assert new_c.dtype == jnp.complex64
    assert y.shape == (T, B, H)
    assert y.dtype == jnp.complex64


def test_reset_resets_carry_to_initial():
    """With reset=1 at t=0, the state fed to the cell at t=0 must equal the
    equal-superposition initial carry. After one cell step, the output is
    no longer the initial vector, but the reset semantics are observable by
    comparing two rollouts that differ only in their pre-scan carry — a
    full reset at t=0 must make the post-scan carry independent of what
    was passed in."""
    T, B, H, D = 3, 2, 16, 16
    mod = ScannedURNN(hidden_size=H)
    vars_, _, ins, _ = _init(mod, T, B, D)
    resets = jnp.zeros((T, B), dtype=bool).at[0, :].set(True)  # full reset at t=0

    carry_A = jnp.ones((B, H), dtype=jnp.complex64) * (3 + 4j)
    carry_B = jnp.ones((B, H), dtype=jnp.complex64) * (-7 + 2j)
    end_c_A, y_A = mod.apply(vars_, carry_A, (ins, resets))
    end_c_B, y_B = mod.apply(vars_, carry_B, (ins, resets))
    # Because the scan starts by forcing the carry to initial_urnn_carry
    # for every env at t=0, the subsequent trajectory must be identical.
    assert jnp.allclose(end_c_A, end_c_B)
    assert jnp.allclose(y_A, y_B)


def test_unitarity_sanity_no_input_embed():
    """When add_input_dense=False and beta=0 (the default init), ModReLU
    is the identity on the post-unitary complex vector for any input with
    non-negative modulus (which is always true, |z|>=0). Then the cell
    output is the unitary image of the carry — the L2 norm per env must
    be preserved up to FP noise. This is a correctness sanity for the
    unitary block."""
    T, B, H, D = 2, 3, 32, 8
    mod = ScannedURNN(hidden_size=H, variant='standard', add_input_dense=False)
    # Set the carry to random non-trivial complex, verify norm is preserved.
    key = jax.random.PRNGKey(42)
    key, k1, k2 = jax.random.split(key, 3)
    carry_re = jax.random.normal(k1, (B, H))
    carry_im = jax.random.normal(k2, (B, H))
    carry = (carry_re + 1j * carry_im).astype(jnp.complex64)
    ins = jax.random.normal(jax.random.PRNGKey(0), (T, B, D))
    resets = jnp.zeros((T, B), dtype=bool)
    vars_ = mod.init(jax.random.PRNGKey(1), carry, (ins, resets))
    # Force beta=0 explicitly (it is by default, but be explicit).
    vars_ = {
        'params': {**vars_['params'],
                   'cell': {**vars_['params']['cell'],
                            'activation': {'beta': jnp.zeros_like(
                                vars_['params']['cell']['activation']['beta'])}}},
    }
    new_c, _ = mod.apply(vars_, carry, (ins, resets))
    norm_in = jnp.linalg.norm(carry, axis=-1)
    norm_out = jnp.linalg.norm(new_c, axis=-1)
    # Loose tolerance for FP drift through FFT + Householder over T=2 steps.
    assert jnp.allclose(norm_in, norm_out, atol=5e-4), \
        f'norm_in={norm_in} norm_out={norm_out}'


def test_jit_and_grad_no_nan():
    T, B, H, D = 4, 2, 16, 16
    mod = ScannedURNN(hidden_size=H, variant='standard')
    vars_, init_carry, ins, resets = _init(mod, T, B, D)

    def loss_fn(params):
        _, y = mod.apply({'params': params}, init_carry, (ins, resets))
        return (y.real ** 2).sum() + (y.imag ** 2).sum()

    grad_fn = jax.jit(jax.grad(loss_fn))
    grads = grad_fn(vars_['params'])
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(not jnp.any(jnp.isnan(g)) for g in leaves), 'NaN grad'
    # Every param leaf must receive non-zero gradient under this loss.
    for path, g in jax.tree_util.tree_flatten_with_path(grads)[0]:
        assert bool(jnp.any(g != 0)), f'zero grad at {path}'


def test_optax_global_norm_handles_complex():
    """Plan-agent item 4b: verify clip_by_global_norm over a mixed
    complex+real pytree computes sqrt(sum |g|^2) — matches torch's
    clip_grad_norm_ semantics. If this ever regresses, swap to a custom
    clip."""
    tree = {
        'c': jnp.array([1 + 2j, 0 + 0j], dtype=jnp.complex64),  # |.|^2 = 5 + 0
        'r': jnp.array([3.0, 0.0], dtype=jnp.float32),          # |.|^2 = 9 + 0
    }
    n = float(optax.global_norm(tree))
    assert np.isclose(n, np.sqrt(14)), f'got {n}, expected {np.sqrt(14)}'


def test_legacy_variant_param_layout():
    T, B, H, D = 3, 2, 16, 16
    mod = ScannedURNN(hidden_size=H, variant='legacy')
    vars_, init_carry, ins, resets = _init(mod, T, B, D)
    cell = vars_['params']['cell']
    assert cell['diag'].dtype == jnp.complex64
    assert cell['diag'].shape == (3 * H,)
    assert cell['rotation'].dtype == jnp.complex64
    assert cell['rotation'].shape == (2 * H,)
    assert cell['input_embed']['kernel'].dtype == jnp.complex64
    assert cell['activation']['beta'].dtype == jnp.float32
    # Smoke the forward to confirm the legacy path computes at all.
    new_c, y = mod.apply(vars_, init_carry, (ins, resets))
    assert new_c.shape == (B, H)
    assert y.shape == (T, B, H)
    assert not jnp.any(jnp.isnan(new_c))


def test_complex_unit_norm_finite_grad_at_zero():
    """Regression guard for the sparse-observation NaN bug.

    ``complex_unit_norm(x)`` must have a finite VJP at ``x=0``.
    The naive ``x / (jnp.linalg.norm(x) + eps)`` form does NOT —
    its gradient is NaN at zero regardless of the outer ``eps``
    because ``jnp.linalg.norm``'s own VJP is ``x*/|x|`` which is
    0/0 there. The fix is to push ``eps`` inside the sqrt.

    This test fails on envs like battleship_10 where the first
    observation of an episode is all-zeros (action_concat's prev-action
    one-hot is zero at reset), making ``rot_embed(0) = 0`` at init and
    producing NaN in the rot_embed gradient.
    """
    from qurl.models.network import complex_unit_norm
    zero = jnp.zeros((4,), dtype=jnp.complex64)
    g = jax.grad(lambda x: jnp.sum(jnp.abs(complex_unit_norm(x))))(zero)
    assert not jnp.any(jnp.isnan(g)), f'complex_unit_norm grad at 0 is NaN: {g}'
    # And on a non-zero input the output is still unit-norm.
    x = jnp.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], dtype=jnp.complex64)
    n = jnp.linalg.norm(complex_unit_norm(x))
    assert np.isclose(float(n), 1.0, atol=1e-4), f'non-unit norm: {float(n)}'


def test_wirtinger_conjugation_descends_on_simple_convex_loss():
    """Regression guard for the JAX-vs-torch Wirtinger mismatch.

    For f(z) = |z|^2, JAX's raw grad returns df/dz = conj(z), but torch
    stores df/dz_bar = z. Gradient descent requires the direction of z,
    so the optimizer must conjugate JAX grads first. If the optimizer
    built for memory_type='urnn' does NOT conjugate, this test fails
    (|z|^2 would grow instead of shrink)."""
    from qurl.algos.ppo import _build_optimizer

    tx = _build_optimizer('urnn', lr_or_schedule=1e-1,
                          complex_lr_or_schedule=1e-1, max_grad_norm=1e9)
    params = {'z': jnp.asarray(2.0 + 3.0j, dtype=jnp.complex64)}
    opt_state = tx.init(params)

    def loss(p):
        return (p['z'] * jnp.conj(p['z'])).real

    for _ in range(5):
        grads = jax.grad(loss)(params)
        upd, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, upd)

    # Started at |z|^2 = 13; after 5 adam steps with conjugation, |z|^2
    # should be substantially smaller. Without conjugation it grows
    # monotonically (verified manually).
    final_mag2 = float(jnp.abs(params['z']) ** 2)
    assert final_mag2 < 13.0 - 1.0, (
        f'|z|^2 did not decrease under memory_type=urnn optimizer: '
        f'final={final_mag2}, start=13.0')


def test_get_memory_initial_carry_dispatch():
    c_gru = get_memory_initial_carry('gru', 4, 8)
    c_urnn = get_memory_initial_carry('urnn', 4, 8)
    assert c_gru.dtype == jnp.float32
    assert c_urnn.dtype == jnp.complex64
    assert c_gru.shape == (4, 8) and c_urnn.shape == (4, 8)
    # The uRNN initial value is sqrt(1/(2*H)) * (1 + 1j) with norm_scale=1
    scalar = np.sqrt(1.0 / (2 * 8))
    expected = (scalar + 1j * scalar)
    assert np.allclose(c_urnn, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
