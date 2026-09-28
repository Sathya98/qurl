"""Tests for the BornRuleActor (QuRNN Born-rule policy head)."""
import jax
import jax.numpy as jnp
import pytest

from qurl.models.discrete import BornRuleActor, _glorot_complex_small
from qurl.models.actor_critic import ActorCritic
from qurl.models.network import initial_urnn_carry


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


def test_init_and_forward_shapes(rng):
    actor = BornRuleActor(action_dim=4, hidden_size=16)
    z = jnp.ones((8, 16), dtype=jnp.complex64) * (0.5 + 0.5j)
    params = actor.init(rng, z)
    pi = actor.apply(params, z)

    logits = pi.logits
    assert logits.shape == (8, 4)
    assert logits.dtype == jnp.float32


def test_action_mask(rng):
    actor = BornRuleActor(action_dim=4, hidden_size=16)
    z = jnp.ones((2, 16), dtype=jnp.complex64) * (0.5 + 0.5j)
    mask = jnp.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=jnp.float32)
    params = actor.init(rng, z, action_mask=mask)
    pi = actor.apply(params, z, action_mask=mask)

    logits = pi.logits
    assert jnp.allclose(logits[0, 2], -1e6, atol=1.0)
    assert jnp.allclose(logits[0, 3], -1e6, atol=1.0)
    assert logits[0, 0] > -1e5
    assert logits[0, 1] > -1e5


def test_near_uniform_initial_logits(rng):
    """With small-scale init, initial entropy should be close to maximum (log(A))."""
    actor = BornRuleActor(action_dim=8, hidden_size=128)
    z = jnp.ones((16, 128), dtype=jnp.complex64) * (0.1 + 0.1j)
    params = actor.init(rng, z)
    pi = actor.apply(params, z)

    max_entropy = jnp.log(jnp.array(8.0))
    actual_entropy = pi.entropy().mean()
    assert actual_entropy > 0.7 * max_entropy, \
        f"Initial entropy ({actual_entropy:.3f}) should be close to max ({max_entropy:.3f})"


def test_hidden_layer_param_shapes(rng):
    """Born head = complex Dense(H→H) → ModReLU → complex Dense(H→A)."""
    hidden_size, action_dim = 16, 4
    actor = BornRuleActor(action_dim=action_dim, hidden_size=hidden_size)
    z = jnp.ones((2, hidden_size), dtype=jnp.complex64)
    params = actor.init(rng, z)['params']
    assert params['Dense_0']['kernel'].shape == (hidden_size, hidden_size)
    assert params['Dense_1']['kernel'].shape == (hidden_size, action_dim)
    assert params['Dense_0']['kernel'].dtype == jnp.complex64
    assert params['Dense_1']['kernel'].dtype == jnp.complex64


def test_gradient_no_nan(rng):
    actor = BornRuleActor(action_dim=4, hidden_size=16)
    z = jnp.ones((4, 16), dtype=jnp.complex64) * (0.5 + 0.5j)
    params = actor.init(rng, z)

    def loss_fn(params):
        pi = actor.apply(params, z)
        return -pi.entropy().mean()

    grads = jax.grad(loss_fn)(params)
    grad_leaves = jax.tree_util.tree_leaves(grads)
    for g in grad_leaves:
        assert not jnp.any(jnp.isnan(g)), f"NaN in gradient: shape={g.shape}, dtype={g.dtype}"


def test_gradient_no_nan_at_zero_input(rng):
    """z=0 is the edge case for Born rule: |0|²=0, log(0+eps) is finite."""
    actor = BornRuleActor(action_dim=4, hidden_size=16)
    z = jnp.zeros((4, 16), dtype=jnp.complex64)
    params = actor.init(rng, z)

    def loss_fn(params):
        pi = actor.apply(params, z)
        return -pi.entropy().mean()

    grads = jax.grad(loss_fn)(params)
    grad_leaves = jax.tree_util.tree_leaves(grads)
    for g in grad_leaves:
        assert not jnp.any(jnp.isnan(g)), f"NaN in gradient at z=0"


def test_full_actor_critic_integration(rng):
    from qurl.envs.wrappers.gymnax import Observation

    hidden_size = 16
    action_dim = 5
    num_envs = 4

    network = ActorCritic(
        env_name='test',
        action_dim=action_dim,
        hidden_size=hidden_size,
        memory_type='urnn',
        urnn_variant='standard',
        policy_head='born',
    )

    init_hstate = initial_urnn_carry(num_envs, hidden_size, norm_scale=1.0)
    obs = Observation(
        obs=jnp.zeros((1, num_envs, 10)),
        action_mask=jnp.ones((1, num_envs, action_dim)),
    )
    dones = jnp.zeros((1, num_envs))
    init_x = (obs, dones)

    params = network.init(rng, init_hstate, init_x)
    hidden, pi, value = network.apply(params, init_hstate, init_x)

    assert hidden.shape == (num_envs, hidden_size)
    assert hidden.dtype == jnp.complex64
    assert pi.logits.shape == (1, num_envs, action_dim)
    assert pi.logits.dtype == jnp.float32
    assert value.shape == (1, num_envs)
    assert value.dtype == jnp.float32


def test_born_rejects_continuous(rng):
    """policy_head='born' requires discrete actions; ActorCritic.setup must raise
    before any forward pass, rather than fail downstream with a shape error."""
    from qurl.envs.wrappers.gymnax import Observation

    hidden_size = 16
    action_dim = 6
    num_envs = 2

    network = ActorCritic(
        env_name='Walker-V-v0',
        action_dim=action_dim,
        hidden_size=hidden_size,
        is_discrete=False,
        memory_type='urnn',
        policy_head='born',
    )

    init_hstate = initial_urnn_carry(num_envs, hidden_size, norm_scale=1.0)
    # Continuous envs have no discrete action mask, so Observation.action_mask
    # is left at its default of None (see qurl/envs/wrappers/gymnax.py).
    obs = Observation(obs=jnp.ones((1, num_envs, 17)))
    dones = jnp.zeros((1, num_envs))
    init_x = (obs, dones)

    with pytest.raises(ValueError, match='discrete action spaces'):
        network.init(rng, init_hstate, init_x)


def test_config_validation_gru_raises():
    from qurl.config import PPOHyperparams

    with pytest.raises(ValueError, match="requires memory_type"):
        PPOHyperparams().parse_args([
            '--policy_head', 'born',
            '--memory_type', 'gru',
        ])


def test_config_validation_memoryless_raises():
    from qurl.config import PPOHyperparams

    with pytest.raises(ValueError, match="incompatible with memoryless"):
        PPOHyperparams().parse_args([
            '--policy_head', 'born',
            '--memory_type', 'urnn',
            '--memoryless',
        ])


def test_complex_params_are_complex_dtype(rng):
    """Verify BornRuleActor's Dense params are complex64 so the optimizer routes them correctly."""
    actor = BornRuleActor(action_dim=4, hidden_size=16)
    z = jnp.ones((2, 16), dtype=jnp.complex64)
    params = actor.init(rng, z)

    param_leaves = jax.tree_util.tree_leaves(params)
    has_complex = any(jnp.iscomplexobj(p) for p in param_leaves)
    assert has_complex, "BornRuleActor should have complex-dtype parameters"


def test_glorot_complex_initializes_both_parts(rng):
    """Flax Linen glorot_uniform(dtype=complex64) must produce nonzero imaginary parts.
    NNX's version is known to only initialize the real part — verify Linen is correct."""
    import flax.linen as nn
    layer = nn.Dense(32, param_dtype=jnp.complex64,
                     kernel_init=nn.initializers.glorot_uniform(dtype=jnp.complex64))
    x = jnp.ones((4, 64), dtype=jnp.complex64)
    params = layer.init(rng, x)
    kernel = params['params']['kernel']

    assert kernel.dtype == jnp.complex64
    assert not jnp.allclose(kernel.imag, 0.0), \
        "glorot_uniform(complex64) should initialize nonzero imaginary parts"
    real_std = kernel.real.std()
    imag_std = kernel.imag.std()
    assert imag_std > 0.5 * real_std, \
        f"Imag std ({imag_std:.4f}) should be comparable to real std ({real_std:.4f})"
