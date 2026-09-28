import distrax
import flax.linen as nn
import jax.numpy as jnp
from jax._src.nn.initializers import orthogonal, constant
import numpy as np

from .network import SmallImageCNN, FullImageCNN, ModReLU, _glorot_complex
from .value import Critic
from qurl.models.transformerXL import Transformer


class DiscreteActor(nn.Module):
    action_dim: int
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x, action_mask=None):
        actor_mean = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            x
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        if action_mask is not None:
            actor_mean = actor_mean * action_mask + (1 - action_mask) * (-1e6)
        pi = distrax.Categorical(logits=actor_mean)
        return pi


def _glorot_complex_small(scale=0.01):
    """Glorot-uniform scaled down (gain=0.01) for near-uniform initial Born-rule logits."""
    base = nn.initializers.glorot_uniform(dtype=jnp.complex64)
    def init(key, shape, dtype=jnp.complex64):
        return scale * base(key, shape, dtype)
    return init


class BornRuleActor(nn.Module):
    """Complex-valued actor using Born-rule logits.

    Two-layer complex head: complex Dense -> ModReLU -> complex Dense,
    producing unnormalised complex action coefficients z. Logits are then
    log(|z|² + ε), so pi(a) = |z_a|² / sum_b |z_b|².

    Takes the raw complex hidden state from uRNN/EUNN and produces a
    Categorical distribution via quantum-inspired measurement (Born rule).
    """
    action_dim: int
    hidden_size: int = 128
    eps: float = 1e-10

    @nn.compact
    def __call__(self, z, action_mask=None):
        z = nn.Dense(self.hidden_size, param_dtype=jnp.complex64,
                        kernel_init=_glorot_complex(), bias_init=constant(0.0))(z)
        z = ModReLU(hidden_size=self.hidden_size)(z)
        z = nn.Dense(self.action_dim, param_dtype=jnp.complex64,
                     kernel_init=_glorot_complex_small(), bias_init=constant(0.0))(z)
        logits = jnp.log(jnp.abs(z) ** 2 + self.eps).real
        if action_mask is not None:
            logits = logits * action_mask + (1 - action_mask) * (-1e6)
        return distrax.Categorical(logits=logits)


class DiscreteActorCriticTransformer(nn.Module):
    action_dim: int
    encoder_size: int
    num_heads: int
    qkv_features: int
    num_layers:int
    hidden_size: int = 128
    gating:bool=False
    gating_bias:float=0.
    double_critic: bool = False

    def setup(self):
        self.transformer = Transformer(
            encoder_size=self.encoder_size,
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            num_layers=self.num_layers,
            gating=self.gating,
            gating_bias=self.gating_bias
        )
        self.actor = DiscreteActor(self.action_dim, hidden_size=self.hidden_size)
        if self.double_critic:
            self.critic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=2,
                             axis_size=2)(hidden_size=self.hidden_size)
        else:
            self.critic = Critic(hidden_size=self.hidden_size)

    def __call__(self, memories,obs,mask):
        embedding, memory_out = self.transformer(memories,obs,mask)
        pi = self.actor(embedding)
        v = self.critic(embedding)
        return pi, jnp.squeeze(v, axis=-1), memory_out
    
    def model_forward_eval(self, memories,obs,mask):
        """Used during environment rollout (single timestep of obs). And return the memory"""
        embedding, memory_out = self.transformer.forward_eval(memories,obs,mask)
        pi = self.actor(embedding)
        v = self.critic(embedding)
        return pi, jnp.squeeze(v, axis=-1), memory_out
    
    def model_forward_train(self, memories,obs,mask): 
        """Used during training: a window of observation is sent. And don't return the memory"""
        embedding = self.transformer.forward_train(memories,obs,mask)
        pi = self.actor(embedding)
        v = self.critic(embedding)
        return pi, jnp.squeeze(v, axis=-1)
    

class ImageDiscreteActorCriticTransformer(nn.Module):
    action_dim: int
    encoder_size: int
    num_heads: int
    qkv_features: int
    num_layers:int
    hidden_size: int = 128
    gating:bool=False
    gating_bias:float=0.
    double_critic: bool = False

    def setup(self):
        self.cnn_full = FullImageCNN(hidden_size=self.hidden_size)
        self.cnn_small = SmallImageCNN(hidden_size=self.hidden_size)
        self.transformer = Transformer(
            encoder_size=self.encoder_size,
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            num_layers=self.num_layers,
            gating=self.gating,
            gating_bias=self.gating_bias
        )
        self.actor = DiscreteActor(self.action_dim, hidden_size=self.hidden_size)
        if self.double_critic:
            self.critic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=2,
                             axis_size=2)(hidden_size=self.hidden_size)
        else:
            self.critic = Critic(hidden_size=self.hidden_size)
    
    def __call__(self, memories,obs,mask):
        if obs.shape[-2] >= 20:
            embedding = self.cnn_full(obs)
        else:
            embedding = self.cnn_small(obs)
        embedding = nn.relu(embedding)
        # embedding = embedding.squeeze(1)
        
        embedding, memory_out = self.transformer(memories,embedding,mask)
        pi = self.actor(embedding)
        v = self.critic(embedding)
        return pi, jnp.squeeze(v, axis=-1),memory_out
    
    def model_forward_eval(self, memories,obs,mask):
        """Used during environment rollout (single timestep of obs). And return the memory"""
        if obs.shape[-2] >= 20:
            embedding = self.cnn_full(obs)
        else:
            embedding = self.cnn_small(obs)
        embedding = nn.relu(embedding)
        # embedding = embedding.squeeze(1)
        embedding,memory_out = self.transformer.forward_eval(memories,embedding,mask)

        pi = self.actor(embedding)

        v = self.critic(embedding)

        return pi, jnp.squeeze(v, axis=-1),memory_out
    
    def model_forward_train(self, memories,obs,mask): 
        """Used during training: a window of observation is sent. And don't return the memory"""
        if obs.shape[-2] >= 20:
            embedding = self.cnn_full(obs)
        else:
            embedding = self.cnn_small(obs)
        embedding = nn.relu(embedding)

        embedding = self.transformer.forward_train(memories,embedding,mask)

        pi = self.actor(embedding)

        v = self.critic(embedding)
        return pi, jnp.squeeze(v, axis=-1)

class BattleShipActorCriticTransformer(nn.Module):
    action_dim: int
    encoder_size: int
    num_heads: int
    qkv_features: int
    num_layers:int
    hidden_size: int = 128
    gating:bool=False
    gating_bias:float=0.
    double_critic: bool = False

    def setup(self):
        self.cnn_small = SmallImageCNN(hidden_size=self.hidden_size)
        self.dense1 = nn.Dense(
                2 * self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )
        self.dense2 = nn.Dense(
                self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )
        self.act1 = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.act2 = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.transformer = Transformer(
            encoder_size=self.encoder_size,
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            num_layers=self.num_layers,
            gating=self.gating,
            gating_bias=self.gating_bias
        )
        if self.double_critic:
            self.critic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=2,
                             axis_size=2)(hidden_size=self.hidden_size)
        else:
            self.critic = Critic(hidden_size=self.hidden_size)

    def __call__(self, memories, obs, mask):
        # Obs is a t x b x obs_size array.
        if len(obs.shape) == 4:
            valid_action_mask = (obs == 0).reshape(*obs.shape[:-2], -1)
            embedding = self.cnn_small(obs)
            embedding = nn.relu(embedding)
        else:
            hit = obs[..., 0:1]
            valid_action_mask = obs[..., 1:self.action_dim + 1]
            obs = jnp.concatenate([hit, obs[..., self.action_dim + 1:]], axis=-1)

            embedding = self.dense1(obs)
            embedding = nn.relu(embedding)

            embedding = jnp.concatenate((hit, embedding), axis=-1)
            embedding = self.dense2(embedding)
            embedding = nn.relu(embedding)

        embedding, memory_out = self.transformer(memories,embedding,mask)

        # MLP actor
        actor_mean = self.act1(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = self.act2(actor_mean)

        # Do masking here for invalid actions.
        actor_mean = actor_mean * valid_action_mask + (1 - valid_action_mask) * (-1e6)

        pi = distrax.Categorical(logits=actor_mean)

        v = self.critic(embedding)

        return pi, jnp.squeeze(v, axis=-1), memory_out

    def model_forward_eval(self, memories, obs, mask):
        """Used during environment rollout (single timestep of obs). And return the memory"""
        if len(obs.shape) == 4:
            valid_action_mask = (obs == 0).reshape(*obs.shape[:-2], -1)
            embedding = self.cnn_small(obs)
            embedding = nn.relu(embedding)
        else:
            hit = obs[..., 0:1]
            valid_action_mask = obs[..., 1:self.action_dim + 1]
            obs = jnp.concatenate([hit, obs[..., self.action_dim + 1:]], axis=-1)

            embedding = self.dense1(obs)
            embedding = nn.relu(embedding)

            embedding = jnp.concatenate((hit, embedding), axis=-1)
            embedding = self.dense2(embedding)
            embedding = nn.relu(embedding)

        embedding,memory_out = self.transformer.forward_eval(memories,embedding,mask)

        # MLP actor
        actor_mean = self.act1(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = self.act2(actor_mean)

        # Do masking here for invalid actions.
        actor_mean = actor_mean * valid_action_mask + (1 - valid_action_mask) * (-1e6)

        pi = distrax.Categorical(logits=actor_mean)

        v = self.critic(embedding)

        return pi, jnp.squeeze(v, axis=-1), memory_out
    
    def model_forward_train(self, memories,obs,mask): 
        """Used during training: a window of observation is sent. And don't return the memory"""
        if len(obs.shape) == 4:
            valid_action_mask = (obs == 0).reshape(*obs.shape[:-2], -1)
            embedding = self.cnn_small(obs)
            embedding = nn.relu(embedding)
        else:
            hit = obs[..., 0:1]
            valid_action_mask = obs[..., 1:self.action_dim + 1]
            obs = jnp.concatenate([hit, obs[..., self.action_dim + 1:]], axis=-1)

            embedding = self.dense1(obs)
            embedding = nn.relu(embedding)

            embedding = jnp.concatenate((hit, embedding), axis=-1)
            embedding = self.dense2(embedding)
            embedding = nn.relu(embedding)

        embedding = self.transformer.forward_train(memories, embedding, mask)

        # MLP actor
        actor_mean = self.act1(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = self.act2(actor_mean)

        # Do masking here for invalid actions.
        actor_mean = actor_mean * valid_action_mask + (1 - valid_action_mask) * (-1e6)

        pi = distrax.Categorical(logits=actor_mean)

        v = self.critic(embedding)

        return pi, jnp.squeeze(v, axis=-1)
