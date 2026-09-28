from flax import linen as nn
from jax import numpy as jnp
from jax._src.nn.initializers import orthogonal, constant

from qurl.models.discrete import DiscreteActor, BornRuleActor
from qurl.models.continuous import ContinuousActor
from qurl.models.value import Critic
from qurl.models.network import SimpleNN, ScannedRNN, ScannedURNN
from qurl.models.embedding import CNN, BattleshipEmbedding


class ActorCritic(nn.Module):
    env_name: str
    action_dim: int
    hidden_size: int = 128
    double_critic: bool = False
    memoryless: bool = False
    is_discrete: bool = True
    is_image: bool = False
    memory_type: str = 'gru'
    urnn_variant: str = 'standard'
    urnn_input_dense: bool = True
    urnn_norm_scale: float = 1.0
    urnn_perm_seed: int = 0
    eunn_capacity: int = 2
    policy_head: str = 'standard'

    def setup(self):
        if self.policy_head == 'born' and not self.is_discrete:
            raise ValueError("policy_head='born' only supports discrete action spaces")
        if self.is_image:
            self.embedding = CNN(hidden_size=self.hidden_size)
        # elif 'battleship' in self.env_name:
        #     self.embedding = BattleshipEmbedding(hidden_size=self.hidden_size, action_dim=self.action_dim)
        elif not self.memoryless:
            self.embedding = nn.Sequential([
                nn.Dense(self.hidden_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
                nn.relu
            ])
        else:
            self.embedding = SimpleNN(hidden_size=self.hidden_size)

        if not self.memoryless:
            if self.memory_type == 'urnn':
                self.memory = ScannedURNN(
                    hidden_size=self.hidden_size,
                    variant=self.urnn_variant,
                    add_input_dense=self.urnn_input_dense,
                    norm_scale=self.urnn_norm_scale,
                    perm_seed=self.urnn_perm_seed,
                )
            elif self.memory_type == 'eunn':
                self.memory = ScannedURNN(
                    hidden_size=self.hidden_size,
                    variant='eunn',
                    norm_scale=self.urnn_norm_scale,
                    capacity=self.eunn_capacity,
                )
            else:
                self.memory = ScannedRNN(hidden_size=self.hidden_size)
        if self.policy_head == 'born':
            self.actor = BornRuleActor(self.action_dim, hidden_size=self.hidden_size)
        elif self.is_discrete:
            self.actor = DiscreteActor(self.action_dim, hidden_size=self.hidden_size)
        else:
            self.actor = ContinuousActor(self.action_dim, hidden_size=self.hidden_size)

        if self.double_critic:
            self.critic = nn.vmap(Critic,
                                  variable_axes={'params': 0},
                                  split_rngs={'params': True},
                                  in_axes=None,
                                  out_axes=2,
                                  axis_size=2)(hidden_size=self.hidden_size)
        else:
            self.critic = Critic(hidden_size=self.hidden_size)

    def __call__(self, hidden, x):
        obs_dict, dones = x
        obs = obs_dict.obs
        action_mask = obs_dict.action_mask
        embedding = self.embedding(obs)
        if not self.memoryless:
            rnn_in = (embedding, dones)
            hidden, mem_out = self.memory(hidden, rnn_in)
            if self.memory_type in ('urnn', 'eunn'):
                critic_embedding = jnp.concatenate([mem_out.real, mem_out.imag], axis=-1)
                actor_input = mem_out if self.policy_head == 'born' else critic_embedding
            else:
                actor_input = mem_out
                critic_embedding = mem_out
        else:
            actor_input = embedding
            critic_embedding = embedding

        pi = self.actor(actor_input, action_mask=action_mask)
        v = self.critic(critic_embedding)

        return hidden, pi, jnp.squeeze(v, axis=-1)
