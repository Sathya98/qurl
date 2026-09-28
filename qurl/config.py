from typing import Literal

from jax import numpy as jnp
from tap import Tap


class Hyperparams(Tap):
    env: str = 'tmaze_5'

    num_envs: int = 4  # Number of environments to run in parallel.
    default_max_steps_in_episode: int = 1000  # Gets overridden if this is defined by the env.
    gamma: float = 0.99  # will be replaced if env has gamma property.

    num_eval_envs: int = 10
    steps_log_freq: int = 1
    update_log_freq: int = 1
    save_runner_state: bool = False  # Do we save the checkpoint in the end?
    seed: int = 2020
    n_seeds: int = 1
    platform: Literal['cpu', 'gpu'] = 'cpu'
    debug: bool = False

    study_name: str = 'test'


class PPOHyperparams(Tap):
    env: str = 'tmaze_5'
    num_envs: int = 4  # Number of environments to run in parallel.
    default_max_steps_in_episode: int = 1000  # Gets overridden if this is defined by the env.
    gamma: float = 0.99  # will be replaced if env has gamma property.

    num_steps: int = 128  # How many steps do we roll out for each update? Also correponds to the largest n for n-step returns.
    num_epochs: int = 50  # How many epochs do we split our training steps into?
    update_epochs: int = 4
    num_minibatches: int = 4

    memoryless: bool = False  # If true, don't use RNNs and use an FNN for func. approximation.
    perfect_memory: bool = False  # [WIP] Do we use the perfect memory version of the environment?
    double_critic: bool = False  # Do we have two critic heads?
    action_concat: bool = False  # Do we concatenate actions to our observation?

    # Memory module selection. 'urnn' = unitary RNN (complex hidden state).
    # 'eunn' = tunable EUNN (Jing et al. 2017, arXiv:1612.05231).
    memory_type: Literal['gru', 'urnn', 'eunn'] = 'gru'
    urnn_variant: Literal['standard', 'legacy'] = 'standard'  # 'standard' = input-dependent d/R, 'legacy' = learnable.
    urnn_input_dense: bool = True  # Add complex input-embed to the hidden state each step (standard variant only).
    urnn_norm_scale: float = 1.0   # Scales the initial complex carry (shared by urnn + eunn).
    urnn_perm_seed: int = 0        # Seed for the fixed permutation inside the uRNN unitary transform (uRNN only; EUNN uses alternating cyclic shifts).
    policy_head: Literal['standard', 'born'] = 'standard'  # 'born' = Born-rule complex actor (requires urnn/eunn, discrete actions).

    # Below are hyperparameters that can be swept with jax.vmap.
    lr: list[float] = [2.5e-4]  # Learning rate (applies to all real params; for uRNN/EUNN, applies only to the real group).
    complex_lr: list[float] = [8e-5]  # LR for the complex param group under memory_type in {'urnn','eunn'}. Ignored otherwise.
    lambda0: list[float] = [0.95]  # GAE lambda_0
    lambda1: list[float] = [0.5]  # GAE lambda_1
    ld_weight: list[float] = [0.0]  # How much to we weight the LD loss? only applies when double_critic is True.
    vf_coeff: list[float] = [0.5]  # How much do we weight value loss?
    entropy_coeff: list[float] = [0.01]  # PPO policy entropy coefficient for exploration
    eunn_capacity: int = 2  # [EUNN] capacity L. Determines tensor shapes so not vmap-sweepable; sweep via separate runs. L=hidden_size gives full U(H).

    hidden_size: int = 128  # Hidden size of our neural net
    total_steps: int = int(1.5e6)  # How many training steps do we run?
    clip_eps: float = 0.2  # PPO policy gradient clip epsilon
    max_grad_norm: float = 0.5  # Maximum grad norm for updates
    anneal_lr: bool = True  # Do we (linearly) anneal learning rate?

    image_size: int = 32  # [MADRONA/CRAFTAX] what is the size of our image?

    num_eval_envs: int = 10  # At the end of our run, how many environments do we run for evaluation?
    steps_log_freq: int = 1  # Over num_steps, how often do we save training statistics (returns etc.)?
    update_log_freq: int = 1  # Over all updates, how often do we save training statistics?
    save_checkpoints: bool = False  # Do we save train_state along with our per timestep outputs?
    save_runner_state: bool = False  # (no effect: the final train state is always saved)
    sweep_type: str = 'grid'  # grid (sweeps all listed hyperparams) | random (randomly rejection samples sets of hyperparams)
    n_random_hparams: int = 1  # [sweep_type = random] How many randomly sampled hyperparams do we use?
    n_run_bins: int = None  # How many bins do we split our runs into? Requires run_bin_idx to be not None
    run_bin_idx: int = 0  # After splitting our runs into n_run_bins bins, which index do we run?
    seed: int = 2020
    n_seeds: int = 5  # How many seeds to run in our experiment?
    platform: Literal['cpu', 'gpu'] = 'cpu'  # use CPU or GPU?
    debug: bool = False  # Do we print run statistics during training?
    show_discounted: bool = False  # For debug plotting, do we show undisc returns or disc returns?

    study_name: str = 'ppo_test'  # Save checkpoints and run statistics into results/{study_name}.

    def process_args(self):
        # Validate n_run_bins and run_bin_idx
        if self.n_run_bins is not None:
            assert self.run_bin_idx is not None
        if self.policy_head == 'born':
            if self.memory_type not in ('urnn', 'eunn'):
                raise ValueError(
                    f"policy_head='born' requires memory_type in {{'urnn','eunn'}}, "
                    f"got '{self.memory_type}'"
                )
            if self.memoryless:
                raise ValueError("policy_head='born' is incompatible with memoryless=True")


class TransformerHyperparams(Tap):
    env: str = 'tmaze_5'
    num_envs: int = 4
    default_max_steps_in_episode: int = 1000
    gamma: float = 0.99  # will be replaced if env has gamma property.

    num_steps: int = 128
    num_epochs: int = 50
    update_epochs: int = 4
    num_minibatches: int = 4

    double_critic: bool = False
    action_concat: bool = False

    lr: list[float] = [2.5e-4]
    lambda0: list[float] = [0.95]  # GAE lambda_0
    lambda1: list[float] = [0.5]  # GAE lambda_1
    ld_weight: list[float] = [0.0]  # how much to we weight the LD loss vs. value loss? only applies when optimize LD is True.
    vf_coeff: list[float] = [0.5]
    entropy_coeff: list[float] = [0.01]  # PPO policy entropy coefficient for exploration

    hidden_size: int = 128
    total_steps: int = int(1.5e6)
    clip_eps: float = 0.2
    max_grad_norm: float = 0.5
    anneal_lr: bool = True

    image_size: int = 32

    num_eval_envs: int = 10
    steps_log_freq: int = 1
    update_log_freq: int = 1
    save_checkpoints: bool = False  # Do we save train_state along with our per timestep outputs?
    save_runner_state: bool = False  # (no effect: the final train state is always saved)
    sweep_type: str = 'grid'  # grid (sweeps all listed hyperparams) | random (randomly rejection samples sets of hyperparams)
    n_random_hparams: int = 1  # [sweep_type = random] How many randomly sampled hyperparams do we use?
    n_run_bins: int = None  # How many bins do we split our runs into? Requires run_bin_idx to be not None
    run_bin_idx: int = 0  # After splitting our runs into n_run_bins bins, which index do we run?
    seed: int = 2020
    n_seeds: int = 5
    platform: Literal['cpu', 'gpu'] = 'cpu'
    debug: bool = False
    show_discounted: bool = False  # For debug plotting, do we show undisc returns or disc returns?

    # transformer hyperparams
    qkv_features: int = 256
    embed_size: int = 256
    num_heads: int = 8
    num_layers: int = 2
    window_mem: int = 128
    window_grad: int = 64
    gating: bool = True
    gating_bias: float = 2.0

    study_name: str = 'ppo_gtrxl_test'

    def process_args(self):
        # Validate n_run_bins and run_bin_idx
        if self.n_run_bins is not None:
            assert self.run_bin_idx is not None
