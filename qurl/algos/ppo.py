from collections import deque
from dataclasses import replace
from functools import partial
import inspect
import os
from time import time
from typing import NamedTuple

import chex
import flax.training.train_state
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint

from qurl.algos.run_helper import vmap_and_train
from qurl.config import PPOHyperparams
from qurl.envs import get_env
from qurl.envs.wrappers.gymnax import LogEnvState
from qurl.models import ScannedRNN, get_network_fn, get_memory_initial_carry
from qurl.utils.file_system import get_results_path, numpyify
from qurl.envs import brax_envs
from qurl.utils.sweep import get_grid_hparams, get_randomly_sampled_hparams


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray = None


class PPO:
    def __init__(self, network,
                 double_critic: bool = False,
                 ld_weight: float = 0.,
                 vf_coeff: float = 0.,
                 entropy_coeff: float = 0.01,
                 clip_eps: float = 0.2):
        self.network = network
        self.double_critic = double_critic
        self.ld_weight = ld_weight
        self.vf_coeff = vf_coeff
        self.entropy_coeff = entropy_coeff
        self.clip_eps = clip_eps
        self.act = jax.jit(self.act)
        self.loss = jax.jit(self.loss)

    def act(self, rng: chex.PRNGKey,
            train_state: flax.training.train_state.TrainState,
            hidden_state: chex.Array,
            obs: chex.Array, done: chex.Array):

        # SELECT ACTION
        ac_in = (jax.tree.map(lambda x: x[jnp.newaxis, ...], obs), done[np.newaxis, :])
        hstate, pi, value = self.network.apply(train_state.params, hidden_state, ac_in)
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )
        return value, action, log_prob, hstate

    def loss(self, params, init_hstate, traj_batch, gae, targets):
        # RERUN NETWORK
        _, pi, value = self.network.apply(
            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
        )
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = traj_batch.value + (
                value - traj_batch.value
        ).clip(-self.clip_eps, self.clip_eps)
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (
            jnp.maximum(value_losses, value_losses_clipped).mean()
        )
        # Lambda discrepancy loss
        if self.double_critic:
            value_loss = self.ld_weight * (jnp.square(value[..., 0] - value[..., 1])).mean() + \
                         (1 - self.ld_weight) * value_loss

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - traj_batch.log_prob)

        # which advantage do we use to update our policy?
        if self.double_critic:
            gae = gae[..., 0]

        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
                jnp.clip(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps,
                    )
                * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = pi.entropy().mean()

        total_loss = (
                loss_actor
                + self.vf_coeff * value_loss
                - self.entropy_coeff * entropy
        )
        return total_loss, (value_loss, loss_actor, entropy)


def env_step(runner_state, unused, agent: PPO, env, env_params):
    train_state, env_state, last_obs, last_done, hstate, rng = runner_state
    rng, _rng = jax.random.split(rng)
    value, action, log_prob, hstate = agent.act(_rng, train_state, hstate, last_obs, last_done)

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, hstate.shape[0])
    obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
    transition = Transition(
        last_done, action, value, reward, log_prob, last_obs, info
    )
    runner_state = (train_state, env_state, obsv, done, hstate, rng)
    return runner_state, transition


def calculate_gae(traj_batch, last_val, last_done, gae_lambda, gamma):
    def _get_advantages(carry, transition):
        gae, next_value, gae_lambda, next_done = carry
        done, value, reward = transition.done, transition.value, transition.reward
        if len(next_value.shape) > 1:
            reward, next_done = reward[..., None], next_done[..., None]
        delta = reward + gamma * next_value * (1 - next_done) - value
        gae = delta + gamma * gae_lambda * (1 - next_done) * gae
        return (gae, value, gae_lambda, done), gae

    _, advantages = jax.lax.scan(_get_advantages,
                                 (jnp.zeros_like(last_val), last_val, gae_lambda, last_done),
                                 traj_batch, reverse=True, unroll=16)
    target = advantages + traj_batch.value
    return advantages, target


def filter_period_first_dim(x, n: int):
    if isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray):
        return x[::n]


def _conjugate_complex_grads() -> optax.GradientTransformation:
    """Conjugate complex-dtype gradient leaves (Wirtinger fix).

    JAX's ``grad`` for f: C -> R returns the holomorphic derivative
    ``df/dz`` — the *ascent* direction under the torch convention is
    ``conj(df/dz) = df/dz_bar``, which is what torch's ``.grad``
    already stores. To make adam's first-moment tracking and the
    final ``z <- z - lr * update`` match torch exactly, we conjugate
    complex grads here before anything else in the optimizer chain.
    Verified empirically: raw JAX grad on ``|z|^2`` moves *away* from
    the minimum; the conjugated grad moves toward it.
    """
    def init_fn(_):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        updates = jax.tree_util.tree_map(
            lambda g: jnp.conj(g) if jnp.iscomplexobj(g) else g,
            updates,
        )
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def _build_optimizer(memory_type: str, lr_or_schedule, complex_lr_or_schedule,
                     max_grad_norm: float):
    """Build the optax optimizer.

    For memory_type='gru': single-group Adam (existing behavior).
    For memory_type in {'urnn','eunn'}: per-dtype param groups — complex
    leaves get ``complex_lr_or_schedule``, real leaves get
    ``lr_or_schedule``. One global-norm clip wraps both, matching the
    PyTorch convention of clip_grad_norm_ followed by per-group Adam.
    Complex grads are conjugated first so the remainder of the chain
    sees PyTorch-convention descent directions.
    """
    if memory_type in ('urnn', 'eunn'):
        def label_fn(params):
            return jax.tree_util.tree_map(
                lambda p: 'complex' if jnp.iscomplexobj(p) else 'real',
                params,
            )
        return optax.chain(
            _conjugate_complex_grads(),
            optax.clip_by_global_norm(max_grad_norm),
            optax.multi_transform(
                {'real':    optax.adam(lr_or_schedule,         eps=1e-5),
                 'complex': optax.adam(complex_lr_or_schedule, eps=1e-5)},
                label_fn,
            ),
        )
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr_or_schedule, eps=1e-5),
    )


def make_train(args: PPOHyperparams, rand_key: jax.random.PRNGKey):
    num_updates = (
            args.total_steps // args.num_steps // args.num_envs
    )
    args.minibatch_size = (
            args.num_envs * args.num_steps // args.num_minibatches
    )
    env_key, rand_key = jax.random.split(rand_key)

    env, env_params = get_env(args.env, env_key,
                              num_envs=args.num_envs,
                              image_size=args.image_size,
                              gamma=args.gamma,
                              perfect_memory=args.perfect_memory,
                              action_concat=args.action_concat)

    if hasattr(env, 'gamma'):
        args.gamma = env.gamma

    assert hasattr(env_params, 'max_steps_in_episode')

    double_critic = args.double_critic
    memoryless = args.memoryless

    network_fn, action_size, is_image, is_discrete = get_network_fn(env, env_params)
    network = network_fn(args.env,
                         action_size,
                         double_critic=double_critic,
                         hidden_size=args.hidden_size,
                         memoryless=memoryless,
                         is_discrete=is_discrete,
                         is_image=is_image,
                         memory_type=args.memory_type,
                         urnn_variant=args.urnn_variant,
                         urnn_input_dense=args.urnn_input_dense,
                         urnn_norm_scale=args.urnn_norm_scale,
                         urnn_perm_seed=args.urnn_perm_seed,
                         eunn_capacity=args.eunn_capacity,
                         policy_head=args.policy_head)

    steps_filter = partial(filter_period_first_dim, n=args.steps_log_freq)
    update_filter = partial(filter_period_first_dim, n=args.update_log_freq)

    # Used for vmapping over our double critic.
    transition_axes_map = Transition(
        None, None, 2, None, None, None, None
    )

    _calculate_gae = calculate_gae
    if double_critic:
        # last_val is index 1 here b/c we squeezed earlier.
        _calculate_gae = jax.vmap(calculate_gae,
                                  in_axes=[transition_axes_map, 1, None, 0, None],
                                  out_axes=2)

    def train(sweep_args_dict, rng):
        lr, complex_lr, ld_weight, vf_coeff, lambda0, lambda1, entropy_coeff = \
            sweep_args_dict['lr'], sweep_args_dict['complex_lr'], sweep_args_dict['ld_weight'], sweep_args_dict['vf_coeff'], \
                sweep_args_dict['lambda0'], sweep_args_dict['lambda1'], sweep_args_dict['entropy_coeff']

        agent = PPO(network, double_critic=double_critic, ld_weight=ld_weight, vf_coeff=vf_coeff,
                    clip_eps=args.clip_eps, entropy_coeff=entropy_coeff)

        # initialize functions
        _env_step = partial(env_step, agent=agent, env=env, env_params=env_params)

        gae_lambda = jnp.array(lambda0)
        if double_critic:
            gae_lambda = jnp.array([lambda0, lambda1])

        def _make_linear_schedule(base_lr):
            def schedule(count):
                frac = (
                    1.0
                    - (count // (args.num_minibatches * args.update_epochs))
                    / num_updates
                )
                return base_lr * frac
            return schedule


        # INIT NETWORK
        rng, _rng = jax.random.split(rng)
        init_x = (
            env.dummy_observation(args.num_envs, env_params),
            jnp.zeros((1, args.num_envs)),
        )
        init_hstate = get_memory_initial_carry(
            args.memory_type, args.num_envs, args.hidden_size, args.urnn_norm_scale
        )
        network_params = agent.network.init(_rng, init_hstate, init_x)
        if args.anneal_lr:
            lr_sched = _make_linear_schedule(lr)
            # Table 1 describes the anneal as applying to the real parameter group only, but
            # both the real and complex LR groups are annealed here — this is how the paper's
            # uRNN/EUNN runs were actually trained.
            complex_lr_sched = _make_linear_schedule(complex_lr)
        else:
            lr_sched = lr
            complex_lr_sched = complex_lr
        tx = _build_optimizer(args.memory_type, lr_sched, complex_lr_sched,
                              args.max_grad_norm)
        train_state = TrainState.create(
            apply_fn=agent.network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, args.num_envs)
        obsv, env_state = env.reset(reset_rng, env_params)
        init_hstate = get_memory_initial_carry(
            args.memory_type, args.num_envs, args.hidden_size, args.urnn_norm_scale
        )

        if args.env in ['craftax', 'craftax_pixels']:
            # We first need to populate our LogEnvState stats.
            rng, _rng = jax.random.split(rng)
            init_rng = jax.random.split(_rng, args.num_envs)
            init_obsv, init_env_state = env.reset(init_rng, env_params)
            init_init_hstate = get_memory_initial_carry(
                args.memory_type, args.num_envs, args.hidden_size, args.urnn_norm_scale
            )

            init_runner_state = (
                train_state,
                env_state,
                init_obsv,
                jnp.zeros(args.num_envs, dtype=bool),
                init_init_hstate,
                _rng,
            )

            starting_runner_state, _ = jax.lax.scan(
                _env_step, init_runner_state, None, env_params.max_steps_in_episode
            )

            def recursive_replace(env_state, new_env_state, names):
                if not isinstance(env_state, LogEnvState):
                    return replace(env_state, env_state=recursive_replace(env_state.env_state, new_env_state.env_state, names))
                new_log_vals = {name: getattr(new_env_state, name) for name in names}
                return replace(env_state, **new_log_vals)

            replace_field_names = ['returned_episode_returns', 'returned_discounted_episode_returns', 'returned_episode_lengths']
            env_state = recursive_replace(env_state, starting_runner_state[1], replace_field_names)

        # TRAIN LOOP
        def _update_step(runner_state, i):
            # COLLECT TRAJECTORIES
            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, jnp.arange(args.num_steps), args.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (jax.tree.map(lambda x: x[jnp.newaxis, ...], last_obs), last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done, gae_lambda, args.gamma)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    grad_fn = jax.value_and_grad(agent.loss, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                # SHUFFLE COLLECTED BATCH
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, args.num_envs)
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], args.num_minibatches, -1]
                            + list(x.shape[2:]),
                            ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, args.update_epochs
            )
            train_state = update_state[0]

            # save metrics only every steps_log_freq
            metric = traj_batch.info
            metric = jax.tree.map(steps_filter, metric)

            # training stats: loss_info is (total_loss, (value_loss, actor_loss, entropy)),
            # each with leading dims (update_epochs, num_minibatches) from the
            # inner scan. Collapse those two axes to get one scalar per update.
            _, (value_loss_info, actor_loss_info, entropy_info) = loss_info
            metric['policy_entropy'] = entropy_info.mean(axis=(-2, -1))
            metric['value_loss'] = value_loss_info.mean(axis=(-2, -1))
            metric['actor_loss'] = actor_loss_info.mean(axis=(-2, -1))

            rng = update_state[-1]

            steps_per_update = args.num_envs * args.num_steps
            log_every = max(1, 1_000_000 // steps_per_update)

            def progress_callback(update_idx, info):
                if int(update_idx) % log_every != 0:
                    return
                env_steps = int(update_idx + 1) * steps_per_update
                returned = info["returned_episode"]
                stats = jax.local_devices()[0].memory_stats()
                mem_str = '' if stats is None else (
                    f" | gpu_mem={stats['bytes_in_use']/2**30:.2f}GB (peak {stats['peak_bytes_in_use']/2**30:.2f}GB)")
                if returned.any():
                    avg_ret = float(jnp.mean(info["returned_episode_returns"][returned]))
                    print(f"[{env_steps:,} env steps] avg return={avg_ret:.2f}{mem_str}", flush=True)
                else:
                    print(f"[{env_steps:,} env steps] (no completed episodes){mem_str}", flush=True)

            jax.debug.callback(progress_callback, i, metric)

            if args.debug:
                def callback(info):
                    timesteps = (
                            info["timestep"][info["returned_episode"]] * args.num_envs
                    )
                    if args.show_discounted:
                        show_str = "avg discounted return"
                        avg_return_values = jnp.mean(info["returned_discounted_episode_returns"][info["returned_episode"]])
                    else:
                        show_str = "avg episodic return"
                        avg_return_values = jnp.mean(info["returned_episode_returns"][info["returned_episode"]])

                    if len(timesteps) > 0:
                        print(
                            f"timesteps={timesteps[0]} - {timesteps[-1]}, {show_str}={avg_return_values:.2f}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((args.num_envs), dtype=bool),
            init_hstate,
            _rng,
        )

        # returned metric has an extra dimension.
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(num_updates), num_updates
        )

        # save metrics only every update_log_freq
        metric = jax.tree.map(update_filter, metric)

        # TODO: offline eval here.
        final_train_state = runner_state[0]

        reset_rng = jax.random.split(runner_state[-1], args.num_envs)
        eval_obsv, eval_env_state = env.reset(reset_rng, env_params)

        eval_init_hstate = get_memory_initial_carry(
            args.memory_type, args.num_envs, args.hidden_size, args.urnn_norm_scale
        )

        eval_runner_state = (
            final_train_state,
            eval_env_state,
            eval_obsv,
            jnp.zeros((args.num_envs), dtype=bool),
            eval_init_hstate,
            _rng,
        )

        # COLLECT EVAL TRAJECTORIES
        eval_runner_state, eval_traj_batch = jax.lax.scan(
            _env_step, eval_runner_state, None, env_params.max_steps_in_episode
        )
        # res = {"runner_state": runner_state, "metric": metric}
        res = {"runner_state": runner_state, "metric": metric, 'final_eval_metric': eval_traj_batch.info}

        return res

    return train


def main(args):
    rng = jax.random.PRNGKey(args.seed)
    make_train_rng, rng = jax.random.split(rng)
    train_fn = make_train(args, make_train_rng)

    if args.sweep_type == 'grid':
        hparams, _ = get_grid_hparams(args)
    elif args.sweep_type == 'random':
        _rng, rng = jax.random.split(rng)
        hparams = get_randomly_sampled_hparams(_rng, args, n_samples=args.n_random_hparams)
    else:
        raise NotImplementedError

    vmap_and_train(args, train_fn, hparams, rng)


def madrona_main(args):
    """
    NOTE: currently Madrona (visual mujoco) runs can only run a single set hyperparameters.
    If you want to sweep over hyperparams, you'll have to sweep over programs.
    """
    rng = jax.random.PRNGKey(args.seed)
    make_train_rng, rng = jax.random.split(rng)
    train_fn = make_train(args, make_train_rng)

    train_args = list(inspect.signature(train_fn).parameters.keys())

    vmaps_train = train_fn
    if args.sweep_type == 'grid':
        hparams, _ = get_grid_hparams(args)
    elif args.sweep_type == 'random':
        _rng, rng = jax.random.split(rng)
        hparams = get_randomly_sampled_hparams(_rng, args, n_samples=args.n_random_hparams)
    else:
        raise NotImplementedError

    def flatten_and_assert_singleton(hparam):
        assert len(hparam) == 1
        return hparams[0]

    hparams = jax.tree.map(flatten_and_assert_singleton, hparams)

    train_jit = jax.jit(vmaps_train)

    t = time()
    out = train_jit(hparams, rng)
    new_t = time()
    total_runtime = new_t - t
    print(f'Training complete. Total runtime: {total_runtime:.1f}s', flush=True)

    final_train_state = out['runner_state'][0]
    if not args.save_runner_state:
        del out['runner_state']

    results_path = get_results_path(args, return_npy=False)  # returns a results directory

    all_results = {
        'argument_order': train_args,
        'out': out,
        'args': args.as_dict(),
        'total_runtime': total_runtime,
        'final_train_state': final_train_state
    }

    # Save all results with Orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(all_results)

    print(f"Saving results to {results_path}", flush=True)
    orbax_checkpointer.save(results_path, all_results, save_args=save_args)

    print("Save complete.", flush=True)


def run(args: PPOHyperparams):
    jax.config.update('jax_platform_name', args.platform)

    if args.env in brax_envs and args.env.endswith('pixels'):
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.10'
        os.environ['MADRONA_DISABLE_CUDA_HEAP_SIZE'] = '1'
        # Tell XLA to use Triton GEMM
        os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_triton_gemm_any=True'
        madrona_main(args)
    else:
        main(args)


if __name__ == "__main__":
    run(PPOHyperparams().parse_args())
