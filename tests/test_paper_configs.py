"""Pin qurl.run's resolved settings to Appendix E of the paper
("Reinforcement Learning with Complex (valued) Memories", EWRL 2026)."""
import jax
import pytest

from qurl.run import build_args, resolve_config

UNITARY = ('urnn', 'eunn_2', 'icurnn', 'qurnn', 'qeunn_2', 'qicurnn')
DISCRETE_ENVS = ('tmaze_75', 'rocksample_11_11', 'rocksample_15_15', 'battleship_10')
CONTINUOUS_ENVS = ('Walker-V-v0', 'HalfCheetah-V-v0')

# Table 2: hidden size, parallel envs, total steps, gamma
TABLE2 = {
    'tmaze_75':         (32, 4, 5_000_000, 0.99),
    'rocksample_11_11': (256, 8, 5_000_000, 0.99),
    'rocksample_15_15': (512, 16, 10_000_000, 0.999),
    'battleship_10':    (512, 32, 10_000_000, 1.0),
    'Walker-V-v0':      (256, 4, 50_000_000, 0.99),
    'HalfCheetah-V-v0': (256, 4, 50_000_000, 0.99),
    'craftax_pixels':   (512, 256, 100_000_000, 0.99),
}
# Table 3: GRU (lr, ent, λ0) and PPO-LD (lr, ent, λ0, λ1, β)
TABLE3_GRU = {
    'tmaze_75': (2.5e-3, 0.01, 0.7), 'rocksample_11_11': (2.5e-3, 0.2, 0.7),
    'rocksample_15_15': (2.5e-3, 0.2, 0.7), 'battleship_10': (2.5e-3, 0.05, 0.7),
    'Walker-V-v0': (2.5e-4, 0.01, 0.95), 'HalfCheetah-V-v0': (2.5e-4, 0.01, 0.9),
    'craftax_pixels': (2.5e-4, 0.01, 0.5),
}
TABLE3_LD = {
    'tmaze_75': (2.5e-4, 0.01, 0.95, 0.95, 0.25), 'rocksample_11_11': (2.5e-3, 0.2, 0.5, 0.5, 0.25),
    'rocksample_15_15': (2.5e-3, 0.2, 0.1, 0.95, 0.5), 'battleship_10': (2.5e-3, 0.05, 0.1, 0.95, 0.5),
    'Walker-V-v0': (2.5e-4, 0.01, 0.95, 0.95, 0.5), 'HalfCheetah-V-v0': (2.5e-5, 0.01, 0.95, 0.7, 0.25),
    'craftax_pixels': (2.5e-4, 0.01, 0.1, 0.95, 0.25),
}
# Tables 4-9: tuned (complex lr, ent, λ0, lr) per unitary variant
TUNED = {
    'tmaze_75': {'urnn': (1e-6, .005, .95, 2.5e-4), 'eunn_2': (5e-5, .01, .8, 2.5e-4), 'icurnn': (5e-5, .01, .95, 2.5e-4),
                 'qurnn': (1e-5, .005, .95, 2.5e-3), 'qeunn_2': (1e-4, .005, .8, 2.5e-4), 'qicurnn': (5e-6, .01, .9, 2.5e-3)},
    'rocksample_11_11': {'urnn': (8e-5, .075, .7, 2.5e-3), 'eunn_2': (1e-6, .15, .7, 2.5e-3), 'icurnn': (1e-6, .075, .7, 2.5e-3),
                         'qurnn': (8e-5, .075, .7, 2.5e-3), 'qeunn_2': (8e-5, .1, .7, 2.5e-3), 'qicurnn': (8e-5, .15, .7, 2.5e-4)},
    'rocksample_15_15': {'urnn': (1e-5, .075, .7, 2.5e-3), 'eunn_2': (1e-6, .1, .7, 2.5e-3), 'icurnn': (8e-5, .15, .7, 2.5e-4),
                         'qurnn': (8e-5, .1, .7, 2.5e-3), 'qeunn_2': (1e-6, .075, .8, 2.5e-3), 'qicurnn': (8e-5, .2, .7, 2.5e-4)},
    'battleship_10': {'urnn': (8e-5, .01, .9, 2.5e-3), 'eunn_2': (1e-6, .01, .7, 2.5e-3), 'icurnn': (8e-5, .01, .95, 2.5e-3),
                      'qurnn': (1e-5, .01, .8, 2.5e-3), 'qeunn_2': (1e-5, .01, .8, 2.5e-3), 'qicurnn': (1e-5, .01, .9, 2.5e-3)},
    'Walker-V-v0': {'urnn': (8e-5, .01, .95, 2.5e-4), 'eunn_2': (1e-5, .01, .95, 2.5e-4), 'icurnn': (8e-5, .005, .95, 2.5e-4)},
    'HalfCheetah-V-v0': {'urnn': (8e-5, .005, .9, 2.5e-4), 'eunn_2': (1e-6, .005, .9, 2.5e-4), 'icurnn': (8e-5, .01, .8, 2.5e-4)},
}
PAPER_RUNS = ([(m, e) for e in TABLE2 for m in ('gru', 'ppo_ld')]
              + [(m, e) for e in DISCRETE_ENVS + ('craftax_pixels',) for m in UNITARY]
              + [(m, e) for e in CONTINUOUS_ENVS for m in ('urnn', 'eunn_2', 'icurnn')])


@pytest.mark.parametrize('method,env', PAPER_RUNS)
def test_paper_table1_and_table2(method, env):
    config = resolve_config(method, env)
    hidden_size, num_envs, total_steps, _ = TABLE2[env]
    assert (config['hidden_size'], config['num_envs'], config['total_steps']) == (hidden_size, num_envs, total_steps)
    assert (config['vf_coeff'], config['n_seeds'], config['action_concat']) == (0.5, 5, True)


@pytest.mark.parametrize('method,env', PAPER_RUNS)
def test_paper_table1_fixed_settings(method, env):
    args = build_args(method, env, [])
    assert (args.update_epochs, args.num_minibatches, args.clip_eps, args.max_grad_norm, args.anneal_lr) == \
        (4, 4, 0.2, 0.5, True)


@pytest.mark.parametrize('method,env', PAPER_RUNS)
def test_paper_table1_rollout_length(method, env):
    expected = 64 if env.startswith('craftax') else 128 if method in ('gru', 'ppo_ld') else 256
    assert resolve_config(method, env)['num_steps'] == expected


@pytest.mark.parametrize('env', TABLE2)
def test_paper_table3_baselines(env):
    gru, ld = resolve_config('gru', env), resolve_config('ppo_ld', env)
    assert (gru['lr'], gru['entropy_coeff'], gru['lambda0']) == TABLE3_GRU[env]
    assert (ld['lr'], ld['entropy_coeff'], ld['lambda0'], ld['lambda1'], ld['ld_weight']) == TABLE3_LD[env]


@pytest.mark.parametrize('env,method', [(e, m) for e, methods in TUNED.items() for m in methods])
def test_paper_tables4_to_9_tuned(env, method):
    config = resolve_config(method, env)
    assert (config['complex_lr'], config['entropy_coeff'], config['lambda0'], config['lr']) == pytest.approx(TUNED[env][method], rel=1e-9)


@pytest.mark.parametrize('method', UNITARY)
def test_craftax_reuses_gru_settings(method):
    config = resolve_config(method, 'craftax_pixels')
    assert (config['lr'], config['entropy_coeff'], config['lambda0']) == TABLE3_GRU['craftax_pixels']


@pytest.mark.parametrize('env', DISCRETE_ENVS)
def test_paper_table2_gamma(env):
    from qurl.config import PPOHyperparams
    from qurl.envs import get_env
    env_obj, _ = get_env(env, jax.random.PRNGKey(0))
    expected = TABLE2[env][3]
    if env == 'rocksample_15_15':
        # App. E Table 2 lists gamma=0.999 here, but RockSample never reads the config's
        # `gamma` key (an upstream POBax behaviour), so every run actually trained at the
        # PPOHyperparams default of 0.99; qurl reproduces the runs, not the table.
        expected = 0.99
    # mirrors qurl.algos.ppo: an env-level gamma overrides the PPOHyperparams default
    assert getattr(env_obj, 'gamma', PPOHyperparams().gamma) == pytest.approx(expected)
