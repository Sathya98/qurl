import re
from pathlib import Path

import pytest
import yaml

from qurl.run import FLOAT_KEYS, build_args, main, method_flags, resolve_config

BEST_DIR = Path(__file__).parents[1] / 'qurl' / 'configs' / 'best'
BEST_NAME = re.compile(r'^(q?(?:urnn|icurnn|eunn_\d+))_(.+)$')


def test_gru_uses_env_row():
    config = resolve_config('gru', 'rocksample_11_11')
    assert (config['hidden_size'], config['num_envs'], config['num_steps']) == (256, 8, 128)
    assert (config['lr'], config['lambda0'], config['entropy_coeff']) == (2.5e-3, 0.7, 0.2)
    assert config['action_concat'] is True and config['seed'] == 2026
    assert 'memory_type' not in config and config['study_name'] == 'gru_rocksample_11_11'


def test_ppo_ld_overrides():
    config = resolve_config('ppo_ld', 'battleship_10')
    assert config['double_critic'] is True
    assert (config['lr'], config['lambda0'], config['lambda1'], config['ld_weight']) == (2.5e-3, 0.1, 0.95, 0.5)
    assert (config['hidden_size'], config['entropy_coeff']) == (512, 0.05)


def test_best_file_overrides_env_row():
    config = resolve_config('icurnn', 'rocksample_11_11')
    best = yaml.safe_load((BEST_DIR / 'icurnn_rocksample_11_11.yaml').read_text())
    assert all(config[key] == float(value) for key, value in best.items())
    assert (config['memory_type'], config['urnn_variant'], config['num_steps']) == ('urnn', 'standard', 256)


def test_method_flags():
    assert method_flags('urnn') == {'memory_type': 'urnn', 'urnn_variant': 'legacy'}
    assert method_flags('icurnn') == {'memory_type': 'urnn', 'urnn_variant': 'standard'}
    assert method_flags('eunn_8') == {'memory_type': 'eunn', 'eunn_capacity': 8}
    assert method_flags('qurnn') == {'memory_type': 'urnn', 'urnn_variant': 'legacy', 'policy_head': 'born'}
    assert method_flags('qicurnn') == {'memory_type': 'urnn', 'urnn_variant': 'standard', 'policy_head': 'born'}
    assert method_flags('qeunn_2') == {'memory_type': 'eunn', 'eunn_capacity': 2, 'policy_head': 'born'}


def test_tmaze_fallback():
    config = resolve_config('eunn_2', 'tmaze_100')
    assert config['env'] == 'tmaze_100' and config['hidden_size'] == 32
    assert resolve_config('ppo_ld', 'tmaze_100')['ld_weight'] == 0.25


def test_craftax_is_untuned_for_unitary_methods():
    config = resolve_config('icurnn', 'craftax_pixels')
    assert (config['lr'], config['lambda0'], config['entropy_coeff'], config['complex_lr']) == (2.5e-4, 0.5, 0.01, 8e-5)
    assert config['num_steps'] == 64 and config['steps_log_freq'] == config['update_log_freq'] == 16
    assert not list(BEST_DIR.glob('*craftax*'))


def test_unknown_method_raises():
    with pytest.raises(ValueError, match='unknown method'):
        resolve_config('lstm', 'tmaze_75')


def test_unknown_env_raises():
    with pytest.raises(ValueError, match='qurl.algos.ppo'):
        resolve_config('gru', 'pocman')


def test_float_keys_are_floats_for_every_best_file():
    for path in BEST_DIR.glob('*.yaml'):
        method, env = BEST_NAME.match(path.stem).groups()
        config = resolve_config(method, env)
        for key in FLOAT_KEYS:
            if key in config:
                assert isinstance(config[key], float), (path.name, key, config[key])


def test_cli_overrides_win():
    args = build_args('gru', 'tmaze_75', ['--n_seeds', '2', '--lr', '1e-3', '--platform', 'cpu'])
    assert (args.n_seeds, args.lr, args.platform) == (2, [1e-3], 'cpu')


def test_main_unknown_method_exits_cleanly():
    with pytest.raises(SystemExit):
        main(['--method', 'lstm', '--env', 'tmaze_75'])
