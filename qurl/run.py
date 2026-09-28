"""Train PPO for one (method, env) pair with the paper's hyperparameters.

    python -m qurl.run --method icurnn --env rocksample_11_11
    python -m qurl.run --method qeunn_2 --env tmaze_75 --dry_run
    python -m qurl.run --method gru --env battleship_10 --n_seeds 3 --platform cpu

Settings are merged in this order (later wins):
  1. configs/envs.yaml `defaults` and `rollout_length` (by method group)
  2. configs/envs.yaml `envs[env]`           (tmaze_<N> falls back to tmaze_75)
  3. method flags                             (memory type, variant, capacity, policy head)
  4. configs/envs.yaml `ppo_ld[env]`         (method ppo_ld only)
  5. configs/best/<method>_<env>.yaml         (tuned unitary methods, when present)
  6. any other PPOHyperparams flag given on the command line
"""
import argparse
import re
import shlex
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent / 'configs'
BASELINE_METHODS = ('gru', 'ppo_ld')
FLOAT_KEYS = ('lr', 'complex_lr', 'lambda0', 'lambda1', 'ld_weight', 'vf_coeff', 'entropy_coeff')
METHODS_HELP = 'gru | ppo_ld | urnn | icurnn | eunn_<L> | qurnn | qicurnn | qeunn_<L>'


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def method_flags(method: str) -> dict:
    """PPOHyperparams fields that define a method, independent of env."""
    if method == 'gru':
        return {}
    if method == 'ppo_ld':
        return {'double_critic': True}
    # optional 'q' prefix = Born-rule policy head
    match = re.fullmatch(r'(q?)(?:(urnn)|(icurnn)|eunn_(\d+))', method)
    if match is None:
        raise ValueError(f"unknown method '{method}'; expected one of: {METHODS_HELP}")
    born, _, input_conditioned, capacity = match.groups()
    if capacity:
        flags = {'memory_type': 'eunn', 'eunn_capacity': int(capacity)}
    else:
        flags = {'memory_type': 'urnn', 'urnn_variant': 'standard' if input_conditioned else 'legacy'}
    if born:
        flags['policy_head'] = 'born'
    return flags


def _env_row(env: str, table: dict) -> dict | None:
    """Look up an env's row; any tmaze_<N> without its own row reuses tmaze_75."""
    if env in table:
        return table[env]
    if re.fullmatch(r'tmaze_\d+', env):
        return table.get('tmaze_75')
    return None


def resolve_config(method: str, env: str) -> dict:
    """Merge the paper's settings for (method, env) into one PPOHyperparams-keyed dict."""
    tables = _load_yaml(CONFIG_DIR / 'envs.yaml')
    flags = method_flags(method)
    base = _env_row(env, tables['envs'])
    if base is None:
        raise ValueError(f"no paper settings for env '{env}' (known: {', '.join(tables['envs'])}). "
                         f"Run `python -m qurl.algos.ppo --env {env} ...` with explicit flags instead.")
    group = 'baseline' if method in BASELINE_METHODS else 'unitary'
    config = {**tables['defaults'], 'num_steps': tables['rollout_length'][group],
              **base, 'env': env, **flags}
    if method == 'ppo_ld':
        config.update(_env_row(env, tables['ppo_ld']))
    best_path = CONFIG_DIR / 'best' / f'{method}_{env}.yaml'
    if best_path.exists():
        config.update(_load_yaml(best_path))
    config['study_name'] = f'{method}_{env}'
    for key in FLOAT_KEYS:  # PyYAML reads e.g. '8e-05' as a string
        if key in config:
            config[key] = float(config[key])
    return config


def to_argv(config: dict) -> list[str]:
    """Turn a config dict into PPOHyperparams CLI flags (True bools become bare flags)."""
    argv = []
    for key, value in config.items():
        if isinstance(value, bool):
            argv += [f'--{key}'] if value else []
        else:
            argv += [f'--{key}', str(value)]
    return argv


def build_args(method: str, env: str, overrides: list[str]):
    from qurl.config import PPOHyperparams
    return PPOHyperparams().parse_args(to_argv(resolve_config(method, env)) + overrides)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--method', required=True, help=METHODS_HELP)
    parser.add_argument('--env', required=True, help='environment id, e.g. rocksample_11_11')
    parser.add_argument('--dry_run', action='store_true', help='print the resolved command and exit')
    cli, overrides = parser.parse_known_args(argv)

    try:
        args = build_args(cli.method, cli.env, overrides)
    except ValueError as e:
        parser.error(str(e))
    if cli.dry_run:
        print('python -m qurl.algos.ppo ' + shlex.join(to_argv(resolve_config(cli.method, cli.env)) + overrides))
        return
    from qurl.algos.ppo import run
    run(args)


if __name__ == '__main__':
    main()
