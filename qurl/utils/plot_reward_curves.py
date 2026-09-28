"""Plot reward curves for qurl PPO runs.

Two modes:
  compare  (default):  overlay mean +/- 95% CI curves for every method with a
                       study directory under results/ for --env.
  per-seed (--method): plot each seed's raw curve for one (env, method).
"""
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint
from tap import Tap

from qurl.definitions import PROJECT_ROOT_DIR, results_dir
from qurl.utils.plot import mean_confidence_interval, smoothen


# --method id -> (study-dir template matching qurl.run's study_name, legend label from the paper)
METHOD_STUDIES = {
    'gru':     ('gru_{env}', 'GRU'),
    'ppo_ld':  ('ppo_ld_{env}', 'PPO-LD'),
    'urnn':    ('urnn_{env}', 'uRNN'),
    'icurnn':  ('icurnn_{env}', 'IC-uRNN'),
    'eunn_2':  ('eunn_2_{env}', 'EUNN (L=2)'),
    'qurnn':   ('qurnn_{env}', 'QuRNN'),
    'qicurnn': ('qicurnn_{env}', 'QIC-uRNN'),
    'qeunn_2': ('qeunn_2_{env}', 'QuEUNN (L=2)'),
}


class PlotArgs(Tap):
    env: str                                              # e.g. tmaze_10, rocksample_11_11
    method: str = None                                    # if set: per-seed mode for this one method
    discounted: bool = False                              # use returned_discounted_episode_returns
    smooth: Literal['none', 'ema', 'savgol'] = 'none'     # smoothing style, see _apply_smooth
    ema_weight: float = 0.9                               # EMA weight (TensorBoard slider convention)
    out_dir: str = None                                   # default: <repo>/plots/
    subsample: int = 1                                    # thin updates axis by this factor (e.g. 16 for heavyweight runs)


def _ema(x: np.ndarray, weight: float) -> np.ndarray:
    """Causal exponential moving average (same recurrence TensorBoard / cleanrl use)."""
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = weight * out[i - 1] + (1 - weight) * x[i]
    return out


def _apply_smooth(y: np.ndarray, args: 'PlotArgs') -> np.ndarray:
    if args.smooth == 'ema':
        return _ema(y, args.ema_weight)
    if args.smooth == 'savgol':
        return smoothen(y)
    return y


def _load_run(study_dir: Path, discounted: bool = False, subsample: int = 1):
    """Load latest ckpt under study_dir; return (returns[n_seeds, n_updates], args).

    returned_(discounted_)episode_returns has shape
    (n_hparams=1, n_seeds, n_updates, n_steps, n_envs); we squeeze the hparams
    axis and mean over (n_steps, n_envs) to get one point per PPO update.
    subsample > 1 thins the update axis to reduce memory and plot weight.
    """
    run_dirs = sorted(p for p in study_dir.iterdir() if p.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f'no run subdirs under {study_dir}')
    restored = orbax.checkpoint.PyTreeCheckpointer().restore(str(run_dirs[-1]))
    key = 'returned_discounted_episode_returns' if discounted else 'returned_episode_returns'
    arr = np.asarray(restored['out']['metric'][key])
    if subsample > 1:
        arr = arr[:, :, ::subsample]
    arr = arr.squeeze(0).mean(axis=(-2, -1))
    return arr, restored['args']


def _x_axis(run_args, n_updates, subsample: int = 1):
    update_log_freq = int(run_args.get('update_log_freq', 1)) * subsample
    return np.arange(n_updates) * update_log_freq * int(run_args['num_steps']) * int(run_args['num_envs'])


def plot_compare(env: str, args: PlotArgs, out_dir: Path):
    """Overlay one mean-curve-with-band per method that has data for `env`."""
    results_root = results_dir()
    loaded, searched = [], []
    for tmpl, label in METHOD_STUDIES.values():
        study_dir = results_root / tmpl.format(env=env)
        searched.append(study_dir)
        if study_dir.exists():
            returns, run_args = _load_run(study_dir, discounted=args.discounted, subsample=args.subsample)
            loaded.append((label, returns, run_args))

    if not loaded:
        raise FileNotFoundError(
            f'no study dirs found for env={env!r}. Searched:\n  '
            + '\n  '.join(str(p) for p in searched)
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    n_seeds = loaded[0][1].shape[0]
    for (label, returns, run_args), color in zip(loaded, [f'C{i}' for i in range(len(loaded))]):
        # seeds on axis 0 -> mean_confidence_interval reduces that axis
        mu, h = mean_confidence_interval(returns, axis=0)
        mu = _apply_smooth(mu, args)
        x = _x_axis(run_args, mu.shape[0], args.subsample)
        ax.plot(x, mu, color=color, label=label)
        ax.fill_between(x, mu - h, mu + h, color=color, alpha=0.2)

    ax.set_xlabel('environment steps')
    ax.set_ylabel('episode return' + (' (discounted)' if args.discounted else ''))
    ax.set_title(env)
    ax.legend()
    fig.tight_layout()
    out = out_dir / f'{env}_reward_{n_seeds}avg.pdf'
    fig.savefig(out)
    print(f'wrote {out}')


def plot_per_seed(env: str, method: str, args: PlotArgs, out_dir: Path):
    """Plot each seed's curve independently for one (env, method)."""
    if method not in METHOD_STUDIES:
        raise ValueError(f'unknown method {method!r}; known: {sorted(METHOD_STUDIES)}')
    tmpl, label = METHOD_STUDIES[method]
    study_dir = results_dir() / tmpl.format(env=env)
    returns, run_args = _load_run(study_dir, discounted=args.discounted, subsample=args.subsample)
    n_seeds, n_updates = returns.shape
    x = _x_axis(run_args, n_updates, args.subsample)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(n_seeds):
        ax.plot(x, _apply_smooth(returns[i], args), label=f'seed {i}', alpha=0.85)
    ax.set_xlabel('environment steps')
    ax.set_ylabel('episode return' + (' (discounted)' if args.discounted else ''))
    ax.set_title(f'{env} - {label} ({n_seeds} seeds)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out_dir / f'{env}_reward_{method}_{n_seeds}seeds.pdf'
    fig.savefig(out)
    print(f'wrote {out}')


def main(args: PlotArgs):
    out_dir = Path(args.out_dir) if args.out_dir else Path(PROJECT_ROOT_DIR, 'plots')
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.method is not None:
        plot_per_seed(args.env, args.method, args, out_dir)
    else:
        plot_compare(args.env, args, out_dir)


if __name__ == '__main__':
    main(PlotArgs().parse_args())
