"""CPU end-to-end smoke test: exercises the full training loop, not just config resolution."""
import jax

import qurl.run


def test_cpu_run_writes_one_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv('QURL_RESULTS_DIR', str(tmp_path))
    # qurl.run.main -> ppo.run calls jax.config.update('jax_platform_name', ...)
    # globally (not scoped), so restore it afterwards rather than leaking it
    # into later tests in the session (same class of leak as the stray x64
    # flag documented in test_eunn.py's test_unitarity_f64).
    prev_platform_name = jax.config.read('jax_platform_name')
    try:
        qurl.run.main([
            '--method', 'icurnn', '--env', 'tmaze_75',
            '--total_steps', '2048', '--n_seeds', '1', '--platform', 'cpu',
        ])
    finally:
        jax.config.update('jax_platform_name', prev_platform_name)
    run_dirs = list((tmp_path / 'icurnn_tmaze_75').iterdir())
    assert len(run_dirs) == 1
