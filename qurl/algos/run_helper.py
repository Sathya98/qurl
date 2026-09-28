import gc
from typing import Callable
from time import time

from flax.training import orbax_utils
import jax
import orbax.checkpoint

from qurl.config import Hyperparams
from qurl.utils.file_system import get_results_path


def _device_get_with_progress(label, tree):
    """Transfer a pytree from device to host, one leaf at a time.

    Avoids the XLA runtime deadlock that occurs when jax.device_get is called
    on very large pytrees (observed at ~7+ GB with JAX 0.6.x).
    """
    leaves, treedef = jax.tree.flatten(tree)
    host_leaves = []
    for leaf in leaves:
        host_leaves.append(jax.device_get(leaf))
    result = treedef.unflatten(host_leaves)
    nbytes = sum(x.nbytes for x in jax.tree.leaves(result))
    print(f"  {label}: {nbytes / 1e9:.2f} GB", flush=True)
    return result


def vmap_and_train(args: Hyperparams,
                   train_fn: Callable,
                   hparams: dict,
                   rng: jax.random.PRNGKey):
    rngs = jax.random.split(rng, args.n_seeds)

    vmap_seeds_train_fn = jax.vmap(train_fn, in_axes=[None, 0])
    vmap_train_fn = jax.vmap(vmap_seeds_train_fn, in_axes=[0, None])
    train_jit = jax.jit(vmap_train_fn)

    t = time()

    # Do NOT call block_until_ready on the full output pytree — JAX 0.6.x
    # deadlocks when synchronising very large pytrees (18+ GB across many
    # leaves).  Instead, let the per-leaf device_get below handle blocking.
    out = train_jit(hparams, rngs)

    final_train_state = out['runner_state'][0]
    if not args.save_runner_state:
        del out['runner_state']

    results_path = get_results_path(args, return_npy=False)

    # Transfer from device to host incrementally (leaf-by-leaf) to avoid
    # XLA runtime deadlock on large pytrees. The first device_get call
    # implicitly blocks until the JIT'd computation finishes. Release
    # device refs between transfers so XLA can reclaim GPU memory.
    print("Transferring results to host...", flush=True)
    metric_np = _device_get_with_progress('metric', out.pop('metric'))

    total_runtime = time() - t
    print(f'Training complete. Total runtime: {total_runtime:.1f}s', flush=True)

    final_eval_np = _device_get_with_progress('final_eval', out.pop('final_eval_metric'))
    train_state_np = _device_get_with_progress('train_state', final_train_state)
    del final_train_state, out
    gc.collect()

    all_results = {
        'swept_hparams': jax.device_get(hparams),
        'out': {'metric': metric_np, 'final_eval_metric': final_eval_np},
        'args': args.as_dict(),
        'total_runtime': total_runtime,
        'final_train_state': train_state_np,
        'final_eval': final_eval_np,
    }

    # Save all results with Orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(all_results)

    print(f"Saving results to {results_path}", flush=True)
    orbax_checkpointer.save(results_path, all_results, save_args=save_args)
    print("Save complete.", flush=True)
