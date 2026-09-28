#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=36:00:00
# Override per env with `sbatch --time=HH:MM:SS ...`; see the README's
# "whole paper matrix" loop for the known-good values per env (Craftax
# needs more, most envs need less).
#SBATCH --output=%x_%j.out
# Train one (method, env) pair with the paper's settings, e.g.
#   sbatch --job-name=icurnn_rs11 --partition=<your_gpu_partition> \
#          --export=ALL,METHOD=icurnn,ENV_NAME=rocksample_11_11 scripts/slurm_example.sbatch
# qurl must be installed in the active environment; set VENV=<path> to activate one.
set -euo pipefail
: "${METHOD:?set METHOD}"; : "${ENV_NAME:?set ENV_NAME}"
[[ -n "${VENV:-}" ]] && source "$VENV/bin/activate"
python -m qurl.run --method "$METHOD" --env "$ENV_NAME" --platform gpu ${EXTRA_ARGS:-}
