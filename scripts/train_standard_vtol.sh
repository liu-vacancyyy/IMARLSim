#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "${ROOT_DIR}/standard_vtol/rl/train_ppo.py" \
  --device "${DEVICE:-cuda:0}" \
  --num-envs "${NUM_ENVS:-1024}" \
  --steps-per-env "${STEPS_PER_ENV:-256}" \
  --iterations "${ITERATIONS:-1000}" \
  --log-dir "${LOG_DIR:-${ROOT_DIR}/runs/standard_vtol_fixed_target}"
