#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  printf 'Missing run environment file: %s\nCopy %s/.env.example to %s and set RUN_DIR.\n' \
    "$ENV_FILE" "$SCRIPT_DIR" "$SCRIPT_DIR" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

: "${RUN_DIR:?Set RUN_DIR in $ENV_FILE}"
ENV_CONFIG="${ENV_CONFIG:-/workspace/finetune/configs/env/env_config.yaml}"
EXP_CONFIG="${EXP_CONFIG:-/workspace/finetune/configs/exp/exp_config.yaml}"

if [[ ! -f "$ENV_CONFIG" || ! -f "$EXP_CONFIG" ]]; then
  printf 'Missing config: ENV_CONFIG=%s EXP_CONFIG=%s\n' "$ENV_CONFIG" "$EXP_CONFIG" >&2
  exit 2
fi

mkdir -p "$RUN_DIR"
date -Is > "$RUN_DIR/train-started-at.txt"
START_EPOCH_SECONDS="$(date +%s)"

PYTHONUNBUFFERED=1 phisonai2 \
  --env_config "$ENV_CONFIG" \
  --exp_config "$EXP_CONFIG" \
  2>&1 | tee "$RUN_DIR/train-wrapper.log"

EXIT_CODE=${PIPESTATUS[0]}
date -Is > "$RUN_DIR/train-finished-at.txt"
END_EPOCH_SECONDS="$(date +%s)"
ELAPSED_SECONDS=$((END_EPOCH_SECONDS - START_EPOCH_SECONDS))

printf 'elapsed=%02d:%02d:%02d\nexit=%s\n' \
  "$((ELAPSED_SECONDS / 3600))" \
  "$(((ELAPSED_SECONDS % 3600) / 60))" \
  "$((ELAPSED_SECONDS % 60))" \
  "$EXIT_CODE" \
  > "$RUN_DIR/train-wall-time.txt"
printf '%s\n' "$EXIT_CODE" > "$RUN_DIR/train-exit-code.txt"

exit "$EXIT_CODE"
