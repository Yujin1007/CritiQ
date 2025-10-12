#!/usr/bin/env bash
set -euo pipefail

# Project root (CritiQ)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Config path (default: push_task/config/push_config.json)
CFG_ARG="${1:-push_task/config/test_config.json}"
if [[ "$CFG_ARG" = /* ]]; then
  ABS_CFG="$CFG_ARG"
else
  ABS_CFG="$SCRIPT_DIR/$CFG_ARG"
fi

# Ensure Python can import both push_task/ (for local modules) and project root (for push_config, etc.)
export PYTHONPATH="$SCRIPT_DIR/push_task:$SCRIPT_DIR:${PYTHONPATH:-}"

# Run from the directory where eval_critiq.py lives
cd "$SCRIPT_DIR/push_task"

echo "[run_push_critiq] Using config: ${ABS_CFG}"
python eval_critiq.py --cfg "${ABS_CFG}"
