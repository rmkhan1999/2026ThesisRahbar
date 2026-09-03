#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p docs/sbc
LOG=docs/sbc8_console.log
echo "=== SBC-8 sequential launch $(date -Iseconds) ===" | tee -a "$LOG"
echo "n=8 sequential warmup=200 draws=100 depth=7 seed_base=20260808 start-index=0" | tee -a "$LOG"
echo "resume: skip-if-exists in run_sbc_replicates.py; restart this script safely" | tee -a "$LOG"
pmset -g batt | head -2 | tee -a "$LOG"
caffeinate -i env PYTHONUNBUFFERED=1 python scripts/run_sbc_replicates.py \
  --n-replicates 8 \
  --start-index 0 \
  --seed-base 20260808 \
  --warmup 200 \
  --draws 100 \
  --max-tree-depth 7 \
  --output-dir docs/sbc \
  2>&1 | tee -a "$LOG"
echo "SBC8_EXIT:${PIPESTATUS[0]} $(date -Iseconds)" | tee -a "$LOG"
ls -la docs/sbc/ | tee -a "$LOG"
