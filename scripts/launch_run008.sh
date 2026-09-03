#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

caffeinate -i env PYTHONUNBUFFERED=1 python scripts/run_mcmc.py \
  --run-id RUN-008 \
  --title "dense mass + tree-depth cap" \
  --question "With correct gradient and a long ridge, does dense mass plus max_tree_depth=7 recover usable mixing?" \
  --design entrained \
  --duration 48 \
  --burn-in 96 \
  --chains 1 \
  --warmup 200 \
  --draws 50 \
  --target-accept 0.8 \
  --max-tree-depth 7 \
  --dense-mass \
  --fix-misclassification 0.01 \
  --seed 0 \
  2>&1 | tee docs/run008_console.log
