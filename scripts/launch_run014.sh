#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== RUN-014 START $(date -Iseconds) ===" | tee docs/run014_console.log
set +e
caffeinate -i env PYTHONUNBUFFERED=1 python scripts/run_mcmc.py \
  --run-id RUN-014 \
  --title "Four-chain R-hat on the entrained fibre" \
  --question "What does R-hat report on fibre coordinates when ESS is order 1-15; do chains separate on chi_w while agreeing on transition times?" \
  --design entrained \
  --duration 24 \
  --burn-in 48 \
  --chains 4 \
  --warmup 200 \
  --draws 100 \
  --max-tree-depth 7 \
  --dense-mass \
  --fix-misclassification 0.01 \
  --seed 0 \
  2>&1 | tee -a docs/run014_console.log
run_exit=${pipestatus[1]}
set -e
echo "RUN014_EXIT:${run_exit} $(date -Iseconds)" | tee -a docs/run014_console.log
if [[ -d runs/RUN-014 ]]; then
  cp docs/run014_console.log runs/RUN-014/stdout.log
fi
exit "${run_exit}"
