#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== PROFILE CAMPAIGN START $(date -Iseconds) ===" | tee docs/profile_campaign_console.log
set +e
caffeinate -i env PYTHONUNBUFFERED=1 python scripts/profile_likelihood.py \
  --parameters chi_wake chi_sleep threshold_gap amplitude \
  --chi-wake-min 8 --chi-wake-max 40 \
  --chi-sleep-min 1.5 --chi-sleep-max 10 \
  --threshold-gap-min 0.25 --threshold-gap-max 0.75 \
  --amplitude-min 0.02 --amplitude-max 0.25 \
  --grid-size 21 \
  --max-fev 1000 \
  --retained-hours 24 \
  --burn-in-hours 48 \
  --seed 31416 \
  --mle-nll-max 15 \
  --profile-dir docs/profile_campaign \
  --output docs/profile_campaign.json \
  2>&1 | tee -a docs/profile_campaign_console.log
prof_exit=${PIPESTATUS[0]}
set -e
echo "PROFILE_EXIT:${prof_exit} $(date -Iseconds)" | tee -a docs/profile_campaign_console.log
if [[ "${prof_exit}" -ne 0 ]]; then
  echo "Profile campaign failed; not starting RUN-014" | tee -a docs/profile_campaign_console.log
  exit "${prof_exit}"
fi

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
run_exit=${PIPESTATUS[0]}
set -e
echo "RUN014_EXIT:${run_exit} $(date -Iseconds)" | tee -a docs/run014_console.log
if [[ -d runs/RUN-014 ]]; then
  cp docs/run014_console.log runs/RUN-014/stdout.log
fi
exit "${run_exit}"
