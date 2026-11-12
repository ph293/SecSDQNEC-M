#!/usr/bin/env bash
# =====================================================================================
#  Extra ablation (kept ready for reviewers / colloquium; not in the submitted paper).
#  Run AFTER run_experiments.sh (needs models_v3/ with the detectors and Proposed models).
#  Usage (from the ns-3.40 folder):   JOBS=4 bash scratch/secsdqnec/run_ablation2.sh
#
#   1. Proposed_wo_mask       : trust in the state + security reward, but NO action masking
#   2. Proposed_wo_secreward  : action masking kept, security term removed from the reward
#   3. Masking threshold      : Proposed evaluated with tau_min = 0.3, 0.5, 0.6 (0.4 = default)
#  Output: results_ablation2/eval.csv, ABLATION2.md and figures/
# =====================================================================================
set -e
JOBS=${JOBS:-4}
WIN=${WIN:-200}
TRAIN_SEEDS=${TRAIN_SEEDS:-"1 2 3"}
SIMT=${SIMT:-40}
MOD=${MOD:-models_v3}; OUT=${OUT:-results_ablation2}
BIN=$(ls build/scratch/secsdqnec/*secsdqnec* | head -1)
mkdir -p $OUT
[ -f $MOD/Proposed_s1.dqn ] || { echo "models_v3 not found: run run_experiments.sh first"; exit 1; }
run() { $BIN --out=$OUT --models=$MOD "$@"; }
export -f run; export BIN OUT MOD

NEW="Proposed_wo_mask Proposed_wo_secreward"
SCEN="No_attack DDoS_on_path Compromised_edge Fake_tasks Mixed"

echo "== Ablation-2 step 1: training the two new variants (3 seeds each)"
for m in $NEW; do for s in $TRAIN_SEEDS; do
  echo "--mode=train --method=$m --windows=$WIN --seed=$s"
done; done | xargs -P $JOBS -I{} bash -c 'run {}'

echo "== Ablation-2 step 2: evaluation (9 runs per method and scenario) + masking threshold"
rm -f $OUT/eval.csv
{
for s in $TRAIN_SEEDS; do
  for e in $((s+100)) $((s+200)) $((s+300)); do
    for sc in $SCEN; do
      for m in Proposed $NEW; do
        echo "--mode=eval --method=$m --trainSeed=$s --nUav=20 --lambda=0.5 --scenario=$sc --seed=$e --simTime=$SIMT"
      done
    done
    for th in 0.3 0.5 0.6; do
      for sc in No_attack Compromised_edge Mixed; do
        echo "--mode=eval --method=Proposed --trainSeed=$s --nUav=20 --lambda=0.5 --scenario=$sc --seed=$e --simTime=$SIMT --maskTh=$th"
      done
    done
  done
done
} | xargs -P $JOBS -I{} bash -c 'run {}'

echo "== Ablation-2 step 3: tables and figures"
python3 scratch/secsdqnec/ablation2_summary.py $OUT
echo "Done: see $OUT/ABLATION2.md and $OUT/figures"
