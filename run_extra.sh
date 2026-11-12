#!/usr/bin/env bash
# =====================================================================================
#  Extra experiments for the paper (run AFTER run_experiments.sh has finished).
#  Re-uses the v3 trained models; does NOT redo the v3 results.
#  Usage (from the ns-3.40 folder):   JOBS=4 bash scratch/secsdqnec/run_extra.sh
#
#   1. Conventional-IDS baseline (SDQNEC_IDS = SDQNEC + 3-sigma threshold IDS + blocking)
#   2. Two more evaluation seeds per trained model -> 9 runs per method, 95% CI, Welch t-test
#   3. Sensitivity to attack intensity (DDoS rate, compromised-server drop probability)
#   4. Detector comparison with Isolation Forest, One-Class SVM and LOF
# =====================================================================================
set -e
JOBS=${JOBS:-4}
WIN=${WIN:-200}
TRAIN_SEEDS=${TRAIN_SEEDS:-"1 2 3"}
SIMT=${SIMT:-40}
V3=${V3:-results_v3}; MOD=${MOD:-models_v3}; OUT=${OUT:-results_extra}
BIN=$(ls build/scratch/secsdqnec/*secsdqnec* | head -1)
mkdir -p $OUT
[ -f $MOD/Proposed_s1.dqn ] || { echo "models_v3 not found: run run_experiments.sh first"; exit 1; }
run() { $BIN --out=$OUT --models=$MOD "$@"; }
export -f run; export BIN OUT MOD

RL="DQNEC DDQNEC SDQNEC Proposed"
ABL="Proposed_wo_trust Proposed_wo_middleware Proposed_wo_security"
BASE="Greedy Random"
SCEN="No_attack DDoS_on_path Compromised_edge Fake_tasks Mixed"

echo "== Extra 1: conventional IDS baseline (threshold statistics + training)"
$BIN --mode=staticdet --out=$V3 --models=$MOD
for s in $TRAIN_SEEDS; do echo "--mode=train --method=SDQNEC_IDS --windows=$WIN --seed=$s"; done |
  xargs -P $JOBS -I{} bash -c 'run {}'

echo "== Extra 2+3: evaluation runs (extra seeds + sensitivity)"
rm -f $OUT/eval.csv
{
for s in $TRAIN_SEEDS; do
  for sc in $SCEN; do
    # the new baseline also gets the original evaluation seed (100+s)
    for e in $((s+100)) $((s+200)) $((s+300)); do
      echo "--mode=eval --method=SDQNEC_IDS --trainSeed=$s --nUav=20 --lambda=0.5 --scenario=$sc --seed=$e --simTime=$SIMT"
    done
    for m in $RL $ABL $BASE; do
      for e in $((s+200)) $((s+300)); do
        echo "--mode=eval --method=$m --trainSeed=$s --nUav=20 --lambda=0.5 --scenario=$sc --seed=$e --simTime=$SIMT"
      done
    done
  done
  for m in Proposed SDQNEC SDQNEC_IDS; do
    for r in 30 60 90 120; do
      echo "--mode=eval --method=$m --trainSeed=$s --nUav=20 --lambda=0.5 --scenario=DDoS_on_path --ddosMbps=$r --seed=$((s+100)) --simTime=$SIMT"
    done
    for p in 0.2 0.4 0.6 0.8; do
      echo "--mode=eval --method=$m --trainSeed=$s --nUav=20 --lambda=0.5 --scenario=Compromised_edge --dropProb=$p --seed=$((s+100)) --simTime=$SIMT"
    done
  done
done
} | xargs -P $JOBS -I{} bash -c 'run {}'

echo "== Extra 4: detector comparison with standard ML detectors"
python3 scratch/secsdqnec/detector_ml.py $V3 $OUT

echo "== Extra 5: statistics, graphs and tables"
python3 scratch/secsdqnec/plots_extra.py $V3 $OUT
echo "Done: see $OUT/EXTRA_RESULTS.md and $OUT/figures"
