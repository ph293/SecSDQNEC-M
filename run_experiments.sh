#!/usr/bin/env bash
# =====================================================================================
#  Runs every experiment of the paper in ns-3 (ns-3.40 + OFSwitch13).
#  Usage (from the ns-3.40 folder):   JOBS=4 bash scratch/secsdqnec/run_experiments.sh
#
#  JOBS         number of CPU cores to use                     (default 4)
#  WIN          training episodes per agent (15 s each)         (default 200)
#  TRAIN_SEEDS  independent trainings per method               (default "1 2 3")
#  SIMT         length of each evaluation run in seconds        (default 40)
#
#  Scenario names use '_' instead of spaces (ns-3's command line stops at spaces).
# =====================================================================================
set -e
JOBS=${JOBS:-4}
WIN=${WIN:-200}
TRAIN_SEEDS=${TRAIN_SEEDS:-"1 2 3"}
SIMT=${SIMT:-40}
OUT=${OUT:-results_v3}; MOD=${MOD:-models_v3}
BIN=$(ls build/scratch/secsdqnec/*secsdqnec* | head -1)
mkdir -p $OUT $MOD
run() { $BIN --out=$OUT --models=$MOD "$@"; }
export -f run; export BIN OUT MOD

RL="DQNEC DDQNEC SDQNEC Proposed"
ABL="Proposed_wo_trust Proposed_wo_middleware Proposed_wo_security"
BASE="Greedy Random"            # non-learning reference baselines (no training needed)
SCEN="No_attack DDoS_on_path Compromised_edge Fake_tasks Mixed"

echo "== Step 1: collect labelled detector windows (train + test)"
rm -f $OUT/det_train.csv $OUT/det_test.csv
printf "%s\n" "--mode=collect --policy=mixed --seed=1 --windows=40" "--mode=collect --policy=mixed --seed=2 --windows=30" |
  xargs -P $JOBS -I{} bash -c 'run {}'

echo "== Step 2: train and evaluate the anomaly detectors"
run --mode=traindet

echo "== Step 3: train the DRL agents inside ns-3 (each method x each training seed)"
for s in $TRAIN_SEEDS; do for m in $RL $ABL; do
  echo "--mode=train --method=$m --windows=$WIN --seed=$s"
done; done | xargs -P $JOBS -I{} bash -c 'run {}'

echo "== Step 4: evaluation runs"
rm -f $OUT/eval.csv
{
for s in $TRAIN_SEEDS; do
  for m in $RL $BASE; do
    for n in 10 20 30 40 50; do echo "--mode=eval --method=$m --trainSeed=$s --nUav=$n --lambda=0.5 --scenario=No_attack --seed=$((s+100)) --simTime=$SIMT"; done
    for l in 0.2 0.4 0.6 0.8 1.0; do echo "--mode=eval --method=$m --trainSeed=$s --nUav=20 --lambda=$l --scenario=No_attack --seed=$((s+100)) --simTime=$SIMT"; done
  done
  for m in $RL $ABL $BASE; do
    for sc in $SCEN; do
      echo "--mode=eval --method=$m --trainSeed=$s --nUav=20 --lambda=0.5 --scenario=$sc --seed=$((s+100)) --simTime=$SIMT"
    done
  done
done
} | xargs -P $JOBS -I{} bash -c 'run {}'

echo "== Step 5: graphs and tables"
python3 scratch/secsdqnec/plots_ns3.py $OUT
echo "Done: see $OUT/figures and $OUT/RESULTS.md"
