#!/bin/bash
# =============================================================================
# Data Curve Ablation — Launch all train + eval jobs with dependency chaining
# =============================================================================
# Submits 9 Slurm jobs:
#   3 training jobs  (N=256, 512, 1024) — independent, can run in parallel
#   6 evaluation jobs (3 sizes x 2 prompts) — each depends on its training job
#
# Pre-requisite: Run create_data_curve_splits.py first!
#
# Usage:
#   cd /mnt/fast/nobackup/users/gb0048/opro3_final
#   bash slurm/jobs/round3_b1/launch_data_curve.sh
# =============================================================================

set -euo pipefail

REPO="/mnt/fast/nobackup/users/gb0048/opro3_final"
SUBMIT="$REPO/slurm/tools/on_submit.sh"
JOB="$REPO/slurm/jobs/round3_b1/b1_data_curve.job"
LOGDIR="$REPO/logs"
SPLIT_DIR="$REPO/audits/round3/data_curve/splits"

mkdir -p "$LOGDIR"

# ---- Pre-flight: verify splits exist ----
echo "Checking prerequisites..."
for N in 256 512 1024; do
    CSV="$SPLIT_DIR/train_n$(printf '%04d' "$N").csv"
    if [ ! -f "$CSV" ]; then
        echo "ERROR: Split CSV not found: $CSV"
        echo "Run first: python3 scripts/create_data_curve_splits.py"
        exit 1
    fi
done

if [ ! -f "$SPLIT_DIR/prompt_opro_tmpl.txt" ]; then
    echo "ERROR: Prompt file not found: $SPLIT_DIR/prompt_opro_tmpl.txt"
    echo "Run first: python3 scripts/create_data_curve_splits.py"
    exit 1
fi
echo "All splits and prompt file found."

echo ""
echo "============================================================"
echo "Data Curve Ablation — Launching 9 jobs (3 train + 6 eval)"
echo "============================================================"
echo ""

SUBMITTED=0

for N in 256 512 1024; do
    # ---- Submit training job ----
    TRAIN_NAME="dc_train_n${N}"
    echo -n "Submitting $TRAIN_NAME ... "
    TRAIN_RESULT=$("$SUBMIT" sbatch \
        --job-name="$TRAIN_NAME" \
        --output="$LOGDIR/${TRAIN_NAME}_%j.out" \
        --error="$LOGDIR/${TRAIN_NAME}_%j.err" \
        --parsable \
        "$JOB" train "$N")
    TRAIN_JOBID=$(echo "$TRAIN_RESULT" | tail -1)
    echo "JOBID=$TRAIN_JOBID"
    SUBMITTED=$((SUBMITTED + 1))

    # ---- Submit eval jobs dependent on training ----
    for PROMPT in hand opro_tmpl; do
        EVAL_NAME="dc_eval_${PROMPT}_n${N}"
        echo -n "  Submitting $EVAL_NAME (after $TRAIN_JOBID) ... "
        EVAL_RESULT=$("$SUBMIT" sbatch \
            --job-name="$EVAL_NAME" \
            --output="$LOGDIR/${EVAL_NAME}_%j.out" \
            --error="$LOGDIR/${EVAL_NAME}_%j.err" \
            --dependency=afterok:"$TRAIN_JOBID" \
            --parsable \
            "$JOB" eval "$N" "$PROMPT")
        EVAL_JOBID=$(echo "$EVAL_RESULT" | tail -1)
        echo "JOBID=$EVAL_JOBID"
        SUBMITTED=$((SUBMITTED + 1))
    done
    echo ""
done

echo "============================================================"
echo "Submitted $SUBMITTED / 9 jobs"
echo ""
echo "Monitor with:"
echo "  $SUBMIT squeue -u gb0048"
echo ""
echo "Results will appear in:"
echo "  audits/round3/data_curve/n{256,512,1024}/{train,eval_hand,eval_opro_tmpl}/"
echo ""
echo "Once all jobs complete, run:"
echo "  python3 scripts/analyze_data_curve.py"
echo "============================================================"
