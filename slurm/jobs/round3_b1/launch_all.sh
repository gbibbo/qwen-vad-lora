#!/bin/bash
# =============================================================================
# B.1 — Launch all 15 multi-seed OPRO-Template jobs
# =============================================================================
# Submits 15 independent SLURM jobs: 3 models × 5 seeds
#
# Usage:
#   cd /mnt/fast/nobackup/users/gb0048/opro3_final
#   bash slurm/jobs/round3_b1/launch_all.sh
#
# To check status:
#   ./slurm/tools/on_submit.sh squeue -u gb0048
# =============================================================================

set -euo pipefail

REPO="/mnt/fast/nobackup/users/gb0048/opro3_final"
SUBMIT="$REPO/slurm/tools/on_submit.sh"
JOB="$REPO/slurm/jobs/round3_b1/b1_opro_multiseed.job"
LOGDIR="$REPO/logs"

# Ensure logs directory exists
mkdir -p "$LOGDIR"

MODELS=("base" "lora" "qwen3")
SEEDS=(42 123 456 789 1024)

echo "============================================================"
echo "B.1 Multi-seed OPRO-Template — Launching 15 jobs"
echo "============================================================"
echo ""

SUBMITTED=0

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        JOB_NAME="b1_${model}_s${seed}"
        echo -n "Submitting ${JOB_NAME}... "

        "$SUBMIT" sbatch \
            --job-name="$JOB_NAME" \
            --output="$LOGDIR/${JOB_NAME}_%j.out" \
            --error="$LOGDIR/${JOB_NAME}_%j.err" \
            "$JOB" "$model" "$seed"

        SUBMITTED=$((SUBMITTED + 1))
    done
    echo ""
done

echo "============================================================"
echo "Submitted $SUBMITTED / 15 jobs"
echo ""
echo "Monitor with:"
echo "  $SUBMIT squeue -u gb0048"
echo ""
echo "Results will appear in:"
echo "  audits/round3/b1_multiseed/{base,lora,qwen3}_seed{42,123,456,789,1024}/"
echo ""
echo "Once all jobs complete, run post-processing:"
echo "  python3 scripts/analyze_multiseed_opro.py"
echo "============================================================"
