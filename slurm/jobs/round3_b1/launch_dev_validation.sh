#!/bin/bash
# =============================================================================
# Launch dev set validation for all 3 model configs (3 jobs total)
# =============================================================================
# Evaluates winning OPRO-Template prompts on full dev set (660 samples)
# to verify that mini-dev selection (20 samples/iter) is not seed-biased.
#
# Usage:
#   cd /mnt/fast/nobackup/users/gb0048/opro3_final
#   bash slurm/jobs/round3_b1/launch_dev_validation.sh
#
# To check status:
#   ./slurm/tools/on_submit.sh squeue -u gb0048
# =============================================================================

set -euo pipefail

REPO="/mnt/fast/nobackup/users/gb0048/opro3_final"
SUBMIT="$REPO/slurm/tools/on_submit.sh"
JOB="$REPO/slurm/jobs/round3_b1/b1_tmpl_dev_validation.job"
LOGDIR="$REPO/logs"

# Ensure logs directory exists
mkdir -p "$LOGDIR"

MODELS=("base" "lora" "qwen3")

echo "============================================================"
echo "Dev Validation — Launching 3 jobs (one per model)"
echo "  Base:  2 templates (T06_forced, T15_simplified)"
echo "  LoRA:  3 templates (T04_contrastive, T11_calibration, T12_delimiters)"
echo "  Qwen3: 3 templates (T01_minimal, T03_verbalizer, T12_delimiters)"
echo "  Total: 8 unique evaluations on 660 dev samples"
echo "============================================================"
echo ""

SUBMITTED=0

for model in "${MODELS[@]}"; do
    JOB_NAME="devval_${model}"
    echo -n "Submitting ${JOB_NAME}... "

    "$SUBMIT" sbatch \
        --job-name="$JOB_NAME" \
        --output="$LOGDIR/${JOB_NAME}_%j.out" \
        --error="$LOGDIR/${JOB_NAME}_%j.err" \
        "$JOB" "$model"

    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "============================================================"
echo "Submitted $SUBMITTED / 3 jobs"
echo ""
echo "Monitor with:"
echo "  $SUBMIT squeue -u gb0048"
echo ""
echo "Results will appear in:"
echo "  results/ablation_opro_tmpl_dev/"
echo ""
echo "Once all jobs complete, run post-processing:"
echo "  python3 scripts/analyze_tmpl_dev_validation.py"
echo "============================================================"
