#!/usr/bin/env python3
"""
B.8 — OPRO Prompts Extraction & Analysis

Extracts all prompts evaluated during OPRO (both LLM and Template)
with their scores, and classifies them by functional type.

Output:
  - audits/round1/B8_opro_all_prompts.csv
  - audits/round1/B8_opro_prompt_analysis.md
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

ROOT = Path("/mnt/fast/nobackup/users/gb0048/opro3_final")
CONSOLIDATED = ROOT / "results" / "BEST_CONSOLIDATED"
AUDITS_DIR = ROOT / "audits" / "round1"
AUDITS_DIR.mkdir(parents=True, exist_ok=True)

OPRO_LLM_CELLS = {
    "02_qwen2_base_opro_llm": ("Base", "OPRO-LLM"),
    "05_qwen2_lora_opro_llm": ("LoRA", "OPRO-LLM"),
    "08_qwen3_omni_opro_llm": ("Qwen3", "OPRO-LLM"),
}

OPRO_TMPL_CELLS = {
    "03_qwen2_base_opro_template": ("Base", "OPRO-Tmpl"),
    "06_qwen2_lora_opro_template": ("LoRA", "OPRO-Tmpl"),
    "09_qwen3_omni_opro_template": ("Qwen3", "OPRO-Tmpl"),
}


def classify_prompt(text):
    """
    Classify a prompt into one of the functional categories defined in B.8.
    Returns a category label.
    """
    text_lower = text.lower().strip()

    # Check patterns in order of specificity

    # One-shot with example
    if "example" in text_lower or "→" in text or "audio→" in text_lower:
        return "One-shot example"

    # Contrastive with definitions
    if ("definition" in text_lower or "= human voice" in text_lower
            or "speech =" in text_lower or "nonspeech =" in text_lower
            or "treat the following as" in text_lower):
        return "Contrastive/definitions"

    # Conservative/Liberal bias
    if ("only if" in text_lower and ("clearly" in text_lower or "confident" in text_lower)):
        return "Conservative bias"
    if "any hint" in text_lower or "even faint" in text_lower:
        return "Liberal bias"

    # Focused instruction (acoustic features, short/noisy)
    if ("focus" in text_lower or "formant" in text_lower
            or "vocal tract" in text_lower or "cues" in text_lower):
        return "Acoustic focus"
    if "short" in text_lower and "noisy" in text_lower:
        return "Robustness focus"

    # Multiple-choice A/B
    if re.search(r'\bA\)|\bB\)|\bA\s*\)', text) or "multiple" in text_lower:
        return "Multiple-choice A/B"

    # Open-ended question
    if text_lower.startswith("what ") or "what type" in text_lower or "what do you hear" in text_lower:
        return "Open-ended question"

    # Binary directive (most common)
    if ("answer" in text_lower or "output" in text_lower or "reply" in text_lower
            or "respond" in text_lower or "classify" in text_lower
            or "detect" in text_lower or "label" in text_lower):
        if "speech" in text_lower and "nonspeech" in text_lower.replace("non-speech", "nonspeech"):
            return "Binary directive"

    # Direct question
    if text_lower.startswith("does ") or text_lower.startswith("is ") or "contain" in text_lower:
        return "Direct question"

    # Confidence calibration
    if "confident" in text_lower or "calibrat" in text_lower:
        return "Confidence calibration"

    # Fallback
    return "Other"


def extract_opro_llm_prompts():
    """Extract all prompts from OPRO-LLM optimization runs."""
    all_prompts = []

    for cell_dir, (model, method) in OPRO_LLM_CELLS.items():
        jsonl_path = CONSOLIDATED / cell_dir / "optimization" / "opro_prompts.jsonl"
        if not jsonl_path.exists():
            print(f"  WARNING: {jsonl_path} not found")
            continue

        with open(jsonl_path) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                prompt = data.get("prompt", "")
                all_prompts.append({
                    "cell": cell_dir,
                    "model": model,
                    "method": method,
                    "prompt": prompt,
                    "reward": data.get("reward"),
                    "ba_clip": data.get("ba_clip"),
                    "ba_conditions": data.get("ba_conditions"),
                    "iteration": data.get("iteration"),
                    "prompt_length": len(prompt),
                    "category": classify_prompt(prompt),
                })

        print(f"  {model} + {method}: {sum(1 for p in all_prompts if p['model'] == model and p['method'] == method)} prompts")

    return all_prompts


def extract_opro_template_prompts():
    """Extract all prompts from OPRO-Template optimization runs."""
    all_prompts = []

    for cell_dir, (model, method) in OPRO_TMPL_CELLS.items():
        history_path = CONSOLIDATED / cell_dir / "optimization" / "optimization_history.json"
        if not history_path.exists():
            print(f"  WARNING: {history_path} not found")
            continue

        with open(history_path) as f:
            data = json.load(f)

        history = data.get("history", [])
        config = data.get("config", {})
        n_candidates = config.get("num_candidates", 8)

        for i, (prompt, accuracy) in enumerate(history):
            iteration = (i // n_candidates) + 1
            candidate = (i % n_candidates) + 1

            all_prompts.append({
                "cell": cell_dir,
                "model": model,
                "method": method,
                "prompt": prompt,
                "reward": accuracy,  # For templates, reward = accuracy on mini-dev
                "ba_clip": accuracy,
                "ba_conditions": None,
                "iteration": iteration,
                "prompt_length": len(prompt),
                "category": classify_prompt(prompt),
            })

        print(f"  {model} + {method}: {len(history)} prompt evaluations")

    return all_prompts


def generate_report(all_prompts_df):
    """Generate analysis report."""
    lines = []
    lines.append("# B.8 — OPRO Prompts Extraction & Analysis")
    lines.append(f"**Date:** 2026-02-17")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    n_llm = len(all_prompts_df[all_prompts_df["method"] == "OPRO-LLM"])
    n_tmpl = len(all_prompts_df[all_prompts_df["method"] == "OPRO-Tmpl"])
    n_unique = all_prompts_df["prompt"].nunique()
    lines.append(f"- **Total prompt evaluations:** {len(all_prompts_df)}")
    lines.append(f"  - OPRO-LLM: {n_llm} (across 3 models)")
    lines.append(f"  - OPRO-Template: {n_tmpl} (across 3 models)")
    lines.append(f"- **Unique prompts:** {n_unique}")
    lines.append(f"- **Categories identified:** {all_prompts_df['category'].nunique()}")
    lines.append("")

    # ---- OPRO-LLM analysis ----
    lines.append("## OPRO-LLM: Generated Prompts")
    lines.append("")

    llm_df = all_prompts_df[all_prompts_df["method"] == "OPRO-LLM"]

    for model in ["Base", "LoRA", "Qwen3"]:
        model_df = llm_df[llm_df["model"] == model]
        if len(model_df) == 0:
            continue

        lines.append(f"### {model}")
        lines.append(f"Total candidates evaluated: {len(model_df)}")
        lines.append(f"Unique prompts: {model_df['prompt'].nunique()}")
        lines.append(f"Iterations: {model_df['iteration'].max() + 1}")
        lines.append("")

        # Best prompt
        best = model_df.loc[model_df["reward"].idxmax()]
        lines.append(f"**Best prompt** (reward={best['reward']:.4f}, BA_clip={best['ba_clip']:.4f}):")
        lines.append(f"> {best['prompt']}")
        lines.append("")

        # Score range by category
        cat_stats = model_df.groupby("category").agg(
            count=("reward", "count"),
            mean_reward=("reward", "mean"),
            min_reward=("reward", "min"),
            max_reward=("reward", "max"),
        ).sort_values("mean_reward", ascending=False)

        lines.append("| Category | Count | Mean Reward | Min | Max |")
        lines.append("|----------|-------|-------------|-----|-----|")
        for cat, row in cat_stats.iterrows():
            lines.append(f"| {cat} | {int(row['count'])} | {row['mean_reward']:.3f} | "
                         f"{row['min_reward']:.3f} | {row['max_reward']:.3f} |")
        lines.append("")

    # ---- OPRO-Template analysis ----
    lines.append("## OPRO-Template: Fixed Library Evaluation")
    lines.append("")

    tmpl_df = all_prompts_df[all_prompts_df["method"] == "OPRO-Tmpl"]

    for model in ["Base", "LoRA", "Qwen3"]:
        model_df = tmpl_df[tmpl_df["model"] == model]
        if len(model_df) == 0:
            continue

        lines.append(f"### {model}")
        lines.append(f"Total evaluations: {len(model_df)}")
        lines.append(f"Unique templates: {model_df['prompt'].nunique()}")
        lines.append("")

        # Best template
        best = model_df.loc[model_df["reward"].idxmax()]
        lines.append(f"**Best template** (accuracy={best['reward']:.2f}):")
        lines.append(f"> {best['prompt']}")
        lines.append("")

        # Perfect scores
        perfect = model_df[model_df["reward"] >= 1.0]
        if len(perfect) > 0:
            lines.append(f"**Templates with perfect score (1.0):** {perfect['prompt'].nunique()}")
            for prompt in perfect["prompt"].unique():
                lines.append(f"- \"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"")
            lines.append("")

        # All unique templates with scores
        unique_tmpl = model_df.groupby("prompt").agg(
            times_evaluated=("reward", "count"),
            mean_acc=("reward", "mean"),
            max_acc=("reward", "max"),
        ).sort_values("mean_acc", ascending=False)

        lines.append("| Template (truncated) | Category | Times | Mean Acc | Max Acc |")
        lines.append("|---------------------|----------|-------|----------|---------|")
        for prompt, row in unique_tmpl.iterrows():
            cat = classify_prompt(prompt)
            short = prompt[:60].replace("\n", " ") + ("..." if len(prompt) > 60 else "")
            lines.append(f"| {short} | {cat} | {int(row['times_evaluated'])} | "
                         f"{row['mean_acc']:.2f} | {row['max_acc']:.2f} |")
        lines.append("")

    # ---- Cross-method category analysis ----
    lines.append("## Category Analysis Across All Methods")
    lines.append("")

    cat_overall = all_prompts_df.groupby("category").agg(
        n_prompts=("prompt", "count"),
        n_unique=("prompt", "nunique"),
        mean_score=("reward", "mean"),
        std_score=("reward", "std"),
        min_score=("reward", "min"),
        max_score=("reward", "max"),
    ).sort_values("mean_score", ascending=False)

    lines.append("| Category | Evaluations | Unique Prompts | Mean Score | Std | Min | Max |")
    lines.append("|----------|-------------|----------------|------------|-----|-----|-----|")
    for cat, row in cat_overall.iterrows():
        lines.append(
            f"| {cat} | {int(row['n_prompts'])} | {int(row['n_unique'])} | "
            f"{row['mean_score']:.3f} | {row['std_score']:.3f} | "
            f"{row['min_score']:.3f} | {row['max_score']:.3f} |"
        )
    lines.append("")

    # Variance assessment
    score_range = cat_overall["max_score"].max() - cat_overall["min_score"].min()
    cat_mean_range = cat_overall["mean_score"].max() - cat_overall["mean_score"].min()

    lines.append("### Variance Assessment")
    lines.append("")
    lines.append(f"- Overall score range: {cat_overall['min_score'].min():.3f} to {cat_overall['max_score'].max():.3f}")
    lines.append(f"- Range of category means: {cat_mean_range:.3f}")

    if cat_mean_range > 0.1:
        lines.append("- **Conclusion:** Significant variation between prompt types. "
                     "A summary table by category would be informative for the paper.")
    else:
        lines.append("- **Conclusion:** Low variation between prompt types. "
                     "A textual note may suffice instead of a full table.")
    lines.append("")

    report = "\n".join(lines)
    output_path = AUDITS_DIR / "B8_opro_prompt_analysis.md"
    output_path.write_text(report)
    print(f"Report written to: {output_path}")
    return report


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("B.8 — OPRO Prompts Extraction & Analysis")
    print("=" * 60)

    print("\nExtracting OPRO-LLM prompts...")
    llm_prompts = extract_opro_llm_prompts()

    print("\nExtracting OPRO-Template prompts...")
    tmpl_prompts = extract_opro_template_prompts()

    # Combine
    all_prompts = llm_prompts + tmpl_prompts
    all_prompts_df = pd.DataFrame(all_prompts)

    # Save CSV
    csv_path = AUDITS_DIR / "B8_opro_all_prompts.csv"
    all_prompts_df.to_csv(csv_path, index=False)
    print(f"\nCSV written to: {csv_path}")

    print("\nGenerating report...")
    report = generate_report(all_prompts_df)
    print("\nDone!")
