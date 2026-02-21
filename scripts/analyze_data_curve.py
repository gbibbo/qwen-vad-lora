#!/usr/bin/env python3
"""Data Curve Ablation — Post-processing Analysis.

Reads metrics from LoRA training at different subset sizes and generates:
  1. Summary CSV with BA_clip and BA_conditions for each (size, prompt)
  2. Markdown report with data efficiency analysis

Usage:
    python3 scripts/analyze_data_curve.py
    python3 scripts/analyze_data_curve.py --results_dir audits/round3/data_curve
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

SUBSET_SIZES = [256, 512, 1024]
FULL_SIZE = 3072
ALL_SIZES = SUBSET_SIZES + [FULL_SIZE]
PROMPT_TYPES = ["hand", "opro_tmpl"]
DIMENSIONS = ["duration", "snr", "reverb", "filter"]

PROMPT_LABELS = {
    "hand": "LoRA+Hand",
    "opro_tmpl": "LoRA+OPRO-Tmpl",
}

# Baselines from the paper for context
REFERENCE_BASELINES = {
    "Base+Hand": 0.7278,
    "Base+OPRO-LLM": 0.8263,
    "Qwen3+Hand": 0.9107,
}

DEFAULT_RESULTS_DIR = "audits/round3/data_curve"
EXISTING_RESULTS = "results/20260204_201138_COMPARATIVE_RUN/experiment_summary.json"

# Cell keys in experiment_summary.json
EXISTING_CELL_MAP = {
    "hand": "2A",       # LoRA + Baseline (Hand prompt)
    "opro_tmpl": "2C",  # LoRA + OPRO-Template
}


def load_existing_3072(repo_root: Path) -> dict:
    """Load n=3072 results from the original comparative run."""
    path = repo_root / EXISTING_RESULTS
    if not path.exists():
        print(f"WARNING: Existing results not found: {path}")
        return {}
    with open(path) as f:
        data = json.load(f)
    return {
        prompt: data["results"][cell]
        for prompt, cell in EXISTING_CELL_MAP.items()
        if cell in data["results"]
    }


def load_metrics(results_dir: Path, n: int, prompt_type: str) -> Optional[dict]:
    """Load metrics.json for a specific (n, prompt_type) run."""
    metrics_file = results_dir / f"n{n}" / f"eval_{prompt_type}" / "metrics.json"
    if not metrics_file.exists():
        return None
    with open(metrics_file) as f:
        return json.load(f)


def build_dataframe(results_dir: Path, existing: dict) -> pd.DataFrame:
    """Build a DataFrame with one row per (n_clips, prompt_type)."""
    rows = []

    for prompt_type in PROMPT_TYPES:
        # Subset sizes from this experiment
        for n in SUBSET_SIZES:
            metrics = load_metrics(results_dir, n, prompt_type)
            row = {
                "n_clips": n,
                "prompt_type": prompt_type,
                "label": PROMPT_LABELS[prompt_type],
            }
            if metrics:
                row.update({
                    "ba_clip": metrics["ba_clip"],
                    "ba_conditions": metrics["ba_conditions"],
                    "speech_acc": metrics["speech_acc"],
                    "nonspeech_acc": metrics["nonspeech_acc"],
                    "n_test_samples": metrics["n_samples"],
                })
                for dim in DIMENSIONS:
                    dm = metrics.get("dimension_metrics", {}).get(dim, {})
                    row[f"ba_{dim}"] = dm.get("ba")
            else:
                row["status"] = "missing"
            rows.append(row)

        # n=3072 from existing results
        m = existing.get(prompt_type)
        if m:
            row = {
                "n_clips": FULL_SIZE,
                "prompt_type": prompt_type,
                "label": PROMPT_LABELS[prompt_type],
                "ba_clip": m["ba_clip"],
                "ba_conditions": m["ba_conditions"],
                "speech_acc": m["speech_acc"],
                "nonspeech_acc": m["nonspeech_acc"],
                "n_test_samples": m["n_samples"],
            }
            for dim in DIMENSIONS:
                dm = m.get("dimension_metrics", {}).get(dim, {})
                row[f"ba_{dim}"] = dm.get("ba")
            rows.append(row)

    return pd.DataFrame(rows)


def generate_report(df: pd.DataFrame) -> str:
    """Generate a markdown report from the data curve results."""
    lines = [
        "# Data Curve Ablation — LoRA Training Size",
        "",
        "## Summary",
        "",
        "This experiment trains LoRA adapters with subset sizes of 256, 512, 1024",
        "clips (3,072 already exists) and evaluates each on the full 21,340-sample",
        "test set with two prompts (Hand-crafted and OPRO-Template T04_contrastive).",
        "",
    ]

    # Main table
    lines.append("## BA_clip by Training Size and Prompt")
    lines.append("")
    lines.append("| Train Size | LoRA+Hand BA | LoRA+OPRO-Tmpl BA |")
    lines.append("|-----------|-------------|-------------------|")

    for n in ALL_SIZES:
        hand_row = df[(df["n_clips"] == n) & (df["prompt_type"] == "hand")]
        opro_row = df[(df["n_clips"] == n) & (df["prompt_type"] == "opro_tmpl")]

        hand_val = _fmt_ba(hand_row)
        opro_val = _fmt_ba(opro_row)
        lines.append(f"| {n:>5}     | {hand_val:>11} | {opro_val:>17} |")

    lines.append("")

    # Reference baselines
    lines.append("### Reference Baselines (from paper)")
    lines.append("")
    for name, ba in REFERENCE_BASELINES.items():
        lines.append(f"- **{name}**: {ba:.4f} ({ba*100:.1f}%)")
    lines.append("")

    # Marginal gains
    lines.append("## Marginal Gains (per doubling of data)")
    lines.append("")
    # Combined marginal gain table
    lines_mg = ["| From → To | LoRA+Hand ΔBA | LoRA+OPRO-Tmpl ΔBA |",
                "|-----------|--------------|-------------------|"]
    for i in range(len(ALL_SIZES) - 1):
        n_from, n_to = ALL_SIZES[i], ALL_SIZES[i + 1]
        deltas = {}
        for prompt in PROMPT_TYPES:
            ba_from = df[(df["n_clips"] == n_from) & (df["prompt_type"] == prompt)]
            ba_to = df[(df["n_clips"] == n_to) & (df["prompt_type"] == prompt)]
            if not ba_from.empty and not ba_to.empty:
                v_from = ba_from.iloc[0].get("ba_clip")
                v_to = ba_to.iloc[0].get("ba_clip")
                if v_from is not None and v_to is not None and pd.notna(v_from) and pd.notna(v_to):
                    deltas[prompt] = v_to - v_from
        hand_d = f"{deltas.get('hand', float('nan')):+.4f}" if "hand" in deltas else "—"
        opro_d = f"{deltas.get('opro_tmpl', float('nan')):+.4f}" if "opro_tmpl" in deltas else "—"
        lines_mg.append(f"| {n_from}→{n_to} | {hand_d:>12} | {opro_d:>17} |")

    # Replace the placeholder marginal gain section
    lines = lines[:lines.index("## Marginal Gains (per doubling of data)") + 2]
    lines.extend(lines_mg)
    lines.append("")

    # Per-dimension breakdown
    lines.append("## Per-Dimension BA (LoRA+OPRO-Tmpl)")
    lines.append("")
    dim_header = "| Train Size | Duration | SNR    | Reverb | Filter |"
    dim_sep = "|-----------|----------|--------|--------|--------|"
    lines.append(dim_header)
    lines.append(dim_sep)
    opro_df = df[df["prompt_type"] == "opro_tmpl"].sort_values("n_clips")
    for _, row in opro_df.iterrows():
        if "ba_clip" not in row or pd.isna(row.get("ba_clip")):
            continue
        n = int(row["n_clips"])
        vals = [f"{row.get(f'ba_{d}', float('nan')):.4f}" if pd.notna(row.get(f"ba_{d}")) else "—"
                for d in DIMENSIONS]
        lines.append(f"| {n:>5}     | {vals[0]:>8} | {vals[1]:>6} | {vals[2]:>6} | {vals[3]:>6} |")
    lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")

    # Check saturation
    opro_vals = df[df["prompt_type"] == "opro_tmpl"].sort_values("n_clips")
    opro_valid = opro_vals.dropna(subset=["ba_clip"])
    if len(opro_valid) >= 2:
        last_two = opro_valid.tail(2)
        ba_vals = last_two["ba_clip"].tolist()
        delta = ba_vals[1] - ba_vals[0]
        n_vals = last_two["n_clips"].tolist()
        if abs(delta) < 0.005:
            lines.append(f"- **Saturation likely**: Gain from {int(n_vals[0])}→{int(n_vals[1])} "
                         f"is only {delta:+.4f} ({delta*100:+.2f} pp)")
        elif delta > 0.01:
            lines.append(f"- **Still improving**: Gain from {int(n_vals[0])}→{int(n_vals[1])} "
                         f"is {delta:+.4f} ({delta*100:+.2f} pp), suggesting more data could help")
        else:
            lines.append(f"- **Diminishing returns**: Gain from {int(n_vals[0])}→{int(n_vals[1])} "
                         f"is {delta:+.4f} ({delta*100:+.2f} pp)")

    # Qwen3 crossover
    qwen3_ba = REFERENCE_BASELINES["Qwen3+Hand"]
    opro_above_qwen3 = opro_valid[opro_valid["ba_clip"] >= qwen3_ba]
    if not opro_above_qwen3.empty:
        min_n = int(opro_above_qwen3["n_clips"].min())
        lines.append(f"- LoRA+OPRO-Tmpl surpasses Qwen3-Omni ({qwen3_ba:.4f}) at **N={min_n}** clips")
    else:
        lines.append(f"- LoRA+OPRO-Tmpl does NOT surpass Qwen3-Omni ({qwen3_ba:.4f}) at any tested size")

    # Base+OPRO-LLM crossover
    base_opro_ba = REFERENCE_BASELINES["Base+OPRO-LLM"]
    hand_above_base = df[(df["prompt_type"] == "hand") & (df["ba_clip"] >= base_opro_ba)]
    if not hand_above_base.empty:
        min_n = int(hand_above_base["n_clips"].min())
        lines.append(f"- LoRA+Hand surpasses Base+OPRO-LLM ({base_opro_ba:.4f}) at **N={min_n}** clips")

    lines.append("")
    return "\n".join(lines)


def _fmt_ba(row_df) -> str:
    """Format a BA value from a single-row DataFrame."""
    if row_df.empty:
        return "—"
    val = row_df.iloc[0].get("ba_clip")
    if val is None or pd.isna(val):
        return "missing"
    return f"{val:.4f} ({val*100:.1f}%)"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR,
                        help="Root of data curve results (default: %(default)s)")
    parser.add_argument("--output_dir", default=None,
                        help="Where to write outputs (default: same as results_dir)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    # Load existing n=3072 results
    existing = load_existing_3072(repo_root)
    if not existing:
        print("WARNING: Could not load existing n=3072 results. Report will be incomplete.")

    # Build DataFrame
    df = build_dataframe(results_dir, existing)

    # Count available results
    available = df.dropna(subset=["ba_clip"]) if "ba_clip" in df.columns else pd.DataFrame()
    missing = df[df["status"] == "missing"] if "status" in df.columns else pd.DataFrame()

    print(f"Data points available: {len(available)} / {len(ALL_SIZES) * len(PROMPT_TYPES)}")
    if not missing.empty:
        for _, row in missing.iterrows():
            print(f"  MISSING: n={int(row['n_clips'])}, prompt={row['prompt_type']}")

    # Save CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "data_curve_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSummary CSV: {csv_path}")

    # Generate and save report
    report = generate_report(df)
    report_path = output_dir / "data_curve_report.md"
    report_path.write_text(report)
    print(f"Report: {report_path}")

    # Print key results to stdout
    print("\n" + "=" * 60)
    print("DATA CURVE RESULTS")
    print("=" * 60)
    if "ba_clip" in df.columns:
        for prompt in PROMPT_TYPES:
            print(f"\n{PROMPT_LABELS[prompt]}:")
            subset = df[df["prompt_type"] == prompt].sort_values("n_clips")
            for _, row in subset.iterrows():
                if pd.notna(row.get("ba_clip")):
                    print(f"  N={int(row['n_clips']):>5}: BA_clip={row['ba_clip']:.4f} "
                          f"({row['ba_clip']*100:.1f}%)")
                else:
                    print(f"  N={int(row['n_clips']):>5}: MISSING")
    print("=" * 60)


if __name__ == "__main__":
    main()
