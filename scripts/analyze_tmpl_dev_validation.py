#!/usr/bin/env python3
"""
Dev Set Validation — Analyze winning OPRO-Template performance on full dev set.

Compares dev set BA (660 samples) with test set BA (21,340 samples) to quantify
the dev-test generalization gap and check if mini-dev template selection is biased.

Reads:
  - results/ablation_opro_tmpl_dev/{model}_{template_id}/metrics.json  (dev evals)
  - audits/round3/b1_multiseed/{model}_seed{seed}/evaluation/metrics.json  (test evals)
  - audits/round3/b1_multiseed/{model}_seed{seed}/optimization/optimization_history.json

Produces:
  - results/ablation_opro_tmpl_dev/summary/dev_validation.csv
  - results/ablation_opro_tmpl_dev/summary/dev_validation_report.md

Usage:
    python3 scripts/analyze_tmpl_dev_validation.py
    python3 scripts/analyze_tmpl_dev_validation.py --dev_dir results/ablation_opro_tmpl_dev
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np


MODEL_LABELS = {
    "base": "Base+OPRO-Tmpl",
    "lora": "LoRA+OPRO-Tmpl",
    "qwen3": "Qwen3+OPRO-Tmpl",
}

# Each evaluation: subdir name, model, template ID, seeds that selected it, reference seed
EVALUATIONS = [
    {
        "subdir": "base_T06_forced",
        "model": "base",
        "template_id": "T06_forced",
        "test_seeds": [42, 123, 456, 1024],
        "test_ref_seed": 42,
    },
    {
        "subdir": "base_T15_simplified",
        "model": "base",
        "template_id": "T15_simplified",
        "test_seeds": [789],
        "test_ref_seed": 789,
    },
    {
        "subdir": "lora_T04_contrastive",
        "model": "lora",
        "template_id": "T04_contrastive",
        "test_seeds": [42, 789, 1024],
        "test_ref_seed": 42,
    },
    {
        "subdir": "lora_T11_calibration",
        "model": "lora",
        "template_id": "T11_calibration",
        "test_seeds": [456],
        "test_ref_seed": 456,
    },
    {
        "subdir": "lora_T12_delimiters",
        "model": "lora",
        "template_id": "T12_delimiters",
        "test_seeds": [123],
        "test_ref_seed": 123,
    },
    {
        "subdir": "qwen3_T01_minimal",
        "model": "qwen3",
        "template_id": "T01_minimal",
        "test_seeds": [789],
        "test_ref_seed": 789,
    },
    {
        "subdir": "qwen3_T03_verbalizer",
        "model": "qwen3",
        "template_id": "T03_verbalizer",
        "test_seeds": [42, 456, 1024],
        "test_ref_seed": 42,
    },
    {
        "subdir": "qwen3_T12_delimiters",
        "model": "qwen3",
        "template_id": "T12_delimiters",
        "test_seeds": [123],
        "test_ref_seed": 123,
    },
]


def load_json(path: Path) -> dict | None:
    """Load a JSON file, returning None if missing."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Dev Set Validation — OPRO-Template winning templates on full dev set"
    )
    parser.add_argument(
        "--dev_dir", type=str,
        default="results/ablation_opro_tmpl_dev",
        help="Directory with dev set evaluation outputs"
    )
    parser.add_argument(
        "--test_dir", type=str,
        default="audits/round3/b1_multiseed",
        help="Directory with multi-seed test set results"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="results/ablation_opro_tmpl_dev/summary",
        help="Output directory for report and CSV"
    )
    args = parser.parse_args()

    dev_dir = Path(args.dev_dir)
    test_dir = Path(args.test_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Dev Set Validation — OPRO-Template Winners on Full Dev (660 samples)")
    print("=" * 70)
    print(f"Dev results:  {dev_dir}")
    print(f"Test results: {test_dir}")
    print(f"Output:       {out_dir}")
    print()

    # ---- Collect data ----
    rows = []
    for ev in EVALUATIONS:
        subdir = ev["subdir"]
        model = ev["model"]
        template_id = ev["template_id"]
        ref_seed = ev["test_ref_seed"]

        # Load dev metrics
        dev_metrics = load_json(dev_dir / subdir / "metrics.json")

        # Load test metrics from the reference seed
        test_seed_dir = f"{model}_seed{ref_seed}"
        test_metrics = load_json(test_dir / test_seed_dir / "evaluation" / "metrics.json")

        # Load OPRO optimization history for mini-dev accuracy
        history = load_json(test_dir / test_seed_dir / "optimization" / "optimization_history.json")
        opro_best_dev_acc = history.get("best_accuracy") if history else None

        # Extract dimension metrics if available
        dev_dimension = dev_metrics.get("dimension_metrics", {}) if dev_metrics else {}
        test_dimension = test_metrics.get("dimension_metrics", {}) if test_metrics else {}

        row = {
            "model": model,
            "model_label": MODEL_LABELS[model],
            "template_id": template_id,
            "n_seeds_winning": len(ev["test_seeds"]),
            "winning_seeds": ", ".join(str(s) for s in ev["test_seeds"]),
            "ref_seed": ref_seed,
            # Dev (660 samples)
            "dev_ba_clip": dev_metrics["ba_clip"] if dev_metrics else None,
            "dev_ba_pct": round(dev_metrics["ba_clip"] * 100, 2) if dev_metrics else None,
            "dev_n_samples": dev_metrics["n_samples"] if dev_metrics else None,
            "dev_speech_acc": dev_metrics.get("speech_acc") if dev_metrics else None,
            "dev_nonspeech_acc": dev_metrics.get("nonspeech_acc") if dev_metrics else None,
            # Test (21,340 samples)
            "test_ba_clip": test_metrics["ba_clip"] if test_metrics else None,
            "test_ba_pct": round(test_metrics["ba_clip"] * 100, 2) if test_metrics else None,
            "test_n_samples": test_metrics["n_samples"] if test_metrics else None,
            "test_speech_acc": test_metrics.get("speech_acc") if test_metrics else None,
            "test_nonspeech_acc": test_metrics.get("nonspeech_acc") if test_metrics else None,
            # OPRO mini-dev (20 samples/iter)
            "opro_minidev_acc": opro_best_dev_acc,
            "opro_minidev_pct": round(opro_best_dev_acc * 100, 2) if opro_best_dev_acc else None,
            # Dimension BAs (dev)
            "dev_ba_duration": dev_dimension.get("duration", {}).get("ba"),
            "dev_ba_snr": dev_dimension.get("snr", {}).get("ba"),
            "dev_ba_reverb": dev_dimension.get("reverb", {}).get("ba"),
            # Dimension BAs (test)
            "test_ba_duration": test_dimension.get("duration", {}).get("ba"),
            "test_ba_snr": test_dimension.get("snr", {}).get("ba"),
            "test_ba_reverb": test_dimension.get("reverb", {}).get("ba"),
        }

        # Gaps
        if row["dev_ba_pct"] is not None and row["test_ba_pct"] is not None:
            row["gap_pp"] = round(row["dev_ba_pct"] - row["test_ba_pct"], 2)
        else:
            row["gap_pp"] = None

        rows.append(row)

        # Print status
        status = "OK" if dev_metrics else "MISSING"
        dev_str = f"{row['dev_ba_pct']:.2f}%" if row["dev_ba_pct"] else "N/A"
        test_str = f"{row['test_ba_pct']:.2f}%" if row["test_ba_pct"] else "N/A"
        gap_str = f"{row['gap_pp']:+.2f}pp" if row["gap_pp"] is not None else "N/A"
        print(f"  [{status}] {subdir}: dev={dev_str}, test={test_str}, gap={gap_str}")

    print()

    df = pd.DataFrame(rows)

    # ---- Save CSV ----
    csv_path = out_dir / "dev_validation.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    # ---- Generate report ----
    lines = []
    lines.append("# Dev Set Validation — OPRO-Template Winning Templates")
    lines.append("")
    lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("**Purpose:** Verify whether templates selected via mini-dev (20 samples/iteration)")
    lines.append("generalize to the full dev set (660 samples), and whether dev-set rankings")
    lines.append("predict test-set rankings.")
    lines.append("")

    # Completeness check
    complete = df[df["dev_ba_clip"].notna()]
    lines.append(f"**Evaluations complete:** {len(complete)} / {len(EVALUATIONS)}")
    if len(complete) < len(EVALUATIONS):
        missing = df[df["dev_ba_clip"].isna()]
        lines.append("")
        lines.append("**Missing evaluations:**")
        for _, row in missing.iterrows():
            lines.append(f"- {row['model']}_{row['template_id']}")
    lines.append("")

    # === Table 1: Main comparison ===
    lines.append("## Table 1: Dev vs Test BA by Model and Template")
    lines.append("")
    lines.append("| Model | Template | Seeds | BA_dev_660 (%) | BA_test_21340 (%) | Gap (pp) |")
    lines.append("|---|---|---|---:|---:|---:|")

    for model in ["base", "lora", "qwen3"]:
        model_df = df[df["model"] == model].sort_values("dev_ba_pct", ascending=False, na_position="last")
        for _, row in model_df.iterrows():
            label = row["model_label"]
            tid = row["template_id"]
            seeds = row["winning_seeds"]
            dev = f"{row['dev_ba_pct']:.2f}" if row["dev_ba_pct"] is not None else "---"
            test = f"{row['test_ba_pct']:.2f}" if row["test_ba_pct"] is not None else "---"
            gap = f"{row['gap_pp']:+.2f}" if row["gap_pp"] is not None else "---"
            lines.append(f"| {label} | {tid} | {seeds} | {dev} | {test} | {gap} |")

    lines.append("")

    # === Table 2: Three-level comparison (mini-dev, full-dev, test) ===
    lines.append("## Table 2: Three-Level Accuracy Comparison")
    lines.append("")
    lines.append("Shows accuracy at each evaluation stage: OPRO mini-dev (20 samples),")
    lines.append("full dev (660 samples), and test (21,340 samples).")
    lines.append("")
    lines.append("| Model | Template | Mini-dev (%) | Full-dev (%) | Test (%) |")
    lines.append("|---|---|---:|---:|---:|")

    for _, row in df.iterrows():
        label = row["model_label"]
        tid = row["template_id"]
        minidev = f"{row['opro_minidev_pct']:.1f}" if row["opro_minidev_pct"] else "---"
        fulldev = f"{row['dev_ba_pct']:.2f}" if row["dev_ba_pct"] is not None else "---"
        test = f"{row['test_ba_pct']:.2f}" if row["test_ba_pct"] is not None else "---"
        lines.append(f"| {label} | {tid} | {minidev} | {fulldev} | {test} |")

    lines.append("")

    # === Per-model ranking analysis ===
    lines.append("## Ranking Analysis: Does Dev Ranking Match Test Ranking?")
    lines.append("")

    for model in ["base", "lora", "qwen3"]:
        label = MODEL_LABELS[model]
        model_df = df[(df["model"] == model) & df["dev_ba_pct"].notna() & df["test_ba_pct"].notna()]

        if len(model_df) < 2:
            lines.append(f"### {label}")
            if len(model_df) == 1:
                lines.append("Only 1 unique template — no ranking to compare.")
            else:
                lines.append("No data available.")
            lines.append("")
            continue

        lines.append(f"### {label}")
        lines.append("")

        # Rank by dev
        dev_ranked = model_df.sort_values("dev_ba_pct", ascending=False)
        dev_order = list(dev_ranked["template_id"])

        # Rank by test
        test_ranked = model_df.sort_values("test_ba_pct", ascending=False)
        test_order = list(test_ranked["template_id"])

        lines.append(f"- **Dev ranking:**  {' > '.join(dev_order)}")
        lines.append(f"- **Test ranking:** {' > '.join(test_order)}")

        if dev_order == test_order:
            lines.append(f"- **Concordance: MATCH** — Dev set correctly predicts test ranking.")
        else:
            lines.append(f"- **Concordance: MISMATCH** — Dev and test rankings differ.")
            # Find the best-dev template and check its test rank
            best_dev_tid = dev_order[0]
            test_rank = test_order.index(best_dev_tid) + 1
            lines.append(f"  - Best dev template ({best_dev_tid}) ranks #{test_rank} on test.")

        lines.append("")

    # === Outlier analysis: Base seed 789 ===
    lines.append("## Outlier Analysis: Base Seed 789 (T15_simplified)")
    lines.append("")

    base_t15 = df[(df["model"] == "base") & (df["template_id"] == "T15_simplified")]
    base_t06 = df[(df["model"] == "base") & (df["template_id"] == "T06_forced")]

    if len(base_t15) == 1 and len(base_t06) == 1:
        t15 = base_t15.iloc[0]
        t06 = base_t06.iloc[0]

        if t15["dev_ba_pct"] is not None and t06["dev_ba_pct"] is not None:
            dev_gap = t06["dev_ba_pct"] - t15["dev_ba_pct"]
            test_gap = (t06["test_ba_pct"] or 0) - (t15["test_ba_pct"] or 0)

            lines.append(f"T06_forced (majority winner, 4/5 seeds):")
            lines.append(f"  - Dev BA:  {t06['dev_ba_pct']:.2f}%")
            lines.append(f"  - Test BA: {t06['test_ba_pct']:.2f}%")
            lines.append("")
            lines.append(f"T15_simplified (seed 789 outlier):")
            lines.append(f"  - Dev BA:  {t15['dev_ba_pct']:.2f}%")
            lines.append(f"  - Test BA: {t15['test_ba_pct']:.2f}%")
            lines.append("")
            lines.append(f"Dev gap (T06 - T15): {dev_gap:+.2f} pp")
            lines.append(f"Test gap (T06 - T15): {test_gap:+.2f} pp")
            lines.append("")

            if dev_gap > 0:
                lines.append("**Interpretation:** T06_forced outperforms T15_simplified on the full dev set.")
                lines.append("The mini-dev (20 samples) made a suboptimal selection for seed 789.")
                lines.append("A full dev set evaluation would have correctly identified T06 as superior.")
            else:
                lines.append("**Interpretation:** T15_simplified performs comparably or better on the full dev set,")
                lines.append("suggesting the test set performance gap is due to distribution differences,")
                lines.append("not a mini-dev selection artifact.")
        else:
            lines.append("Awaiting dev evaluation results for full analysis.")
    else:
        lines.append("Awaiting dev evaluation results for full analysis.")
    lines.append("")

    # === Dimension breakdown (if available) ===
    has_dim = df["dev_ba_duration"].notna().any()
    if has_dim:
        lines.append("## Dimension-Level Dev vs Test Comparison")
        lines.append("")
        lines.append("| Model | Template | Dim | Dev BA | Test BA | Gap (pp) |")
        lines.append("|---|---|---|---:|---:|---:|")

        for _, row in df.iterrows():
            for dim in ["duration", "snr", "reverb"]:
                dev_val = row.get(f"dev_ba_{dim}")
                test_val = row.get(f"test_ba_{dim}")
                if dev_val is not None and test_val is not None:
                    gap = (dev_val - test_val) * 100
                    lines.append(
                        f"| {row['model_label']} | {row['template_id']} | {dim} | "
                        f"{dev_val*100:.1f} | {test_val*100:.1f} | {gap:+.1f} |"
                    )
        lines.append("")

    # === Summary ===
    lines.append("## Summary")
    lines.append("")

    if len(complete) == len(EVALUATIONS):
        # All data available — generate summary
        gaps = df["gap_pp"].dropna()
        mean_gap = gaps.mean()
        max_gap = gaps.abs().max()

        lines.append(f"- **Mean dev-test gap:** {mean_gap:+.2f} pp (across all 8 evaluations)")
        lines.append(f"- **Max absolute gap:** {max_gap:.2f} pp")
        lines.append("")

        # Check ranking concordance
        concordant = 0
        total_models = 0
        for model in ["base", "lora", "qwen3"]:
            model_df = df[(df["model"] == model) & df["dev_ba_pct"].notna() & df["test_ba_pct"].notna()]
            if len(model_df) >= 2:
                total_models += 1
                dev_order = list(model_df.sort_values("dev_ba_pct", ascending=False)["template_id"])
                test_order = list(model_df.sort_values("test_ba_pct", ascending=False)["template_id"])
                if dev_order == test_order:
                    concordant += 1

        lines.append(f"- **Ranking concordance:** {concordant}/{total_models} models have matching dev/test rankings")
    else:
        lines.append(f"*Awaiting {len(EVALUATIONS) - len(complete)} remaining evaluations.*")

    lines.append("")

    # Write report
    report_path = out_dir / "dev_validation_report.md"
    report_path.write_text("\n".join(lines))
    print(f"Report saved: {report_path}")

    print()
    print("Done! Check:")
    print(f"  {out_dir}/dev_validation_report.md")
    print(f"  {out_dir}/dev_validation.csv")


if __name__ == "__main__":
    main()
