#!/usr/bin/env python3
"""
B.7 — ESC-50 Per-Category NONSPEECH Accuracy

Extracts accuracy broken down by ESC-50 category for each configuration.

Output:
  - audits/round1/B7_esc50_category_accuracy.csv
  - audits/round1/B7_esc50_accuracy_report.md
"""

import os
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path("/mnt/fast/nobackup/users/gb0048/opro3_final")
CONSOLIDATED = ROOT / "results" / "BEST_CONSOLIDATED"
AUDITS_DIR = ROOT / "audits" / "round1"
AUDITS_DIR.mkdir(parents=True, exist_ok=True)

CELLS = {
    "01_qwen2_base_baseline": "Base+Hand",
    "02_qwen2_base_opro_llm": "Base+OPRO-LLM",
    "03_qwen2_base_opro_template": "Base+OPRO-Tmpl",
    "04_qwen2_lora_baseline": "LoRA+Hand",
    "05_qwen2_lora_opro_llm": "LoRA+OPRO-LLM",
    "06_qwen2_lora_opro_template": "LoRA+OPRO-Tmpl",
    "07_qwen3_omni_baseline": "Qwen3+Hand",
    "08_qwen3_omni_opro_llm": "Qwen3+OPRO-LLM",
    "09_qwen3_omni_opro_template": "Qwen3+OPRO-Tmpl",
}

ESC50_CATEGORIES = {
    0: "dog", 1: "rooster", 2: "pig", 3: "cow", 4: "frog",
    5: "cat", 6: "hen", 7: "insects", 8: "sheep", 9: "crow",
    10: "rain", 11: "sea_waves", 12: "crackling_fire", 13: "crickets",
    14: "chirping_birds", 15: "water_drops", 16: "wind", 17: "pouring_water",
    18: "toilet_flush", 19: "thunderstorm", 20: "crying_baby", 21: "sneezing",
    22: "clapping", 23: "breathing", 24: "coughing", 25: "footsteps",
    26: "laughing", 27: "brushing_teeth", 28: "snoring", 29: "drinking_sipping",
    30: "door_knock", 31: "mouse_click", 32: "keyboard", 33: "door_wood_creaks",
    34: "can_opening", 35: "washing_machine", 36: "vacuum_cleaner",
    37: "clock_alarm", 38: "clock_tick", 39: "glass_breaking",
    40: "helicopter", 41: "chainsaw", 42: "siren", 43: "car_horn",
    44: "engine", 45: "train", 46: "church_bells", 47: "airplane",
    48: "fireworks", 49: "hand_saw",
}

CATEGORY_GROUPS = {
    "Human vocalizations": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    "Animal vocalizations": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Mechanical/domestic": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    "Natural/ambient": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    "Machinery/transport": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
}


def extract_esc50_category(audio_path):
    """Extract ESC-50 category ID from filename."""
    filename = os.path.basename(str(audio_path))
    match = re.search(r"esc50_\d+-\d+-[A-Z]-(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


def process_all_configs():
    """Process all 9 configs and extract per-category accuracy."""
    all_data = {}

    for cell_dir, label in CELLS.items():
        csv_path = CONSOLIDATED / cell_dir / "evaluation" / "predictions.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)

        # Filter to NONSPEECH ground truth only
        nonspeech = df[df["ground_truth"].str.upper().str.strip().isin(
            ["NONSPEECH", "NON-SPEECH", "NON SPEECH"]
        )].copy()

        # Extract ESC-50 category
        nonspeech["esc50_cat"] = nonspeech["audio_path"].apply(extract_esc50_category)
        esc50_only = nonspeech.dropna(subset=["esc50_cat"]).copy()
        esc50_only["esc50_cat"] = esc50_only["esc50_cat"].astype(int)

        # Normalize predictions
        pred = esc50_only["prediction"].str.upper().str.strip()
        pred = pred.str.replace("NON-SPEECH", "NONSPEECH", regex=False)
        pred = pred.str.replace("NON SPEECH", "NONSPEECH", regex=False)
        esc50_only["correct"] = (pred == "NONSPEECH").astype(int)

        # Per-category accuracy
        cat_stats = esc50_only.groupby("esc50_cat").agg(
            n_samples=("correct", "count"),
            n_correct=("correct", "sum"),
        )
        cat_stats["accuracy"] = cat_stats["n_correct"] / cat_stats["n_samples"]

        all_data[label] = cat_stats

        print(f"  {label}: {len(esc50_only)} ESC-50 samples, "
              f"overall NONSPEECH acc = {esc50_only['correct'].mean():.3f}")

    return all_data


def build_accuracy_table(all_data):
    """Build a DataFrame with categories as rows and configs as columns."""
    # Create full table
    rows = []
    for cat_id in sorted(ESC50_CATEGORIES.keys()):
        cat_name = ESC50_CATEGORIES[cat_id]
        group = next((g for g, ids in CATEGORY_GROUPS.items() if cat_id in ids), "Other")
        row = {"cat_id": cat_id, "category": cat_name, "group": group}

        for label, cat_stats in all_data.items():
            if cat_id in cat_stats.index:
                row[label] = cat_stats.loc[cat_id, "accuracy"]
                row[f"{label}_n"] = int(cat_stats.loc[cat_id, "n_samples"])
            else:
                row[label] = None
                row[f"{label}_n"] = 0

        # Mean accuracy across all configs
        accs = [row[label] for label in all_data.keys() if row.get(label) is not None]
        row["mean_accuracy"] = np.mean(accs) if accs else None

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def generate_report(all_data, table_df):
    """Generate markdown report."""
    lines = []
    lines.append("# B.7 — ESC-50 Per-Category NONSPEECH Accuracy")
    lines.append(f"**Date:** 2026-02-17")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append("NONSPEECH accuracy broken down by ESC-50 category for all 9 configurations.")
    lines.append("Each category has 22 variants per base clip in the test set.")
    lines.append("")

    # Config labels for the accuracy columns
    config_labels = list(all_data.keys())

    # Grouped table
    lines.append("## Full Results by Acoustic Group")
    lines.append("")

    for group_name, cat_ids in CATEGORY_GROUPS.items():
        lines.append(f"### {group_name}")
        lines.append("")

        # Header
        header = "| Category |"
        sep = "|----------|"
        for label in config_labels:
            short = label.replace("+", " ")
            header += f" {short} |"
            sep += "--------|"
        header += " Mean |"
        sep += "------|"
        lines.append(header)
        lines.append(sep)

        group_df = table_df[table_df["cat_id"].isin(cat_ids)].sort_values("mean_accuracy")

        for _, row in group_df.iterrows():
            line = f"| {row['category']:<20} |"
            for label in config_labels:
                val = row.get(label)
                if val is not None:
                    pct = val * 100
                    marker = " **" if pct < 80 else ""
                    marker2 = "**" if pct < 80 else ""
                    line += f" {marker2}{pct:.1f}%{marker} |"
                else:
                    line += " N/A |"
            mean_val = row.get("mean_accuracy")
            if mean_val is not None:
                line += f" {mean_val*100:.1f}% |"
            else:
                line += " N/A |"
            lines.append(line)

        lines.append("")

    # Top 10 hardest categories
    lines.append("## Top 10 Hardest Categories (by mean accuracy)")
    lines.append("")
    hardest = table_df.dropna(subset=["mean_accuracy"]).nsmallest(10, "mean_accuracy")

    lines.append("| Rank | Category | Group | Mean Acc | Hardest Config | Easiest Config |")
    lines.append("|------|----------|-------|----------|----------------|----------------|")

    for rank, (_, row) in enumerate(hardest.iterrows(), 1):
        # Find hardest and easiest config
        config_accs = {label: row.get(label) for label in config_labels
                       if row.get(label) is not None}
        if config_accs:
            hardest_cfg = min(config_accs, key=config_accs.get)
            easiest_cfg = max(config_accs, key=config_accs.get)
            lines.append(
                f"| {rank} | {row['category']} | {row['group']} | "
                f"{row['mean_accuracy']*100:.1f}% | "
                f"{hardest_cfg} ({config_accs[hardest_cfg]*100:.1f}%) | "
                f"{easiest_cfg} ({config_accs[easiest_cfg]*100:.1f}%) |"
            )

    lines.append("")

    # Paper-cited categories verification
    lines.append("## Verification of Paper Claims (Section 5.8)")
    lines.append("")
    lines.append("Paper cites three hard categories for Base+Hand:")
    lines.append("")

    paper_claims = [
        (26, "laughing", 31.8),
        (24, "coughing", 56.6),
        (20, "crying_baby", 77.3),
    ]

    for cat_id, cat_name, paper_pct in paper_claims:
        base_hand = all_data.get("Base+Hand")
        if base_hand is not None and cat_id in base_hand.index:
            actual = base_hand.loc[cat_id, "accuracy"] * 100
            diff = actual - paper_pct
            status = "MATCH" if abs(diff) < 1.0 else f"DIFF ({diff:+.1f}pp)"
            lines.append(f"- **{cat_name}:** paper={paper_pct}%, actual={actual:.1f}% [{status}]")

            # Also show Qwen3 comparison if available
            qwen3_hand = all_data.get("Qwen3+Hand")
            if qwen3_hand is not None and cat_id in qwen3_hand.index:
                q3_val = qwen3_hand.loc[cat_id, "accuracy"] * 100
                lines.append(f"  - Qwen3+Hand: {q3_val:.1f}%")
        else:
            lines.append(f"- **{cat_name}:** DATA NOT FOUND")

    lines.append("")

    # Group-level summary
    lines.append("## Group-Level Summary")
    lines.append("")
    lines.append("| Group | Mean Acc (all configs) | Hardest Category | Easiest Category |")
    lines.append("|-------|-----------------------|------------------|------------------|")

    for group_name, cat_ids in CATEGORY_GROUPS.items():
        group_df = table_df[table_df["cat_id"].isin(cat_ids)]
        mean_all = group_df["mean_accuracy"].mean()
        if len(group_df) > 0:
            hardest_row = group_df.loc[group_df["mean_accuracy"].idxmin()]
            easiest_row = group_df.loc[group_df["mean_accuracy"].idxmax()]
            lines.append(
                f"| {group_name} | {mean_all*100:.1f}% | "
                f"{hardest_row['category']} ({hardest_row['mean_accuracy']*100:.1f}%) | "
                f"{easiest_row['category']} ({easiest_row['mean_accuracy']*100:.1f}%) |"
            )

    lines.append("")

    report = "\n".join(lines)
    output_path = AUDITS_DIR / "B7_esc50_accuracy_report.md"
    output_path.write_text(report)
    print(f"Report written to: {output_path}")
    return report


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("B.7 — ESC-50 Per-Category NONSPEECH Accuracy")
    print("=" * 60)

    print("\nProcessing all configs...")
    all_data = process_all_configs()

    print("\nBuilding accuracy table...")
    table_df = build_accuracy_table(all_data)

    # Save CSV (accuracy columns only, no _n suffix columns)
    csv_cols = ["cat_id", "category", "group"] + list(CELLS.values()) + ["mean_accuracy"]
    csv_df = table_df[csv_cols].copy()
    csv_path = AUDITS_DIR / "B7_esc50_category_accuracy.csv"
    csv_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nCSV written to: {csv_path}")

    print("\nGenerating report...")
    report = generate_report(all_data, table_df)
    print("\nDone!")
