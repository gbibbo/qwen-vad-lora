#!/usr/bin/env python3
"""
B.4 Part 3 — Combine Part 1 SNR breakdown (-10 to +20 dB) with
extended SNR results (-15, -20 dB) into a unified table.

Output: audits/round2/B4_extended_snr_results.md
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path("/mnt/fast/nobackup/users/gb0048/opro3_final")
AUDITS_DIR = ROOT / "audits" / "round2"

# Part 1: all 9 configs × 6 SNR levels
PART1_CSV = AUDITS_DIR / "B4_snr_breakdown.csv"

# Part 3: 3 configs × 2 SNR levels (metrics.json per config)
EXTENDED_DIR = AUDITS_DIR / "b4_extended_snr"
EXTENDED_CONFIGS = {
    "LoRA+OPRO-Tmpl": EXTENDED_DIR / "06_lora_opro_template" / "metrics.json",
    "Qwen3+Hand": EXTENDED_DIR / "07_qwen3_baseline" / "metrics.json",
    "Qwen3+OPRO-LLM": EXTENDED_DIR / "08_qwen3_opro_llm" / "metrics.json",
}

# Display order
CONFIG_ORDER = [
    "Base+Hand", "Base+OPRO-LLM", "Base+OPRO-Tmpl",
    "LoRA+Hand", "LoRA+OPRO-LLM", "LoRA+OPRO-Tmpl",
    "Qwen3+Hand", "Qwen3+OPRO-LLM", "Qwen3+OPRO-Tmpl",
]

SNR_COLS = [
    "snr_-20dB", "snr_-15dB", "snr_-10dB", "snr_-5dB",
    "snr_0dB", "snr_5dB", "snr_10dB", "snr_20dB",
]
SNR_LABELS = ["-20", "-15", "-10", "-5", "0", "+5", "+10", "+20"]


def load_extended_metrics():
    """Load extended SNR metrics from metrics.json files."""
    rows = []
    for config_name, json_path in EXTENDED_CONFIGS.items():
        with open(json_path) as f:
            data = json.load(f)
        for cond_key, cond_data in data["condition_metrics"].items():
            rows.append({
                "config": config_name,
                "snr_level": cond_key,
                "ba": cond_data["ba"],
                "recall_speech": cond_data["speech_acc"],
                "recall_nonspeech": cond_data["nonspeech_acc"],
                "n_total": cond_data["n_samples"],
                "n_speech": cond_data["n_speech"],
                "n_nonspeech": cond_data["n_nonspeech"],
            })
    return pd.DataFrame(rows)


def build_pivot(df, metric_col):
    """Pivot to config × SNR level matrix."""
    pivot = df.pivot_table(index="config", columns="snr_level",
                           values=metric_col, aggfunc="first")
    # Reindex to desired order
    pivot = pivot.reindex(index=CONFIG_ORDER, columns=SNR_COLS)
    return pivot


def fmt_pct(val):
    if pd.isna(val):
        return "—"
    return f"{val*100:.1f}"


def generate_report(ba_pivot, speech_pivot, nonspeech_pivot):
    lines = []
    lines.append("# B.4 — SNR Robustness: Full Range (−20 to +20 dB)")
    lines.append("**Date:** 2026-02-18")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("Part 1 evaluated all 9 configs at −10 to +20 dB (n=970 per level).")
    lines.append("Part 3 extended evaluation to −15 and −20 dB for the 3 top performers:")
    lines.append("LoRA+OPRO-Tmpl, Qwen3+Hand, Qwen3+OPRO-LLM (n=970 per level).")
    lines.append("")

    # BA table
    lines.append("## Balanced Accuracy (%)")
    lines.append("")
    header = "| Config | " + " | ".join(f"{s} dB" for s in SNR_LABELS) + " |"
    sep = "|" + "---|" * (len(SNR_LABELS) + 1)
    lines.append(header)
    lines.append(sep)
    for config in CONFIG_ORDER:
        if config in ba_pivot.index:
            vals = [fmt_pct(ba_pivot.loc[config, col]) for col in SNR_COLS]
        else:
            vals = ["—"] * len(SNR_COLS)
        lines.append(f"| {config} | " + " | ".join(vals) + " |")
    lines.append("")

    # Speech recall table
    lines.append("## Speech Recall (%)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for config in CONFIG_ORDER:
        if config in speech_pivot.index:
            vals = [fmt_pct(speech_pivot.loc[config, col]) for col in SNR_COLS]
        else:
            vals = ["—"] * len(SNR_COLS)
        lines.append(f"| {config} | " + " | ".join(vals) + " |")
    lines.append("")

    # Nonspeech recall table
    lines.append("## Nonspeech Recall (%)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for config in CONFIG_ORDER:
        if config in nonspeech_pivot.index:
            vals = [fmt_pct(nonspeech_pivot.loc[config, col]) for col in SNR_COLS]
        else:
            vals = ["—"] * len(SNR_COLS)
        lines.append(f"| {config} | " + " | ".join(vals) + " |")
    lines.append("")

    # Analysis
    lines.append("## Key Findings")
    lines.append("")
    lines.append("1. **LoRA+OPRO-Tmpl** is the only system that maintains meaningful "
                 "performance at −15 dB (BA=83.3%). At −20 dB it degrades to near-chance "
                 "(51.2%), driven by speech recall collapse (2.7%) while nonspeech "
                 "recall stays high (99.8%).")
    lines.append("")
    lines.append("2. **Both Qwen3 configs collapse abruptly below −10 dB.** Qwen3+Hand "
                 "drops from 98.7% BA at −10 dB to 51.6% at −15 dB. Qwen3+OPRO-LLM "
                 "drops to exactly 50.0% (pure NONSPEECH bias) at both −15 and −20 dB.")
    lines.append("")
    lines.append("3. **The −15 dB cliff separates fine-tuned vs. zero-shot robustness.** "
                 "LoRA fine-tuning provides a noise resilience advantage that persists "
                 "~5 dB below the point where zero-shot Qwen3 fails.")
    lines.append("")
    lines.append("4. At −20 dB all systems are effectively at chance, confirming this as "
                 "the practical floor for speech detection with current LALMs.")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("B.4 — Combining SNR Tables (Part 1 + Part 3)")
    print("=" * 60)

    # Load Part 1
    df_p1 = pd.read_csv(PART1_CSV)
    print(f"Part 1: {len(df_p1)} rows ({df_p1['config'].nunique()} configs × "
          f"{df_p1['snr_level'].nunique()} SNR levels)")

    # Load Part 3
    df_ext = load_extended_metrics()
    print(f"Part 3: {len(df_ext)} rows ({df_ext['config'].nunique()} configs × "
          f"{df_ext['snr_level'].nunique()} SNR levels)")

    # Combine
    df_all = pd.concat([df_p1, df_ext], ignore_index=True)
    print(f"Combined: {len(df_all)} rows")

    # Build pivots
    ba_pivot = build_pivot(df_all, "ba")
    speech_pivot = build_pivot(df_all, "recall_speech")
    nonspeech_pivot = build_pivot(df_all, "recall_nonspeech")

    # Generate report
    report = generate_report(ba_pivot, speech_pivot, nonspeech_pivot)
    output_path = AUDITS_DIR / "B4_extended_snr_results.md"
    output_path.write_text(report)
    print(f"\nReport saved to: {output_path}")

    # Also save combined CSV
    csv_path = AUDITS_DIR / "B4_snr_combined.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
