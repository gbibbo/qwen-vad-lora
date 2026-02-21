#!/usr/bin/env python3
"""
B.4 Part 1 — Extract BA by SNR level for all 9 configs.

Reads existing predictions.csv files from BEST_CONSOLIDATED.
Filters to variant_type == 'snr', groups by condition (snr_-10dB ... snr_20dB),
computes BA, speech recall, and nonspeech recall per SNR level per config.

Output: audits/round2/B4_snr_breakdown.md + .csv
"""

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/mnt/fast/nobackup/users/gb0048/opro3_final")
sys.path.insert(0, str(ROOT))

from scripts.stats import load_predictions, extract_clip_id

CONSOLIDATED = ROOT / "results" / "BEST_CONSOLIDATED"
AUDITS_DIR = ROOT / "audits" / "round2"
AUDITS_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "Base+Hand":       ("01_qwen2_base_baseline",      "evaluation/predictions.csv"),
    "Base+OPRO-LLM":   ("02_qwen2_base_opro_llm",      "evaluation/predictions.csv"),
    "Base+OPRO-Tmpl":  ("03_qwen2_base_opro_template",  "evaluation/predictions.csv"),
    "LoRA+Hand":       ("04_qwen2_lora_baseline",       "evaluation/predictions.csv"),
    "LoRA+OPRO-LLM":   ("05_qwen2_lora_opro_llm",      "evaluation/predictions.csv"),
    "LoRA+OPRO-Tmpl":  ("06_qwen2_lora_opro_template",  "evaluation/predictions.csv"),
    "Qwen3+Hand":      ("07_qwen3_omni_baseline",       "evaluation/predictions.csv"),
    "Qwen3+OPRO-LLM":  ("08_qwen3_omni_opro_llm",      "evaluation/predictions.csv"),
    "Qwen3+OPRO-Tmpl": ("09_qwen3_omni_opro_template",  "evaluation/predictions.csv"),
}

# SNR levels in the test set (ascending order)
SNR_LEVELS = ["snr_-10dB", "snr_-5dB", "snr_0dB", "snr_5dB", "snr_10dB", "snr_20dB"]
SNR_DISPLAY = ["-10", "-5", "0", "+5", "+10", "+20"]


def compute_metrics_at_snr(df_snr):
    """Compute BA, speech recall, nonspeech recall for a subset of predictions."""
    speech_mask = df_snr["ground_truth"] == "SPEECH"
    nonspeech_mask = df_snr["ground_truth"] == "NONSPEECH"

    n_speech = speech_mask.sum()
    n_nonspeech = nonspeech_mask.sum()

    speech_correct = (speech_mask & (df_snr["correct"] == 1)).sum()
    nonspeech_correct = (nonspeech_mask & (df_snr["correct"] == 1)).sum()

    recall_speech = speech_correct / n_speech if n_speech > 0 else 0.0
    recall_nonspeech = nonspeech_correct / n_nonspeech if n_nonspeech > 0 else 0.0
    ba = (recall_speech + recall_nonspeech) / 2

    return {
        "ba": float(ba),
        "recall_speech": float(recall_speech),
        "recall_nonspeech": float(recall_nonspeech),
        "n_total": len(df_snr),
        "n_speech": int(n_speech),
        "n_nonspeech": int(n_nonspeech),
    }


def main():
    print("=" * 70)
    print("B.4 Part 1 — BA × SNR Breakdown")
    print("=" * 70)

    all_results = {}
    csv_rows = []

    for config_name, (cell_dir, csv_rel) in CONFIGS.items():
        csv_path = CONSOLIDATED / cell_dir / csv_rel
        if not csv_path.exists():
            print(f"  WARNING: {config_name} not found at {csv_path}")
            continue

        df = load_predictions(str(csv_path))
        print(f"\n{config_name}: {len(df)} total samples")

        # Filter to SNR variants
        snr_mask = df["condition"].str.startswith("snr_")
        df_snr = df[snr_mask].copy()
        print(f"  SNR samples: {len(df_snr)}")

        config_results = {}
        for snr_level in SNR_LEVELS:
            level_mask = df_snr["condition"] == snr_level
            df_level = df_snr[level_mask]

            if len(df_level) == 0:
                # Try alternative format (some CSVs might use different format)
                print(f"  WARNING: No samples for {snr_level}")
                continue

            metrics = compute_metrics_at_snr(df_level)
            config_results[snr_level] = metrics
            print(f"  {snr_level}: BA={metrics['ba']*100:.1f}%, "
                  f"Rspeech={metrics['recall_speech']*100:.1f}%, "
                  f"Rnonspeech={metrics['recall_nonspeech']*100:.1f}%, "
                  f"n={metrics['n_total']}")

            csv_rows.append({
                "config": config_name,
                "snr_level": snr_level,
                **metrics,
            })

        all_results[config_name] = config_results

    # Save detailed CSV
    df_csv = pd.DataFrame(csv_rows)
    csv_path = AUDITS_DIR / "B4_snr_breakdown.csv"
    df_csv.to_csv(csv_path, index=False)
    print(f"\nCSV saved to: {csv_path}")

    # Generate report
    generate_report(all_results)


def generate_report(all_results):
    lines = []
    lines.append("# B.4 Part 1 — BA × SNR Breakdown (All 9 Configs)")
    lines.append(f"**Date:** 2026-02-17")
    lines.append("")
    lines.append("## Motivation")
    lines.append("")
    lines.append("All adapted models (LoRA variants + Qwen3) have SNR75 < −10 dB (censored).")
    lines.append("This table shows the complete BA at each tested SNR level to understand")
    lines.append("how much headroom remains and whether extension to lower SNRs is warranted.")
    lines.append("")

    # BA table
    lines.append("## Table: Balanced Accuracy by SNR Level (% BA)")
    lines.append("")
    header = "| Configuration | " + " | ".join(f"{s} dB" for s in SNR_DISPLAY) + " |"
    sep = "|" + "|".join(["---"] * (len(SNR_DISPLAY) + 1)) + "|"
    lines.append(header)
    lines.append(sep)

    for config_name, config_results in all_results.items():
        values = []
        for snr_level in SNR_LEVELS:
            if snr_level in config_results:
                values.append(f"{config_results[snr_level]['ba']*100:.1f}")
            else:
                values.append("—")
        lines.append(f"| {config_name} | " + " | ".join(values) + " |")

    lines.append("")

    # Speech recall table
    lines.append("## Table: Speech Recall by SNR Level (%)")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    for config_name, config_results in all_results.items():
        values = []
        for snr_level in SNR_LEVELS:
            if snr_level in config_results:
                values.append(f"{config_results[snr_level]['recall_speech']*100:.1f}")
            else:
                values.append("—")
        lines.append(f"| {config_name} | " + " | ".join(values) + " |")

    lines.append("")

    # Nonspeech recall table
    lines.append("## Table: Nonspeech Recall by SNR Level (%)")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    for config_name, config_results in all_results.items():
        values = []
        for snr_level in SNR_LEVELS:
            if snr_level in config_results:
                values.append(f"{config_results[snr_level]['recall_nonspeech']*100:.1f}")
            else:
                values.append("—")
        lines.append(f"| {config_name} | " + " | ".join(values) + " |")

    lines.append("")

    # Analysis
    lines.append("## Analysis")
    lines.append("")

    # Check which models are >90% at -10dB
    at_minus10 = {}
    for config_name, config_results in all_results.items():
        if "snr_-10dB" in config_results:
            at_minus10[config_name] = config_results["snr_-10dB"]["ba"]

    above_90 = {k: v for k, v in at_minus10.items() if v >= 0.90}
    if above_90:
        lines.append(f"**Configs with BA ≥ 90% at −10 dB ({len(above_90)}/{len(at_minus10)}):**")
        for name, ba in sorted(above_90.items(), key=lambda x: -x[1]):
            lines.append(f"- {name}: {ba*100:.1f}%")
        lines.append("")
        lines.append("These systems have significant headroom at the lowest tested SNR.")
        lines.append("Extension to −15 and −20 dB would differentiate them further.")
    else:
        lines.append("No adapted models reach 90% BA at −10 dB.")
        lines.append("SNR extension may not be warranted.")

    lines.append("")

    # Feasibility section
    lines.append("## Feasibility of SNR Extension to −15 and −20 dB")
    lines.append("")
    lines.append("- **Audio generation:** 970 clips × 2 new levels = 1,940 new files")
    lines.append("- **Generation time:** ~5 min (white noise addition, trivial computation)")
    lines.append("- **Evaluation time per model:** ~40 min (Qwen2), ~2h (Qwen3)")
    lines.append("- **Top 3 systems to evaluate:** LoRA+OPRO-Tmpl, Qwen3+Hand, Qwen3+OPRO-LLM")
    lines.append("- **Total GPU time:** ~4-5 hours on a single A100")
    lines.append("")
    lines.append("The SNR formula `noise_rms = signal_rms / 10^(snr_db/20)` works for any value.")
    lines.append("At −20 dB, noise is 10× louder than signal — very challenging but physically meaningful.")

    report = "\n".join(lines)
    output_path = AUDITS_DIR / "B4_snr_breakdown.md"
    output_path.write_text(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
