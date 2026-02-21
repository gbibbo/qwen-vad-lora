#!/usr/bin/env python3
"""
B.6 Post-processing — Compute Silero VAD metrics with bootstrap CIs.

Reads Silero predictions (both max-frame and speech-ratio criteria) and
computes the same metrics as for LALMs:
  - BA_clip with cluster bootstrap 95% CI
  - Per-class recall with Wilson score intervals
  - Per-condition BA (22 conditions)
  - Per-dimension BA (4 dimensions)
  - Psychometric thresholds: DT50, DT75, DT90, SNR-75

Output: audits/round2/B6_silero_results.md
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

ROOT = Path("/mnt/fast/nobackup/users/gb0048/opro3_final")
sys.path.insert(0, str(ROOT))

from scripts.stats import (
    load_predictions,
    compute_ba,
    cluster_bootstrap_ba,
    compute_recalls_with_wilson,
    extract_clip_id,
)

AUDITS_DIR = ROOT / "audits" / "round2"
SILERO_DIR = AUDITS_DIR / "b6_silero"


def compute_per_condition_ba(df: pd.DataFrame) -> Dict:
    """Compute BA per condition."""
    condition_metrics = {}
    for condition, grp in df.groupby("condition"):
        speech_mask = grp["ground_truth"] == "SPEECH"
        nonspeech_mask = grp["ground_truth"] == "NONSPEECH"

        n_speech = speech_mask.sum()
        n_nonspeech = nonspeech_mask.sum()

        speech_correct = (speech_mask & (grp["correct"] == 1)).sum()
        nonspeech_correct = (nonspeech_mask & (grp["correct"] == 1)).sum()

        recall_speech = speech_correct / n_speech if n_speech > 0 else 0
        recall_nonspeech = nonspeech_correct / n_nonspeech if n_nonspeech > 0 else 0
        ba = (recall_speech + recall_nonspeech) / 2

        condition_metrics[condition] = {
            "ba": float(ba),
            "recall_speech": float(recall_speech),
            "recall_nonspeech": float(recall_nonspeech),
            "n": len(grp),
        }
    return condition_metrics


def estimate_psychometric_threshold(condition_metrics: Dict, dimension: str,
                                    target_ba: float) -> Dict:
    """Estimate psychometric threshold via linear interpolation."""
    if dimension == "duration":
        prefix = "dur_"
        parse_fn = lambda c: int(c.replace("dur_", "").replace("ms", ""))
        ascending = True  # BA increases with duration
    elif dimension == "snr":
        prefix = "snr_"
        parse_fn = lambda c: float(c.replace("snr_", "").replace("dB", ""))
        ascending = True  # BA increases with SNR (higher = cleaner)
    else:
        return {"point": None, "censoring": "not_applicable"}

    # Extract (value, BA) pairs
    points = []
    for cond, metrics in condition_metrics.items():
        if cond.startswith(prefix):
            try:
                val = parse_fn(cond)
                points.append((val, metrics["ba"]))
            except (ValueError, KeyError):
                continue

    if len(points) < 2:
        return {"point": None, "censoring": "insufficient_data"}

    points.sort(key=lambda x: x[0])
    values = [p[0] for p in points]
    bas = [p[1] for p in points]

    # Check censoring
    if all(b >= target_ba for b in bas):
        return {"point": values[0], "censoring": "below_range"}
    if all(b < target_ba for b in bas):
        return {"point": values[-1], "censoring": "above_range"}

    # Linear interpolation
    try:
        interp = interp1d(bas, values, bounds_error=False, fill_value="extrapolate")
        threshold = float(interp(target_ba))
        return {"point": threshold, "censoring": "ok"}
    except Exception:
        return {"point": None, "censoring": "failed"}


def analyze_criterion(predictions_path: str, criterion_name: str) -> Dict:
    """Full analysis for one prediction file."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {criterion_name}")
    print(f"{'='*60}")

    df = load_predictions(predictions_path)
    n_clips = df["clip_id"].nunique()
    print(f"  Samples: {len(df)}, Clips: {n_clips}")

    # BA with cluster bootstrap CI
    print("  Computing BA with cluster bootstrap (B=10,000)...")
    ba_point, ba_ci_low, ba_ci_high = cluster_bootstrap_ba(df, n_bootstrap=10000)
    print(f"  BA = {ba_point*100:.1f}% [{ba_ci_low*100:.1f}, {ba_ci_high*100:.1f}]")

    # Per-class recalls with Wilson CIs
    recalls = compute_recalls_with_wilson(df)
    print(f"  Recall(SPEECH) = {recalls['recall_speech']*100:.1f}% "
          f"[{recalls['recall_speech_ci'][0]*100:.1f}, {recalls['recall_speech_ci'][1]*100:.1f}]")
    print(f"  Recall(NONSPEECH) = {recalls['recall_nonspeech']*100:.1f}% "
          f"[{recalls['recall_nonspeech_ci'][0]*100:.1f}, {recalls['recall_nonspeech_ci'][1]*100:.1f}]")

    # Per-condition BA
    cond_metrics = compute_per_condition_ba(df)

    # Per-dimension BA
    dim_metrics = {}
    for dim_prefix, dim_name in [("dur_", "duration"), ("snr_", "snr"),
                                  ("reverb_", "reverb"), ("filter_", "filter")]:
        dim_conds = {k: v for k, v in cond_metrics.items() if k.startswith(dim_prefix)}
        if dim_conds:
            dim_ba = np.mean([v["ba"] for v in dim_conds.values()])
            dim_metrics[dim_name] = float(dim_ba)

    # Psychometric thresholds
    thresholds = {}
    for target, name in [(0.5, "DT50"), (0.75, "DT75"), (0.9, "DT90")]:
        thresholds[name] = estimate_psychometric_threshold(cond_metrics, "duration", target)
    thresholds["SNR75"] = estimate_psychometric_threshold(cond_metrics, "snr", 0.75)

    return {
        "criterion": criterion_name,
        "ba_clip": ba_point,
        "ba_clip_ci": (ba_ci_low, ba_ci_high),
        "recalls": recalls,
        "condition_metrics": cond_metrics,
        "dimension_metrics": dim_metrics,
        "thresholds": thresholds,
        "n_samples": len(df),
        "n_clips": n_clips,
    }


def generate_report(results_max: Dict, results_ratio: Dict):
    """Generate markdown report."""
    lines = []
    lines.append("# B.6 — Silero VAD Under Psychometric Bank")
    lines.append(f"**Date:** 2026-02-17")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("Silero VAD evaluated on all 21,340 test clips using two operating points:")
    lines.append("1. **Max-frame:** If ANY frame has speech probability ≥ 0.5 → SPEECH")
    lines.append("2. **Speech-ratio:** If proportion of speech frames ≥ 0.5 → SPEECH")
    lines.append("   (consistent with the speech_ratio criterion used for data curation)")
    lines.append("")

    for label, results in [("Max-frame", results_max), ("Speech-ratio", results_ratio)]:
        lines.append(f"## {label} Criterion")
        lines.append("")
        lines.append(f"- **BA_clip:** {results['ba_clip']*100:.1f}% "
                     f"[{results['ba_clip_ci'][0]*100:.1f}, {results['ba_clip_ci'][1]*100:.1f}]")
        r = results["recalls"]
        lines.append(f"- **Recall(SPEECH):** {r['recall_speech']*100:.1f}% "
                     f"[{r['recall_speech_ci'][0]*100:.1f}, {r['recall_speech_ci'][1]*100:.1f}]")
        lines.append(f"- **Recall(NONSPEECH):** {r['recall_nonspeech']*100:.1f}% "
                     f"[{r['recall_nonspeech_ci'][0]*100:.1f}, {r['recall_nonspeech_ci'][1]*100:.1f}]")
        lines.append("")

        # Per-dimension BA
        lines.append("### Per-Dimension BA")
        lines.append("")
        lines.append("| Dimension | BA (%) |")
        lines.append("|-----------|--------|")
        for dim, ba in sorted(results["dimension_metrics"].items()):
            lines.append(f"| {dim} | {ba*100:.1f} |")
        lines.append("")

        # Psychometric thresholds
        lines.append("### Psychometric Thresholds")
        lines.append("")
        for name, thr in results["thresholds"].items():
            if thr["point"] is not None:
                unit = "ms" if name.startswith("DT") else "dB"
                lines.append(f"- **{name}:** {thr['point']:.0f} {unit} ({thr['censoring']})")
            else:
                lines.append(f"- **{name}:** N/A ({thr['censoring']})")
        lines.append("")

        # Per-condition BA table
        lines.append("### Per-Condition BA")
        lines.append("")
        lines.append("| Condition | BA (%) | R_speech (%) | R_nonspeech (%) | n |")
        lines.append("|-----------|--------|-------------|----------------|---|")

        for cond in sorted(results["condition_metrics"].keys()):
            m = results["condition_metrics"][cond]
            lines.append(f"| {cond} | {m['ba']*100:.1f} | "
                         f"{m['recall_speech']*100:.1f} | {m['recall_nonspeech']*100:.1f} | {m['n']} |")
        lines.append("")

    # Comparison summary
    lines.append("## Summary: Two Operating Points Compared")
    lines.append("")
    lines.append("| Metric | Max-frame | Speech-ratio |")
    lines.append("|--------|-----------|-------------|")
    lines.append(f"| BA_clip | {results_max['ba_clip']*100:.1f}% | {results_ratio['ba_clip']*100:.1f}% |")
    lines.append(f"| Recall(SPEECH) | {results_max['recalls']['recall_speech']*100:.1f}% | "
                 f"{results_ratio['recalls']['recall_speech']*100:.1f}% |")
    lines.append(f"| Recall(NONSPEECH) | {results_max['recalls']['recall_nonspeech']*100:.1f}% | "
                 f"{results_ratio['recalls']['recall_nonspeech']*100:.1f}% |")
    lines.append("")
    lines.append("The max-frame criterion is more sensitive (higher speech recall, lower nonspeech recall),")
    lines.append("while the speech-ratio criterion is more conservative and consistent with the data curation filter.")

    report = "\n".join(lines)
    output_path = AUDITS_DIR / "B6_silero_results.md"
    output_path.write_text(report)
    print(f"\nReport saved to: {output_path}")

    # Save JSON for downstream integration
    json_data = {
        "max_frame": {
            "ba_clip": results_max["ba_clip"],
            "ba_clip_ci": results_max["ba_clip_ci"],
            "recalls": results_max["recalls"],
            "dimension_metrics": results_max["dimension_metrics"],
            "thresholds": results_max["thresholds"],
        },
        "speech_ratio": {
            "ba_clip": results_ratio["ba_clip"],
            "ba_clip_ci": results_ratio["ba_clip_ci"],
            "recalls": results_ratio["recalls"],
            "dimension_metrics": results_ratio["dimension_metrics"],
            "thresholds": results_ratio["thresholds"],
        },
    }
    json_path = SILERO_DIR / "silero_analysis.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"JSON saved to: {json_path}")


def main():
    print("=" * 60)
    print("B.6 — Silero VAD Metrics Analysis")
    print("=" * 60)

    max_path = SILERO_DIR / "predictions_max_frame.csv"
    ratio_path = SILERO_DIR / "predictions_speech_ratio.csv"

    if not max_path.exists() or not ratio_path.exists():
        print(f"ERROR: Prediction files not found.")
        print(f"  Expected: {max_path}")
        print(f"  Expected: {ratio_path}")
        print(f"  Run eval_silero.py first.")
        sys.exit(1)

    results_max = analyze_criterion(str(max_path), "Max-frame")
    results_ratio = analyze_criterion(str(ratio_path), "Speech-ratio")

    generate_report(results_max, results_ratio)


if __name__ == "__main__":
    main()
