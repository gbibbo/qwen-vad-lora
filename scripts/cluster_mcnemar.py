#!/usr/bin/env python3
"""
B.3 — Cluster-Aware McNemar Tests

The current McNemar test in stats.py treats 21,340 samples as i.i.d.,
but each of the 970 base clips generates 22 degraded variants, inducing
within-clip correlation. This script implements:

1. Standard i.i.d. McNemar (21,340 samples) — baseline/existing
2. Cluster bootstrap McNemar (resample 970 clips, B=10,000)
3. Majority-vote collapsed McNemar (~970 clips)

Reads existing predictions.csv files; no GPU needed.

Output: audits/round2/B3_cluster_mcnemar.md + .csv
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Add project root to path
ROOT = Path("/mnt/fast/nobackup/users/gb0048/opro3_final")
sys.path.insert(0, str(ROOT))

from scripts.stats import (
    load_predictions,
    extract_clip_id,
    mcnemar_exact_test,
    compute_ba,
)

CONSOLIDATED = ROOT / "results" / "BEST_CONSOLIDATED"
AUDITS_DIR = ROOT / "audits" / "round2"
AUDITS_DIR.mkdir(parents=True, exist_ok=True)

# Config paths and display names
CONFIGS = {
    "Base+Hand":      CONSOLIDATED / "01_qwen2_base_baseline" / "evaluation" / "predictions.csv",
    "Base+OPRO-LLM":  CONSOLIDATED / "02_qwen2_base_opro_llm" / "evaluation" / "predictions.csv",
    "Base+OPRO-Tmpl": CONSOLIDATED / "03_qwen2_base_opro_template" / "evaluation" / "predictions.csv",
    "LoRA+Hand":      CONSOLIDATED / "04_qwen2_lora_baseline" / "evaluation" / "predictions.csv",
    "LoRA+OPRO-LLM":  CONSOLIDATED / "05_qwen2_lora_opro_llm" / "evaluation" / "predictions.csv",
    "LoRA+OPRO-Tmpl": CONSOLIDATED / "06_qwen2_lora_opro_template" / "evaluation" / "predictions.csv",
    "Qwen3+Hand":     CONSOLIDATED / "07_qwen3_omni_baseline" / "evaluation" / "predictions.csv",
    "Qwen3+OPRO-LLM": CONSOLIDATED / "08_qwen3_omni_opro_llm" / "evaluation" / "predictions.csv",
    "Qwen3+OPRO-Tmpl":CONSOLIDATED / "09_qwen3_omni_opro_template" / "evaluation" / "predictions.csv",
}

# Primary comparisons matching Table 5
COMPARISONS = [
    ("Base+Hand",      "Base+OPRO-LLM",  "Baseline vs Base+OPRO-LLM"),
    ("Base+Hand",      "LoRA+Hand",       "Baseline vs LoRA+Hand"),
    ("LoRA+Hand",      "LoRA+OPRO-Tmpl",  "LoRA+Hand vs LoRA+OPRO-Tmpl"),
    ("LoRA+OPRO-Tmpl", "LoRA+OPRO-LLM",  "LoRA+OPRO-Tmpl vs LoRA+OPRO-LLM"),
    ("Qwen3+Hand",     "Qwen3+OPRO-LLM", "Qwen3+Hand vs Qwen3+OPRO-LLM"),
    ("LoRA+OPRO-Tmpl", "Qwen3+Hand",     "LoRA+OPRO-Tmpl vs Qwen3+Hand"),
]


def cluster_bootstrap_mcnemar(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    n_bootstrap: int = 10000,
    random_state: int = 42,
) -> Dict:
    """
    Cluster bootstrap McNemar test.

    Resamples 970 base clips (with replacement), constructs discordant
    pairs from all samples of each resampled clip, computes the discordant
    ratio per bootstrap sample.

    Returns:
        Dict with:
        - n_01_observed, n_10_observed: observed discordant pair counts
        - ratio_observed: n_01 / (n_01 + n_10)
        - ratio_ci: 95% CI on the ratio from cluster bootstrap
        - p_value: two-tailed bootstrap p-value
    """
    # Merge on audio_path
    merged = pd.merge(
        df_a[["audio_path", "correct", "clip_id"]].rename(columns={"correct": "correct_a"}),
        df_b[["audio_path", "correct"]].rename(columns={"correct": "correct_b"}),
        on="audio_path",
        how="inner",
    )

    # Observed discordant counts
    n_01_obs = int(((merged["correct_a"] == 1) & (merged["correct_b"] == 0)).sum())
    n_10_obs = int(((merged["correct_a"] == 0) & (merged["correct_b"] == 1)).sum())
    n_disc_obs = n_01_obs + n_10_obs

    if n_disc_obs == 0:
        return {
            "n_01_observed": n_01_obs,
            "n_10_observed": n_10_obs,
            "n_discordant_observed": 0,
            "ratio_observed": 0.5,
            "ratio_ci": (0.5, 0.5),
            "p_value": 1.0,
            "n_bootstrap": n_bootstrap,
        }

    ratio_obs = n_01_obs / n_disc_obs

    # Precompute per-clip discordant counts as numpy arrays for fast resampling
    clip_ids_unique = merged["clip_id"].unique()
    n_clips = len(clip_ids_unique)
    clip_id_to_idx = {cid: i for i, cid in enumerate(clip_ids_unique)}

    # Per-clip n_01 and n_10 counts
    clip_n01 = np.zeros(n_clips, dtype=np.int64)
    clip_n10 = np.zeros(n_clips, dtype=np.int64)

    for cid, grp in merged.groupby("clip_id", sort=False):
        idx = clip_id_to_idx[cid]
        clip_n01[idx] = int(((grp["correct_a"] == 1) & (grp["correct_b"] == 0)).sum())
        clip_n10[idx] = int(((grp["correct_a"] == 0) & (grp["correct_b"] == 1)).sum())

    # Vectorized bootstrap: resample clip indices and sum
    rng = np.random.RandomState(random_state)
    # Generate all bootstrap indices at once: (n_bootstrap, n_clips)
    boot_indices = rng.choice(n_clips, size=(n_bootstrap, n_clips), replace=True)

    # Sum per-clip counts for each bootstrap sample
    boot_n01 = clip_n01[boot_indices].sum(axis=1)  # shape: (n_bootstrap,)
    boot_n10 = clip_n10[boot_indices].sum(axis=1)

    boot_disc = boot_n01 + boot_n10
    # Compute ratio, handling zero-discordant case
    ratio_samples = np.where(boot_disc > 0, boot_n01 / boot_disc, 0.5)

    # CI on the ratio
    ci_lower = float(np.percentile(ratio_samples, 2.5))
    ci_upper = float(np.percentile(ratio_samples, 97.5))

    # Two-tailed p-value: proportion of bootstrap samples where ratio
    # is on the opposite side of 0.5 from the observed ratio
    # (or more extreme in the same direction from the null)
    # Using the percentile method: if CI doesn't include 0.5, it's significant
    # More precisely: compute how often |ratio_boot - 0.5| >= |ratio_obs - 0.5|
    obs_deviation = abs(ratio_obs - 0.5)
    boot_deviations = np.abs(ratio_samples - 0.5)
    # But this tests under the empirical distribution, not the null.
    # Better approach: shift the bootstrap distribution to center at 0.5,
    # then compute the tail probability
    shifted = ratio_samples - np.mean(ratio_samples) + 0.5
    shifted_deviations = np.abs(shifted - 0.5)
    p_value = float(np.mean(shifted_deviations >= obs_deviation))
    # Ensure p_value is not exactly 0 (could happen with finite bootstrap)
    p_value = max(p_value, 1.0 / (n_bootstrap + 1))

    return {
        "n_01_observed": n_01_obs,
        "n_10_observed": n_10_obs,
        "n_discordant_observed": n_disc_obs,
        "ratio_observed": float(ratio_obs),
        "ratio_ci": (ci_lower, ci_upper),
        "p_value": float(p_value),
        "n_bootstrap": n_bootstrap,
    }


def majority_vote_mcnemar(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> Dict:
    """
    Collapse predictions to clip-level via majority vote (over 22 variants),
    then run standard McNemar on ~970 independent clips.

    For each clip_id: majority vote over predictions -> clip-level prediction.
    Ground truth is the same for all variants of a clip.
    """
    # Collapse A: per-clip majority vote
    def collapse_to_clip(df):
        clip_results = []
        for clip_id, grp in df.groupby("clip_id"):
            gt = grp["ground_truth"].iloc[0]  # Same for all variants
            # Majority vote: count SPEECH vs NONSPEECH predictions
            pred_counts = grp["prediction"].value_counts()
            # Among SPEECH and NONSPEECH only (UNKNOWN counts as incorrect)
            n_speech = pred_counts.get("SPEECH", 0)
            n_nonspeech = pred_counts.get("NONSPEECH", 0)
            if n_speech >= n_nonspeech:
                majority_pred = "SPEECH"
            else:
                majority_pred = "NONSPEECH"
            clip_results.append({
                "clip_id": clip_id,
                "ground_truth": gt,
                "prediction": majority_pred,
                "correct": int(majority_pred == gt),
            })
        return pd.DataFrame(clip_results)

    clips_a = collapse_to_clip(df_a)
    clips_b = collapse_to_clip(df_b)

    # Merge on clip_id
    merged = pd.merge(
        clips_a[["clip_id", "correct"]].rename(columns={"correct": "correct_a"}),
        clips_b[["clip_id", "correct"]].rename(columns={"correct": "correct_b"}),
        on="clip_id",
        how="inner",
    )

    n_total = len(merged)
    n_00 = int(((merged["correct_a"] == 1) & (merged["correct_b"] == 1)).sum())
    n_01 = int(((merged["correct_a"] == 1) & (merged["correct_b"] == 0)).sum())
    n_10 = int(((merged["correct_a"] == 0) & (merged["correct_b"] == 1)).sum())
    n_11 = int(((merged["correct_a"] == 0) & (merged["correct_b"] == 0)).sum())

    n_discordant = n_01 + n_10

    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = scipy_stats.binomtest(
            k=n_01, n=n_discordant, p=0.5, alternative="two-sided"
        ).pvalue

    return {
        "n_clips": n_total,
        "n_00": n_00,
        "n_01": n_01,
        "n_10": n_10,
        "n_11": n_11,
        "n_discordant": n_discordant,
        "p_value": float(p_value),
    }


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Apply Holm-Bonferroni correction to a list of p-values."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(1.0, p * (n - rank))
        cummax = max(cummax, adj_p)
        adjusted[orig_idx] = cummax
    return adjusted


def main():
    print("=" * 70)
    print("B.3 — Cluster-Aware McNemar Tests")
    print("=" * 70)

    # Load all configs
    print("\nLoading predictions for all 9 configs...")
    all_configs = {}
    for name, path in CONFIGS.items():
        if path.exists():
            all_configs[name] = load_predictions(str(path))
            n_clips = all_configs[name]["clip_id"].nunique()
            print(f"  {name}: {len(all_configs[name])} samples, {n_clips} clips")
        else:
            print(f"  WARNING: {name} not found at {path}")

    # Run comparisons
    print(f"\nRunning {len(COMPARISONS)} comparisons with 3 methods each...")
    results = []

    for config_a, config_b, label in COMPARISONS:
        if config_a not in all_configs or config_b not in all_configs:
            print(f"  SKIP: {label} (missing config)")
            continue

        df_a = all_configs[config_a]
        df_b = all_configs[config_b]

        print(f"\n--- {label} ---")

        # BA for context
        ba_a = compute_ba(df_a)
        ba_b = compute_ba(df_b)
        delta_ba = ba_a - ba_b

        # Method 1: Standard i.i.d. McNemar
        print("  [1/3] Standard i.i.d. McNemar...")
        iid_result = mcnemar_exact_test(df_a, df_b)
        p_iid = iid_result["p_value"]
        print(f"        n_01={iid_result['n_01']}, n_10={iid_result['n_10']}, "
              f"p={p_iid:.2e}")

        # Method 2: Cluster bootstrap McNemar
        print("  [2/3] Cluster bootstrap McNemar (B=10,000)...")
        cluster_result = cluster_bootstrap_mcnemar(df_a, df_b, n_bootstrap=10000)
        p_cluster = cluster_result["p_value"]
        print(f"        ratio={cluster_result['ratio_observed']:.4f}, "
              f"CI=[{cluster_result['ratio_ci'][0]:.4f}, {cluster_result['ratio_ci'][1]:.4f}], "
              f"p={p_cluster:.4f}")

        # Method 3: Majority-vote collapsed McNemar
        print("  [3/3] Majority-vote collapsed McNemar...")
        mv_result = majority_vote_mcnemar(df_a, df_b)
        p_mv = mv_result["p_value"]
        print(f"        n_clips={mv_result['n_clips']}, "
              f"n_01={mv_result['n_01']}, n_10={mv_result['n_10']}, "
              f"p={p_mv:.2e}")

        results.append({
            "comparison": label,
            "config_a": config_a,
            "config_b": config_b,
            "ba_a": ba_a,
            "ba_b": ba_b,
            "delta_ba_pp": delta_ba * 100,
            "p_iid": p_iid,
            "n_01_iid": iid_result["n_01"],
            "n_10_iid": iid_result["n_10"],
            "n_discordant_iid": iid_result["n_01"] + iid_result["n_10"],
            "p_cluster": p_cluster,
            "ratio_cluster": cluster_result["ratio_observed"],
            "ratio_ci_low": cluster_result["ratio_ci"][0],
            "ratio_ci_high": cluster_result["ratio_ci"][1],
            "p_majority_vote": p_mv,
            "n_01_mv": mv_result["n_01"],
            "n_10_mv": mv_result["n_10"],
            "n_clips_mv": mv_result["n_clips"],
        })

    # Apply Holm-Bonferroni correction
    p_iids = [r["p_iid"] for r in results]
    p_clusters = [r["p_cluster"] for r in results]
    p_mvs = [r["p_majority_vote"] for r in results]

    p_iid_adj = holm_bonferroni(p_iids)
    p_cluster_adj = holm_bonferroni(p_clusters)
    p_mv_adj = holm_bonferroni(p_mvs)

    for i, r in enumerate(results):
        r["p_iid_adj"] = p_iid_adj[i]
        r["p_cluster_adj"] = p_cluster_adj[i]
        r["p_mv_adj"] = p_mv_adj[i]

    # Save CSV
    df_results = pd.DataFrame(results)
    csv_path = AUDITS_DIR / "B3_cluster_mcnemar.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nCSV saved to: {csv_path}")

    # Generate report
    generate_report(results)


def generate_report(results: List[Dict]):
    lines = []
    lines.append("# B.3 — Cluster-Aware McNemar Tests")
    lines.append(f"**Date:** 2026-02-17")
    lines.append("")
    lines.append("## Motivation")
    lines.append("")
    lines.append("Each of the 970 base clips generates 22 degraded variants, inducing within-clip")
    lines.append("correlation. The standard McNemar test treats all 21,340 samples as i.i.d.,")
    lines.append("which may inflate significance for small effect sizes. We implement two")
    lines.append("cluster-aware alternatives to verify robustness of p-values.")
    lines.append("")
    lines.append("## Methods")
    lines.append("")
    lines.append("1. **Standard i.i.d. McNemar** — Exact binomial test on 21,340 discordant pairs")
    lines.append("2. **Cluster bootstrap McNemar** (B=10,000) — Resample 970 base clips with replacement,")
    lines.append("   compute discordant ratio per bootstrap sample, derive p-value from bootstrap null")
    lines.append("3. **Majority-vote collapsed McNemar** — Collapse 22 variants per clip to single")
    lines.append("   prediction (majority vote), then standard McNemar on ~970 independent clips")
    lines.append("")
    lines.append("All p-values corrected with Holm-Bonferroni for multiple comparisons.")
    lines.append("")

    # Summary table
    lines.append("## Results")
    lines.append("")
    lines.append("| Comparison | ΔBA (pp) | p (i.i.d.) | p (cluster) | p (majority) | Sig changes? |")
    lines.append("|-----------|---------|-----------|------------|-------------|-------------|")

    for r in results:
        sig_iid = r["p_iid_adj"] < 0.05
        sig_cluster = r["p_cluster_adj"] < 0.05
        sig_mv = r["p_mv_adj"] < 0.05

        change = ""
        if sig_iid and not sig_cluster:
            change = "i.i.d.→NS (cluster)"
        elif sig_iid and not sig_mv:
            change = "i.i.d.→NS (majority)"
        elif not sig_iid:
            change = "All NS"
        else:
            change = "All significant"

        lines.append(
            f"| {r['comparison']} | {r['delta_ba_pp']:+.1f} | "
            f"{r['p_iid_adj']:.2e} | {r['p_cluster_adj']:.4f} | "
            f"{r['p_mv_adj']:.2e} | {change} |"
        )

    lines.append("")

    # Detailed results
    lines.append("## Detailed Results")
    lines.append("")

    for r in results:
        lines.append(f"### {r['comparison']}")
        lines.append(f"- {r['config_a']}: BA = {r['ba_a']*100:.1f}%")
        lines.append(f"- {r['config_b']}: BA = {r['ba_b']*100:.1f}%")
        lines.append(f"- ΔBA = {r['delta_ba_pp']:+.1f} pp")
        lines.append("")
        lines.append(f"**i.i.d. McNemar:** n_01={r['n_01_iid']}, n_10={r['n_10_iid']}, "
                     f"n_disc={r['n_discordant_iid']}, p={r['p_iid']:.2e} "
                     f"(adj={r['p_iid_adj']:.2e})")
        lines.append("")
        lines.append(f"**Cluster bootstrap:** ratio={r['ratio_cluster']:.4f}, "
                     f"CI=[{r['ratio_ci_low']:.4f}, {r['ratio_ci_high']:.4f}], "
                     f"p={r['p_cluster']:.4f} (adj={r['p_cluster_adj']:.4f})")
        lines.append("")
        lines.append(f"**Majority-vote:** n_clips={r['n_clips_mv']}, "
                     f"n_01={r['n_01_mv']}, n_10={r['n_10_mv']}, "
                     f"p={r['p_majority_vote']:.2e} (adj={r['p_mv_adj']:.2e})")
        lines.append("")

    # Key finding
    lines.append("## Key Finding: Qwen3+Hand vs Qwen3+OPRO-LLM")
    lines.append("")
    qwen3_result = next((r for r in results if "Qwen3+Hand" in r["comparison"]), None)
    if qwen3_result:
        lines.append(f"This is the most vulnerable comparison (ΔBA = {qwen3_result['delta_ba_pp']:+.1f} pp).")
        lines.append(f"- i.i.d. p = {qwen3_result['p_iid']:.2e} (adjusted: {qwen3_result['p_iid_adj']:.2e})")
        lines.append(f"- Cluster p = {qwen3_result['p_cluster']:.4f} (adjusted: {qwen3_result['p_cluster_adj']:.4f})")
        lines.append(f"- Majority p = {qwen3_result['p_majority_vote']:.2e} (adjusted: {qwen3_result['p_mv_adj']:.2e})")
        lines.append("")
        if qwen3_result["p_cluster_adj"] >= 0.05:
            lines.append("**FINDING:** The cluster-aware correction renders this comparison non-significant.")
            lines.append("The paper text should be updated to reflect this.")
        elif qwen3_result["p_mv_adj"] >= 0.05:
            lines.append("**FINDING:** The majority-vote test renders this comparison non-significant,")
            lines.append("while the cluster bootstrap remains significant. The paper should note the")
            lines.append("sensitivity to clustering assumptions.")
        else:
            lines.append("**FINDING:** All three methods agree that this comparison is significant.")
            lines.append("Clustering does not change the conclusion.")

    lines.append("")

    report = "\n".join(lines)
    output_path = AUDITS_DIR / "B3_cluster_mcnemar.md"
    output_path.write_text(report)
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
