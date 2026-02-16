#!/usr/bin/env python3
"""Generate calibration (reliability) figure from evaluation predictions CSVs."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Match paper figure style from scripts/plot_final_figures.py
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 12,
    "font.size": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (7, 4.5),
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# Keep same config-to-color/style mapping used in final paper figures
CONFIG_MAP = {
    "01_qwen2_base_baseline":      {"label": "Baseline",           "color": "#7f7f7f", "style": "--", "marker": "o"},
    "02_qwen2_base_opro_llm":      {"label": "Base + OPRO",        "color": "#1f77b4", "style": "-",  "marker": "o"},
    "03_qwen2_base_opro_template": {"label": "Base + OPRO-Tmpl",   "color": "#aec7e8", "style": ":",  "marker": "o"},
    "04_qwen2_lora_baseline":      {"label": "LoRA",               "color": "#ff7f0e", "style": "--", "marker": "s"},
    "05_qwen2_lora_opro_llm":      {"label": "LoRA + OPRO-LLM",    "color": "#d62728", "style": "-",  "marker": "s"},
    "06_qwen2_lora_opro_template": {"label": "LoRA + OPRO",        "color": "#ff9896", "style": ":",  "marker": "s"},
    "07_qwen3_omni_baseline":      {"label": "Qwen3 Baseline",     "color": "#2ca02c", "style": "--", "marker": "^"},
    "08_qwen3_omni_opro_llm":      {"label": "Qwen3 + OPRO",       "color": "#9467bd", "style": "-",  "marker": "^"},
    "09_qwen3_omni_opro_template": {"label": "Qwen3 + OPRO-Tmpl",  "color": "#c5b0d5", "style": ":",  "marker": "^"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate reliability diagram from predictions CSVs")
    parser.add_argument(
        "--prediction_csvs",
        nargs="+",
        required=True,
        help="Paths to predictions.csv files",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Legend labels corresponding to --prediction_csvs",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output PDF path (PNG with same stem is also saved)",
    )
    return parser.parse_args()


def infer_style(csv_path: Path, fallback_idx: int):
    for part in csv_path.parts:
        if part in CONFIG_MAP:
            return CONFIG_MAP[part]
    fallback_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    return {
        "color": fallback_colors[fallback_idx % len(fallback_colors)],
        "style": "-",
        "marker": "o",
    }


def load_and_validate_predictions(csv_path: Path):
    df = pd.read_csv(csv_path)

    required = ["prediction", "ground_truth", "p_first_token"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"{csv_path}: missing required columns: {missing_str}. "
            "Expected predictions.csv with p_first_token logged by scripts/eval.py."
        )

    conf = pd.to_numeric(df["p_first_token"], errors="coerce")
    correct = (df["prediction"] == df["ground_truth"]).astype(float)

    valid_mask = conf.notna() & (conf >= 0.0) & (conf <= 1.0)
    conf_np = conf[valid_mask].to_numpy(dtype=float)
    correct_np = correct[valid_mask].to_numpy(dtype=float)

    if conf_np.size == 0:
        raise ValueError(
            f"{csv_path}: no valid p_first_token values in [0, 1]. "
            "Cannot compute calibration."
        )

    return conf_np, correct_np


def calibration_stats(conf, correct, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, edges[1:-1], right=False)  # 0..9

    mean_conf = np.full(n_bins, np.nan, dtype=float)
    acc = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    total = conf.size
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        n = int(mask.sum())
        counts[b] = n
        if n == 0:
            continue
        c = float(conf[mask].mean())
        a = float(correct[mask].mean())
        mean_conf[b] = c
        acc[b] = a
        ece += (n / total) * abs(a - c)

    return edges, mean_conf, acc, counts, ece


MIN_BIN_SAMPLES = 30  # suppress noisy bins with fewer samples


def plot_calibration(results, output_path: Path):
    n_cfg = len(results)
    fig, (ax_rel, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(7, 6.5),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        sharex=True,
    )

    # ------------------------------------------------------------------
    # Top panel: reliability diagram
    # ------------------------------------------------------------------
    ax_rel.plot(
        [0, 1], [0, 1],
        color="gray", linestyle=":", linewidth=1,
        label="Perfect calibration",
    )

    for r in results:
        valid = r["counts"] >= MIN_BIN_SAMPLES
        mc = r["mean_conf"][valid]
        ac = r["acc"][valid]
        sty = r["style"]

        # (Fix 4) Shade gap between reliability curve and diagonal
        sort_idx = np.argsort(mc)
        mc_s, ac_s = mc[sort_idx], ac[sort_idx]
        ax_rel.fill_between(
            mc_s, ac_s, mc_s,  # area between curve and y=x
            color=sty["color"], alpha=0.10,
        )

        # (Fix 3) Markers only — no connecting line through empty regions
        ax_rel.plot(
            mc, ac,
            color=sty["color"],
            linestyle=sty["style"],
            marker=sty["marker"],
            alpha=0.9,
            label=f'{r["label"]} (ECE = {r["ece"]:.3f})',
        )

    # (Fix 5) Lightweight text labels for the two regimes
    _label_bbox = dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="gray", alpha=0.8)
    ax_rel.text(
        0.28, 0.92, "Under-confident",
        fontsize=8, color="#1f77b4", bbox=_label_bbox,
        ha="center", va="center",
    )
    ax_rel.text(
        0.82, 0.88, "Well-calibrated",
        fontsize=8, color="#ff9896", bbox=_label_bbox,
        ha="center", va="center",
    )

    ax_rel.set_xlim(0.0, 1.0)
    ax_rel.set_ylim(0.0, 1.0)
    ax_rel.set_ylabel("Empirical Accuracy")
    ax_rel.grid(True, linestyle=":", alpha=0.5)
    ax_rel.legend(loc="lower right", framealpha=0.95)
    # (Fix 1) Remove x-tick labels from top panel (shared axis)
    plt.setp(ax_rel.get_xticklabels(), visible=False)

    # ------------------------------------------------------------------
    # Bottom panel: side-by-side histogram
    # ------------------------------------------------------------------
    edges = results[0]["edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    bar_w = 0.035  # (Fix 2) narrow bars so they sit side-by-side
    for i, r in enumerate(results):
        offset = (i - (n_cfg - 1) / 2) * bar_w
        ax_hist.bar(
            centers + offset,
            r["counts"],
            width=bar_w * 0.92,
            color=r["style"]["color"],
            edgecolor="black",
            linewidth=0.4,
            label=r["label"],
        )

    ax_hist.set_xlim(0.0, 1.0)
    ax_hist.set_xticks(np.linspace(0.0, 1.0, 11))
    ax_hist.set_ylabel("Count")
    # (Fix 1 & 5) Shared x-axis label only on bottom panel
    ax_hist.set_xlabel(r"Mean Predicted Confidence ($p_{\mathrm{first\_token}}$)")
    ax_hist.grid(axis="y", linestyle=":", alpha=0.5)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_path}")
    print(f"Saved {png_path}")


def main():
    args = parse_args()

    if len(args.prediction_csvs) != len(args.labels):
        raise ValueError(
            f"--prediction_csvs count ({len(args.prediction_csvs)}) must match "
            f"--labels count ({len(args.labels)})."
        )

    output_path = Path(args.output)
    results = []
    for i, (csv, label) in enumerate(zip(args.prediction_csvs, args.labels)):
        csv_path = Path(csv)
        conf, correct = load_and_validate_predictions(csv_path)
        edges, mean_conf, acc, counts, ece = calibration_stats(conf, correct, n_bins=10)
        results.append({
            "label": label,
            "csv_path": csv_path,
            "style": infer_style(csv_path, i),
            "edges": edges,
            "mean_conf": mean_conf,
            "acc": acc,
            "counts": counts,
            "ece": ece,
        })
        print(f"{label}: N={int(counts.sum())}, ECE={ece:.4f}")

    plot_calibration(results, output_path)


if __name__ == "__main__":
    main()
