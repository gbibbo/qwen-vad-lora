#!/usr/bin/env python3
"""Generate publication-quality figures for the OPRO3 comparative study.

Produces 5 PDF figures:
  1. Fig_Duration.pdf  — Psychometric curve (BA vs duration, semilog) with CI bands
  2. Fig_SNR.pdf       — Psychometric curve (BA vs SNR, linear) with CI bands
  3. Fig_Reverb.pdf    — Psychometric curve (BA vs reverb T60, linear) with CI bands
  4. Fig_Tradeoff.pdf  — Sensitivity vs Specificity scatter (9 configs)
  5. Fig_Overall_BA.pdf — Grouped bar chart of BA_clip with bootstrap CI error bars
"""

import argparse
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'axes.labelsize': 12,
    'font.size': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (7, 4.5),
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ---------------------------------------------------------------------------
# 9-cell configuration map (folder name -> visual style)
# ---------------------------------------------------------------------------
CONFIG_MAP = {
    '01_qwen2_base_baseline':      {'label': 'Qwen2 Base + Hand',      'color': '#7f7f7f', 'style': '--', 'marker': 'o'},
    '02_qwen2_base_opro_llm':      {'label': 'Qwen2 Base + OPRO-LLM',  'color': '#1f77b4', 'style': '-',  'marker': 'o'},
    '03_qwen2_base_opro_template': {'label': 'Qwen2 Base + OPRO-Tmpl', 'color': '#aec7e8', 'style': ':',  'marker': 'o'},
    '04_qwen2_lora_baseline':      {'label': 'Qwen2 LoRA + Hand',      'color': '#ff7f0e', 'style': '--', 'marker': 's'},
    '05_qwen2_lora_opro_llm':      {'label': 'Qwen2 LoRA + OPRO-LLM',  'color': '#d62728', 'style': '-',  'marker': 's'},
    '06_qwen2_lora_opro_template': {'label': 'Qwen2 LoRA + OPRO-Tmpl', 'color': '#ff9896', 'style': ':',  'marker': 's'},
    '07_qwen3_omni_baseline':      {'label': 'Qwen3 Omni + Hand',      'color': '#2ca02c', 'style': '--', 'marker': '^'},
    '08_qwen3_omni_opro_llm':      {'label': 'Qwen3 Omni + OPRO-LLM',  'color': '#9467bd', 'style': '-',  'marker': '^'},
    '09_qwen3_omni_opro_template': {'label': 'Qwen3 Omni + OPRO-Tmpl', 'color': '#c5b0d5', 'style': ':',  'marker': '^'},
}

# Folder name -> key in statistical_analysis.json config_metrics
FOLDER_TO_STATS = {
    '01_qwen2_base_baseline':      'baseline',
    '02_qwen2_base_opro_llm':      'base_opro',
    '04_qwen2_lora_baseline':      'lora',
    '05_qwen2_lora_opro_llm':      'lora_opro',
    '06_qwen2_lora_opro_template': 'lora_opro_classic',
    '07_qwen3_omni_baseline':      'qwen3_baseline',
    '08_qwen3_omni_opro_llm':      'qwen3_opro',
}

# Bar chart layout: (folder, group_index, method_index)
BAR_LAYOUT = [
    ('01_qwen2_base_baseline',      0, 0),
    ('02_qwen2_base_opro_llm',      0, 1),
    ('03_qwen2_base_opro_template', 0, 2),
    ('04_qwen2_lora_baseline',      1, 0),
    ('05_qwen2_lora_opro_llm',      1, 1),
    ('06_qwen2_lora_opro_template', 1, 2),
    ('07_qwen3_omni_baseline',      2, 0),
    ('08_qwen3_omni_opro_llm',      2, 1),
    ('09_qwen3_omni_opro_template', 2, 2),
]
GROUP_LABELS = ['Qwen2-Audio\n(Base)', 'Qwen2-Audio\n(LoRA)', 'Qwen3-Omni\n(Frozen)']
METHOD_LABELS = ['Hand-crafted', 'OPRO-LLM', 'OPRO-Template']
METHOD_COLORS = ['#95a5a6', '#3498db', '#e74c3c']

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(results_dir):
    """Load metrics.json from all 9 experimental cells."""
    data = {}
    results_path = Path(results_dir)
    for subdir in sorted(results_path.iterdir()):
        if subdir.is_dir() and subdir.name in CONFIG_MAP:
            for candidate in [subdir / "evaluation" / "metrics.json", subdir / "metrics.json"]:
                if candidate.exists():
                    with open(candidate) as f:
                        data[subdir.name] = json.load(f)
                    break
            else:
                print(f"  Warning: No metrics found for {subdir.name}")
    return data


def load_stats(results_dir):
    """Load statistical_analysis.json (for bootstrap CIs on BA_clip)."""
    stats_file = Path(results_dir) / 'stats' / 'statistical_analysis.json'
    if stats_file.exists():
        with open(stats_file) as f:
            return json.load(f)
    print("  Warning: statistical_analysis.json not found; bar chart CIs will use normal approximation")
    return None

# ---------------------------------------------------------------------------
# Psychometric curves (with CI bands)
# ---------------------------------------------------------------------------

def extract_curve_data(metrics_data, axis_prefix):
    """Extract (x, y, y_lower, y_upper) for one config on one degradation axis.

    CI is a normal approximation on BA:
      SE(BA) = 0.5 * sqrt( p_s(1-p_s)/n_s + p_ns(1-p_ns)/n_ns )
    """
    points = []
    cond_metrics = metrics_data.get('condition_metrics', {})
    for cond_key, st in cond_metrics.items():
        if not cond_key.startswith(axis_prefix):
            continue
        raw = cond_key[len(axis_prefix):]
        if raw == 'none':
            val = 0.0
        else:
            raw = raw.replace('ms', '').replace('dB', '')
            # strip trailing 's' only (for reverb seconds), but avoid breaking negative signs
            if raw.endswith('s'):
                raw = raw[:-1]
            try:
                val = float(raw)
            except ValueError:
                continue
        ba = st['ba']
        n_s = st.get('n_speech', 0)
        n_ns = st.get('n_nonspeech', 0)
        p_s = st.get('speech_acc', 0)
        p_ns = st.get('nonspeech_acc', 0)
        if n_s > 0 and n_ns > 0:
            se = 0.5 * math.sqrt(p_s * (1 - p_s) / n_s + p_ns * (1 - p_ns) / n_ns)
            ci_lo = max(0.0, ba - 1.96 * se)
            ci_hi = min(1.0, ba + 1.96 * se)
        else:
            ci_lo = ci_hi = ba
        points.append((val, ba, ci_lo, ci_hi))
    points.sort(key=lambda p: p[0])
    if not points:
        return [], [], [], []
    x, y, lo, hi = zip(*points)
    return list(x), list(y), list(lo), list(hi)


def plot_psychometric_curve(data, axis_type, output_path):
    """Psychometric robustness curve with shaded CI bands."""
    fig, ax = plt.subplots()
    prefixes = {'duration': 'dur_', 'snr': 'snr_', 'reverb': 'reverb_'}
    x_labels = {'duration': 'Duration (ms)', 'snr': 'SNR (dB)', 'reverb': r'Reverb $T_{60}$ (s)'}
    prefix = prefixes[axis_type]

    for folder in sorted(CONFIG_MAP.keys()):
        if folder not in data:
            continue
        cfg = CONFIG_MAP[folder]
        x, y, lo, hi = extract_curve_data(data[folder], prefix)
        if not x:
            continue
        xa, ya, la, ha = np.array(x), np.array(y), np.array(lo), np.array(hi)

        plot_fn = ax.semilogx if axis_type == 'duration' else ax.plot
        plot_fn(xa, ya, label=cfg['label'], color=cfg['color'],
                linestyle=cfg['style'], marker=cfg['marker'], alpha=0.85)
        ax.fill_between(xa, la, ha, color=cfg['color'], alpha=0.10)

    ax.set_xlabel(x_labels[axis_type])
    ax.set_ylabel('Balanced Accuracy')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, label='Chance')

    if axis_type == 'duration':
        ax.set_xticks([20, 50, 100, 200, 500, 1000])
        ax.set_xticklabels(['20', '50', '100', '200', '500', '1000'])

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_path}")

# ---------------------------------------------------------------------------
# Recall trade-off scatter
# ---------------------------------------------------------------------------

def plot_recall_tradeoff(data, output_path):
    """Sensitivity vs Specificity scatter plot."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for folder in sorted(CONFIG_MAP.keys()):
        if folder not in data:
            continue
        cfg = CONFIG_MAP[folder]
        rec_s = data[folder].get('speech_acc', 0)
        rec_ns = data[folder].get('nonspeech_acc', 0)
        ax.scatter(rec_ns, rec_s, color=cfg['color'], marker=cfg['marker'],
                   s=120, label=cfg['label'], zorder=5, edgecolors='white', linewidth=0.5)

    ax.set_xlabel('Recall Non-Speech (Specificity)')
    ax.set_ylabel('Recall Speech (Sensitivity)')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_path}")

# ---------------------------------------------------------------------------
# Overall BA bar chart (grouped, with bootstrap CI error bars)
# ---------------------------------------------------------------------------

def plot_overall_ba(data, stats, output_path):
    """Grouped bar chart: 3 model groups x 3 methods, with CI error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))

    n_groups = 3
    n_methods = 3
    bar_width = 0.24
    group_gap = 0.18

    config_metrics = stats.get('config_metrics', {}) if stats else {}

    for folder, g_idx, m_idx in BAR_LAYOUT:
        if folder not in data:
            continue
        metrics = data[folder]
        ba = metrics.get('ba_clip', 0)

        # Bootstrap CI from stats.json when available, else normal approximation
        stats_key = FOLDER_TO_STATS.get(folder)
        if stats_key and stats_key in config_metrics:
            ci = config_metrics[stats_key].get('ba_clip_ci', [ba, ba])
            err_lo = max(0.0, ba - ci[0])
            err_hi = max(0.0, ci[1] - ba)
        else:
            n = metrics.get('n_samples', 21340) or 21340
            se = math.sqrt(ba * (1 - ba) / n) if n > 0 else 0
            err_lo = err_hi = 1.96 * se

        x_pos = g_idx * (n_methods * bar_width + group_gap) + m_idx * bar_width

        ax.bar(x_pos, ba, bar_width * 0.88, color=METHOD_COLORS[m_idx],
               yerr=[[err_lo], [err_hi]], capsize=3,
               error_kw={'linewidth': 1.2, 'capthick': 1.2},
               edgecolor='white', linewidth=0.5,
               label=METHOD_LABELS[m_idx] if g_idx == 0 else None)

        ax.text(x_pos, ba + err_hi + 0.008, f'{ba:.3f}',
                ha='center', va='bottom', fontsize=7.5)

    # X-axis group labels
    group_centers = [g * (n_methods * bar_width + group_gap) + bar_width
                     for g in range(n_groups)]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(GROUP_LABELS)

    ax.set_ylabel(r'Balanced Accuracy ($BA_{\mathrm{clip}}$)')
    ax.set_ylim(0.40, 1.02)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.legend(loc='upper left', framealpha=0.9)

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate OPRO3 publication figures (PDF)')
    parser.add_argument('--results_dir', required=True, help='Path to BEST_CONSOLIDATED')
    parser.add_argument('--output_dir', required=True, help='Output directory for figures')
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = load_metrics(args.results_dir)
    stats = load_stats(args.results_dir)
    print(f"Loaded metrics for {len(data)} / 9 configurations\n")

    # Psychometric curves with CI bands
    plot_psychometric_curve(data, 'duration', out / 'Fig_Duration.pdf')
    plot_psychometric_curve(data, 'snr',      out / 'Fig_SNR.pdf')
    plot_psychometric_curve(data, 'reverb',   out / 'Fig_Reverb.pdf')

    # Recall trade-off scatter
    plot_recall_tradeoff(data, out / 'Fig_Tradeoff.pdf')

    # Overall BA grouped bar chart
    plot_overall_ba(data, stats, out / 'Fig_Overall_BA.pdf')

    print(f"\nAll 5 figures saved to {out}/")


if __name__ == '__main__':
    main()
