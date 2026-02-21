#!/usr/bin/env python3
"""Generate ESC-50 category accuracy heatmap for the OPRO3 paper.

Reads the per-category accuracy CSV and produces a heatmap (PDF) with
50 ESC-50 categories on the y-axis grouped by acoustic type and
9 experimental configurations on the x-axis.
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'axes.labelsize': 10,
    'font.size': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 7,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Config columns in desired order
CONFIG_COLS = [
    'Base+Hand', 'Base+OPRO-LLM', 'Base+OPRO-Tmpl',
    'LoRA+Hand', 'LoRA+OPRO-LLM', 'LoRA+OPRO-Tmpl',
    'Qwen3+Hand', 'Qwen3+OPRO-LLM', 'Qwen3+OPRO-Tmpl',
]

# Group ordering (must match CSV group names exactly)
GROUP_ORDER = [
    'Human vocalizations',
    'Animal vocalizations',
    'Natural/ambient',
    'Mechanical/domestic',
    'Machinery/transport',
]


def main():
    parser = argparse.ArgumentParser(description='Generate ESC-50 accuracy heatmap')
    parser.add_argument('--csv', required=True, help='Path to B7_esc50_category_accuracy.csv')
    parser.add_argument('--output', required=True, help='Output PDF path')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Sort by group order, then by mean accuracy within each group
    group_rank = {g: i for i, g in enumerate(GROUP_ORDER)}
    df['group_rank'] = df['group'].map(group_rank).fillna(99)
    df = df.sort_values(['group_rank', 'mean_accuracy'], ascending=[True, True])

    # Extract heatmap data
    categories = df['category'].tolist()
    groups = df['group'].tolist()
    data = df[CONFIG_COLS].values  # shape: (50, 9)

    fig, ax = plt.subplots(figsize=(8, 12))

    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0.0, vmax=1.0,
                   interpolation='nearest')

    # X-axis: config names
    ax.set_xticks(range(len(CONFIG_COLS)))
    ax.set_xticklabels(CONFIG_COLS, rotation=45, ha='right')

    # Y-axis: category names with group separators
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)

    # Add group separators and labels
    current_group = None
    group_starts = []
    for i, g in enumerate(groups):
        if g != current_group:
            if current_group is not None:
                ax.axhline(i - 0.5, color='white', linewidth=2)
            group_starts.append((i, g))
            current_group = g

    # Add group labels on the right
    group_short = {
        'Human vocalizations': 'Human vocal.',
        'Animal vocalizations': 'Animal vocal.',
        'Natural/ambient': 'Natural/amb.',
        'Mechanical/domestic': 'Mech./domestic',
        'Machinery/transport': 'Mach./transport',
    }
    for idx, (start, group_name) in enumerate(group_starts):
        if idx + 1 < len(group_starts):
            end = group_starts[idx + 1][0]
        else:
            end = len(categories)
        mid = (start + end - 1) / 2
        label = group_short.get(group_name, group_name)
        ax.text(len(CONFIG_COLS) + 0.3, mid, label,
                va='center', ha='left', fontsize=7, style='italic',
                clip_on=False)

    # Colorbar — use axes_divider so it doesn't compete with group labels
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=1.6)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Accuracy')

    ax.set_xlabel('Configuration')
    ax.set_ylabel('ESC-50 Category')

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == '__main__':
    main()
