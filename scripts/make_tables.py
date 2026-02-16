#!/usr/bin/env python3
"""
generate_latex_tables.py
Created: 2026-01-26
Updated: 2026-01-26 - Added all 7 tables
Purpose: Generate LaTeX tables from statistical analysis results.

Generates ALL 7 tables:
1. Tab_R02_OverallPerformance.tex - BA, recalls with CIs
2. Tab_R04_dimension_means.tex - BA by degradation axis
3. Tab_R05_ErrorCounts.tex - TP/TN/FP/FN, FPR/FNR
4. Tab_PromptSummary.tex - Prompt text summary
5. tab_psychometric_thresholds.tex - DT50/75/90, SNR75
6. tab_primary_comparisons.tex - ΔBA, McNemar, Holm-Bonferroni
7. tab_prelim_multimodel_comparison.tex - Qwen2 + Qwen3 comparison
"""

import json
from pathlib import Path

# Paths (updated for opro3_final BEST_CONSOLIDATED)
REPO = Path("/mnt/fast/nobackup/users/gb0048/opro3_final")
STATS_FILE = REPO / "results/BEST_CONSOLIDATED/stats/statistical_analysis.json"
RAW_RESULTS_FILE = REPO / "results/BEST_CONSOLIDATED/all_experiment_results.json"
TABLES_DIR = REPO / "tables"
OUTPUT_CONSOLIDATED = REPO / "tables/LATEX_TABLES_UPDATED.txt"

# Paths to metrics.json for dimension means (all 9 cells)
METRICS_PATHS = {
    'baseline':            REPO / "results/BEST_CONSOLIDATED/01_qwen2_base_baseline/evaluation/metrics.json",
    'base_opro':           REPO / "results/BEST_CONSOLIDATED/02_qwen2_base_opro_llm/evaluation/metrics.json",
    'base_opro_template':  REPO / "results/BEST_CONSOLIDATED/03_qwen2_base_opro_template/evaluation/metrics.json",
    'lora':                REPO / "results/BEST_CONSOLIDATED/04_qwen2_lora_baseline/evaluation/metrics.json",
    'lora_opro_open':      REPO / "results/BEST_CONSOLIDATED/05_qwen2_lora_opro_llm/evaluation/metrics.json",
    'lora_opro':           REPO / "results/BEST_CONSOLIDATED/06_qwen2_lora_opro_template/evaluation/metrics.json",
    'lora_opro_classic':   REPO / "results/BEST_CONSOLIDATED/06_qwen2_lora_opro_template/evaluation/metrics.json",
    'qwen3_baseline':      REPO / "results/BEST_CONSOLIDATED/07_qwen3_omni_baseline/evaluation/metrics.json",
    'qwen3_opro':          REPO / "results/BEST_CONSOLIDATED/08_qwen3_omni_opro_llm/evaluation/metrics.json",
    'qwen3_opro_template': REPO / "results/BEST_CONSOLIDATED/09_qwen3_omni_opro_template/evaluation/metrics.json",
}


def load_data():
    """Load statistical analysis and raw results."""
    with open(STATS_FILE) as f:
        stats = json.load(f)

    if RAW_RESULTS_FILE.exists():
        with open(RAW_RESULTS_FILE) as f:
            raw = json.load(f)
    else:
        print(f"  Note: {RAW_RESULTS_FILE.name} not found, deriving error counts from config_metrics")
        raw = _derive_raw_from_stats(stats)

    return stats, raw


def _derive_raw_from_stats(stats):
    """Derive TP/TN/FP/FN from config_metrics when all_experiment_results.json is absent."""
    config_metrics = stats.get('config_metrics', {})

    # Map stats keys -> raw JSON keys
    qwen2_map = {
        'baseline': 'Baseline',
        'base_opro': 'Base+OPRO',
        'lora': 'LoRA',
        'lora_opro_classic': 'LoRA+OPRO',
    }
    qwen3_map = {
        'qwen3_baseline': 'Baseline',
        'qwen3_opro': 'OPRO',
    }

    def compute_counts(m):
        n_sp = m.get('n_speech', 0)
        n_ns = m.get('n_nonspeech', 0)
        tp = round(m.get('recall_speech', 0) * n_sp)
        fn = n_sp - tp
        tn = round(m.get('recall_nonspeech', 0) * n_ns)
        fp = n_ns - tn
        return {
            'ba': m.get('ba_clip', 0),
            'recall_speech': m.get('recall_speech', 0),
            'recall_nonspeech': m.get('recall_nonspeech', 0),
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        }

    qwen2_metrics = {}
    for stats_key, raw_key in qwen2_map.items():
        if stats_key in config_metrics:
            qwen2_metrics[raw_key] = compute_counts(config_metrics[stats_key])

    qwen3_metrics = {}
    for stats_key, raw_key in qwen3_map.items():
        if stats_key in config_metrics:
            qwen3_metrics[raw_key] = compute_counts(config_metrics[stats_key])

    return {
        'qwen2': {'metrics': qwen2_metrics},
        'qwen3': {'metrics': qwen3_metrics},
    }


def load_metrics_files():
    """Load metrics.json files for dimension means."""
    metrics = {}
    for name, path in METRICS_PATHS.items():
        if path.exists():
            with open(path) as f:
                metrics[name] = json.load(f)
    return metrics


def format_ci(point, ci_low, ci_high, decimals=3):
    """Format point estimate with CI: value [low, high]"""
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(point)} [{fmt.format(ci_low)}, {fmt.format(ci_high)}]"


def format_pvalue(p):
    """Format p-value with scientific notation for very small values."""
    if p < 1e-10:
        return "$<$10$^{-10}$"
    elif p < 0.001:
        exp = int(f"{p:.0e}".split('e')[1])
        return f"$<$10$^{{{exp}}}$"
    elif p < 0.01:
        return f"{p:.4f}"
    else:
        return f"{p:.3f}"


# =============================================================================
# TABLE 1: Overall Performance
# =============================================================================
def generate_overall_performance_table(stats):
    """Generate Tab_R02_OverallPerformance.tex — all 9 matrix cells."""

    # 9 configs in matrix order, grouped by model block
    config_rows = [
        ('baseline',           'Base + Hand',       'qwen2_base'),
        ('base_opro',          'Base + OPRO-LLM',   'qwen2_base'),
        ('base_opro_template', 'Base + OPRO-Tmpl',  'qwen2_base'),
        ('lora',               'LoRA + Hand',       'qwen2_lora'),
        ('lora_opro_open',     'LoRA + OPRO-LLM',   'qwen2_lora'),
        ('lora_opro',          'LoRA + OPRO-Tmpl',  'qwen2_lora'),
        ('qwen3_baseline',     'Frozen + Hand',     'qwen3'),
        ('qwen3_opro',         'Frozen + OPRO-LLM', 'qwen3'),
        ('qwen3_opro_template','Frozen + OPRO-Tmpl','qwen3'),
    ]

    model_display = {
        'qwen2_base': 'Qwen2-Audio-7B',
        'qwen2_lora': 'Qwen2-Audio-7B',
        'qwen3': 'Qwen3-Omni-30B',
    }

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Overall performance on the test set (21,340 samples). BA = Balanced Accuracy with 95\% CI (10,000 cluster-bootstrap resamples). Per-class recalls with Wilson score 95\% CI.}")
    lines.append(r"\label{tab:overall_performance}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Configuration} & \textbf{BA$_{\text{clip}}$ [95\% CI]} & \textbf{Recall$_{\text{SPEECH}}$ [95\% CI]} & \textbf{Recall$_{\text{NON-SPEECH}}$ [95\% CI]} \\")
    lines.append(r"\midrule")

    config_metrics = stats.get('config_metrics', {})

    prev_group = None
    for cfg_key, display_name, group in config_rows:
        if cfg_key not in config_metrics:
            continue

        if prev_group is not None and group != prev_group:
            lines.append(r"\midrule")
        prev_group = group

        model = model_display[group]
        m = config_metrics[cfg_key]

        ba = m.get('ba_clip', 0)
        ba_ci = m.get('ba_clip_ci', [ba, ba])
        ba_str = format_ci(ba, ba_ci[0], ba_ci[1])

        rs = m.get('recall_speech', 0)
        rs_ci = m.get('recall_speech_ci', [rs, rs])
        rs_str = format_ci(rs, rs_ci[0], rs_ci[1])

        rn = m.get('recall_nonspeech', 0)
        rn_ci = m.get('recall_nonspeech_ci', [rn, rn])
        rn_str = format_ci(rn, rn_ci[0], rn_ci[1])

        lines.append(f"{model} & {display_name} & {ba_str} & {rs_str} & {rn_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return '\n'.join(lines)


# =============================================================================
# TABLE 2: Dimension Means
# =============================================================================
def generate_dimension_means_table(metrics_data):
    """Generate Tab_R04_dimension_means.tex — transposed: configs as rows, axes as columns."""

    axes = ['duration', 'snr', 'reverb', 'filter']

    # 9 configs in matrix order, grouped by model
    config_rows = [
        ('baseline',           'Base + Hand',       'qwen2_base'),
        ('base_opro',          'Base + OPRO-LLM',   'qwen2_base'),
        ('base_opro_template', 'Base + OPRO-Tmpl',  'qwen2_base'),
        ('lora',               'LoRA + Hand',       'qwen2_lora'),
        ('lora_opro_open',     'LoRA + OPRO-LLM',   'qwen2_lora'),
        ('lora_opro',          'LoRA + OPRO-Tmpl',  'qwen2_lora'),
        ('qwen3_baseline',     'Frozen + Hand',     'qwen3'),
        ('qwen3_opro',         'Frozen + OPRO-LLM', 'qwen3'),
        ('qwen3_opro_template','Frozen + OPRO-Tmpl','qwen3'),
    ]

    model_display = {
        'qwen2_base': 'Qwen2-Audio-7B',
        'qwen2_lora': 'Qwen2-Audio-7B',
        'qwen3': 'Qwen3-Omni-30B',
    }

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Mean balanced accuracy (\%) within each degradation axis (macro-average over the conditions in that axis). Bold indicates column-wise best.}")
    lines.append(r"\label{tab:r04_dimension_means}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Configuration} & \textbf{Duration} & \textbf{SNR} & \textbf{Reverb} & \textbf{Filter} \\")
    lines.append(r"\midrule")

    # Pre-compute all values to find column maxima
    all_values = {axis: [] for axis in axes}
    row_data = []
    for cfg_key, display_name, group in config_rows:
        row_vals = {}
        for axis in axes:
            if cfg_key in metrics_data:
                dim_metrics = metrics_data[cfg_key].get('dimension_metrics', {})
                if axis in dim_metrics:
                    val = dim_metrics[axis].get('ba', 0) * 100
                    row_vals[axis] = val
                    all_values[axis].append(val)
                else:
                    row_vals[axis] = None
            else:
                row_vals[axis] = None
        row_data.append(row_vals)

    col_max = {axis: max(all_values[axis]) if all_values[axis] else 0 for axis in axes}

    prev_group = None
    for i, (cfg_key, display_name, group) in enumerate(config_rows):
        if prev_group is not None and group != prev_group:
            lines.append(r"\midrule")
        prev_group = group

        model = model_display[group]
        cells = []
        for axis in axes:
            val = row_data[i][axis]
            if val is None:
                cells.append("--")
            else:
                s = f"{val:.1f}"
                if abs(val - col_max[axis]) < 0.05:
                    s = r"\textbf{" + s + "}"
                cells.append(s)

        lines.append(f"{model} & {display_name} & {' & '.join(cells)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return '\n'.join(lines)


# =============================================================================
# TABLE 3: Error Counts
# =============================================================================
def generate_error_counts_table(raw):
    """Generate Tab_R05_ErrorCounts.tex"""

    config_display = {
        'Baseline': ('Qwen2-Audio-7B', 'Base + Hand'),
        'Base+OPRO': ('Qwen2-Audio-7B', 'Base + OPRO'),
        'LoRA': ('Qwen2-Audio-7B', 'LoRA + Hand'),
        'LoRA+OPRO': ('Qwen2-Audio-7B', 'LoRA + OPRO'),
    }

    qwen2_metrics = raw.get('qwen2', {}).get('metrics', {})
    qwen3_metrics = raw.get('qwen3', {}).get('metrics', {})

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Error profile (confusion matrix counts and derived error rates) for each configuration, aggregated over the full test set.}")
    lines.append(r"\label{tab:error-counts}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Config} & \textbf{TP} & \textbf{TN} & \textbf{FP} & \textbf{FN} & \textbf{FPR} & \textbf{FNR} \\")
    lines.append(r"\midrule")

    # Qwen2 configs
    for cfg_key in ['Baseline', 'Base+OPRO', 'LoRA', 'LoRA+OPRO']:
        if cfg_key not in qwen2_metrics:
            continue
        m = qwen2_metrics[cfg_key]
        model, config = config_display.get(cfg_key, ('Qwen2-Audio-7B', cfg_key))

        tp = m.get('TP', 0)
        tn = m.get('TN', 0)
        fp = m.get('FP', 0)
        fn = m.get('FN', 0)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        lines.append(f"{model} & {config} & {tp} & {tn} & {fp} & {fn} & {fpr:.3f} & {fnr:.3f} \\\\")

    # Qwen3 configs
    qwen3_display = {
        'Baseline': ('Qwen3-Omni-30B', 'Frozen + Hand'),
        'OPRO': ('Qwen3-Omni-30B', 'Frozen + OPRO'),
    }

    for cfg_key in ['Baseline', 'OPRO']:
        if cfg_key not in qwen3_metrics:
            continue
        m = qwen3_metrics[cfg_key]
        model, config = qwen3_display.get(cfg_key, ('Qwen3-Omni-30B', cfg_key))

        tp = m.get('TP', 0)
        tn = m.get('TN', 0)
        fp = m.get('FP', 0)
        fn = m.get('FN', 0)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        lines.append(f"{model} & {config} & {tp} & {tn} & {fp} & {fn} & {fpr:.3f} & {fnr:.3f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return '\n'.join(lines)


# =============================================================================
# TABLE 4: Prompt Summary
# =============================================================================
def generate_prompt_summary_table():
    """Generate Tab_PromptSummary.tex (static content)"""

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Summary of prompts used in each configuration. Length in characters (excluding whitespace normalization).}")
    lines.append(r"\label{tab:prompt_summary}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{llp{0.55\linewidth}c}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Config} & \textbf{Prompt} & \textbf{Length} \\")
    lines.append(r"\midrule")
    lines.append(r"Qwen2-Audio-7B & Baseline & \textit{Does this audio contain human speech? Reply with ONLY one word: SPEECH or NON-SPEECH.} & 85 \\")
    lines.append(r"Qwen2-Audio-7B & Base+OPRO & \textit{Listen carefully, is this very short clip human speech or noise? Respond: SPEECH or NON-SPEECH.} & 97 \\")
    lines.append(r"Qwen2-Audio-7B & LoRA+Hand & \textit{Does this audio contain human speech? Reply with ONLY one word: SPEECH or NON-SPEECH.} & 85 \\")
    lines.append(r"Qwen2-Audio-7B & LoRA+OPRO & \textit{Pay attention to this clip, is it human speech? Just answer: SPEECH or NON-SPEECH.} & 89 \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return '\n'.join(lines)


# =============================================================================
# TABLE 5: Psychometric Thresholds
# =============================================================================
def generate_psychometric_thresholds_table(stats):
    """Generate tab_psychometric_thresholds.tex"""

    thresholds = stats.get('psychometric_thresholds', {})

    config_display = {
        'baseline': 'Baseline',
        'base_opro': 'Base+OPRO',
        'lora': 'LoRA+Hand',
        'lora_opro': 'LoRA+OPRO',
        'lora_opro_classic': 'LoRA+OPRO',
        'qwen3_baseline': 'Qwen3 Baseline',
        'qwen3_opro': 'Qwen3+OPRO-LLM',
        'qwen3_opro_template': 'Qwen3+OPRO-Tmpl',
    }

    display_order = ['baseline', 'base_opro', 'lora', 'lora_opro_classic', 'qwen3_baseline', 'qwen3_opro', 'qwen3_opro_template']

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Psychometric thresholds with 95\% bootstrap confidence intervals. Values are reported as point estimate [CI]. Flags indicate when the true threshold lies outside the tested range (censored).}")
    lines.append(r"\label{tab:psychometric_thresholds}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Configuration} & \textbf{DT50 (ms)} & \textbf{DT75 (ms)} & \textbf{DT90 (ms)} & \textbf{SNR75 (dB)} \\")
    lines.append(r"\midrule")

    prev_was_qwen2 = False
    for cfg_key in display_order:
        if cfg_key not in thresholds:
            continue

        # Insert separator between Qwen2 and Qwen3 sections
        if cfg_key.startswith('qwen3') and prev_was_qwen2:
            lines.append(r"\midrule")
        prev_was_qwen2 = not cfg_key.startswith('qwen3')

        cfg_thresh = thresholds[cfg_key]
        config_name = config_display.get(cfg_key, cfg_key)

        # Duration thresholds
        dur_thresh = cfg_thresh.get('duration', {})

        def format_threshold(key, thresh_dict, is_snr=False):
            if key not in thresh_dict:
                return "TBD"
            t = thresh_dict[key]
            point = t.get('point', None)
            censoring = t.get('censoring', 'ok')
            ci = t.get('ci', [point, point] if point else [None, None])

            if point is None:
                return "TBD"

            if censoring == 'below_range':
                if is_snr:
                    return f"$<${int(point)}"
                else:
                    return f"$<${int(point)}"
            elif censoring == 'above_range':
                if is_snr:
                    return f"$>${int(point)}"
                else:
                    return f"$>${int(point)}"
            else:
                if is_snr:
                    return f"{point:.0f} [{ci[0]:.0f}, {ci[1]:.0f}]"
                else:
                    return f"{point:.0f} [{ci[0]:.0f}, {ci[1]:.0f}]"

        dt50 = format_threshold('DT50', dur_thresh)
        dt75 = format_threshold('DT75', dur_thresh)
        dt90 = format_threshold('DT90', dur_thresh)

        # SNR threshold
        snr_thresh = cfg_thresh.get('snr', {})
        snr75 = format_threshold('SNR75', snr_thresh, is_snr=True)

        lines.append(f"{config_name} & {dt50} & {dt75} & {dt90} & {snr75} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"\vspace{0.4em}")
    lines.append(r"\begin{flushleft}")
    lines.append(r"\footnotesize")
    lines.append(r"\textit{Censoring notation:} $<$20 ms indicates the criterion is already met at the most challenging tested duration (true threshold is better than 20 ms); $>$1000 ms indicates the criterion is not met even at the longest tested duration (true threshold is worse than 1000 ms). Similarly, $<$$-$10 dB and $>$+20 dB indicate censoring below and above the tested SNR range, respectively. Numeric values without inequality symbols indicate successful interpolation within the tested range.")
    lines.append(r"\end{flushleft}")
    lines.append(r"\end{table}")

    return '\n'.join(lines)


# =============================================================================
# TABLE 6: Primary Comparisons
# =============================================================================
def generate_primary_comparisons_table(stats):
    """Generate tab_primary_comparisons.tex"""

    comparisons = stats.get('primary_comparisons', {})

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Primary paired comparisons on the evaluation set. $\Delta$BA $=$ BA(A)$-$BA(B).")
    lines.append(r"Discordant counts are from McNemar contingency tables: $n_{01}$ counts clips where A is")
    lines.append(r"correct and B is incorrect; $n_{10}$ counts clips where A is incorrect and B is correct.}")
    lines.append(r"\label{tab:primary_comparisons}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{lccccccc}")
    lines.append(r"\toprule")
    lines.append(r"Comparison (A vs.\ B) & $\Delta$BA & 95\% CI & $p$ (raw) & $p_{\mathrm{Holm}}$ & Disc.\ rate & $n_{01}$ & $n_{10}$ \\")
    lines.append(r"\midrule")

    for label, comp in comparisons.items():
        delta_ba = comp.get('delta_ba', 0)
        delta_ci = comp.get('delta_ba_ci', [delta_ba, delta_ba])
        p_raw = comp.get('p_value_raw', 1.0)
        p_adj = comp.get('p_value_adjusted', 1.0)
        mcnemar = comp.get('mcnemar', {})
        disc_rate = mcnemar.get('discordant_rate', 0)
        n01 = mcnemar.get('n_01', 0)
        n10 = mcnemar.get('n_10', 0)

        delta_str = f"{delta_ba:+.3f}" if delta_ba != 0 else "0.000"
        ci_str = f"[{delta_ci[0]:.3f}, {delta_ci[1]:.3f}]"

        lines.append(f"{label} & {delta_str} & {ci_str} & {format_pvalue(p_raw)} & {format_pvalue(p_adj)} & {disc_rate:.3f} & {n01} & {n10} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    return '\n'.join(lines)


# =============================================================================
# TABLE 7: Multi-Model Comparison (Appendix)
# =============================================================================
def generate_multimodel_comparison_table(stats, raw, metrics_data):
    """Generate tab_prelim_multimodel_comparison.tex"""

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Cross-model comparison on the full 21,340-sample test set.}")
    lines.append(r"\label{tab:prelim_multimodel_comparison}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(r"\begin{tabular}{llll}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Prompt} & \textbf{BA$_{\text{clip}}$} & \textbf{BA$_{\text{conditions}}$} \\")
    lines.append(r"\midrule")

    config_metrics = stats.get('config_metrics', {})

    # Qwen2 configs
    qwen2_rows = [
        ('baseline', 'Qwen2-Audio-7B Base', 'Baseline'),
        ('base_opro', 'Qwen2-Audio-7B Base', 'OPRO'),
        ('lora', 'Qwen2-Audio-7B LoRA', 'Baseline'),
        ('lora_opro_classic', 'Qwen2-Audio-7B LoRA', 'OPRO'),
    ]

    for cfg_key, model_name, prompt_type in qwen2_rows:
        if cfg_key in metrics_data:
            m = metrics_data[cfg_key]
            ba_clip = m.get('ba_clip', 0)
            ba_cond = m.get('ba_conditions', ba_clip)
            ba_clip_pct = f"{ba_clip*100:.1f}\\%"
            ba_cond_pct = f"{ba_cond*100:.1f}\\%"
            lines.append(f"{model_name} & {prompt_type} & {ba_clip_pct} & {ba_cond_pct} \\\\")

    # Qwen3 configs
    qwen3_rows = [
        ('qwen3_baseline', 'Qwen3-Omni-30B (Frozen)', 'Baseline'),
        ('qwen3_opro', 'Qwen3-Omni-30B (Frozen)', 'OPRO'),
    ]

    for cfg_key, model_name, prompt_type in qwen3_rows:
        if cfg_key in metrics_data:
            m = metrics_data[cfg_key]
            ba_clip = m.get('ba_clip', 0)
            ba_cond = m.get('ba_conditions', ba_clip)
            ba_clip_pct = f"{ba_clip*100:.1f}\\%"
            ba_cond_pct = f"{ba_cond*100:.1f}\\%"
            lines.append(f"{model_name} & {prompt_type} & {ba_clip_pct} & {ba_cond_pct} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return '\n'.join(lines)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("Loading data...")
    stats, raw = load_data()
    metrics_data = load_metrics_files()

    print("Generating 7 tables...")

    # Generate all tables
    tab1_overall = generate_overall_performance_table(stats)
    tab2_dimension = generate_dimension_means_table(metrics_data)
    tab3_errors = generate_error_counts_table(raw)
    tab4_prompts = generate_prompt_summary_table()
    tab5_psychometric = generate_psychometric_thresholds_table(stats)
    tab6_comparisons = generate_primary_comparisons_table(stats)
    tab7_multimodel = generate_multimodel_comparison_table(stats, raw, metrics_data)

    # Save individual tables
    TABLES_DIR.mkdir(exist_ok=True)

    tables = [
        ("Tab_R02_OverallPerformance.tex", tab1_overall),
        ("Tab_R04_dimension_means.tex", tab2_dimension),
        ("Tab_R05_ErrorCounts.tex", tab3_errors),
        ("Tab_PromptSummary.tex", tab4_prompts),
        ("tab_psychometric_thresholds.tex", tab5_psychometric),
        ("tab_primary_comparisons.tex", tab6_comparisons),
        ("tab_prelim_multimodel_comparison.tex", tab7_multimodel),
    ]

    for filename, content in tables:
        with open(TABLES_DIR / filename, 'w') as f:
            f.write(content)
        print(f"  ✓ Saved {filename}")

    # Generate consolidated file
    consolidated = []
    consolidated.append("=" * 80)
    consolidated.append("TODAS LAS TABLAS EN LATEX - VALORES ACTUALIZADOS")
    consolidated.append(f"Generado desde: {STATS_FILE}")
    consolidated.append("=" * 80)
    consolidated.append("")

    table_titles = [
        "TABLE 1: Overall Performance (Tab_R02_OverallPerformance.tex)",
        "TABLE 2: Dimension Means (Tab_R04_dimension_means.tex)",
        "TABLE 3: Error Counts (Tab_R05_ErrorCounts.tex)",
        "TABLE 4: Prompt Summary (Tab_PromptSummary.tex)",
        "TABLE 5: Psychometric Thresholds (tab_psychometric_thresholds.tex)",
        "TABLE 6: Primary Comparisons (tab_primary_comparisons.tex)",
        "TABLE 7: Multi-Model Comparison (tab_prelim_multimodel_comparison.tex)",
    ]

    table_contents = [tab1_overall, tab2_dimension, tab3_errors, tab4_prompts, tab5_psychometric, tab6_comparisons, tab7_multimodel]

    for title, content in zip(table_titles, table_contents):
        consolidated.append("=" * 80)
        consolidated.append(title)
        consolidated.append("=" * 80)
        consolidated.append("")
        consolidated.append(content)
        consolidated.append("")

    consolidated.append("=" * 80)
    consolidated.append("FIN")
    consolidated.append("=" * 80)

    with open(OUTPUT_CONSOLIDATED, 'w') as f:
        f.write('\n'.join(consolidated))
    print(f"  ✓ Saved LATEX_TABLES_UPDATED.txt")

    print("\n✓ All 7 tables generated successfully!")


if __name__ == '__main__':
    main()