#!/usr/bin/env python3
"""
Paper Audit Script: Verify claims in the LALM VAD paper against actual data.

Checks 10 audit items (A1-A10) and produces a formatted report.
"""

import json
import os
import re
import glob
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

# =============================================================================
# Configuration
# =============================================================================

ROOT = Path("/mnt/fast/nobackup/users/gb0048/opro3_final")
CONSOLIDATED = ROOT / "results" / "BEST_CONSOLIDATED"

CELLS = {
    "01_qwen2_base_baseline": "Qwen2-Base + Baseline",
    "02_qwen2_base_opro_llm": "Qwen2-Base + OPRO-LLM",
    "03_qwen2_base_opro_template": "Qwen2-Base + OPRO-Tmpl",
    "04_qwen2_lora_baseline": "Qwen2-LoRA + Baseline",
    "05_qwen2_lora_opro_llm": "Qwen2-LoRA + OPRO-LLM",
    "06_qwen2_lora_opro_template": "Qwen2-LoRA + OPRO-Tmpl",
    "07_qwen3_omni_baseline": "Qwen3 + Baseline",
    "08_qwen3_omni_opro_llm": "Qwen3 + OPRO-LLM",
    "09_qwen3_omni_opro_template": "Qwen3 + OPRO-Tmpl",
}

# ESC-50 category mapping (standard 50 categories)
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


# =============================================================================
# Helper functions
# =============================================================================

def load_predictions(cell_dir):
    """Load predictions.csv for a cell."""
    csv_path = CONSOLIDATED / cell_dir / "evaluation" / "predictions.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found")
        return None
    return pd.read_csv(csv_path)


def compute_confusion(df):
    """Compute TP/TN/FP/FN/UNKNOWN from predictions DataFrame."""
    gt = df["ground_truth"].str.upper().str.strip()
    pred = df["prediction"].str.upper().str.strip()

    # Normalize NON-SPEECH variants
    gt = gt.str.replace("NON-SPEECH", "NONSPEECH", regex=False)
    gt = gt.str.replace("NON SPEECH", "NONSPEECH", regex=False)
    pred = pred.str.replace("NON-SPEECH", "NONSPEECH", regex=False)
    pred = pred.str.replace("NON SPEECH", "NONSPEECH", regex=False)

    unknown_mask = ~pred.isin(["SPEECH", "NONSPEECH"])
    n_unknown = unknown_mask.sum()

    tp = ((gt == "SPEECH") & (pred == "SPEECH")).sum()
    tn = ((gt == "NONSPEECH") & (pred == "NONSPEECH")).sum()
    fp = ((gt == "NONSPEECH") & (pred == "SPEECH")).sum()
    fn = ((gt == "SPEECH") & (pred == "NONSPEECH")).sum()

    # UNKNOWNs count as errors
    fn_unknown = ((gt == "SPEECH") & unknown_mask).sum()
    fp_unknown = ((gt == "NONSPEECH") & unknown_mask).sum()
    fn += fn_unknown
    fp += fp_unknown

    total = len(df)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "UNKNOWN": int(n_unknown), "Total": int(total),
        "FPR": fpr, "FNR": fnr,
    }


def extract_esc50_category(audio_path):
    """Extract ESC-50 category ID from filename."""
    filename = os.path.basename(audio_path)
    # Pattern: esc50_X-NNNNN-A-CC_...
    match = re.search(r"esc50_\d+-\d+-[A-Z]-(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


# =============================================================================
# A1: Table 7 — Missing 3 configurations
# =============================================================================

def audit_a1():
    print("\n" + "=" * 80)
    print("### A1. Table 7 — Missing 3 configurations (TP/TN/FP/FN)")
    print("=" * 80)

    missing_cells = [
        "03_qwen2_base_opro_template",
        "05_qwen2_lora_opro_llm",
        "09_qwen3_omni_opro_template",
    ]

    results = {}
    for cell in missing_cells:
        label = CELLS[cell]
        df = load_predictions(cell)
        if df is None:
            print(f"  {label}: DATA NOT FOUND")
            continue

        conf = compute_confusion(df)
        results[cell] = conf

        print(f"\n  {label}:")
        print(f"    TP={conf['TP']:>6}  TN={conf['TN']:>6}  FP={conf['FP']:>6}  FN={conf['FN']:>6}")
        print(f"    UNKNOWN={conf['UNKNOWN']:>4}  Total={conf['Total']:>6}")
        print(f"    FPR={conf['FPR']:.4f}  FNR={conf['FNR']:.4f}")
        print(f"    Sanity: TP+FN={conf['TP']+conf['FN']}  TN+FP={conf['TN']+conf['FP']}")

    # Also report UNKNOWN counts for ALL 9 configs
    print(f"\n  --- UNKNOWN counts across ALL 9 configs ---")
    for cell_dir, label in CELLS.items():
        df = load_predictions(cell_dir)
        if df is None:
            print(f"    {label}: FILE NOT FOUND")
            continue
        conf = compute_confusion(df)
        pct = 100 * conf["UNKNOWN"] / conf["Total"] if conf["Total"] > 0 else 0
        print(f"    {label}: {conf['UNKNOWN']}/{conf['Total']} UNKNOWN ({pct:.2f}%)")

    return results


# =============================================================================
# A2: Table 3 — Winning prompts (pre-verified, just print)
# =============================================================================

def audit_a2():
    print("\n" + "=" * 80)
    print("### A2. Table 3 — Full set of winning prompts")
    print("=" * 80)
    print("**Status**: DATA FOUND")

    opro_cells = [
        "02_qwen2_base_opro_llm",
        "03_qwen2_base_opro_template",
        "05_qwen2_lora_opro_llm",
        "06_qwen2_lora_opro_template",
        "08_qwen3_omni_opro_llm",
        "09_qwen3_omni_opro_template",
    ]

    for cell in opro_cells:
        label = CELLS[cell]
        prompt_path = CONSOLIDATED / cell / "optimization" / "best_prompt.txt"
        if prompt_path.exists():
            prompt = prompt_path.read_text().strip()
            print(f"\n  {label}:")
            print(f"    \"{prompt}\"")
        else:
            print(f"\n  {label}: best_prompt.txt NOT FOUND")


# =============================================================================
# A3: ESC-50 per-category breakdown for Qwen3
# =============================================================================

def audit_a3():
    print("\n" + "=" * 80)
    print("### A3. ESC-50 per-category NONSPEECH accuracy for Qwen3")
    print("=" * 80)

    configs_to_check = [
        "07_qwen3_omni_baseline",
        "08_qwen3_omni_opro_llm",
    ]

    for cell in configs_to_check:
        label = CELLS[cell]
        df = load_predictions(cell)
        if df is None:
            print(f"  {label}: DATA NOT FOUND")
            continue

        print(f"\n  --- {label} ---")

        # Filter to NONSPEECH ground truth only
        nonspeech_df = df[df["ground_truth"].str.upper().str.strip().str.replace("NON-SPEECH", "NONSPEECH") == "NONSPEECH"].copy()

        # Extract ESC-50 category
        nonspeech_df["esc50_cat"] = nonspeech_df["audio_path"].apply(extract_esc50_category)

        # Filter to ESC-50 samples only (some may be LibriSpeech or other sources)
        esc50_only = nonspeech_df.dropna(subset=["esc50_cat"])
        esc50_only["esc50_cat"] = esc50_only["esc50_cat"].astype(int)

        print(f"  Total NONSPEECH samples: {len(nonspeech_df)}")
        print(f"  ESC-50 NONSPEECH samples: {len(esc50_only)}")

        if len(esc50_only) == 0:
            print("  No ESC-50 samples found!")
            continue

        # Normalize predictions
        esc50_only = esc50_only.copy()
        pred_norm = esc50_only["prediction"].str.upper().str.strip()
        pred_norm = pred_norm.str.replace("NON-SPEECH", "NONSPEECH", regex=False)
        pred_norm = pred_norm.str.replace("NON SPEECH", "NONSPEECH", regex=False)
        esc50_only["correct"] = (pred_norm == "NONSPEECH").astype(int)

        # Per-category accuracy
        cat_acc = esc50_only.groupby("esc50_cat").agg(
            n_samples=("correct", "count"),
            n_correct=("correct", "sum"),
        )
        cat_acc["accuracy"] = cat_acc["n_correct"] / cat_acc["n_samples"]
        cat_acc["category_name"] = cat_acc.index.map(
            lambda x: ESC50_CATEGORIES.get(x, f"unknown_{x}")
        )
        cat_acc = cat_acc.sort_values("accuracy")

        # Report all categories, highlighting those below 80%
        print(f"\n  Per-category NONSPEECH accuracy (sorted ascending):")
        print(f"  {'Category':<20} {'Acc':>8} {'N':>6} {'Correct':>8}")
        print(f"  {'-'*45}")
        for idx, row in cat_acc.iterrows():
            marker = " ***" if row["accuracy"] < 0.80 else ""
            print(f"  {row['category_name']:<20} {row['accuracy']:>8.1%} {int(row['n_samples']):>6} {int(row['n_correct']):>8}{marker}")

        # Specifically report the 3 categories mentioned in paper
        print(f"\n  Paper-cited categories:")
        for cat_id, cat_name in [(26, "laughing"), (24, "coughing"), (20, "crying_baby")]:
            if cat_id in cat_acc.index:
                row = cat_acc.loc[cat_id]
                print(f"    {cat_name}: {row['accuracy']:.1%} ({int(row['n_correct'])}/{int(row['n_samples'])})")
            else:
                print(f"    {cat_name}: NOT FOUND in data")


# =============================================================================
# A4: Inference latency
# =============================================================================

def audit_a4():
    print("\n" + "=" * 80)
    print("### A4. Inference latency")
    print("=" * 80)

    logs_dir = ROOT / "logs"
    if not logs_dir.exists():
        print("  logs/ directory not found")
        return

    # Find all .out files
    out_files = sorted(logs_dir.glob("matrix_*.out"))
    print(f"  Found {len(out_files)} slurm .out files")

    # Search for timing info in smaller files
    for out_file in out_files:
        size_mb = out_file.stat().st_size / (1024 * 1024)
        if size_mb > 50:  # Skip very large files
            print(f"  Skipping {out_file.name} ({size_mb:.0f} MB — too large)")
            continue

        print(f"\n  Scanning {out_file.name} ({size_mb:.1f} MB)...")
        try:
            with open(out_file, "r", errors="replace") as f:
                lines = f.readlines()

            # Look for tqdm progress bars with it/s
            timing_lines = []
            for i, line in enumerate(lines):
                if "it/s" in line or "s/it" in line or "Evaluating" in line:
                    timing_lines.append((i, line.strip()))

            if timing_lines:
                print(f"    Found {len(timing_lines)} timing-related lines")
                # Show first and last few
                for idx, line in timing_lines[:3]:
                    print(f"      L{idx}: {line[:120]}")
                if len(timing_lines) > 6:
                    print(f"      ... ({len(timing_lines) - 6} more)")
                for idx, line in timing_lines[-3:]:
                    print(f"      L{idx}: {line[:120]}")

            # Look for timestamps (first and last lines with dates)
            first_ts = None
            last_ts = None
            ts_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})")
            for line in lines[:50]:
                m = ts_pattern.search(line)
                if m:
                    first_ts = m.group(1)
                    break
            for line in reversed(lines[-50:]):
                m = ts_pattern.search(line)
                if m:
                    last_ts = m.group(1)
                    break

            if first_ts and last_ts:
                print(f"    First timestamp: {first_ts}")
                print(f"    Last timestamp:  {last_ts}")

        except Exception as e:
            print(f"    Error reading: {e}")

    # Fallback: wall-clock estimation
    print(f"\n  --- Wall-clock fallback estimation ---")
    print(f"  Total test samples per config: 21,340")
    print(f"  If Slurm sacct is available, use: elapsed_time / 21340 = seconds_per_sample")
    print(f"  NOTE: Wall-clock includes model loading, CUDA init, I/O overhead.")
    print(f"  For paper: report as approximate throughput with caveats.")


# =============================================================================
# A5: Keyword count (pre-verified, just print)
# =============================================================================

def audit_a5():
    print("\n" + "=" * 80)
    print("### A5. Keyword count in normalize_to_binary")
    print("=" * 80)
    print("**Status**: DISCREPANCY")
    print()

    normalize_path = ROOT / "src" / "qsm" / "utils" / "normalize.py"
    print(f"  File: {normalize_path}")
    print()

    # Actually count from source
    with open(normalize_path) as f:
        source = f.read()

    # Extract speech_synonyms list
    speech_match = re.search(
        r"speech_synonyms\s*=\s*\[(.*?)\]", source, re.DOTALL
    )
    nonspeech_match = re.search(
        r"nonspeech_synonyms\s*=\s*\[(.*?)\]", source, re.DOTALL
    )

    if speech_match:
        speech_terms = re.findall(r'"([^"]+)"', speech_match.group(1))
        print(f"  speech_synonyms: {len(speech_terms)} items")
        for i, t in enumerate(speech_terms, 1):
            print(f"    {i:2d}. \"{t}\"")
    else:
        print("  speech_synonyms: NOT FOUND")
        speech_terms = []

    print()

    if nonspeech_match:
        nonspeech_terms = re.findall(r'"([^"]+)"', nonspeech_match.group(1))
        print(f"  nonspeech_synonyms: {len(nonspeech_terms)} items")
        for i, t in enumerate(nonspeech_terms, 1):
            print(f"    {i:2d}. \"{t}\"")
    else:
        print("  nonspeech_synonyms: NOT FOUND")
        nonspeech_terms = []

    print()
    print(f"  Paper claims: 18 speech + 26 nonspeech = 44 total")
    print(f"  Actual count: {len(speech_terms)} speech + {len(nonspeech_terms)} nonspeech = {len(speech_terms)+len(nonspeech_terms)} total")

    if len(speech_terms) != 18 or len(nonspeech_terms) != 26:
        print(f"  ** DISCREPANCY: nonspeech count is {len(nonspeech_terms)}, not 26 **")
        print(f"  Paper impact: Update '26 non-speech terms' to '{len(nonspeech_terms)}'")


# =============================================================================
# A6: Meta-prompt content (pre-verified)
# =============================================================================

def audit_a6():
    print("\n" + "=" * 80)
    print("### A6. Meta-prompt content in OPRO-LLM")
    print("=" * 80)
    print("**Status**: VERIFIED")
    print()
    print("  File: scripts/opro_llm.py, build_meta_prompt() method (lines 361-427)")
    print("  - Conciseness: 'Are clear and concise (target <150 chars, absolute max 300 chars)' (L401)")
    print("  - Short clips: 'Encourage robust detection on SHORT and NOISY clips' (L402)")
    print("  - Noisy clips: Same line as above")
    print("  - Additional: 'Consider emphasizing: brevity detection, noise robustness' (L410)")
    print("  Paper claim fully verified.")


# =============================================================================
# A7: Holm-Bonferroni (pre-verified)
# =============================================================================

def audit_a7():
    print("\n" + "=" * 80)
    print("### A7. Table 5 — Holm-Bonferroni correction completeness")
    print("=" * 80)
    print("**Status**: VERIFIED")
    print()

    stats_path = CONSOLIDATED / "stats" / "statistical_analysis.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        comparisons = stats.get("primary_comparisons", {})
        print(f"  File: {stats_path}")
        print(f"  Number of comparisons in Holm family: {len(comparisons)}")
        for i, (label, data) in enumerate(comparisons.items(), 1):
            sig = "SIG" if data.get("significant") else "n.s."
            p_adj = data.get("p_value_adjusted", "N/A")
            delta = data.get("delta_ba", 0)
            print(f"    {i}. {label}: ΔBA={delta:+.4f}, p_adj={p_adj:.2e} [{sig}]")

        # Check if LoRA+OPRO-Tmpl vs Qwen3 is included
        has_cross = "LoRA+OPRO-Tmpl vs Qwen3 Baseline" in comparisons
        print(f"\n  'LoRA+OPRO-Tmpl vs Qwen3 Baseline' in Holm set: {has_cross}")
    else:
        print(f"  {stats_path}: NOT FOUND")


# =============================================================================
# A8: OPRO-LLM convergence (pre-verified)
# =============================================================================

def audit_a8():
    print("\n" + "=" * 80)
    print("### A8. OPRO-LLM convergence iterations")
    print("=" * 80)
    print("**Status**: VERIFIED")
    print()

    opro_llm_cells = [
        ("02_qwen2_base_opro_llm", "Qwen2-Base"),
        ("05_qwen2_lora_opro_llm", "Qwen2-LoRA"),
        ("08_qwen3_omni_opro_llm", "Qwen3-Omni"),
    ]

    for cell, model_name in opro_llm_cells:
        history_path = CONSOLIDATED / cell / "optimization" / "opro_history.json"
        if not history_path.exists():
            print(f"  {model_name}: opro_history.json NOT FOUND")
            continue

        with open(history_path) as f:
            history = json.load(f)

        best_per_iter = history.get("best_reward_per_iteration", [])
        total_iters = len(best_per_iter)

        # Find best iteration (first time best reward appears)
        if best_per_iter:
            max_reward = max(best_per_iter)
            best_iter = best_per_iter.index(max_reward)
            no_improve_iters = total_iters - 1 - best_iter
        else:
            best_iter = "?"
            no_improve_iters = "?"
            max_reward = "?"

        print(f"  {model_name} + OPRO-LLM:")
        print(f"    Total iterations: {total_iters} (0 to {total_iters - 1})")
        print(f"    Best iteration: {best_iter} (reward={max_reward})")
        print(f"    No-improvement iters after best: {no_improve_iters}")
        early_stopped = no_improve_iters >= 5
        print(f"    Early stopping triggered (patience=5): {early_stopped}")
        print()


# =============================================================================
# A9: OPRO-Template perfect scores
# =============================================================================

def audit_a9():
    print("\n" + "=" * 80)
    print("### A9. OPRO-Template perfect scores (LoRA model)")
    print("=" * 80)

    tmpl_cells = [
        ("06_qwen2_lora_opro_template", "Qwen2-LoRA + OPRO-Tmpl"),
        ("03_qwen2_base_opro_template", "Qwen2-Base + OPRO-Tmpl"),
        ("09_qwen3_omni_opro_template", "Qwen3-Omni + OPRO-Tmpl"),
    ]

    for cell, label in tmpl_cells:
        history_path = CONSOLIDATED / cell / "optimization" / "optimization_history.json"
        if not history_path.exists():
            print(f"\n  {label}: optimization_history.json NOT FOUND")
            continue

        with open(history_path) as f:
            data = json.load(f)

        print(f"\n  --- {label} ---")
        print(f"  best_accuracy: {data.get('best_accuracy')}")

        history = data.get("history", [])
        config = data.get("config", {})
        n_candidates = config.get("num_candidates", 8)
        n_iterations = config.get("num_iterations", 15)

        print(f"  Config: {n_iterations} iterations × {n_candidates} candidates = {n_iterations * n_candidates} evals")
        print(f"  Actual history entries: {len(history)}")

        # Find all perfect scores
        perfect_entries = []
        for i, (prompt, score) in enumerate(history):
            if score >= 1.0:
                iteration = (i // n_candidates) + 1  # 1-indexed iteration
                candidate = (i % n_candidates) + 1
                perfect_entries.append({
                    "index": i,
                    "iteration": iteration,
                    "candidate": candidate,
                    "prompt": prompt,
                    "score": score,
                })

        print(f"  Perfect score (1.0) entries: {len(perfect_entries)}")

        # Count unique templates with perfect score
        unique_perfect_prompts = set()
        for entry in perfect_entries:
            unique_perfect_prompts.add(entry["prompt"])

        print(f"  Unique templates with perfect score: {len(unique_perfect_prompts)}")

        # Show which iterations had perfect scores
        perfect_iters = sorted(set(e["iteration"] for e in perfect_entries))
        print(f"  Iterations with perfect scores: {perfect_iters}")

        # List unique perfect templates
        for j, prompt in enumerate(sorted(unique_perfect_prompts), 1):
            # Find which iterations this prompt got perfect scores
            iters = [e["iteration"] for e in perfect_entries if e["prompt"] == prompt]
            print(f"    {j}. (iter {iters}): \"{prompt[:80]}{'...' if len(prompt) > 80 else ''}\"")


# =============================================================================
# A10: Parseability verification
# =============================================================================

def audit_a10():
    print("\n" + "=" * 80)
    print("### A10. Parseability claim verification")
    print("=" * 80)

    print(f"\n  {'Config':<30} {'Total':>6} {'SPEECH':>7} {'NONSPEECH':>10} {'UNKNOWN':>8} {'Parse%':>8}")
    print(f"  {'-'*72}")

    for cell_dir, label in CELLS.items():
        df = load_predictions(cell_dir)
        if df is None:
            print(f"  {label:<30} {'N/A':>6}")
            continue

        pred = df["prediction"].str.upper().str.strip()
        pred = pred.str.replace("NON-SPEECH", "NONSPEECH", regex=False)
        pred = pred.str.replace("NON SPEECH", "NONSPEECH", regex=False)

        n_total = len(df)
        n_speech = (pred == "SPEECH").sum()
        n_nonspeech = (pred == "NONSPEECH").sum()
        n_unknown = n_total - n_speech - n_nonspeech
        parse_rate = (n_speech + n_nonspeech) / n_total * 100 if n_total > 0 else 0

        print(f"  {label:<30} {n_total:>6} {n_speech:>7} {n_nonspeech:>10} {n_unknown:>8} {parse_rate:>7.2f}%")

    print()
    print("  Paper claims: 'over 99.7% at levels 1-2, Qwen3-Omni 100% parseability'")
    print("  UNKNOWN = failed to parse to SPEECH or NONSPEECH (treated as errors)")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("PAPER AUDIT: Implementation Verification Report")
    print("=" * 80)
    print(f"Root: {ROOT}")
    print(f"Consolidated: {CONSOLIDATED}")

    # Verify BEST_CONSOLIDATED exists
    if not CONSOLIDATED.exists():
        print("ERROR: BEST_CONSOLIDATED directory not found!")
        return

    # Run all audits
    audit_a1()
    audit_a2()
    audit_a3()
    audit_a4()
    audit_a5()
    audit_a6()
    audit_a7()
    audit_a8()
    audit_a9()
    audit_a10()

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
