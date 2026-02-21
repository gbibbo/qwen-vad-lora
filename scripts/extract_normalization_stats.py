#!/usr/bin/env python3
"""
B.2 — Normalization Pathway Stats

Extracts how many responses resolved at each normalization level.

Two data sources:
1. Test set predictions.csv (9 configs × 21,340 samples) — only final labels,
   so we can only determine parse rate (SPEECH/NONSPEECH vs UNKNOWN).
2. OPRO-Template optimization iter*.csv files — contain raw_text, so we can
   run the normalization hierarchy with level tracking.

Output: audits/round1/B2_normalization_stats.md
"""

import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

ROOT = Path("/mnt/fast/nobackup/users/gb0048/opro3_final")
CONSOLIDATED = ROOT / "results" / "BEST_CONSOLIDATED"
AUDITS_DIR = ROOT / "audits" / "round1"
AUDITS_DIR.mkdir(parents=True, exist_ok=True)

# Add src to path for normalize import
sys.path.insert(0, str(ROOT))

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


def normalize_with_level(text):
    """
    Re-implements normalize_to_binary with level tracking.
    Returns (label, level_name) where level_name is one of:
      L1_NONSPEECH, L2_SPEECH, L3_LETTER, L4_YESNO, L5_KEYWORDS, L6_UNKNOWN
    """
    if not text:
        return None, "L6_UNKNOWN"

    text_clean = text.strip().upper()
    text_lower = text.strip().lower()

    # Level 1: NONSPEECH substring
    if ("NONSPEECH" in text_clean or "NON-SPEECH" in text_clean
            or "NON SPEECH" in text_clean or "NO SPEECH" in text_clean):
        if "NOT NONSPEECH" not in text_clean and "NOT NON-SPEECH" not in text_clean:
            return "NONSPEECH", "L1_NONSPEECH"

    # Level 2: SPEECH substring
    if "SPEECH" in text_clean:
        if "NOT SPEECH" not in text_clean:
            return "SPEECH", "L2_SPEECH"

    # Level 3: Letter mapping (A/B/C/D) — we don't have the mapping here,
    # so check for leading letter patterns
    letter_match = re.match(r"^([A-D])[\s\)\.]", text_clean)
    if letter_match:
        return f"LETTER_{letter_match.group(1)}", "L3_LETTER"

    # Level 4: YES/NO
    yes_patterns = ["YES", "SÍ", "SI", "AFFIRMATIVE", "TRUE", "CORRECT", "PRESENT"]
    no_patterns = ["NO", "NEGATIVE", "FALSE", "INCORRECT", "ABSENT", "NOT PRESENT"]

    for pattern in yes_patterns:
        if re.search(r'\b' + re.escape(pattern) + r'\b', text_clean):
            return "SPEECH", "L4_YESNO"

    for pattern in no_patterns:
        if re.search(r'\b' + re.escape(pattern) + r'\b', text_clean):
            return "NONSPEECH", "L4_YESNO"

    # Level 5: Keywords
    speech_synonyms = [
        "voice", "voices", "talking", "spoken", "speaking", "speaker",
        "conversation", "conversational", "words", "utterance", "vocal",
        "human voice", "person talking", "dialogue", "speech", "syllables",
        "phonemes", "formants",
    ]
    nonspeech_synonyms = [
        "music", "musical", "song", "melody", "instrumental", "beep", "beeps",
        "tone", "tones", "pitch", "sine wave", "noise", "noisy", "static",
        "hiss", "white noise", "silence", "silent", "quiet", "nothing", "empty",
        "ambient", "environmental", "background", "click", "clicks", "clock",
        "tick", "ticking",
    ]

    speech_score = sum(1 for syn in speech_synonyms if syn in text_lower)
    nonspeech_score = sum(1 for syn in nonspeech_synonyms if syn in text_lower)

    if speech_score > nonspeech_score:
        return "SPEECH", "L5_KEYWORDS"
    elif nonspeech_score > speech_score:
        return "NONSPEECH", "L5_KEYWORDS"

    # Level 6: Unknown
    return None, "L6_UNKNOWN"


# =============================================================================
# Part 1: Test set parse rates (from predictions.csv)
# =============================================================================

def analyze_test_set_parse_rates():
    """Parse rates from the final test set predictions."""
    results = {}

    for cell_dir, label in CELLS.items():
        csv_path = CONSOLIDATED / cell_dir / "evaluation" / "predictions.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        pred = df["prediction"].str.upper().str.strip()
        pred = pred.str.replace("NON-SPEECH", "NONSPEECH", regex=False)
        pred = pred.str.replace("NON SPEECH", "NONSPEECH", regex=False)

        n_total = len(df)
        n_speech = (pred == "SPEECH").sum()
        n_nonspeech = (pred == "NONSPEECH").sum()
        n_unknown = n_total - n_speech - n_nonspeech
        parse_rate = (n_speech + n_nonspeech) / n_total * 100

        results[label] = {
            "total": n_total,
            "speech": int(n_speech),
            "nonspeech": int(n_nonspeech),
            "unknown": int(n_unknown),
            "parse_rate": parse_rate,
            "levels_1_2_pct": parse_rate,  # Best we can say: parsed = levels 1-2+ (at least)
        }

    return results


# =============================================================================
# Part 2: Detailed level breakdown from OPRO-Template iter*.csv files
# =============================================================================

def analyze_opro_template_iterations():
    """Detailed level breakdown from OPRO-Template optimization iterations."""
    template_cells = {
        "03_qwen2_base_opro_template": "Base+OPRO-Tmpl",
        "06_qwen2_lora_opro_template": "LoRA+OPRO-Tmpl",
        "09_qwen3_omni_opro_template": "Qwen3+OPRO-Tmpl",
    }

    all_results = {}

    for cell_dir, label in template_cells.items():
        opt_dir = CONSOLIDATED / cell_dir / "optimization"
        if not opt_dir.exists():
            continue

        level_counts = defaultdict(int)
        total_samples = 0

        # Process all iter*.csv files
        iter_files = sorted(opt_dir.glob("iter*_all_predictions.csv"))
        for iter_file in iter_files:
            try:
                df = pd.read_csv(iter_file)
            except Exception as e:
                print(f"  Error reading {iter_file}: {e}")
                continue

            if "raw_text" not in df.columns:
                print(f"  WARNING: {iter_file.name} has no raw_text column")
                continue

            for _, row in df.iterrows():
                raw_text = str(row.get("raw_text", ""))
                _, level = normalize_with_level(raw_text)
                level_counts[level] += 1
                total_samples += 1

        if total_samples > 0:
            all_results[label] = {
                "total": total_samples,
                "levels": dict(level_counts),
                "level_pcts": {k: v / total_samples * 100 for k, v in level_counts.items()},
            }

    return all_results


# =============================================================================
# Generate report
# =============================================================================

def generate_report(test_results, template_results):
    lines = []
    lines.append("# B.2 — Normalization Pathway Stats")
    lines.append(f"**Date:** 2026-02-17")
    lines.append("")

    # ---- Table 1: Test set parse rates ----
    lines.append("## Table 1: Test Set Parse Rates (21,340 samples per config)")
    lines.append("")
    lines.append("These are from the FINAL test evaluation `predictions.csv`. Since raw model")
    lines.append("outputs were not saved, we can only determine whether a response was")
    lines.append("successfully parsed (SPEECH or NONSPEECH) vs unparseable (UNKNOWN).")
    lines.append("Successfully parsed responses went through levels 1-5; we cannot distinguish")
    lines.append("which specific level resolved them.")
    lines.append("")
    lines.append("| Configuration | Total | SPEECH | NONSPEECH | UNKNOWN | Parse Rate |")
    lines.append("|---------------|-------|--------|-----------|---------|------------|")

    for label, data in test_results.items():
        lines.append(
            f"| {label} | {data['total']:,} | {data['speech']:,} | "
            f"{data['nonspeech']:,} | {data['unknown']} | "
            f"{data['parse_rate']:.2f}% |"
        )

    lines.append("")

    # Verify paper claims
    lines.append("### Verification of Paper Claims (Section 3.3)")
    lines.append("")

    # Check ">99.7% at levels 1-2 for 7 of 9 configs"
    above_997 = [label for label, d in test_results.items() if d["parse_rate"] >= 99.7]
    below_997 = [label for label, d in test_results.items() if d["parse_rate"] < 99.7]

    lines.append(f"- **Paper claim:** \">99.7% of responses resolved at levels 1-2 for 7 of 9 configs\"")
    lines.append(f"- **Configs with >=99.7% parse rate:** {len(above_997)}/9 — {', '.join(above_997)}")
    if below_997:
        lines.append(f"- **Configs below 99.7%:** {', '.join(below_997)}")
        for label in below_997:
            d = test_results[label]
            lines.append(f"  - {label}: {d['unknown']} unparseable ({100 - d['parse_rate']:.2f}%)")
    lines.append("")

    # Check "Qwen3-Omni 100% parseability"
    qwen3_configs = [l for l in test_results if l.startswith("Qwen3")]
    qwen3_100 = all(test_results[l]["unknown"] == 0 for l in qwen3_configs)
    lines.append(f"- **Paper claim:** \"Qwen3-Omni achieving 100% parseability across all three prompting conditions\"")
    lines.append(f"- **Verified:** {'YES' if qwen3_100 else 'NO'} — " +
                 ", ".join(f"{l}: {test_results[l]['unknown']} unknown" for l in qwen3_configs))
    lines.append("")

    # Check "Base+OPRO-LLM: 320 of 21,340 (1.5%) required lower-level normalization"
    base_opro = test_results.get("Base+OPRO-LLM", {})
    if base_opro:
        lines.append(f"- **Paper claim:** \"Base + OPRO-LLM, where 320 of 21,340 responses (1.5%) required lower-level normalization\"")
        lines.append(f"- **UNKNOWN count from predictions.csv:** {base_opro['unknown']}")
        lines.append(f"- **Note:** The paper's 320 figure refers to responses that needed levels 3-6 "
                     f"(but were still resolved), NOT to UNKNOWN/unparseable responses. "
                     f"We cannot verify this exact number without raw model outputs.")
    lines.append("")

    # ---- Table 2: OPRO-Template level breakdown ----
    lines.append("## Table 2: Detailed Level Breakdown (OPRO-Template Optimization, Dev Set)")
    lines.append("")
    lines.append("These statistics come from the `iter*_all_predictions.csv` files generated")
    lines.append("during OPRO-Template optimization. Unlike the test set, these files contain")
    lines.append("the `raw_text` column, allowing us to trace each response through the")
    lines.append("normalization hierarchy.")
    lines.append("")
    lines.append("**Note:** This is dev set data during optimization (not the final test set),")
    lines.append("and covers multiple prompts per iteration. It provides a representative")
    lines.append("distribution of normalization levels but is not directly comparable to the")
    lines.append("paper's test-set claims.")
    lines.append("")

    level_order = ["L1_NONSPEECH", "L2_SPEECH", "L3_LETTER", "L4_YESNO", "L5_KEYWORDS", "L6_UNKNOWN"]
    level_names = {
        "L1_NONSPEECH": "L1: NONSPEECH substr",
        "L2_SPEECH": "L2: SPEECH substr",
        "L3_LETTER": "L3: Letter mapping",
        "L4_YESNO": "L4: YES/NO",
        "L5_KEYWORDS": "L5: Keywords",
        "L6_UNKNOWN": "L6: Unknown/fallback",
    }

    for label, data in template_results.items():
        lines.append(f"### {label}")
        lines.append(f"Total responses analyzed: {data['total']:,}")
        lines.append("")
        lines.append("| Level | Count | Percentage |")
        lines.append("|-------|-------|------------|")

        for level in level_order:
            count = data["levels"].get(level, 0)
            pct = data["level_pcts"].get(level, 0.0)
            name = level_names.get(level, level)
            lines.append(f"| {name} | {count:,} | {pct:.2f}% |")

        l1_l2 = data["levels"].get("L1_NONSPEECH", 0) + data["levels"].get("L2_SPEECH", 0)
        l1_l2_pct = l1_l2 / data["total"] * 100 if data["total"] > 0 else 0
        lines.append(f"| **L1+L2 combined** | **{l1_l2:,}** | **{l1_l2_pct:.2f}%** |")
        lines.append("")

    # ---- What's missing ----
    lines.append("## What Is Missing & How to Complete")
    lines.append("")
    lines.append("### Missing Data")
    lines.append("The final test evaluation (`scripts/eval.py`) saves only the normalized label")
    lines.append("in `predictions.csv`, not the raw model output. Therefore:")
    lines.append("")
    lines.append("- We **can** report parse rates (SPEECH/NONSPEECH/UNKNOWN) for all 9 configs")
    lines.append("- We **cannot** determine which normalization level (1-5) resolved each response")
    lines.append("- The paper's claim \"320 responses required lower-level normalization\" for")
    lines.append("  Base+OPRO-LLM cannot be verified from saved data")
    lines.append("")
    lines.append("### How to Get Complete Data (Ronda 2)")
    lines.append("Modify `scripts/eval.py` to save `raw_text` in predictions.csv:")
    lines.append("")
    lines.append("```python")
    lines.append("# In eval.py evaluate_samples(), add raw_text to results dict (line ~119):")
    lines.append("results.append({")
    lines.append('    "audio_path": audio_path,')
    lines.append('    "ground_truth": ground_truth,')
    lines.append('    "raw_text": response,          # <-- ADD THIS')
    lines.append('    "prediction": prediction,')
    lines.append('    "condition": condition_key,')
    lines.append('    "variant_type": row.get("variant_type", "unknown"),')
    lines.append("})")
    lines.append("```")
    lines.append("")
    lines.append("Then re-run evaluation for all 9 configs (requires GPU). After that,")
    lines.append("re-run this script to get the complete level breakdown for the test set.")
    lines.append("")
    lines.append("**Estimated GPU time:** ~2-3 hours per config × 9 configs on A100.")

    report = "\n".join(lines)
    output_path = AUDITS_DIR / "B2_normalization_stats.md"
    output_path.write_text(report)
    print(f"Report written to: {output_path}")
    return report


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("B.2 — Normalization Pathway Stats")
    print("=" * 60)

    print("\nPart 1: Test set parse rates...")
    test_results = analyze_test_set_parse_rates()

    print("\nPart 2: OPRO-Template level breakdown...")
    template_results = analyze_opro_template_iterations()

    print("\nGenerating report...")
    report = generate_report(test_results, template_results)

    print("\n" + report)
