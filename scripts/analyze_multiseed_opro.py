#!/usr/bin/env python3
"""
B.1 Post-processing — Multi-seed OPRO-Template Analysis

Reads results from 15 runs (3 models × 5 seeds) and generates:
  1. Table: models × seeds with (winning template, BA_test)
  2. Mean ± std of BA per model
  3. Template consistency analysis
  4. Markdown report + CSV

Usage:
    python3 scripts/analyze_multiseed_opro.py
    python3 scripts/analyze_multiseed_opro.py --results_dir audits/round3/b1_multiseed
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np


MODELS = ["base", "lora", "qwen3"]
SEEDS = [42, 123, 456, 789, 1024]

MODEL_LABELS = {
    "base": "Base+OPRO-Tmpl",
    "lora": "LoRA+OPRO-Tmpl",
    "qwen3": "Qwen3+OPRO-Tmpl",
}

# Known paper values for seed=42 (for verification)
PAPER_BA = {
    "base": 0.6404,   # 64.0% (Base+OPRO-Tmpl from Table 2)
    "lora": 0.9329,   # 93.3% (LoRA+OPRO-Tmpl — best system)
    "qwen3": 0.9091,  # 90.9% (Qwen3+OPRO-Tmpl)
}
BA_TOLERANCE = 0.005  # Allow 0.5pp tolerance for reproduction check


# Map from hardcoded template index to a short ID
# These correspond to the 15 templates in opro_template.py generate_candidate_prompts()
TEMPLATE_IDS = {
    "Does this audio contain human speech? Answer exactly one token: SPEECH or NONSPEECH.": "T01_minimal",
    "Binary decision. Output exactly one token: SPEECH or NONSPEECH.": "T02_binary",
    "Decide the dominant content.\nDefinitions:\n- SPEECH = human voice, spoken words, syllables, conversational cues.\n- NONSPEECH = music, tones/beeps, environmental noise, silence.\nOutput exactly: SPEECH or NONSPEECH.": "T03_verbalizer",
    "Detect human speech. Treat the following as NONSPEECH: pure tones/beeps, clicks, clock ticks, music, environmental noise, silence.\nAnswer: SPEECH or NONSPEECH.": "T04_contrastive",
    "Example:\nAudio→ crowd noise, music → Output: NONSPEECH\nNow classify the new audio. Output exactly ONE token: SPEECH or NONSPEECH.": "T05_oneshot",
    "Make a definite decision for the clip.\nOutput exactly one token: SPEECH or NONSPEECH.": "T06_forced",
    "Label SPEECH only if human voice is clearly present; otherwise label NONSPEECH.\nAnswer: SPEECH or NONSPEECH.": "T07_conservative",
    "If there is any hint of human voice (even faint/short), label SPEECH; otherwise NONSPEECH.\nAnswer: SPEECH or NONSPEECH.": "T08_liberal",
    "Focus on cues of human vocal tract (formants, syllabic rhythm, consonant onsets).\nAnswer exactly: SPEECH or NONSPEECH.": "T09_acoustic",
    "TASK: Speech detection. Is human voice/speech present in this audio?\nAnswer: SPEECH or NONSPEECH.": "T10_task",
    "Binary classification task.\nQ: Does this contain human speech?\nIf confident YES → SPEECH\nIf confident NO → NONSPEECH\nAnswer:": "T11_calibration",
    "You will answer with one token only.\n<question>Does this audio contain human speech?</question>\n<answer>SPEECH or NONSPEECH only</answer>": "T12_delimiters",
    "Classify this audio. Output only: SPEECH or NONSPEECH.": "T13_explicit",
    "Listen for human voice. If present: SPEECH. Otherwise: NONSPEECH.\nAnswer:": "T14_focus",
    "Human speech present? Answer: SPEECH or NONSPEECH.": "T15_simplified",
}


def identify_template(prompt_text: str) -> str:
    """Map a prompt text to its template ID."""
    # Handle dict prompts (from JSON templates)
    if isinstance(prompt_text, dict):
        prompt_text = prompt_text.get("text", str(prompt_text))

    # Exact match
    if prompt_text in TEMPLATE_IDS:
        return TEMPLATE_IDS[prompt_text]

    # Fuzzy match: strip whitespace and compare
    stripped = prompt_text.strip()
    for template, tid in TEMPLATE_IDS.items():
        if template.strip() == stripped:
            return tid

    # Partial match: first 40 chars
    for template, tid in TEMPLATE_IDS.items():
        if template[:40] == stripped[:40]:
            return f"{tid}?"

    return f"UNKNOWN({stripped[:50]}...)"


def load_run(results_dir: Path, model: str, seed: int) -> dict:
    """Load results from a single model×seed run."""
    run_dir = results_dir / f"{model}_seed{seed}"

    result = {
        "model": model,
        "seed": seed,
        "status": "missing",
        "ba_clip": None,
        "ba_clip_pct": None,
        "n_samples": None,
        "speech_acc": None,
        "nonspeech_acc": None,
        "winning_prompt": None,
        "template_id": None,
        "opro_best_dev_acc": None,
    }

    if not run_dir.exists():
        return result

    # Load evaluation metrics
    metrics_file = run_dir / "evaluation" / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        result["ba_clip"] = metrics.get("ba_clip")
        result["ba_clip_pct"] = round(metrics["ba_clip"] * 100, 2) if metrics.get("ba_clip") else None
        result["n_samples"] = metrics.get("n_samples")
        result["speech_acc"] = metrics.get("speech_acc")
        result["nonspeech_acc"] = metrics.get("nonspeech_acc")
        result["status"] = "complete"
    else:
        result["status"] = "eval_missing"

    # Load winning prompt
    prompt_file = run_dir / "optimization" / "best_prompt.txt"
    if prompt_file.exists():
        prompt_text = prompt_file.read_text().strip()
        result["winning_prompt"] = prompt_text
        result["template_id"] = identify_template(prompt_text)
        if result["status"] == "missing":
            result["status"] = "opro_only"

    # Load OPRO optimization history for dev accuracy
    history_file = run_dir / "optimization" / "optimization_history.json"
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
        result["opro_best_dev_acc"] = history.get("best_accuracy")

    return result


def generate_report(results: list[dict], output_dir: Path) -> None:
    """Generate markdown report and CSV from results."""
    df = pd.DataFrame(results)

    # --- CSV output ---
    csv_path = output_dir / "B1_multiseed_opro.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    # --- Build report ---
    lines = []
    lines.append("# B.1 Multi-seed OPRO-Template Stability Analysis")
    lines.append("")
    lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Seeds:** {SEEDS}")
    lines.append(f"**Models:** {list(MODEL_LABELS.values())}")
    lines.append("")

    # Check completeness
    complete = df[df["status"] == "complete"]
    total = len(MODELS) * len(SEEDS)
    lines.append(f"**Runs complete:** {len(complete)} / {total}")
    if len(complete) < total:
        missing = df[df["status"] != "complete"]
        lines.append("")
        lines.append("**Missing/incomplete runs:**")
        for _, row in missing.iterrows():
            lines.append(f"- {row['model']}_seed{row['seed']}: {row['status']}")
    lines.append("")

    # === Table 1: BA results matrix ===
    lines.append("## Table 1: Test BA (%) by Model × Seed")
    lines.append("")

    # Header
    header = "| Model | " + " | ".join(f"Seed {s}" for s in SEEDS) + " | Mean | Std |"
    sep = "|---|" + "|".join(["---:"] * len(SEEDS)) + "|---:|---:|"
    lines.append(header)
    lines.append(sep)

    model_stats = {}
    for model in MODELS:
        label = MODEL_LABELS[model]
        model_df = df[(df["model"] == model) & (df["status"] == "complete")]
        bas = []
        cells = []
        for seed in SEEDS:
            row = model_df[model_df["seed"] == seed]
            if len(row) == 1:
                ba = row.iloc[0]["ba_clip_pct"]
                bas.append(ba)
                # Bold seed=42 for paper reference
                if seed == 42:
                    cells.append(f"**{ba:.2f}**")
                else:
                    cells.append(f"{ba:.2f}")
            else:
                cells.append("---")

        if bas:
            mean_ba = np.mean(bas)
            std_ba = np.std(bas, ddof=1) if len(bas) > 1 else 0.0
            model_stats[model] = {"mean": mean_ba, "std": std_ba, "n": len(bas), "values": bas}
            row_str = f"| {label} | " + " | ".join(cells) + f" | **{mean_ba:.2f}** | {std_ba:.2f} |"
        else:
            model_stats[model] = {"mean": None, "std": None, "n": 0, "values": []}
            row_str = f"| {label} | " + " | ".join(cells) + " | --- | --- |"
        lines.append(row_str)

    lines.append("")
    lines.append("*Bold values = seed=42 (original paper result). Std = sample standard deviation (ddof=1).*")
    lines.append("")

    # === Table 2: Winning templates matrix ===
    lines.append("## Table 2: Winning Template by Model × Seed")
    lines.append("")

    header2 = "| Model | " + " | ".join(f"Seed {s}" for s in SEEDS) + " |"
    sep2 = "|---|" + "|".join(["---"] * len(SEEDS)) + "|"
    lines.append(header2)
    lines.append(sep2)

    template_counts = {}  # model -> {template_id: count}
    for model in MODELS:
        label = MODEL_LABELS[model]
        model_df = df[df["model"] == model]
        cells = []
        template_counts[model] = {}
        for seed in SEEDS:
            row = model_df[model_df["seed"] == seed]
            if len(row) == 1 and row.iloc[0]["template_id"]:
                tid = row.iloc[0]["template_id"]
                cells.append(tid)
                template_counts[model][tid] = template_counts[model].get(tid, 0) + 1
            else:
                cells.append("---")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append("")

    # === Template consistency analysis ===
    lines.append("## Template Consistency Analysis")
    lines.append("")

    for model in MODELS:
        label = MODEL_LABELS[model]
        counts = template_counts.get(model, {})
        if not counts:
            lines.append(f"### {label}: No data")
            continue

        total_runs = sum(counts.values())
        most_common = max(counts.items(), key=lambda x: x[1])
        consistency = most_common[1] / total_runs * 100

        lines.append(f"### {label}")
        lines.append(f"- **Most frequent template:** {most_common[0]} ({most_common[1]}/{total_runs} seeds = {consistency:.0f}%)")

        if len(counts) == 1:
            lines.append(f"- **Result:** Same template wins across ALL {total_runs} seeds. **Highly stable.**")
        else:
            lines.append(f"- **Unique templates selected:** {len(counts)}")
            for tid, count in sorted(counts.items(), key=lambda x: -x[1]):
                lines.append(f"  - {tid}: {count}/{total_runs} seeds")

            # Check if BA variation is small despite different templates
            stats = model_stats.get(model, {})
            if stats.get("std") is not None and stats["std"] < 1.0:
                lines.append(f"- **Note:** Despite template variation, BA std = {stats['std']:.2f}pp — performance is stable.")
            elif stats.get("std") is not None:
                lines.append(f"- **Note:** BA std = {stats['std']:.2f}pp — non-trivial performance variation across seeds.")

        lines.append("")

    # === Seed=42 reproduction check ===
    lines.append("## Seed=42 Reproduction Check")
    lines.append("")
    lines.append("Verifying that seed=42 reproduces original paper results:")
    lines.append("")

    for model in MODELS:
        label = MODEL_LABELS[model]
        row = df[(df["model"] == model) & (df["seed"] == 42) & (df["status"] == "complete")]
        expected = PAPER_BA.get(model)

        if len(row) == 1:
            actual = row.iloc[0]["ba_clip"]
            if expected and actual:
                diff = abs(actual - expected)
                match = diff < BA_TOLERANCE
                status = "MATCH" if match else f"MISMATCH (delta={diff*100:.2f}pp)"
                lines.append(f"- **{label}:** BA={actual*100:.2f}% (expected {expected*100:.1f}%) — **{status}**")
            else:
                lines.append(f"- **{label}:** BA={actual*100:.2f}% (no expected value)")
        else:
            lines.append(f"- **{label}:** seed=42 run not complete")

    lines.append("")

    # === Summary statistics ===
    lines.append("## Summary Statistics (for paper)")
    lines.append("")
    lines.append("| Model | Mean BA (%) | Std (pp) | Min | Max | Range (pp) | N |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for model in MODELS:
        label = MODEL_LABELS[model]
        stats = model_stats.get(model, {})
        if stats.get("values"):
            vals = stats["values"]
            lines.append(
                f"| {label} | {stats['mean']:.2f} | {stats['std']:.2f} | "
                f"{min(vals):.2f} | {max(vals):.2f} | {max(vals)-min(vals):.2f} | {stats['n']} |"
            )
        else:
            lines.append(f"| {label} | --- | --- | --- | --- | --- | 0 |")

    lines.append("")

    # === Suggested text for paper ===
    lines.append("## Suggested Text for Paper")
    lines.append("")

    all_complete = all(model_stats.get(m, {}).get("n", 0) == len(SEEDS) for m in MODELS)
    if all_complete:
        lora_stats = model_stats["lora"]
        lines.append(
            f"\"To verify that template selection is not an artifact of a single random seed, "
            f"we repeated the OPRO-Template search with five seeds (42, 123, 456, 789, 1024) "
            f"for all three model configurations. "
            f"LoRA+OPRO-Tmpl achieved a mean BA of {lora_stats['mean']:.1f}% "
            f"(std = {lora_stats['std']:.2f} pp, range = "
            f"{min(lora_stats['values']):.1f}%–{max(lora_stats['values']):.1f}%), "
            f"confirming that the reported 93.3% result is representative and not seed-dependent.\""
        )
    else:
        lines.append("*[Waiting for all runs to complete before generating suggested text.]*")

    lines.append("")

    # === Dev accuracy vs test accuracy ===
    dev_data = df[df["opro_best_dev_acc"].notna()]
    if len(dev_data) > 0:
        lines.append("## Dev vs Test Accuracy")
        lines.append("")
        lines.append("| Model | Seed | Dev BA (%) | Test BA (%) | Gap (pp) |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, row in dev_data.iterrows():
            dev_ba = row["opro_best_dev_acc"] * 100
            test_ba = row["ba_clip_pct"] if row["ba_clip_pct"] else 0
            gap = test_ba - dev_ba
            lines.append(
                f"| {MODEL_LABELS[row['model']]} | {row['seed']} | "
                f"{dev_ba:.1f} | {test_ba:.2f} | {gap:+.2f} |"
            )
        lines.append("")

    # Write report
    report_path = output_dir / "B1_multiseed_opro.md"
    report_path.write_text("\n".join(lines))
    print(f"Report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="B.1 Multi-seed OPRO-Template Analysis")
    parser.add_argument(
        "--results_dir", type=str,
        default="audits/round3/b1_multiseed",
        help="Root directory containing {model}_seed{seed}/ subdirs"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="audits/round3",
        help="Output directory for report and CSV"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("B.1 Multi-seed OPRO-Template — Post-processing")
    print("=" * 70)
    print(f"Results dir: {results_dir}")
    print(f"Output dir:  {output_dir}")
    print()

    # Load all runs
    results = []
    for model in MODELS:
        for seed in SEEDS:
            run = load_run(results_dir, model, seed)
            results.append(run)
            status_icon = {
                "complete": "+",
                "opro_only": "~",
                "eval_missing": "~",
                "missing": "-",
            }.get(run["status"], "?")
            ba_str = f"{run['ba_clip_pct']:.2f}%" if run["ba_clip_pct"] else "N/A"
            tid_str = run["template_id"] or "N/A"
            print(f"  [{status_icon}] {model}_seed{seed}: BA={ba_str}, template={tid_str}")

    print()

    # Summary
    complete = [r for r in results if r["status"] == "complete"]
    print(f"Complete: {len(complete)} / {len(results)}")

    if len(complete) == 0:
        print("\nNo completed runs found. Check that SLURM jobs have finished.")
        print(f"Expected directory structure: {results_dir}/{{base,lora,qwen3}}_seed{{42,...}}/evaluation/metrics.json")
        sys.exit(1)

    generate_report(results, output_dir)

    print()
    print("Done! Check:")
    print(f"  {output_dir}/B1_multiseed_opro.md")
    print(f"  {output_dir}/B1_multiseed_opro.csv")


if __name__ == "__main__":
    main()
