#!/usr/bin/env python3
"""Create nested stratified subsets for LoRA data-curve ablation.

Produces three training CSVs (256, 512, 1024 clips) that are strict nested
subsets of each other, maintaining 50/50 class balance.  Also writes the
OPRO-Template prompt file used by the evaluation Slurm job.

Usage:
    python3 scripts/create_data_curve_splits.py
    python3 scripts/create_data_curve_splits.py --output_dir audits/round3/data_curve/splits
"""

import argparse
import json
from pathlib import Path

import pandas as pd

SEED = 42
SUBSET_SIZES = [256, 512, 1024]
TRAIN_CSV = "data/processed/experimental_variants/train_metadata.csv"
DEFAULT_OUTPUT_DIR = "audits/round3/data_curve/splits"

OPRO_TMPL_PROMPT = (
    "Detect human speech. Treat the following as NONSPEECH: "
    "pure tones/beeps, clicks, clock ticks, music, environmental noise, silence.\n"
    "Answer: SPEECH or NONSPEECH."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_csv", default=TRAIN_CSV,
                        help="Full training manifest (default: %(default)s)")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Where to write subset CSVs (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for shuffling (default: %(default)s)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load and validate source data
    # ------------------------------------------------------------------
    df = pd.read_csv(args.train_csv)
    speech = df[df["ground_truth"] == "SPEECH"]
    nonspeech = df[df["ground_truth"] == "NONSPEECH"]

    print(f"Source: {args.train_csv}")
    print(f"  Total rows  : {len(df)}")
    print(f"  SPEECH       : {len(speech)}")
    print(f"  NONSPEECH    : {len(nonspeech)}")
    assert len(speech) == 1536, f"Expected 1536 SPEECH rows, got {len(speech)}"
    assert len(nonspeech) == 1536, f"Expected 1536 NONSPEECH rows, got {len(nonspeech)}"

    # ------------------------------------------------------------------
    # Shuffle each class independently (fixed seed)
    # ------------------------------------------------------------------
    speech = speech.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    nonspeech = nonspeech.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Create nested subsets via prefix slicing
    # ------------------------------------------------------------------
    summary = {"seed": args.seed, "source": args.train_csv, "subsets": {}}
    prev_ids = set()

    for n in SUBSET_SIZES:
        per_class = n // 2
        subset = pd.concat([
            speech.iloc[:per_class],
            nonspeech.iloc[:per_class],
        ], ignore_index=True)
        # Shuffle combined for training order
        subset = subset.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

        fname = f"train_n{n:04d}.csv"
        subset.to_csv(output_dir / fname, index=False)

        current_ids = set(subset["variant_id"])

        # Verify nestedness
        if prev_ids:
            assert prev_ids.issubset(current_ids), (
                f"Nestedness violation: previous {len(prev_ids)}-set "
                f"is not a subset of current {n}-set"
            )

        summary["subsets"][str(n)] = {
            "file": fname,
            "total": len(subset),
            "speech": int((subset["ground_truth"] == "SPEECH").sum()),
            "nonspeech": int((subset["ground_truth"] == "NONSPEECH").sum()),
            "unique_clip_ids": int(subset["clip_id"].nunique()),
            "unique_durations": sorted(subset["duration_ms"].unique().tolist()),
            "unique_snrs": sorted(subset["snr_db"].unique().tolist()),
        }

        print(f"\n  {fname}:")
        print(f"    Rows: {len(subset)} (SPEECH={summary['subsets'][str(n)]['speech']}, "
              f"NONSPEECH={summary['subsets'][str(n)]['nonspeech']})")
        print(f"    Unique clip_ids: {summary['subsets'][str(n)]['unique_clip_ids']}")
        print(f"    Durations covered: {len(summary['subsets'][str(n)]['unique_durations'])}/8")
        print(f"    SNRs covered: {len(summary['subsets'][str(n)]['unique_snrs'])}")
        if prev_ids:
            print(f"    Nested in previous: YES ({len(prev_ids)} ids verified)")

        prev_ids = current_ids

    # ------------------------------------------------------------------
    # Write OPRO-Template prompt file
    # ------------------------------------------------------------------
    prompt_path = output_dir / "prompt_opro_tmpl.txt"
    prompt_path.write_text(OPRO_TMPL_PROMPT)
    summary["prompt_file"] = str(prompt_path)
    print(f"\n  Prompt file: {prompt_path}")

    # ------------------------------------------------------------------
    # Write summary JSON
    # ------------------------------------------------------------------
    summary_path = output_dir / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")

    print(f"\nDone. All splits written to {output_dir}")


if __name__ == "__main__":
    main()
