#!/usr/bin/env python3
"""
B.6 — Evaluate Silero VAD on the full psychometric test bank (21,340 clips).

Uses the existing SileroVAD class from src/qsm/vad/silero.py.
Computes two operating points:
  1. max(frame_probs) >= threshold → SPEECH (any-frame criterion)
  2. speech_ratio >= threshold → SPEECH (proportion criterion, consistent with data curation)

Outputs predictions.csv for each criterion in the same format as eval.py
for compatibility with stats.py and analyze_silero.py.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

ROOT = Path("/mnt/fast/nobackup/users/gb0048/opro3_final")
sys.path.insert(0, str(ROOT))

from src.qsm.vad.silero import SileroVAD


def get_condition_key(row):
    """Extract condition key from row (same logic as eval.py)."""
    variant_type = row.get("variant_type", "")

    if variant_type == "duration":
        dur = row.get("duration_ms", "")
        return f"dur_{int(dur)}ms"
    elif variant_type == "snr":
        snr = row.get("snr_db", "")
        return f"snr_{float(snr):.0f}dB"
    elif variant_type == "reverb":
        t60 = row.get("T60", "")
        if pd.isna(t60) or t60 == "" or t60 == "none":
            return "reverb_none"
        return f"reverb_{t60}s"
    elif variant_type == "filter":
        filt = row.get("band_filter", "none")
        return f"filter_{filt}"
    else:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Evaluate Silero VAD on psychometric test bank")
    parser.add_argument("--manifest", type=str,
                        default=str(ROOT / "data/processed/variants_validated_1000/test_metadata.csv"),
                        help="Path to test metadata CSV")
    parser.add_argument("--output_dir", type=str,
                        default=str(ROOT / "audits/round2/b6_silero"),
                        help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Silero speech probability threshold")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for Silero (cpu or cuda)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("B.6 — Silero VAD Evaluation")
    print("=" * 60)
    print(f"Manifest: {args.manifest}")
    print(f"Output: {args.output_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Device: {args.device}")

    # Load manifest
    print("\nLoading manifest...")
    df = pd.read_csv(args.manifest)
    print(f"  Total samples: {len(df)}")

    # Initialize Silero VAD
    print("\nInitializing Silero VAD...")
    vad = SileroVAD(threshold=args.threshold, device=args.device)
    print(f"  Model: {vad.name}")
    print(f"  Frame duration: {vad.frame_duration_ms}ms")

    # Evaluate all clips
    print(f"\nEvaluating {len(df)} clips...")
    results_max = []       # any-frame criterion (max prob >= threshold)
    results_ratio = []     # speech_ratio criterion (proportion >= threshold)

    start_time = time.time()
    errors = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Silero VAD"):
        audio_path = row["audio_path"]
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(str(ROOT), audio_path)

        ground_truth = row["ground_truth"]
        condition = get_condition_key(row)
        variant_type = row.get("variant_type", "unknown")

        try:
            # Load audio and get frame-level probabilities
            audio, sr = sf.read(audio_path, dtype="float32")
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Resample if needed
            if sr != 16000:
                ratio = 16000 / sr
                new_length = int(len(audio) * ratio)
                indices = np.linspace(0, len(audio) - 1, new_length).astype(int)
                audio = audio[indices]
                sr = 16000

            frame_decisions, frame_probs = vad.predict_frames(audio, sr)

            # --- Criterion 1: Any-frame (max prob >= threshold) ---
            max_prob = max(frame_probs) if frame_probs else 0.0
            has_speech_frame = max_prob >= args.threshold
            label_max = "SPEECH" if has_speech_frame else "NONSPEECH"
            conf_max = max_prob if label_max == "SPEECH" else (1.0 - max_prob)

            # --- Criterion 2: Speech ratio (proportion of frames >= threshold) ---
            n_frames = len(frame_decisions)
            n_speech_frames = sum(frame_decisions)
            speech_ratio = n_speech_frames / n_frames if n_frames > 0 else 0.0
            label_ratio = "SPEECH" if speech_ratio >= args.threshold else "NONSPEECH"
            conf_ratio = speech_ratio if label_ratio == "SPEECH" else (1.0 - speech_ratio)

        except Exception as e:
            if errors < 5:
                print(f"\n  Error processing {audio_path}: {e}")
            errors += 1
            label_max = "ERROR"
            conf_max = 0.0
            label_ratio = "ERROR"
            conf_ratio = 0.0
            max_prob = 0.0
            speech_ratio = 0.0

        results_max.append({
            "audio_path": audio_path,
            "ground_truth": ground_truth,
            "prediction": label_max,
            "condition": condition,
            "variant_type": variant_type,
            "p_first_token": conf_max,
        })

        results_ratio.append({
            "audio_path": audio_path,
            "ground_truth": ground_truth,
            "prediction": label_ratio,
            "condition": condition,
            "variant_type": variant_type,
            "p_first_token": conf_ratio,
        })

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/len(df)*1000:.1f}ms per clip)")
    if errors > 0:
        print(f"Errors: {errors}/{len(df)}")

    # Save predictions for both criteria
    df_max = pd.DataFrame(results_max)
    max_path = output_dir / "predictions_max_frame.csv"
    df_max.to_csv(max_path, index=False)
    print(f"\nMax-frame predictions saved to: {max_path}")

    df_ratio = pd.DataFrame(results_ratio)
    ratio_path = output_dir / "predictions_speech_ratio.csv"
    df_ratio.to_csv(ratio_path, index=False)
    print(f"Speech-ratio predictions saved to: {ratio_path}")

    # Quick summary
    for criterion_name, df_pred in [("Max-frame", df_max), ("Speech-ratio", df_ratio)]:
        n_speech = (df_pred["prediction"] == "SPEECH").sum()
        n_nonspeech = (df_pred["prediction"] == "NONSPEECH").sum()
        n_error = (df_pred["prediction"] == "ERROR").sum()

        gt_speech = (df_pred["ground_truth"] == "SPEECH")
        gt_nonspeech = (df_pred["ground_truth"] == "NONSPEECH")

        tp = ((df_pred["prediction"] == "SPEECH") & gt_speech).sum()
        tn = ((df_pred["prediction"] == "NONSPEECH") & gt_nonspeech).sum()

        recall_speech = tp / gt_speech.sum() if gt_speech.sum() > 0 else 0
        recall_nonspeech = tn / gt_nonspeech.sum() if gt_nonspeech.sum() > 0 else 0
        ba = (recall_speech + recall_nonspeech) / 2

        print(f"\n--- {criterion_name} (threshold={args.threshold}) ---")
        print(f"  Predicted: SPEECH={n_speech}, NONSPEECH={n_nonspeech}, ERROR={n_error}")
        print(f"  BA = {ba*100:.1f}%")
        print(f"  Speech recall = {recall_speech*100:.1f}%")
        print(f"  Nonspeech recall = {recall_nonspeech*100:.1f}%")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
