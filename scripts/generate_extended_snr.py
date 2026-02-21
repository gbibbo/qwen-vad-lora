#!/usr/bin/env python3
"""
B.4 Part 3 — Generate -15 dB and -20 dB SNR variants for 970 test base clips.

Replicates the SNR degradation pipeline from the original experiment:
1. Load base 1000ms clip
2. Pad to 2000ms container (centered, padding noise σ=0.0001)
3. Compute signal RMS from the 1000ms region
4. Add white Gaussian noise at target SNR
5. Save as WAV

Also generates a metadata CSV compatible with eval.py's manifest format.

Includes sanity check: verifies actual SNR of generated audio matches target (±1 dB).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path("/mnt/fast/nobackup/users/gb0048/opro3_final")

# Constants matching the original pipeline
CONTAINER_DURATION_MS = 2000
SAMPLE_RATE = 16000
PADDING_NOISE_AMPLITUDE = 0.0001  # σ for container padding
CONTAINER_SAMPLES = int(CONTAINER_DURATION_MS / 1000 * SAMPLE_RATE)  # 32000


def pad_to_container(audio: np.ndarray, sr: int, seed: int = None) -> np.ndarray:
    """Pad audio to 2000ms container, centered, with low-amplitude Gaussian noise padding."""
    target_len = CONTAINER_SAMPLES
    audio_len = len(audio)

    if audio_len >= target_len:
        # Trim to container length (centered)
        start = (audio_len - target_len) // 2
        return audio[start:start + target_len]

    # Create padding noise
    rng = np.random.RandomState(seed)
    container = rng.normal(0, PADDING_NOISE_AMPLITUDE, target_len).astype(np.float32)

    # Center the audio in the container
    start = (target_len - audio_len) // 2
    container[start:start + audio_len] = audio

    return container


def add_noise_at_snr(audio: np.ndarray, signal_region: np.ndarray, snr_db: float,
                     seed: int = None) -> np.ndarray:
    """
    Add white Gaussian noise to audio at specified SNR.

    SNR is computed relative to the signal region (not the full container).
    Noise is added to the full container.
    """
    signal_rms = np.sqrt(np.mean(signal_region ** 2))

    if signal_rms < 1e-10:
        # Near-silent signal: use fallback noise floor
        signal_rms = PADDING_NOISE_AMPLITUDE

    noise_rms = signal_rms / (10 ** (snr_db / 20))

    rng = np.random.RandomState(seed)
    noise = rng.normal(0, noise_rms, len(audio)).astype(np.float32)

    return audio + noise


def measure_snr(clean_signal: np.ndarray, noisy_signal: np.ndarray) -> float:
    """Measure actual SNR between clean and noisy signals."""
    noise = noisy_signal - clean_signal
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-20:
        return float('inf')

    return 10 * np.log10(signal_power / noise_power)


def main():
    parser = argparse.ArgumentParser(description="Generate extended SNR variants")
    parser.add_argument("--base_csv", type=str,
                        default=str(ROOT / "data/processed/base_validated_1000/test_base.csv"),
                        help="CSV with base clip metadata")
    parser.add_argument("--output_dir", type=str,
                        default=str(ROOT / "audits/round2/b4_extended_snr"),
                        help="Output directory")
    parser.add_argument("--snr_levels", type=float, nargs="+", default=[-15.0, -20.0],
                        help="SNR levels to generate (dB)")
    parser.add_argument("--sanity_check_n", type=int, default=5,
                        help="Number of clips for sanity check (0 to skip)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Load base clip metadata
    base_df = pd.read_csv(args.base_csv)
    print(f"Loaded {len(base_df)} base clips from {args.base_csv}")
    print(f"SNR levels to generate: {args.snr_levels} dB")

    # --- Sanity Check ---
    if args.sanity_check_n > 0:
        print(f"\n{'='*60}")
        print(f"SANITY CHECK: Verifying SNR on {args.sanity_check_n} clips")
        print(f"{'='*60}")

        check_clips = base_df.head(args.sanity_check_n)
        all_ok = True

        for _, row in check_clips.iterrows():
            audio_path = row["audio_path"]
            if not os.path.isabs(audio_path):
                audio_path = str(ROOT / audio_path)

            audio, sr = sf.read(audio_path, dtype="float32")
            if sr != SAMPLE_RATE:
                print(f"  WARNING: {row['clip_id']} has sr={sr}, expected {SAMPLE_RATE}")
                continue

            # Pad to container
            container = pad_to_container(audio, sr, seed=args.seed)
            signal_start = (CONTAINER_SAMPLES - len(audio)) // 2
            signal_region = container[signal_start:signal_start + len(audio)]

            for snr_db in args.snr_levels:
                noisy = add_noise_at_snr(container.copy(), signal_region, snr_db,
                                         seed=args.seed + int(abs(snr_db) * 100))
                # Measure actual SNR on the signal region
                actual_snr = measure_snr(
                    signal_region,
                    noisy[signal_start:signal_start + len(audio)]
                )
                diff = abs(actual_snr - snr_db)
                status = "OK" if diff <= 1.0 else "FAIL"
                if diff > 1.0:
                    all_ok = False
                print(f"  {row['clip_id']}: target={snr_db:+.0f}dB, actual={actual_snr:+.1f}dB, "
                      f"Δ={diff:.2f}dB [{status}]")

        if all_ok:
            print(f"\nSanity check PASSED: all {args.sanity_check_n} clips within ±1 dB tolerance")
        else:
            print(f"\nSanity check FAILED: some clips exceed ±1 dB tolerance")
            print("Aborting generation. Please investigate.")
            sys.exit(1)

    # --- Generate all variants ---
    print(f"\n{'='*60}")
    print(f"Generating {len(base_df) * len(args.snr_levels)} audio files...")
    print(f"{'='*60}")

    metadata_rows = []
    n_generated = 0

    for idx, row in base_df.iterrows():
        clip_id = row["clip_id"]
        audio_path = row["audio_path"]
        if not os.path.isabs(audio_path):
            audio_path = str(ROOT / audio_path)

        ground_truth = row["ground_truth"]
        dataset = row.get("dataset", "unknown")
        group_id = row.get("group_id", "")

        # Load base audio
        audio, sr = sf.read(audio_path, dtype="float32")
        if sr != SAMPLE_RATE:
            # Simple resampling
            ratio = SAMPLE_RATE / sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length).astype(int)
            audio = audio[indices]
            sr = SAMPLE_RATE

        # Pad to container
        container = pad_to_container(audio, sr, seed=args.seed + idx)
        signal_start = (CONTAINER_SAMPLES - len(audio)) // 2
        signal_region = container[signal_start:signal_start + len(audio)]

        for snr_db in args.snr_levels:
            snr_str = f"{snr_db:+.0f}".replace("+", "+").replace("-", "-")
            variant_id = f"{clip_id}_snr{snr_str}dB"
            out_filename = f"{variant_id}.wav"
            out_path = audio_dir / out_filename

            # Generate noisy variant
            noisy = add_noise_at_snr(
                container.copy(), signal_region, snr_db,
                seed=args.seed + idx * 100 + int(abs(snr_db))
            )

            # Save
            sf.write(str(out_path), noisy, sr)

            metadata_rows.append({
                "clip_id": clip_id,
                "variant_id": variant_id,
                "variant_type": "snr",
                "duration_ms": "",
                "snr_db": snr_db,
                "T60": "",
                "band_filter": "",
                "audio_path": str(out_path),
                "label": ground_truth,
                "ground_truth": ground_truth,
                "dataset": dataset,
                "group_id": group_id,
                "sr": sr,
                "rms": float(np.sqrt(np.mean(noisy ** 2))),
                "container_duration_ms": CONTAINER_DURATION_MS,
            })
            n_generated += 1

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(base_df)} clips ({n_generated} files generated)")

    # Save metadata CSV
    meta_df = pd.DataFrame(metadata_rows)
    meta_path = output_dir / "extended_snr_metadata.csv"
    meta_df.to_csv(meta_path, index=False)

    print(f"\nGenerated {n_generated} audio files in {audio_dir}")
    print(f"Metadata saved to {meta_path}")
    print(f"SNR levels: {args.snr_levels}")


if __name__ == "__main__":
    main()
