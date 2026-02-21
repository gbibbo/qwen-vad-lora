# B.4 Part 1 — BA × SNR Breakdown (All 9 Configs)
**Date:** 2026-02-17

## Motivation

All adapted models (LoRA variants + Qwen3) have SNR75 < −10 dB (censored).
This table shows the complete BA at each tested SNR level to understand
how much headroom remains and whether extension to lower SNRs is warranted.

## Table: Balanced Accuracy by SNR Level (% BA)

| Configuration | -10 dB | -5 dB | 0 dB | +5 dB | +10 dB | +20 dB |
|---|---|---|---|---|---|---|
| Base+Hand | 62.2 | 66.2 | 64.0 | 62.6 | 61.2 | 60.5 |
| Base+OPRO-LLM | 88.0 | 88.9 | 88.4 | 85.4 | 83.0 | 82.2 |
| Base+OPRO-Tmpl | 68.1 | 72.9 | 75.8 | 77.7 | 76.6 | 77.7 |
| LoRA+Hand | 91.9 | 91.9 | 91.5 | 88.8 | 88.5 | 88.4 |
| LoRA+OPRO-LLM | 78.2 | 81.4 | 83.6 | 85.2 | 82.6 | 82.2 |
| LoRA+OPRO-Tmpl | 96.3 | 96.2 | 97.1 | 97.7 | 98.5 | 98.5 |
| Qwen3+Hand | 98.7 | 98.5 | 98.8 | 98.6 | 98.6 | 98.1 |
| Qwen3+OPRO-LLM | 95.7 | 98.7 | 98.9 | 98.6 | 97.6 | 97.5 |
| Qwen3+OPRO-Tmpl | 96.1 | 97.1 | 96.1 | 94.8 | 94.3 | 93.2 |

## Table: Speech Recall by SNR Level (%)

| Configuration | -10 dB | -5 dB | 0 dB | +5 dB | +10 dB | +20 dB |
|---|---|---|---|---|---|---|
| Base+Hand | 28.0 | 37.3 | 33.4 | 28.5 | 26.0 | 24.3 |
| Base+OPRO-LLM | 96.1 | 92.0 | 84.5 | 75.7 | 71.1 | 70.7 |
| Base+OPRO-Tmpl | 79.0 | 76.9 | 73.8 | 70.9 | 69.3 | 73.6 |
| LoRA+Hand | 94.2 | 93.6 | 89.7 | 84.3 | 83.9 | 82.5 |
| LoRA+OPRO-LLM | 100.0 | 99.0 | 98.1 | 96.1 | 92.6 | 92.6 |
| LoRA+OPRO-Tmpl | 99.6 | 100.0 | 99.8 | 99.8 | 99.8 | 100.0 |
| Qwen3+Hand | 99.0 | 99.8 | 100.0 | 100.0 | 100.0 | 100.0 |
| Qwen3+OPRO-LLM | 92.2 | 99.6 | 100.0 | 100.0 | 100.0 | 100.0 |
| Qwen3+OPRO-Tmpl | 95.3 | 99.6 | 100.0 | 100.0 | 100.0 | 100.0 |

## Table: Nonspeech Recall by SNR Level (%)

| Configuration | -10 dB | -5 dB | 0 dB | +5 dB | +10 dB | +20 dB |
|---|---|---|---|---|---|---|
| Base+Hand | 96.3 | 95.1 | 94.6 | 96.7 | 96.5 | 96.7 |
| Base+OPRO-LLM | 80.0 | 85.8 | 92.2 | 95.1 | 94.8 | 93.6 |
| Base+OPRO-Tmpl | 57.3 | 68.9 | 77.7 | 84.5 | 83.9 | 81.9 |
| LoRA+Hand | 89.5 | 90.1 | 93.4 | 93.2 | 93.0 | 94.2 |
| LoRA+OPRO-LLM | 56.5 | 63.9 | 69.1 | 74.2 | 72.6 | 71.8 |
| LoRA+OPRO-Tmpl | 93.0 | 92.4 | 94.4 | 95.7 | 97.1 | 96.9 |
| Qwen3+Hand | 98.4 | 97.1 | 97.5 | 97.1 | 97.1 | 96.3 |
| Qwen3+OPRO-LLM | 99.2 | 97.7 | 97.7 | 97.1 | 95.3 | 95.1 |
| Qwen3+OPRO-Tmpl | 96.9 | 94.6 | 92.2 | 89.7 | 88.7 | 86.4 |

## Analysis

**Configs with BA ≥ 90% at −10 dB (5/9):**
- Qwen3+Hand: 98.7%
- LoRA+OPRO-Tmpl: 96.3%
- Qwen3+OPRO-Tmpl: 96.1%
- Qwen3+OPRO-LLM: 95.7%
- LoRA+Hand: 91.9%

These systems have significant headroom at the lowest tested SNR.
Extension to −15 and −20 dB would differentiate them further.

## Feasibility of SNR Extension to −15 and −20 dB

- **Audio generation:** 970 clips × 2 new levels = 1,940 new files
- **Generation time:** ~5 min (white noise addition, trivial computation)
- **Evaluation time per model:** ~40 min (Qwen2), ~2h (Qwen3)
- **Top 3 systems to evaluate:** LoRA+OPRO-Tmpl, Qwen3+Hand, Qwen3+OPRO-LLM
- **Total GPU time:** ~4-5 hours on a single A100

The SNR formula `noise_rms = signal_rms / 10^(snr_db/20)` works for any value.
At −20 dB, noise is 10× louder than signal — very challenging but physically meaningful.