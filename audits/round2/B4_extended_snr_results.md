# B.4 — SNR Robustness: Full Range (−20 to +20 dB)
**Date:** 2026-02-18

## Overview

Part 1 evaluated all 9 configs at −10 to +20 dB (n=970 per level).
Part 3 extended evaluation to −15 and −20 dB for the 3 top performers:
LoRA+OPRO-Tmpl, Qwen3+Hand, Qwen3+OPRO-LLM (n=970 per level).

## Balanced Accuracy (%)

| Config | -20 dB | -15 dB | -10 dB | -5 dB | 0 dB | +5 dB | +10 dB | +20 dB |
|---|---|---|---|---|---|---|---|---|
| Base+Hand | — | — | 62.2 | 66.2 | 64.0 | 62.6 | 61.2 | 60.5 |
| Base+OPRO-LLM | — | — | 88.0 | 88.9 | 88.4 | 85.4 | 83.0 | 82.2 |
| Base+OPRO-Tmpl | — | — | 68.1 | 72.9 | 75.8 | 77.7 | 76.6 | 77.7 |
| LoRA+Hand | — | — | 91.9 | 91.9 | 91.5 | 88.8 | 88.5 | 88.4 |
| LoRA+OPRO-LLM | — | — | 78.2 | 81.4 | 83.6 | 85.2 | 82.6 | 82.2 |
| LoRA+OPRO-Tmpl | 51.2 | 83.3 | 96.3 | 96.2 | 97.1 | 97.7 | 98.5 | 98.5 |
| Qwen3+Hand | 50.0 | 51.6 | 98.7 | 98.5 | 98.8 | 98.6 | 98.6 | 98.1 |
| Qwen3+OPRO-LLM | 50.0 | 50.0 | 95.7 | 98.7 | 98.9 | 98.6 | 97.6 | 97.5 |
| Qwen3+OPRO-Tmpl | — | — | 96.1 | 97.1 | 96.1 | 94.8 | 94.3 | 93.2 |

## Speech Recall (%)

| Config | -20 dB | -15 dB | -10 dB | -5 dB | 0 dB | +5 dB | +10 dB | +20 dB |
|---|---|---|---|---|---|---|---|---|
| Base+Hand | — | — | 28.0 | 37.3 | 33.4 | 28.5 | 26.0 | 24.3 |
| Base+OPRO-LLM | — | — | 96.1 | 92.0 | 84.5 | 75.7 | 71.1 | 70.7 |
| Base+OPRO-Tmpl | — | — | 79.0 | 76.9 | 73.8 | 70.9 | 69.3 | 73.6 |
| LoRA+Hand | — | — | 94.2 | 93.6 | 89.7 | 84.3 | 83.9 | 82.5 |
| LoRA+OPRO-LLM | — | — | 100.0 | 99.0 | 98.1 | 96.1 | 92.6 | 92.6 |
| LoRA+OPRO-Tmpl | 2.7 | 69.5 | 99.6 | 100.0 | 99.8 | 99.8 | 99.8 | 100.0 |
| Qwen3+Hand | 0.0 | 3.3 | 99.0 | 99.8 | 100.0 | 100.0 | 100.0 | 100.0 |
| Qwen3+OPRO-LLM | 0.0 | 0.0 | 92.2 | 99.6 | 100.0 | 100.0 | 100.0 | 100.0 |
| Qwen3+OPRO-Tmpl | — | — | 95.3 | 99.6 | 100.0 | 100.0 | 100.0 | 100.0 |

## Nonspeech Recall (%)

| Config | -20 dB | -15 dB | -10 dB | -5 dB | 0 dB | +5 dB | +10 dB | +20 dB |
|---|---|---|---|---|---|---|---|---|
| Base+Hand | — | — | 96.3 | 95.1 | 94.6 | 96.7 | 96.5 | 96.7 |
| Base+OPRO-LLM | — | — | 80.0 | 85.8 | 92.2 | 95.1 | 94.8 | 93.6 |
| Base+OPRO-Tmpl | — | — | 57.3 | 68.9 | 77.7 | 84.5 | 83.9 | 81.9 |
| LoRA+Hand | — | — | 89.5 | 90.1 | 93.4 | 93.2 | 93.0 | 94.2 |
| LoRA+OPRO-LLM | — | — | 56.5 | 63.9 | 69.1 | 74.2 | 72.6 | 71.8 |
| LoRA+OPRO-Tmpl | 99.8 | 97.1 | 93.0 | 92.4 | 94.4 | 95.7 | 97.1 | 96.9 |
| Qwen3+Hand | 100.0 | 100.0 | 98.4 | 97.1 | 97.5 | 97.1 | 97.1 | 96.3 |
| Qwen3+OPRO-LLM | 100.0 | 100.0 | 99.2 | 97.7 | 97.7 | 97.1 | 95.3 | 95.1 |
| Qwen3+OPRO-Tmpl | — | — | 96.9 | 94.6 | 92.2 | 89.7 | 88.7 | 86.4 |

## Key Findings

1. **LoRA+OPRO-Tmpl** is the only system that maintains meaningful performance at −15 dB (BA=83.3%). At −20 dB it degrades to near-chance (51.2%), driven by speech recall collapse (2.7%) while nonspeech recall stays high (99.8%).

2. **Both Qwen3 configs collapse abruptly below −10 dB.** Qwen3+Hand drops from 98.7% BA at −10 dB to 51.6% at −15 dB. Qwen3+OPRO-LLM drops to exactly 50.0% (pure NONSPEECH bias) at both −15 and −20 dB.

3. **The −15 dB cliff separates fine-tuned vs. zero-shot robustness.** LoRA fine-tuning provides a noise resilience advantage that persists ~5 dB below the point where zero-shot Qwen3 fails.

4. At −20 dB all systems are effectively at chance, confirming this as the practical floor for speech detection with current LALMs.