# B.6 — Silero VAD Under Psychometric Bank
**Date:** 2026-02-17

## Overview

Silero VAD evaluated on all 21,340 test clips using two operating points:
1. **Max-frame:** If ANY frame has speech probability ≥ 0.5 → SPEECH
2. **Speech-ratio:** If proportion of speech frames ≥ 0.5 → SPEECH
   (consistent with the speech_ratio criterion used for data curation)

## Max-frame Criterion

- **BA_clip:** 88.9% [88.4, 89.5]
- **Recall(SPEECH):** 78.8% [78.0, 79.5]
- **Recall(NONSPEECH):** 99.1% [98.9, 99.3]

### Per-Dimension BA

| Dimension | BA (%) |
|-----------|--------|
| duration | 77.5 |
| filter | 98.9 |
| reverb | 99.6 |
| snr | 90.4 |

### Psychometric Thresholds

- **DT50:** 20 ms (below_range)
- **DT75:** 81 ms (ok)
- **DT90:** 196 ms (ok)
- **SNR75:** -7 dB (ok)

### Per-Condition BA

| Condition | BA (%) | R_speech (%) | R_nonspeech (%) | n |
|-----------|--------|-------------|----------------|---|
| dur_1000ms | 98.6 | 99.8 | 97.3 | 970 |
| dur_100ms | 73.5 | 48.0 | 99.0 | 970 |
| dur_200ms | 90.5 | 86.8 | 94.2 | 970 |
| dur_20ms | 55.7 | 13.4 | 97.9 | 970 |
| dur_40ms | 61.4 | 24.1 | 98.8 | 970 |
| dur_500ms | 97.6 | 96.7 | 98.6 | 970 |
| dur_60ms | 68.1 | 37.1 | 99.2 | 970 |
| dur_80ms | 74.8 | 53.4 | 96.3 | 970 |
| filter_bandpass | 97.6 | 95.3 | 100.0 | 970 |
| filter_highpass | 98.9 | 97.7 | 100.0 | 970 |
| filter_lowpass | 99.5 | 99.2 | 99.8 | 970 |
| filter_none | 99.6 | 99.4 | 99.8 | 970 |
| reverb_0.3s | 99.5 | 99.2 | 99.8 | 970 |
| reverb_1.0s | 99.7 | 99.4 | 100.0 | 970 |
| reverb_2.5s | 99.7 | 99.6 | 99.8 | 970 |
| reverb_none | 99.5 | 99.0 | 100.0 | 970 |
| snr_-10dB | 57.0 | 14.0 | 100.0 | 970 |
| snr_-5dB | 89.7 | 79.4 | 100.0 | 970 |
| snr_0dB | 97.9 | 95.9 | 100.0 | 970 |
| snr_10dB | 99.5 | 99.0 | 100.0 | 970 |
| snr_20dB | 99.5 | 99.0 | 100.0 | 970 |
| snr_5dB | 99.0 | 97.9 | 100.0 | 970 |

## Speech-ratio Criterion

- **BA_clip:** 54.9% [54.4, 55.4]
- **Recall(SPEECH):** 9.8% [9.3, 10.4]
- **Recall(NONSPEECH):** 100.0% [100.0, 100.0]

### Per-Dimension BA

| Dimension | BA (%) |
|-----------|--------|
| duration | 50.1 |
| filter | 53.7 |
| reverb | 52.2 |
| snr | 64.0 |

### Psychometric Thresholds

- **DT50:** 20 ms (below_range)
- **DT75:** 1000 ms (above_range)
- **DT90:** 1000 ms (above_range)
- **SNR75:** 20 dB (above_range)

### Per-Condition BA

| Condition | BA (%) | R_speech (%) | R_nonspeech (%) | n |
|-----------|--------|-------------|----------------|---|
| dur_1000ms | 50.8 | 1.6 | 100.0 | 970 |
| dur_100ms | 50.0 | 0.0 | 100.0 | 970 |
| dur_200ms | 50.0 | 0.0 | 100.0 | 970 |
| dur_20ms | 50.0 | 0.0 | 100.0 | 970 |
| dur_40ms | 50.0 | 0.0 | 100.0 | 970 |
| dur_500ms | 50.0 | 0.0 | 100.0 | 970 |
| dur_60ms | 50.0 | 0.0 | 100.0 | 970 |
| dur_80ms | 50.0 | 0.0 | 100.0 | 970 |
| filter_bandpass | 52.1 | 4.1 | 100.0 | 970 |
| filter_highpass | 56.8 | 13.6 | 100.0 | 970 |
| filter_lowpass | 52.4 | 4.7 | 100.0 | 970 |
| filter_none | 53.4 | 6.8 | 100.0 | 970 |
| reverb_0.3s | 52.9 | 5.8 | 100.0 | 970 |
| reverb_1.0s | 50.2 | 0.4 | 100.0 | 970 |
| reverb_2.5s | 52.9 | 5.8 | 100.0 | 970 |
| reverb_none | 52.8 | 5.6 | 100.0 | 970 |
| snr_-10dB | 50.0 | 0.0 | 100.0 | 970 |
| snr_-5dB | 52.9 | 5.8 | 100.0 | 970 |
| snr_0dB | 67.4 | 34.8 | 100.0 | 970 |
| snr_10dB | 73.5 | 47.0 | 100.0 | 970 |
| snr_20dB | 65.3 | 30.5 | 100.0 | 970 |
| snr_5dB | 74.8 | 49.7 | 100.0 | 970 |

## Summary: Two Operating Points Compared

| Metric | Max-frame | Speech-ratio |
|--------|-----------|-------------|
| BA_clip | 88.9% | 54.9% |
| Recall(SPEECH) | 78.8% | 9.8% |
| Recall(NONSPEECH) | 99.1% | 100.0% |

The max-frame criterion is more sensitive (higher speech recall, lower nonspeech recall),
while the speech-ratio criterion is more conservative and consistent with the data curation filter.