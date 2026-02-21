# B.2 — Normalization Level Breakdown (Test Set)
**Date:** 2026-02-17

## Base+OPRO-LLM

### Reproducibility Check
- Predictions matched original: **100.00%** (21340/21340)
- **Perfect match** — greedy decoding is deterministic

### Normalization Level Breakdown (n=21,340)

| Level | Count | Percentage |
|-------|-------|------------|
| L1: NONSPEECH substring | 12,070 | 56.56% |
| L2: SPEECH substring | 8,949 | 41.94% |
| L5: Semantic keywords | 1 | 0.00% |
| L6b: Unknown/unparseable | 320 | 1.50% |
| **L1+L2 combined** | **21,019** | **98.50%** |

### Unparseable/Fallback Responses (first 20)

| Audio | Level | Prediction | Raw Text (truncated) |
|-------|-------|------------|---------------------|
| voxconverse_ktzmw_0377_1000ms_filterhighpass.wav | L6_UNKNOWN | UNKNOWN | PredictionResult(label='UNKNOWN', confidence=0.0, raw_output='人类的语音。', latency_ms=342.745304107666,  |
| voxconverse_wewoz_0454_1000ms_dur500ms.wav | L6_UNKNOWN | UNKNOWN | PredictionResult(label='UNKNOWN', confidence=0.0, raw_output='人类的语音。', latency_ms=346.68803215026855 |
| voxconverse_wewoz_0454_1000ms_dur1000ms.wav | L6_UNKNOWN | UNKNOWN | PredictionResult(label='UNKNOWN', confidence=0.0, raw_output='人类的语音。', latency_ms=342.7097797393799, |
| voxconverse_wewoz_0454_1000ms_reverbnone.wav | L6_UNKNOWN | UNKNOWN | PredictionResult(label='UNKNOWN', confidence=0.0, raw_output='人类的语音。', latency_ms=348.2394218444824, |
| voxconverse_wewoz_0454_1000ms_reverb0.3s.wav | L6_UNKNOWN | UNKNOWN | PredictionResult(label='UNKNOWN', confidence=0.0, raw_output='人类的语音。', latency_ms=341.19486808776855 |
| voxconverse_wewoz_0454_1000ms_reverb2.5s.wav | L6_UNKNOWN | UNKNOWN | PredictionResult(label='UNKNOWN', confidence=0.0, raw_output='人类的语音。', latency_ms=341.1548137664795, |
| voxconverse_wewoz_0454_1000ms_filternone.wav | L6_UNKNOWN | UNKNOWN | PredictionResult(label='UNKNOWN', confidence=0.0, raw_output='人类的语音。', latency_ms=340.8987522125244, |
| voxconverse_hycgx_0274_1000ms_dur500ms.wav | L6_UNKNOWN | UNKNOWN | PredictionResult(label='UNKNOWN', confidence=0.0, raw_output='人类的语音。', latency_ms=341.9327735900879, |
| voxconverse_afjiv_0263_1000ms_dur200ms.wav | L6_UNKNOWN | UNKNOWN | PredictionResult(label='UNKNOWN', confidence=0.0, raw_output='人类的语音。', latency_ms=342.73338317871094 |
| voxconverse_ldkmv_0483_1000ms_dur200ms.wav | L6_UNKNOWN | UNKNOWN | PredictionResult(label='UNKNOWN', confidence=0.0, raw_output='人类的语音。', latency_ms=343.4782028198242, |
| voxconverse_ldkmv_0483_1000ms_dur500ms.wav | L6_UNKNOWN | UNKNOWN | PredictionResult(label='UNKNOWN', confidence=0.0, raw_output='人类的语音。', latency_ms=344.30670738220215 |
| voxconverse_ldkmv_0483_1000ms_snr-5dB.wav | L6_UNKNOWN | UNKNOWN | PredictionResult(label='UNKNOWN', confidence=0.0, raw_output='人类的说话声。', latency_ms=395.5862522125244 |
| voxconverse_kszpd_0079_1000ms_dur100ms.wav | L6_UNKNOWN | UNKNOWN | PredictionResult(label='UNKNOWN', confidence=0.0, raw_output='人类的语音。', latency_ms=356.63843154907227 |
| voxconverse_kszpd_0079_1000ms_dur200ms.wav | L6_UNKNOWN | UNKNOWN | PredictionResult(label='UNKNOWN', confidence=0.0, raw_output='人类的语音。', latency_ms=354.7971248626709, |
| voxconverse_dhorc_0307_1000ms_dur1000ms.wav | L6_UNKNOWN | UNKNOWN | PredictionResult(label='UNKNOWN', confidence=0.0, raw_output='人类的说话声。', latency_ms=408.1721305847168 |
| voxconverse_dhorc_0307_1000ms_snr+10dB.wav | L6_UNKNOWN | UNKNOWN | PredictionResult(label='UNKNOWN', confidence=0.0, raw_output='人类的说话声。', latency_ms=404.4334888458252 |
| voxconverse_dhorc_0307_1000ms_reverbnone.wav | L6_UNKNOWN | UNKNOWN | PredictionResult(label='UNKNOWN', confidence=0.0, raw_output='人类的说话声。', latency_ms=407.2782993316650 |
| voxconverse_dhorc_0307_1000ms_reverb0.3s.wav | L6_UNKNOWN | UNKNOWN | PredictionResult(label='UNKNOWN', confidence=0.0, raw_output='人类的语音。', latency_ms=354.2478084564209, |
| voxconverse_dhorc_0307_1000ms_reverb2.5s.wav | L6_UNKNOWN | UNKNOWN | PredictionResult(label='UNKNOWN', confidence=0.0, raw_output='人类的说话声。', latency_ms=403.4066200256347 |
| voxconverse_dhorc_0307_1000ms_filternone.wav | L6_UNKNOWN | UNKNOWN | PredictionResult(label='UNKNOWN', confidence=0.0, raw_output='人类的说话声。', latency_ms=405.6315422058105 |

## LoRA+OPRO-Tmpl

### Reproducibility Check
- Predictions matched original: **100.00%** (21340/21340)
- **Perfect match** — greedy decoding is deterministic

### Normalization Level Breakdown (n=21,340)

| Level | Count | Percentage |
|-------|-------|------------|
| L1: NONSPEECH substring | 10,770 | 50.47% |
| L2: SPEECH substring | 10,569 | 49.53% |
| L6b: Unknown/unparseable | 1 | 0.00% |
| **L1+L2 combined** | **21,339** | **100.00%** |

### Unparseable/Fallback Responses (first 1)

| Audio | Level | Prediction | Raw Text (truncated) |
|-------|-------|------------|---------------------|
| esc50_5-202220-A-21_0040_1000ms_reverb2.5s.wav | L6_UNKNOWN | UNKNOWN | PredictionResult(label='UNKNOWN', confidence=0.0, raw_output='SNEEZE', latency_ms=474.84493255615234 |

## Qwen3+Hand

### Reproducibility Check
- Predictions matched original: **100.00%** (21340/21340)
- **Perfect match** — greedy decoding is deterministic

### Normalization Level Breakdown (n=21,340)

| Level | Count | Percentage |
|-------|-------|------------|
| L1: NONSPEECH substring | 11,451 | 53.66% |
| L2: SPEECH substring | 9,889 | 46.34% |
| **L1+L2 combined** | **21,340** | **100.00%** |
