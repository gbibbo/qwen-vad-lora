# Data Curve Ablation — LoRA Training Size

## Summary

This experiment trains LoRA adapters with subset sizes of 256, 512, 1024
clips (3,072 already exists) and evaluates each on the full 21,340-sample
test set with two prompts (Hand-crafted and OPRO-Template T04_contrastive).

## BA_clip by Training Size and Prompt

| Train Size | LoRA+Hand BA | LoRA+OPRO-Tmpl BA |
|-----------|-------------|-------------------|
|   256     |     missing |           missing |
|   512     |     missing |           missing |
|  1024     |     missing |           missing |
|  3072     | 0.8640 (86.4%) |    0.9329 (93.3%) |

### Reference Baselines (from paper)

- **Base+Hand**: 0.7278 (72.8%)
- **Base+OPRO-LLM**: 0.8263 (82.6%)
- **Qwen3+Hand**: 0.9107 (91.1%)

## Marginal Gains (per doubling of data)

| From → To | LoRA+Hand ΔBA | LoRA+OPRO-Tmpl ΔBA |
|-----------|--------------|-------------------|
| 256→512 |            — |                 — |
| 512→1024 |            — |                 — |
| 1024→3072 |            — |                 — |

## Per-Dimension BA (LoRA+OPRO-Tmpl)

| Train Size | Duration | SNR    | Reverb | Filter |
|-----------|----------|--------|--------|--------|
|  3072     |   0.8737 | 0.9737 | 0.9606 | 0.9624 |

## Interpretation

- LoRA+OPRO-Tmpl surpasses Qwen3-Omni (0.9107) at **N=3072** clips
- LoRA+Hand surpasses Base+OPRO-LLM (0.8263) at **N=3072** clips
