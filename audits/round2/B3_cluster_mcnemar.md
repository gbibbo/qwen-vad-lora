# B.3 — Cluster-Aware McNemar Tests
**Date:** 2026-02-17

## Motivation

Each of the 970 base clips generates 22 degraded variants, inducing within-clip
correlation. The standard McNemar test treats all 21,340 samples as i.i.d.,
which may inflate significance for small effect sizes. We implement two
cluster-aware alternatives to verify robustness of p-values.

## Methods

1. **Standard i.i.d. McNemar** — Exact binomial test on 21,340 discordant pairs
2. **Cluster bootstrap McNemar** (B=10,000) — Resample 970 base clips with replacement,
   compute discordant ratio per bootstrap sample, derive p-value from bootstrap null
3. **Majority-vote collapsed McNemar** — Collapse 22 variants per clip to single
   prediction (majority vote), then standard McNemar on ~970 independent clips

All p-values corrected with Holm-Bonferroni for multiple comparisons.

## Results

| Comparison | ΔBA (pp) | p (i.i.d.) | p (cluster) | p (majority) | Sig changes? |
|-----------|---------|-----------|------------|-------------|-------------|
| Baseline vs Base+OPRO-LLM | -18.6 | 0.00e+00 | 0.0006 | 9.46e-78 | All significant |
| Baseline vs LoRA+Hand | -22.4 | 0.00e+00 | 0.0006 | 2.37e-82 | All significant |
| LoRA+Hand vs LoRA+OPRO-Tmpl | -6.9 | 3.97e-231 | 0.0006 | 1.17e-09 | All significant |
| LoRA+OPRO-Tmpl vs LoRA+OPRO-LLM | +9.3 | 5.43e-250 | 0.0006 | 5.24e-17 | All significant |
| Qwen3+Hand vs Qwen3+OPRO-LLM | -0.3 | 1.36e-02 | 0.0929 | 1.00e+00 | i.i.d.→NS (cluster) |
| LoRA+OPRO-Tmpl vs Qwen3+Hand | +2.2 | 3.05e-32 | 0.0006 | 1.00e+00 | i.i.d.→NS (majority) |

## Detailed Results

### Baseline vs Base+OPRO-LLM
- Base+Hand: BA = 64.0%
- Base+OPRO-LLM: BA = 82.6%
- ΔBA = -18.6 pp

**i.i.d. McNemar:** n_01=869, n_10=4840, n_disc=5709, p=0.00e+00 (adj=0.00e+00)

**Cluster bootstrap:** ratio=0.1522, CI=[0.1290, 0.1773], p=0.0001 (adj=0.0006)

**Majority-vote:** n_clips=970, n_01=6, n_10=293, p=1.89e-78 (adj=9.46e-78)

### Baseline vs LoRA+Hand
- Base+Hand: BA = 64.0%
- LoRA+Hand: BA = 86.4%
- ΔBA = -22.4 pp

**i.i.d. McNemar:** n_01=650, n_10=5428, n_disc=6078, p=0.00e+00 (adj=0.00e+00)

**Cluster bootstrap:** ratio=0.1069, CI=[0.0887, 0.1273], p=0.0001 (adj=0.0006)

**Majority-vote:** n_clips=970, n_01=6, n_10=309, p=3.95e-83 (adj=2.37e-82)

### LoRA+Hand vs LoRA+OPRO-Tmpl
- LoRA+Hand: BA = 86.4%
- LoRA+OPRO-Tmpl: BA = 93.3%
- ΔBA = -6.9 pp

**i.i.d. McNemar:** n_01=379, n_10=1850, n_disc=2229, p=1.32e-231 (adj=3.97e-231)

**Cluster bootstrap:** ratio=0.1700, CI=[0.1436, 0.1998], p=0.0001 (adj=0.0006)

**Majority-vote:** n_clips=970, n_01=5, n_10=49, p=3.89e-10 (adj=1.17e-09)

### LoRA+OPRO-Tmpl vs LoRA+OPRO-LLM
- LoRA+OPRO-Tmpl: BA = 93.3%
- LoRA+OPRO-LLM: BA = 84.0%
- ΔBA = +9.3 pp

**i.i.d. McNemar:** n_01=2787, n_10=813, n_disc=3600, p=1.36e-250 (adj=5.43e-250)

**Cluster bootstrap:** ratio=0.7742, CI=[0.7416, 0.8033], p=0.0001 (adj=0.0006)

**Majority-vote:** n_clips=970, n_01=80, n_10=6, p=1.31e-17 (adj=5.24e-17)

### Qwen3+Hand vs Qwen3+OPRO-LLM
- Qwen3+Hand: BA = 91.1%
- Qwen3+OPRO-LLM: BA = 91.4%
- ΔBA = -0.3 pp

**i.i.d. McNemar:** n_01=275, n_10=337, n_disc=612, p=1.36e-02 (adj=1.36e-02)

**Cluster bootstrap:** ratio=0.4493, CI=[0.3900, 0.5086], p=0.0929 (adj=0.0929)

**Majority-vote:** n_clips=970, n_01=6, n_10=6, p=1.00e+00 (adj=1.00e+00)

### LoRA+OPRO-Tmpl vs Qwen3+Hand
- LoRA+OPRO-Tmpl: BA = 93.3%
- Qwen3+Hand: BA = 91.1%
- ΔBA = +2.2 pp

**i.i.d. McNemar:** n_01=1030, n_10=559, n_disc=1589, p=1.53e-32 (adj=3.05e-32)

**Cluster bootstrap:** ratio=0.6482, CI=[0.5989, 0.6975], p=0.0001 (adj=0.0006)

**Majority-vote:** n_clips=970, n_01=12, n_10=13, p=1.00e+00 (adj=1.00e+00)

## Key Finding: Qwen3+Hand vs Qwen3+OPRO-LLM

This is the most vulnerable comparison (ΔBA = -0.3 pp).
- i.i.d. p = 1.36e-02 (adjusted: 1.36e-02)
- Cluster p = 0.0929 (adjusted: 0.0929)
- Majority p = 1.00e+00 (adjusted: 1.00e+00)

**FINDING:** The cluster-aware correction renders this comparison non-significant.
The paper text should be updated to reflect this.
