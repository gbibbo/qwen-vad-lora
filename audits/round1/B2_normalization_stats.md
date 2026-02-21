# B.2 — Normalization Pathway Stats
**Date:** 2026-02-17

## Table 1: Test Set Parse Rates (21,340 samples per config)

These are from the FINAL test evaluation `predictions.csv`. Since raw model
outputs were not saved, we can only determine whether a response was
successfully parsed (SPEECH or NONSPEECH) vs unparseable (UNKNOWN).
Successfully parsed responses went through levels 1-5; we cannot distinguish
which specific level resolved them.

| Configuration | Total | SPEECH | NONSPEECH | UNKNOWN | Parse Rate |
|---------------|-------|--------|-----------|---------|------------|
| Base+Hand | 21,340 | 3,852 | 17,479 | 9 | 99.96% |
| Base+OPRO-LLM | 21,340 | 8,950 | 12,070 | 320 | 98.50% |
| Base+OPRO-Tmpl | 21,340 | 9,932 | 11,381 | 27 | 99.87% |
| LoRA+Hand | 21,340 | 9,822 | 11,511 | 7 | 99.97% |
| LoRA+OPRO-LLM | 21,340 | 12,952 | 8,332 | 56 | 99.74% |
| LoRA+OPRO-Tmpl | 21,340 | 10,569 | 10,770 | 1 | 100.00% |
| Qwen3+Hand | 21,340 | 9,889 | 11,451 | 0 | 100.00% |
| Qwen3+OPRO-LLM | 21,340 | 10,207 | 11,133 | 0 | 100.00% |
| Qwen3+OPRO-Tmpl | 21,340 | 11,185 | 10,155 | 0 | 100.00% |

### Verification of Paper Claims (Section 3.3)

- **Paper claim:** ">99.7% of responses resolved at levels 1-2 for 7 of 9 configs"
- **Configs with >=99.7% parse rate:** 8/9 — Base+Hand, Base+OPRO-Tmpl, LoRA+Hand, LoRA+OPRO-LLM, LoRA+OPRO-Tmpl, Qwen3+Hand, Qwen3+OPRO-LLM, Qwen3+OPRO-Tmpl
- **Configs below 99.7%:** Base+OPRO-LLM (98.50%)
- **DISCREPANCY:** Paper says "7 of 9" but data shows **8 of 9** configs meet ≥99.7%. The borderline case is LoRA+OPRO-LLM at 99.74%. Possible explanations: (a) the paper used a stricter threshold, (b) rounding difference, or (c) minor error in manuscript. **Recommend verifying in Ronda 2 with raw_text data.**

- **Paper claim:** "Qwen3-Omni achieving 100% parseability across all three prompting conditions"
- **Verified:** YES — Qwen3+Hand: 0 unknown, Qwen3+OPRO-LLM: 0 unknown, Qwen3+OPRO-Tmpl: 0 unknown

- **Paper claim:** "Base + OPRO-LLM, where 320 of 21,340 responses (1.5%) required lower-level normalization"
- **UNKNOWN count from predictions.csv:** 320
- **DISCREPANCY in wording:** The paper says these 320 responses "required lower-level normalization," implying they were eventually resolved (at levels 3-6). However, predictions.csv shows these 320 as **UNKNOWN** — meaning they failed ALL 6 normalization levels and could NOT be resolved. The count matches exactly (320), so these are the same responses, but the paper's phrasing is misleading. The manuscript should clarify that these 320 responses **could not be parsed** by the normalization hierarchy. **Recommend correcting in Phase A revisions.**

## Table 2: Detailed Level Breakdown (OPRO-Template Optimization, Dev Set)

These statistics come from the `iter*_all_predictions.csv` files generated
during OPRO-Template optimization. Unlike the test set, these files contain
the `raw_text` column, allowing us to trace each response through the
normalization hierarchy.

**Note:** This is dev set data during optimization (not the final test set),
and covers multiple prompts per iteration. It provides a representative
distribution of normalization levels but is not directly comparable to the
paper's test-set claims.

### Base+OPRO-Tmpl
Total responses analyzed: 2,400

| Level | Count | Percentage |
|-------|-------|------------|
| L1: NONSPEECH substr | 1,560 | 65.00% |
| L2: SPEECH substr | 833 | 34.71% |
| L3: Letter mapping | 0 | 0.00% |
| L4: YES/NO | 0 | 0.00% |
| L5: Keywords | 0 | 0.00% |
| L6: Unknown/fallback | 7 | 0.29% |
| **L1+L2 combined** | **2,393** | **99.71%** |

### LoRA+OPRO-Tmpl
Total responses analyzed: 2,400

| Level | Count | Percentage |
|-------|-------|------------|
| L1: NONSPEECH substr | 1,221 | 50.88% |
| L2: SPEECH substr | 1,170 | 48.75% |
| L3: Letter mapping | 0 | 0.00% |
| L4: YES/NO | 2 | 0.08% |
| L5: Keywords | 0 | 0.00% |
| L6: Unknown/fallback | 7 | 0.29% |
| **L1+L2 combined** | **2,391** | **99.62%** |

### Qwen3+OPRO-Tmpl
Total responses analyzed: 2,400

| Level | Count | Percentage |
|-------|-------|------------|
| L1: NONSPEECH substr | 849 | 35.38% |
| L2: SPEECH substr | 1,551 | 64.62% |
| L3: Letter mapping | 0 | 0.00% |
| L4: YES/NO | 0 | 0.00% |
| L5: Keywords | 0 | 0.00% |
| L6: Unknown/fallback | 0 | 0.00% |
| **L1+L2 combined** | **2,400** | **100.00%** |

## What Is Missing & How to Complete

### Missing Data
The final test evaluation (`scripts/eval.py`) saves only the normalized label
in `predictions.csv`, not the raw model output. Therefore:

- We **can** report parse rates (SPEECH/NONSPEECH/UNKNOWN) for all 9 configs
- We **cannot** determine which normalization level (1-5) resolved each response
- The paper's claim "320 responses required lower-level normalization" for
  Base+OPRO-LLM cannot be verified from saved data

### How to Get Complete Data (Ronda 2)
Modify `scripts/eval.py` to save `raw_text` in predictions.csv:

```python
# In eval.py evaluate_samples(), add raw_text to results dict (line ~119):
results.append({
    "audio_path": audio_path,
    "ground_truth": ground_truth,
    "raw_text": response,          # <-- ADD THIS
    "prediction": prediction,
    "condition": condition_key,
    "variant_type": row.get("variant_type", "unknown"),
})
```

Then re-run evaluation for all 9 configs (requires GPU). After that,
re-run this script to get the complete level breakdown for the test set.

**Estimated GPU time:** ~2-3 hours per config × 9 configs on A100.