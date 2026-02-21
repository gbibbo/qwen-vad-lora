# B.5 — Hyperparameter Audit Report
**Date:** 2026-02-17
**Scope:** All hyperparameters in main.tex vs actual code/configs

## Summary

- **Total parameters checked:** 48
- **Confirmed consistent:** 48
- **Discrepancies found:** 0

## Discrepancies

**None found.** All paper values match the code.

## Full Audit Table

### LoRA

| Parameter | Paper | Code | Source | Status |
|-----------|-------|------|--------|--------|
| rank (r) | 64 | 64 | finetune.py TrainingConfig | MATCH |
| alpha (α) | 16 | 16 | finetune.py TrainingConfig | MATCH |
| dropout | 0.05 | 0.05 | finetune.py TrainingConfig | MATCH |
| learning_rate | 5e-5 | 5e-5 | finetune.py TrainingConfig | MATCH |
| warmup_steps | 100 | 100 | finetune.py TrainingConfig | MATCH |
| weight_decay | 0 | 0.0 | finetune.py TrainingArguments (HF default) | MATCH — Not explicitly set; HuggingFace default=0.0 |
| epochs | 3 | 3 | finetune.py TrainingConfig | MATCH |
| per_device_batch_size | 2 | 2 | finetune.py TrainingConfig | MATCH |
| gradient_accumulation_steps | 8 | 8 | finetune.py TrainingConfig | MATCH |
| effective_batch_size | 16 | 16 | derived: 2 × 8 | MATCH |
| gradient_checkpointing | True | True | finetune.py | MATCH |
| quantization_type | NF4 | nf4 | finetune.py BitsAndBytesConfig | MATCH |
| 4-bit quantization | True | True | finetune.py BitsAndBytesConfig | MATCH |
| target_modules | k_proj, o_proj, q_proj, v_proj | k_proj, o_proj, q_proj, v_proj | finetune.py line 633 | MATCH |
### LoRA (saved)

| Parameter | Paper | Code | Source | Status |
|-----------|-------|------|--------|--------|
| rank in adapter_config.json | 64 | 64 | results/20260204_201138_COMPARATIVE_RUN/00_lora_training/checkpoints/final/adapter_config.json | MATCH |
| alpha in adapter_config.json | 16 | 16 | results/20260204_201138_COMPARATIVE_RUN/00_lora_training/checkpoints/final/adapter_config.json | MATCH |
| dropout in adapter_config.json | 0.05 | 0.05 | results/20260204_201138_COMPARATIVE_RUN/00_lora_training/checkpoints/final/adapter_config.json | MATCH |
| target_modules in adapter_config.json | k_proj, o_proj, q_proj, v_proj | k_proj, o_proj, q_proj, v_proj | results/20260204_201138_COMPARATIVE_RUN/00_lora_training/checkpoints/final/adapter_config.json | MATCH |
### OPRO-LLM

| Parameter | Paper | Code | Source | Status |
|-----------|-------|------|--------|--------|
| max_iterations | 30 | 30 | opro_llm.py argparse | MATCH |
| candidates_per_iter | 3 | 3 | opro_llm.py argparse | MATCH |
| top_k | 10 | 10 | opro_llm.py argparse | MATCH |
| early_stopping | 5 | 5 | opro_llm.py argparse | MATCH |
| temperature | 0.7 | 0.7 | opro_llm.py argparse | MATCH |
| max_new_tokens | 2000 | 2000 | opro_llm.py argparse | MATCH |
| top_p | 0.9 | 0.9 | opro_llm.py hardcoded in generate() | MATCH |
| meta-LLM | Qwen2.5-7B-Instruct | Qwen/Qwen2.5-7B-Instruct | opro_llm.py OPROClassicOptimizer | MATCH |
| reward λ (ba_cond weight) | 0.25 | 0.25 | opro_llm.py argparse | MATCH |
### OPRO-Tmpl

| Parameter | Paper | Code | Source | Status |
|-----------|-------|------|--------|--------|
| iterations (I) | 15 | 15 | opro_template.py argparse | MATCH |
| candidates/iter (K) | 8 | 8 | opro_template.py argparse | MATCH |
| mini-dev samples (N) | 20 | 20 | opro_template.py argparse | MATCH |
| seed | 42 | 42 | opro_template.py argparse | MATCH |
| templates in library | 15 | 15 | opro_template.py generate_candidate_prompts() | MATCH |
### Eval

| Parameter | Paper | Code | Source | Status |
|-----------|-------|------|--------|--------|
| max_new_tokens (open) | 128 | 128 | qwen_audio.py predict() | MATCH |
| max_new_tokens (constrained A/B) | 1 | 1 | qwen_audio.py predict() — '1 if use_constrained' | MATCH |
| greedy decoding (Qwen2) | True (temp=0) | do_sample=False (greedy) | qwen_audio.py generate() | MATCH |
| greedy decoding (Qwen3) | True (temp=0) | do_sample=False, temperature=0.0 | qwen3_omni.py generate() | MATCH |
| max_new_tokens (Qwen3-Omni) | 128 | 128 | qwen3_omni.py generate() | MATCH |
### Audio

| Parameter | Paper | Code | Source | Status |
|-----------|-------|------|--------|--------|
| sample_rate | 16000 | 16000 | config.yaml + test_metadata.csv sr column | MATCH |
| container_duration_ms | 2000 | 2000 | test_metadata.csv container_duration_ms column | MATCH |
| padding σ | 0.0001 | 0.0001 | main.tex line 378; verified in degradation code | MATCH |
### Data

| Parameter | Paper | Code | Source | Status |
|-----------|-------|------|--------|--------|
| training set size | 3072 | 3072 | data/processed/experimental_variants/train_metadata.csv | MATCH — This is the base clip count. Paper says 3,072 balanced SPEECH/NONSPEECH. |
| dev set size | 660 | 660 | variants_validated_1000/dev_metadata.csv | MATCH |
| dev base clips | 30 | 30 | base_validated_1000/dev_base.csv | MATCH |
| test set size | 21340 | 21340 | variants_validated_1000/test_metadata.csv | MATCH |
| test base clips | 970 | 970 | base_validated_1000/test_base.csv | MATCH |
| Silero speech_ratio threshold | 0.8 | 0.8 | finetune.py argparse/config | MATCH |
### Normalize

| Parameter | Paper | Code | Source | Status |
|-----------|-------|------|--------|--------|
| speech keyword terms | 18 | 18 | normalize.py speech_synonyms list | MATCH |
| nonspeech keyword terms | 29 | 29 | normalize.py nonspeech_synonyms list | MATCH |

## config.yaml Divergences (Informational)

These values in `config.yaml` differ from the actual script defaults and from the paper. They represent early prototype settings that were superseded by the script argparse defaults. They do NOT affect the actual experiment runs.

- **PROTOTYPE_MODE:** true
- config.yaml lora.r=8 (actual runs use 64)
- config.yaml learning_rate=2e-4 (actual runs use 5e-5)
- config.yaml batch_size=4 (actual runs use 2)
- config.yaml gradient_accumulation_steps=4 (actual runs use 8)

**Severity: INFORMATIONAL** — config.yaml is not used by the scripts for these parameters. The scripts use their own argparse defaults.

## Conclusion

All hyperparameters reported in the paper are consistent with the code defaults and the saved LoRA adapter configuration. No blocking discrepancies were found.

The config.yaml file contains stale prototype values (e.g., LoRA r=8 instead of 64, lr=2e-4 instead of 5e-5) but these are overridden by the script argparse defaults and do not affect the actual experiments. Recommendation: either update config.yaml to match or add a clear comment marking it as deprecated/prototype-only.