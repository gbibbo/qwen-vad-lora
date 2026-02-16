# Paper Audit Report — OPRO3 Final

Date: 2026-02-13  
Scope: Cross-check paper claims against code, SLURM logs, checkpoint metadata, and consolidated result artifacts.  
Output: Read-only audit report (no source/result/data modifications).

## Executive Summary

- **Critical (2):**
  - **Audit 1 (Decoding Mode):** Section 5.6 claim about constrained single-token decoding for OPRO-Template is incorrect in final runs.
  - **Audit 2 (Figure CIs):** Figures 1-3 CI bands are normal-approximation CIs, not cluster bootstrap CIs.
- **Confirmed correct (4):** Audits 3, 6, 8, and most of 4.
- **Needs clarification (2):** Audit 5 (where VAD filtering was applied) and one gap in Audit 4 (weight decay not set explicitly in training script).

---

## AUDIT 1: Decoding Mode

### Finding
**Paper claim partially incorrect.**

- Section 4.2.4 claim (“all nine cells use open-ended generation with post-hoc normalization”) matches runtime behavior.
- Section 5.6 claim that OPRO-Template activates constrained single-token decoding for LoRA is incorrect for the evaluated runs.

### Evidence

- `scripts/eval.py` does not expose/pass a `--decoding_mode` argument; inference calls `predict(..., return_scores=True)` with no decoding override: `scripts/eval.py:34`, `scripts/eval.py:45`, `scripts/eval.py:97`.
- Qwen2 classifier defaults to `decoding_mode="auto"`: `src/qsm/models/qwen_audio.py:332`.
- In `auto`, constrained decoding is enabled only when detected format is `ab` or `mc`: `src/qsm/models/qwen_audio.py:431`, `src/qsm/models/qwen_audio.py:436`, `src/qsm/models/qwen_audio.py:437`.
- `detect_format()` returns `labels` when prompt includes `SPEECH` + `NONSPEECH`; `ab`/`mc` require explicit A/B or A–D patterns: `src/qsm/utils/normalize.py:199`, `src/qsm/utils/normalize.py:203`, `src/qsm/utils/normalize.py:209`.
- Generation length is `max_new_tokens=1` only when constrained path is active; otherwise `max_new_tokens=128`: `src/qsm/models/qwen_audio.py:443`.
- Logged eval commands show no `--decoding_mode` flag in matrix/Qwen3 jobs: `logs/matrix_2055498.out:4720`, `logs/matrix_2055498.out:4748`, `logs/matrix_2054058.out:6533`, `logs/qwen3_3B_2055499.out:457`, `logs/qwen3_3C_2055500.out:961`.
- Best prompts used label outputs (not A/B/MC): `results/BEST_CONSOLIDATED/03_qwen2_base_opro_template/optimization/best_prompt.txt:2`, `results/BEST_CONSOLIDATED/06_qwen2_lora_opro_template/optimization/best_prompt.txt:2`, `results/BEST_CONSOLIDATED/09_qwen3_omni_opro_template/optimization/best_prompt.txt:5`.

### Paper Impact
- Correct Section 5.6 wording: decoding was effectively open-ended across evaluated cells; constrained single-token decoding was not activated by prompt format in final eval runs.

---

## AUDIT 2: Figure CI Bands

### Finding
**Paper claim incorrect for Figures 1-3.**

- Paper states 95% cluster bootstrap CIs for psychometric figures.
- Actual psychometric CI bands are normal-approximation CIs (`BA ± 1.96*SE`) computed from per-condition accuracies.

### Evidence

- Psychometric curve CI math implemented as normal approximation: `scripts/plot_final_figures.py:117`, `scripts/plot_final_figures.py:143`, `scripts/plot_final_figures.py:144`, `scripts/plot_final_figures.py:145`.
- Shaded CI bands in psychometric curves come from those values: `scripts/plot_final_figures.py:167`, `scripts/plot_final_figures.py:175`.
- Cluster bootstrap exists in stats code (`n_bootstrap=10000`, 2.5/97.5 percentiles): `scripts/stats.py:250`, `scripts/stats.py:252`, `scripts/stats.py:289`, `scripts/stats.py:290`.
- Bar chart explicitly consumes bootstrap CIs from `statistical_analysis.json` (`ba_clip_ci`), while psychometric curves do not: `scripts/plot_final_figures.py:102`, `scripts/plot_final_figures.py:222`, `scripts/plot_final_figures.py:239`, `scripts/plot_final_figures.py:242`, `results/BEST_CONSOLIDATED/stats/statistical_analysis.json:5`.

### Paper Impact
- Update Figures 1-3 captions/text to “normal approximation CI,” or modify figure generation to use bootstrap CIs for those curves.

---

## AUDIT 3: LoRA Target Modules

### Finding
**Paper claim correct.**

LoRA targets are q/k/v/o projections only in final run artifacts.

### Evidence

- Training script target defaults: `scripts/finetune.py:633`.
- Optional MLP targets are only added with `--add_mlp_targets`: `scripts/finetune.py:634`.
- Matrix training command does not include `--add_mlp_targets`: `scripts/run_matrix.py:174`, `scripts/run_matrix.py:183`.
- Final LoRA log shows target modules `['q_proj', 'v_proj', 'k_proj', 'o_proj']`: `logs/matrix_2055498.out:77`.
- Final adapter config confirms `target_modules` and rank: `results/20260204_201138_COMPARATIVE_RUN/00_lora_training/checkpoints/final/adapter_config.json:28`, `results/20260204_201138_COMPARATIVE_RUN/00_lora_training/checkpoints/final/adapter_config.json:31`.

### Paper Impact
- No correction required.

---

## AUDIT 4: Training Hyperparameters

### Finding
**Mostly correct, with one reporting gap.**

Final LoRA cells in `BEST_CONSOLIDATED` map to run `20260204_201138_COMPARATIVE_RUN`; key paper hyperparameters match. `weight_decay` is not explicitly set in script.

### Evidence

- Consolidated symlinks point LoRA cells to Feb 4 run: `results/BEST_CONSOLIDATED:5`, `results/BEST_CONSOLIDATED:6`, `results/BEST_CONSOLIDATED:7`.
- Final run config in SLURM log: `r=64`, `alpha=16`, epochs `3`, batch `2`, LR `5e-5`, seed `42`: `logs/matrix_2055498.out:52`, `logs/matrix_2055498.out:53`, `logs/matrix_2055498.out:54`, `logs/matrix_2055498.out:55`, `logs/matrix_2055498.out:56`.
- Training samples and class balance: `logs/matrix_2055498.out:63`, `logs/matrix_2055498.out:65`, `logs/matrix_2055498.out:66`.
- Trainable params fraction: `logs/matrix_2055498.out:78`.
- Adapter metadata: `lora_dropout=0.05`, `r=64`, `lora_alpha=16`: `results/20260204_201138_COMPARATIVE_RUN/00_lora_training/checkpoints/final/adapter_config.json:19`, `results/20260204_201138_COMPARATIVE_RUN/00_lora_training/checkpoints/final/adapter_config.json:21`, `results/20260204_201138_COMPARATIVE_RUN/00_lora_training/checkpoints/final/adapter_config.json:28`.
- Effective batch size = `2 * 8 = 16`: `scripts/run_matrix.py:181`, `scripts/run_matrix.py:182`.
- Warmup default is 100 and used in `TrainingArguments`: `scripts/finetune.py:244`, `scripts/finetune.py:662`.
- `TrainingArguments` block does not set `weight_decay`: `scripts/finetune.py:656` through `scripts/finetune.py:674`.
- Earlier run differed (not final): `r=16`, `alpha=32`, `lr=0.0002`: `logs/matrix_2053228.out:24`, `logs/matrix_2053228.out:27`.

### Paper Impact
- Keep final hyperparameter claims, but explicitly state `weight_decay` was not set in script (effectively default behavior).

---

## AUDIT 5: Silero VAD Threshold

### Finding
**Paper claim needs clarification on stage of application.**

Silero threshold values are consistent, but OPRO3 training explicitly skipped in-script VAD filtering (`--skip_vad_filter`).

### Evidence

- Silero threshold default is `0.5`: `src/qsm/vad/silero.py:25`.
- Training-side filter threshold default is `min_speech_ratio=0.8`: `scripts/finetune.py:138`, `scripts/finetune.py:499`, `scripts/finetune.py:501`.
- Matrix pipeline passes `--skip_vad_filter`: `scripts/run_matrix.py:183`.
- Final run log confirms filtering skipped with ratio shown in log: `logs/matrix_2055498.out:60`.
- Final command line includes `--skip_vad_filter`: `logs/matrix_2055498.out:4711`.

### Paper Impact
- Clarify that speech-ratio filtering was enforced during upstream dataset curation (not inside OPRO3 fine-tuning execution), while reporting Silero threshold values used in tooling.

---

## AUDIT 6: OPRO-LLM Convergence

### Finding
**Paper claims correct.**

Iteration counts, earliest-best iteration index, and early-stopping behavior match consolidated histories.

### Evidence

- Base run has 10 iterations (0-9), first best at iteration 4 (`best_reward_per_iteration` jump at index 4): `results/BEST_CONSOLIDATED/02_qwen2_base_opro_llm/optimization/opro_history.json:2`, `results/BEST_CONSOLIDATED/02_qwen2_base_opro_llm/optimization/opro_history.json:62`, `results/BEST_CONSOLIDATED/02_qwen2_base_opro_llm/optimization/opro_history.json:67`.
- LoRA run has 9 iterations (0-8), first best at iteration 3: `results/BEST_CONSOLIDATED/05_qwen2_lora_opro_llm/optimization/opro_history.json:2`, `results/BEST_CONSOLIDATED/05_qwen2_lora_opro_llm/optimization/opro_history.json:56`, `results/BEST_CONSOLIDATED/05_qwen2_lora_opro_llm/optimization/opro_history.json:60`.
- Qwen3 run has 8 iterations (0-7), first best at iteration 2: `results/BEST_CONSOLIDATED/08_qwen3_omni_opro_llm/optimization/opro_history.json:2`, `results/BEST_CONSOLIDATED/08_qwen3_omni_opro_llm/optimization/opro_history.json:50`, `results/BEST_CONSOLIDATED/08_qwen3_omni_opro_llm/optimization/opro_history.json:53`.
- Early-stopping implementation (patience counter, break condition): `scripts/opro_llm.py:558`, `scripts/opro_llm.py:661`, `scripts/opro_llm.py:663`.
- Best prompts match report text: `results/BEST_CONSOLIDATED/02_qwen2_base_opro_llm/optimization/best_prompt.txt:1`, `results/BEST_CONSOLIDATED/05_qwen2_lora_opro_llm/optimization/best_prompt.txt:1`, `results/BEST_CONSOLIDATED/08_qwen3_omni_opro_llm/optimization/best_prompt.txt:1`.

### Paper Impact
- No correction required.

---

## AUDIT 7: Computational Cost

### Finding
**Partial data available; enough for coarse wall-clock estimates, not full per-cell cost table.**

### Evidence

- GPU type in matrix/Qwen3 logs: `NVIDIA A100-SXM4-80GB`: `logs/matrix_2054058.out:9`, `logs/matrix_2055498.out:10`, `logs/qwen3_3B_2054869.out:10`, `logs/qwen3_3C_2054870.out:10`.
- Jan 30 full matrix run start is logged, but no explicit end marker in file: `logs/matrix_2054058.out:2`.
- File modification timestamp indicates last write near `2026-02-01 14:07:26 UTC` for `matrix_2054058.out` (coarse duration estimate from start).
- Feb 4 LoRA matrix run has explicit start/end: `logs/matrix_2055498.out:2`, `logs/matrix_2055498.out:4755` (~13.4h).
- Qwen3 3B run explicit start/end: `logs/qwen3_3B_2054869.out:2`, `logs/qwen3_3B_2054869.out:463` (end `Wed Feb 4 17:21:18 GMT 2026`).
- Qwen3 3C run explicit start/end: `logs/qwen3_3C_2054870.out:2`, `logs/qwen3_3C_2054870.out:971` (end `Wed Feb 4 06:12:25 GMT 2026`).
- Test-set size for final eval workload basis: `21340` samples logged: `logs/matrix_2054058.out:7010`.

### Paper Impact
- Add a dedicated cost table from per-cell SLURM start/end pairs (or scheduler accounting export) before finalizing computational-cost claims.

---

## AUDIT 8: OPRO-Template Parameters

### Finding
**Paper claims correct.**

Final consolidated template runs share the same optimization configuration (`I=15, K=8, N=20, seed=42`), and winning prompts/accuracies match report values.

### Evidence

- Base template config and winner: `results/BEST_CONSOLIDATED/03_qwen2_base_opro_template/optimization/optimization_history.json:2`, `results/BEST_CONSOLIDATED/03_qwen2_base_opro_template/optimization/optimization_history.json:3`, `results/BEST_CONSOLIDATED/03_qwen2_base_opro_template/optimization/optimization_history.json:488`, `results/BEST_CONSOLIDATED/03_qwen2_base_opro_template/optimization/optimization_history.json:489`, `results/BEST_CONSOLIDATED/03_qwen2_base_opro_template/optimization/optimization_history.json:490`, `results/BEST_CONSOLIDATED/03_qwen2_base_opro_template/optimization/optimization_history.json:491`.
- LoRA template config and winner: `results/BEST_CONSOLIDATED/06_qwen2_lora_opro_template/optimization/optimization_history.json:2`, `results/BEST_CONSOLIDATED/06_qwen2_lora_opro_template/optimization/optimization_history.json:3`, `results/BEST_CONSOLIDATED/06_qwen2_lora_opro_template/optimization/optimization_history.json:488`, `results/BEST_CONSOLIDATED/06_qwen2_lora_opro_template/optimization/optimization_history.json:489`, `results/BEST_CONSOLIDATED/06_qwen2_lora_opro_template/optimization/optimization_history.json:490`, `results/BEST_CONSOLIDATED/06_qwen2_lora_opro_template/optimization/optimization_history.json:491`.
- Qwen3 template config and winner: `results/BEST_CONSOLIDATED/09_qwen3_omni_opro_template/optimization/optimization_history.json:2`, `results/BEST_CONSOLIDATED/09_qwen3_omni_opro_template/optimization/optimization_history.json:3`, `results/BEST_CONSOLIDATED/09_qwen3_omni_opro_template/optimization/optimization_history.json:488`, `results/BEST_CONSOLIDATED/09_qwen3_omni_opro_template/optimization/optimization_history.json:489`, `results/BEST_CONSOLIDATED/09_qwen3_omni_opro_template/optimization/optimization_history.json:490`, `results/BEST_CONSOLIDATED/09_qwen3_omni_opro_template/optimization/optimization_history.json:491`.
- Winning prompt texts: `results/BEST_CONSOLIDATED/03_qwen2_base_opro_template/optimization/best_prompt.txt:1`, `results/BEST_CONSOLIDATED/03_qwen2_base_opro_template/optimization/best_prompt.txt:2`, `results/BEST_CONSOLIDATED/06_qwen2_lora_opro_template/optimization/best_prompt.txt:1`, `results/BEST_CONSOLIDATED/06_qwen2_lora_opro_template/optimization/best_prompt.txt:2`, `results/BEST_CONSOLIDATED/09_qwen3_omni_opro_template/optimization/best_prompt.txt:1`, `results/BEST_CONSOLIDATED/09_qwen3_omni_opro_template/optimization/best_prompt.txt:5`.

### Paper Impact
- No correction required.

---

## Summary Hyperparameter Table (Final LoRA Run)

| Item | Value | Evidence |
|---|---:|---|
| LoRA rank (`r`) | 64 | `logs/matrix_2055498.out:52`, `results/20260204_201138_COMPARATIVE_RUN/00_lora_training/checkpoints/final/adapter_config.json:28` |
| LoRA alpha | 16 | `logs/matrix_2055498.out:52`, `results/20260204_201138_COMPARATIVE_RUN/00_lora_training/checkpoints/final/adapter_config.json:19` |
| LoRA dropout | 0.05 | `results/20260204_201138_COMPARATIVE_RUN/00_lora_training/checkpoints/final/adapter_config.json:21` |
| Target modules | q_proj, k_proj, v_proj, o_proj | `logs/matrix_2055498.out:77`, `scripts/finetune.py:633` |
| Epochs | 3 | `logs/matrix_2055498.out:53` |
| Batch size (device) | 2 | `logs/matrix_2055498.out:54` |
| Grad accumulation | 8 | `scripts/run_matrix.py:182` |
| Effective batch size | 16 | `scripts/run_matrix.py:181`, `scripts/run_matrix.py:182` |
| Learning rate | 5e-5 | `logs/matrix_2055498.out:55` |
| Warmup steps | 100 | `scripts/finetune.py:244`, `scripts/finetune.py:662` |
| Weight decay | Not explicitly set | `scripts/finetune.py:656` through `scripts/finetune.py:674` |
| Seed | 42 | `logs/matrix_2055498.out:56` |
| Train samples | 3,072 (1,536/1,536) | `logs/matrix_2055498.out:63`, `logs/matrix_2055498.out:65`, `logs/matrix_2055498.out:66` |
| Trainable parameters | 82,837,504 / 8,479,932,416 (0.9769%) | `logs/matrix_2055498.out:78` |
| VAD behavior in OPRO3 training | skipped (`--skip_vad_filter`) | `scripts/run_matrix.py:183`, `logs/matrix_2055498.out:60`, `logs/matrix_2055498.out:4711` |

---

## Notes on Open Items

1. **Decoding wording mismatch:** Correct Section 5.6 to avoid describing constrained single-token decoding as active in final evaluated runs.
2. **CI wording mismatch:** Align figure captions for psychometric curves with actual normal-approximation method, or change plotting implementation.
3. **VAD provenance clarity:** Distinguish upstream dataset curation filtering from in-training filtering.
4. **Weight decay reporting:** Add explicit statement in methods/hyperparameters.

