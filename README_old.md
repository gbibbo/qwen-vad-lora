# Audio-Language Models under Psychometric Degradations: Optimization for Voice Activity Detection

**Authors:** Gabriel Bibbó, Mark D. Plumbley, Simone Spagnol

Can Large Audio-Language Models (LALMs) reliably detect speech in degraded audio? We evaluate how model architecture interacts with optimization strategies for Voice Activity Detection (VAD) under controlled psychometric degradations — and find that **a fine-tuned 7B model outperforms a frozen 30B model**.

---

## Overview

We test a **3 × 3 experimental matrix** crossing three model configurations with three prompting strategies, evaluated on **21,340 samples** under 22 psychometric degradation conditions spanning four acoustic axes.

| Model Config | (A) Baseline | (B) OPRO-LLM | (C) OPRO-Template |
|:---|:---|:---|:---|
| **Qwen2-Audio 7B (Base)** | Direct Eval | Opt → Eval | Opt → Eval |
| **Qwen2-Audio 7B (LoRA)** | Train → Eval | Opt → Eval | Opt → Eval |
| **Qwen3-Omni 30B (Frozen MoE)** | Direct Eval | Opt → Eval | Opt → Eval |

**Degradation bank (22 conditions per sample):**

| Axis | Conditions |
|:---|:---|
| Segment duration | 20, 40, 60, 80, 100, 200, 500, 1000 ms (8) |
| SNR | −10, −5, 0, +5, +10, +20 dB (6) |
| Reverberation (RT60) | 0.0, 0.3, 1.0, 2.5 s (4) |
| Spectral filtering | None, bandpass, lowpass, highpass (4) |

---

## Key Results

### Overall Performance

A clear **adaptation hierarchy** emerges: LoRA + OPRO > Qwen3-Omni (Frozen) > LoRA > OPRO > Base.

| Model | Configuration | BA_clip [95% CI] | Recall_SPEECH | Recall_NONSPEECH |
|:---|:---|:---:|:---:|:---:|
| Qwen2-Audio-7B | Base + Hand | 0.640 [0.626, 0.654] | 0.321 | 0.959 |
| Qwen2-Audio-7B | Base + OPRO-LLM | 0.826 [0.814, 0.838] | 0.747 | 0.906 |
| Qwen2-Audio-7B | LoRA + Hand | 0.864 [0.852, 0.875] | 0.824 | 0.903 |
| Qwen2-Audio-7B | **LoRA + OPRO-Tmpl** | **0.933 [0.925, 0.940]** | **0.928** | **0.938** |
| Qwen3-Omni-30B | Frozen + Hand | 0.911 [0.904, 0.918] | 0.874 | 0.947 |
| Qwen3-Omni-30B | Frozen + OPRO-LLM | 0.914 [0.906, 0.921] | 0.892 | 0.935 |

> A **7B dense model** with LoRA + OPRO (**93.3% BA**) surpasses a **frozen 30B MoE model** (**91.1% BA**) by 2.2 pp — statistically significant (McNemar, p < 10⁻¹⁰). The 7B model runs on consumer RTX 3090 with 4-bit quantization; the 30B model requires A100 80GB unquantized.

### Temporal Resolution (Duration Curves)

![Duration curves](figures/Fig_Duration.png)

LoRA + OPRO achieves a **DT90 of 96 ms** [88, 133] — the minimum segment duration for 90% balanced accuracy. This approaches human temporal integration limits (~75–100 ms). The unoptimized baseline never reaches 90% at any tested duration.

| Configuration | DT90 (ms) | SNR75 (dB) |
|:---|:---:|:---:|
| Baseline | >1000 | >+20 |
| Base + OPRO | >1000 | <−10 |
| LoRA + Hand | 329 [87, 1000] | <−10 |
| **LoRA + OPRO** | **96 [88, 133]** | **<−10** |
| Qwen3-Omni (Frozen) | 175 [146, 222] | <−10 |

### Noise Robustness (SNR Curves)

![SNR curves](figures/Fig_SNR.png)

LoRA fine-tuning renders the model **nearly invariant to additive noise** (SNR75 below −10 dB across all LoRA configurations). Prompt optimization alone cannot replicate this robustness.

### Reverberation Robustness

![Reverberation curves](figures/Fig_Reverb.pdf)

Adapted models maintain stable performance across all reverberation conditions (RT60 from 0 to 2.5 s). The baseline shows marked sensitivity.

### Sensitivity–Specificity Trade-off

![Recall trade-off](figures/Fig_Tradeoff.png)

Three operating regimes emerge:
- **Conservative baseline** (Qwen2-Base): very high specificity, very low sensitivity — defaults to NONSPEECH under uncertainty.
- **Sensitivity-recovered** (OPRO): recovers speech recall but introduces more false alarms.
- **Balanced** (LoRA + OPRO): maximizes both recalls simultaneously, closest to ideal upper-right corner.

### Per-Axis Breakdown (Mean Balanced Accuracy %)

| Configuration | Duration | SNR | Reverb | Filter |
|:---|:---:|:---:|:---:|:---:|
| Base + Hand | 65.9 | 62.8 | 64.0 | 62.1 |
| Base + OPRO-LLM | 82.5 | 86.0 | 81.8 | 78.7 |
| LoRA + OPRO-Tmpl | **87.4** | 97.4 | 96.1 | 96.2 |
| Qwen3-Omni + Hand | 79.8 | **98.5** | **97.0** | 96.7 |

### Prompt Optimization Insight

A striking **interaction between adaptation and prompt style**:
- **Frozen models** prefer natural-language prompts (OPRO-LLM)
- **Fine-tuned models** prefer structured templates (OPRO-Template)

Applying OPRO-LLM to the LoRA model *degrades* performance (84.0%) below the unoptimized LoRA baseline (86.4%), while OPRO-Template yields the best result (93.3%). Prompt strategy must match the model's training history.

---

## Practical Recommendations

1. **Robustness-sensitive deployment** (forensic audio, hearing aids, industrial monitoring): use LoRA + OPRO-Template on the 7B model — highest robustness, lowest inference cost.
2. **Rapid prototyping** (no labeled data available): use a frozen MoE model with OPRO-LLM — strong baseline with only a small dev set for prompt search.
3. **OPRO as diagnostic**: the magnitude of OPRO gain signals how much failure is due to instruction misalignment (large gain = recoverable) vs. representational limits (small gain = weight adaptation needed).
4. **Match prompt style to training**: generative search for frozen models, template search for fine-tuned models.

---

## Repository Structure

```
opro3_final/
├── data/                     # Symlink to preprocessed audio data (read-only)
├── scripts/
│   ├── run_matrix.py         # Orchestrator: runs the full 3×3 experiment
│   ├── eval.py               # Universal evaluator (Base, LoRA, Qwen3)
│   ├── finetune.py           # LoRA training (Qwen2-Audio only)
│   ├── opro_llm.py           # Method B: generative prompt optimization
│   ├── opro_template.py      # Method C: deterministic template search
│   ├── stats.py              # Bootstrap CIs & McNemar tests
│   ├── make_tables.py        # Generate LaTeX tables from results
│   └── plot_final_figures.py # Generate publication figures
├── src/qsm/                  # Core library (models, normalization, etc.)
├── slurm/
│   ├── templates/            # Slurm job templates
│   ├── tools/                # Slurm wrapper (on_submit.sh)
│   └── jobs/                 # Submitted job files
├── results/                  # Timestamped output directories
├── tables/                   # Generated LaTeX tables
├── figures/                  # Generated figures (PDF + PNG)
├── main.tex                  # Paper manuscript
└── CLAUDE.md                 # Development instructions
```

---

## Reproducing from Scratch

### 1. Clone and set up the environment

```bash
git clone <repo-url> opro3_final
cd opro3_final
```

### 2. Data setup

The data directory should be a symlink to the preprocessed audio data from the `opro2_clean` pipeline. If running locally, you need access to the preprocessed VoxConverse + ESC-50 data:

```bash
# Verify the data symlink exists and is valid
ls -la data/
# Should show: data -> ../opro2_clean/data

# If the symlink is broken, re-create it pointing to your data directory:
# ln -s /path/to/preprocessed/data data
```

The data directory contains:
- `processed/experimental_variants/` — LoRA training and dev splits (3,072 + 3,456 samples)
- `processed/variants_validated_1000/` — OPRO dev set (660 samples) and test set (21,340 samples)
- Source audio from VoxConverse (speech) and ESC-50 (non-speech), resampled to 16 kHz mono

### 3. Install dependencies

**For Qwen2-Audio (Base + LoRA):**

```bash
pip install torch transformers peft bitsandbytes accelerate
pip install pandas numpy scipy tqdm scikit-learn soundfile librosa
pip install matplotlib seaborn  # for figures
```

**For Qwen3-Omni (requires dev transformers):**

```bash
pip install git+https://github.com/huggingface/transformers.git
```

### 4. Run the full 3×3 matrix (orchestrator)

The orchestrator manages the entire pipeline — LoRA training, OPRO optimization, and evaluation for all 9 cells:

```bash
# Dry run (shows what would execute without running anything)
python scripts/run_matrix.py --dry_run

# Run all 9 cells
python scripts/run_matrix.py --cells all

# Run specific cells (e.g., only Qwen2-Base baseline and LoRA cells)
python scripts/run_matrix.py --cells 1A,2A,2B,2C
```

Output is written to `results/<TIMESTAMP>_COMPARATIVE_RUN/` with subfolders for each cell.

### 5. Run individual pipeline stages

If you prefer manual control, you can run each stage independently:

**a) LoRA fine-tuning (Qwen2-Audio only):**

```bash
python scripts/finetune.py \
    --train_csv data/processed/experimental_variants/train_metadata.csv \
    --val_csv data/processed/experimental_variants/dev_metadata.csv \
    --output_dir results/lora_training/checkpoints \
    --seed 42
```

**b) OPRO-LLM optimization:**

```bash
python scripts/opro_llm.py \
    --manifest data/processed/variants_validated_1000/dev_metadata.csv \
    --output_dir results/opro_llm_base/ \
    --model_type qwen2
```

**c) OPRO-Template optimization:**

```bash
python scripts/opro_template.py \
    --manifest data/processed/variants_validated_1000/dev_metadata.csv \
    --output_dir results/opro_template_base/ \
    --model_type qwen2
```

**d) Evaluation (universal for all models):**

```bash
# Qwen2-Audio base with a specific prompt
python scripts/eval.py \
    --manifest data/processed/variants_validated_1000/test_metadata.csv \
    --prompt "Is this audio human speech? Answer: SPEECH or NON-SPEECH." \
    --output_dir results/eval_base_opro/ \
    --model_type qwen2

# Qwen2-Audio with LoRA checkpoint
python scripts/eval.py \
    --manifest data/processed/variants_validated_1000/test_metadata.csv \
    --prompt "Detect human speech. Answer: SPEECH or NONSPEECH." \
    --output_dir results/eval_lora/ \
    --checkpoint results/lora_training/checkpoints/final \
    --model_type qwen2

# Qwen3-Omni (frozen)
python scripts/eval.py \
    --manifest data/processed/variants_validated_1000/test_metadata.csv \
    --prompt "What type of sound is this? Respond: SPEECH or NON-SPEECH." \
    --output_dir results/eval_qwen3/ \
    --model_type qwen3_omni
```

### 6. Statistical analysis and figures

```bash
# Run bootstrap CIs and McNemar tests on the consolidated results
python scripts/stats.py --results_dir results/<TIMESTAMP>_COMPARATIVE_RUN/

# Generate LaTeX tables
python scripts/make_tables.py

# Generate publication figures
python scripts/plot_final_figures.py
```

### 7. Slurm (HPC only)

On the Surrey HPC cluster, Slurm commands are not in PATH. Use the provided wrapper:

```bash
# Submit a job
./slurm/tools/on_submit.sh sbatch slurm/jobs/<job_file>.job

# Check queue
./slurm/tools/on_submit.sh squeue --me

# Cancel a job
./slurm/tools/on_submit.sh scancel <job_id>
```

### Hardware Requirements

| Model | GPU | Quantization | VRAM |
|:---|:---|:---|:---|
| Qwen2-Audio-7B (Base/LoRA) | NVIDIA RTX 3090 | 4-bit NF4 | ~24 GB |
| Qwen3-Omni-30B (Frozen) | NVIDIA A100 | None (fp16) | ~80 GB |

Total wall-clock time for the full Qwen2-Audio matrix (LoRA training + OPRO + 6-cell evaluation): ~13.4 hours on a single A100-SXM4-80GB.

---

## Citation

```bibtex
@article{bibbo2025audio,
  title={Audio-Language Models under Psychometric Degradations: Optimization for Voice Activity Detection},
  author={Bibb{\'o}, Gabriel and Plumbley, Mark D.},
  year={2025}
}
```
