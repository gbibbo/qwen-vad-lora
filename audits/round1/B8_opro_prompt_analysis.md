# B.8 — OPRO Prompts Extraction & Analysis
**Date:** 2026-02-17

## Overview

- **Total prompt evaluations:** 435
  - OPRO-LLM: 75 (across 3 models)
  - OPRO-Template: 360 (across 3 models)
- **Unique prompts:** 71
- **Categories identified:** 9

## OPRO-LLM: Generated Prompts

### Base
Total candidates evaluated: 28
Unique prompts: 24
Iterations: 10

**Best prompt** (reward=1.1102, BA_clip=0.8864):
> Is this audio human speech? Answer: SPEECH or NON-SPEECH.

| Category | Count | Mean Reward | Min | Max |
|----------|-------|-------------|-----|-----|
| Acoustic focus | 5 | 0.844 | 0.635 | 1.018 |
| Direct question | 3 | 0.825 | 0.625 | 1.088 |
| Binary directive | 10 | 0.803 | 0.644 | 1.110 |
| Open-ended question | 9 | 0.708 | 0.629 | 0.925 |
| Robustness focus | 1 | 0.625 | 0.625 | 0.625 |

### LoRA
Total candidates evaluated: 25
Unique prompts: 22
Iterations: 9

**Best prompt** (reward=1.1705, BA_clip=0.9364):
> Classify this audio as SPEECH or NON-SPEECH, focusing on short and noisy clips.

| Category | Count | Mean Reward | Min | Max |
|----------|-------|-------------|-----|-----|
| Acoustic focus | 7 | 0.959 | 0.628 | 1.171 |
| Binary directive | 8 | 0.858 | 0.625 | 1.162 |
| Robustness focus | 3 | 0.772 | 0.625 | 0.974 |
| Open-ended question | 6 | 0.663 | 0.625 | 0.685 |
| Direct question | 1 | 0.625 | 0.625 | 0.625 |

### Qwen3
Total candidates evaluated: 22
Unique prompts: 19
Iterations: 8

**Best prompt** (reward=1.1941, BA_clip=0.9530):
> What type of sound is this? Respond: SPEECH or NON-SPEECH.

| Category | Count | Mean Reward | Min | Max |
|----------|-------|-------------|-----|-----|
| Open-ended question | 7 | 1.167 | 1.061 | 1.194 |
| Binary directive | 12 | 1.135 | 1.059 | 1.158 |
| Acoustic focus | 2 | 0.983 | 0.825 | 1.140 |
| Direct question | 1 | 0.935 | 0.935 | 0.935 |

## OPRO-Template: Fixed Library Evaluation

### Base
Total evaluations: 120
Unique templates: 15

**Best template** (accuracy=0.85):
> Make a definite decision for the clip.
Output exactly one token: SPEECH or NONSPEECH.

| Template (truncated) | Category | Times | Mean Acc | Max Acc |
|---------------------|----------|-------|----------|---------|
| Binary classification task. Q: Does this contain human speec... | One-shot example | 6 | 0.77 | 0.85 |
| Make a definite decision for the clip. Output exactly one to... | Binary directive | 21 | 0.75 | 0.85 |
| Focus on cues of human vocal tract (formants, syllabic rhyth... | Acoustic focus | 8 | 0.73 | 0.85 |
| Human speech present? Answer: SPEECH or NONSPEECH. | Binary directive | 7 | 0.72 | 0.85 |
| Detect human speech. Treat the following as NONSPEECH: pure ... | Contrastive/definitions | 7 | 0.70 | 0.75 |
| Binary decision. Output exactly one token: SPEECH or NONSPEE... | Binary directive | 6 | 0.67 | 0.75 |
| Does this audio contain human speech? Answer exactly one tok... | Binary directive | 9 | 0.67 | 0.75 |
| You will answer with one token only. <question>Does this aud... | Binary directive | 7 | 0.66 | 0.75 |
| Classify this audio. Output only: SPEECH or NONSPEECH. | Binary directive | 7 | 0.62 | 0.70 |
| TASK: Speech detection. Is human voice/speech present in thi... | Binary directive | 6 | 0.58 | 0.65 |
| Decide the dominant content. Definitions: - SPEECH = human v... | Contrastive/definitions | 8 | 0.53 | 0.55 |
| Example: Audio→ crowd noise, music → Output: NONSPEECH Now c... | One-shot example | 3 | 0.50 | 0.50 |
| If there is any hint of human voice (even faint/short), labe... | Liberal bias | 10 | 0.50 | 0.50 |
| Label SPEECH only if human voice is clearly present; otherwi... | Conservative bias | 9 | 0.50 | 0.50 |
| Listen for human voice. If present: SPEECH. Otherwise: NONSP... | Binary directive | 6 | 0.50 | 0.50 |

### LoRA
Total evaluations: 120
Unique templates: 15

**Best template** (accuracy=1.00):
> Detect human speech. Treat the following as NONSPEECH: pure tones/beeps, clicks, clock ticks, music, environmental noise, silence.
Answer: SPEECH or NONSPEECH.

**Templates with perfect score (1.0):** 4
- "Detect human speech. Treat the following as NONSPEECH: pure tones/beeps, clicks, clock ticks, music,..."
- "You will answer with one token only.
<question>Does this audio contain human speech?</question>
<ans..."
- "Binary classification task.
Q: Does this contain human speech?
If confident YES → SPEECH
If confiden..."
- "Decide the dominant content.
Definitions:
- SPEECH = human voice, spoken words, syllables, conversat..."

| Template (truncated) | Category | Times | Mean Acc | Max Acc |
|---------------------|----------|-------|----------|---------|
| You will answer with one token only. <question>Does this aud... | Binary directive | 7 | 0.96 | 1.00 |
| Detect human speech. Treat the following as NONSPEECH: pure ... | Contrastive/definitions | 21 | 0.93 | 1.00 |
| Binary classification task. Q: Does this contain human speec... | One-shot example | 6 | 0.92 | 1.00 |
| Does this audio contain human speech? Answer exactly one tok... | Binary directive | 9 | 0.91 | 0.95 |
| Human speech present? Answer: SPEECH or NONSPEECH. | Binary directive | 7 | 0.91 | 0.95 |
| Decide the dominant content. Definitions: - SPEECH = human v... | Contrastive/definitions | 8 | 0.88 | 1.00 |
| Focus on cues of human vocal tract (formants, syllabic rhyth... | Acoustic focus | 8 | 0.71 | 0.85 |
| Make a definite decision for the clip. Output exactly one to... | Binary directive | 7 | 0.69 | 0.75 |
| Label SPEECH only if human voice is clearly present; otherwi... | Conservative bias | 9 | 0.65 | 0.75 |
| Binary decision. Output exactly one token: SPEECH or NONSPEE... | Binary directive | 6 | 0.64 | 0.70 |
| Classify this audio. Output only: SPEECH or NONSPEECH. | Binary directive | 7 | 0.62 | 0.70 |
| TASK: Speech detection. Is human voice/speech present in thi... | Binary directive | 6 | 0.56 | 0.65 |
| Listen for human voice. If present: SPEECH. Otherwise: NONSP... | Binary directive | 6 | 0.55 | 0.60 |
| Example: Audio→ crowd noise, music → Output: NONSPEECH Now c... | One-shot example | 3 | 0.53 | 0.55 |
| If there is any hint of human voice (even faint/short), labe... | Liberal bias | 10 | 0.50 | 0.50 |

### Qwen3
Total evaluations: 120
Unique templates: 15

**Best template** (accuracy=1.00):
> Decide the dominant content.
Definitions:
- SPEECH = human voice, spoken words, syllables, conversational cues.
- NONSPEECH = music, tones/beeps, environmental noise, silence.
Output exactly: SPEECH or NONSPEECH.

**Templates with perfect score (1.0):** 1
- "Decide the dominant content.
Definitions:
- SPEECH = human voice, spoken words, syllables, conversat..."

| Template (truncated) | Category | Times | Mean Acc | Max Acc |
|---------------------|----------|-------|----------|---------|
| Decide the dominant content. Definitions: - SPEECH = human v... | Contrastive/definitions | 22 | 0.92 | 1.00 |
| Does this audio contain human speech? Answer exactly one tok... | Binary directive | 9 | 0.90 | 0.95 |
| Listen for human voice. If present: SPEECH. Otherwise: NONSP... | Binary directive | 6 | 0.88 | 0.95 |
| You will answer with one token only. <question>Does this aud... | Binary directive | 7 | 0.87 | 0.95 |
| Classify this audio. Output only: SPEECH or NONSPEECH. | Binary directive | 7 | 0.86 | 0.95 |
| Detect human speech. Treat the following as NONSPEECH: pure ... | Contrastive/definitions | 7 | 0.83 | 0.95 |
| TASK: Speech detection. Is human voice/speech present in thi... | Binary directive | 6 | 0.83 | 0.90 |
| Make a definite decision for the clip. Output exactly one to... | Binary directive | 7 | 0.82 | 0.90 |
| Example: Audio→ crowd noise, music → Output: NONSPEECH Now c... | One-shot example | 3 | 0.80 | 0.85 |
| Binary decision. Output exactly one token: SPEECH or NONSPEE... | Binary directive | 6 | 0.77 | 0.80 |
| Focus on cues of human vocal tract (formants, syllabic rhyth... | Acoustic focus | 8 | 0.76 | 0.90 |
| Label SPEECH only if human voice is clearly present; otherwi... | Conservative bias | 9 | 0.72 | 0.80 |
| Human speech present? Answer: SPEECH or NONSPEECH. | Binary directive | 7 | 0.71 | 0.85 |
| Binary classification task. Q: Does this contain human speec... | One-shot example | 6 | 0.63 | 0.75 |
| If there is any hint of human voice (even faint/short), labe... | Liberal bias | 10 | 0.57 | 0.70 |

## Category Analysis Across All Methods

| Category | Evaluations | Unique Prompts | Mean Score | Std | Min | Max |
|----------|-------------|----------------|------------|-----|-----|-----|
| Contrastive/definitions | 73 | 2 | 0.845 | 0.145 | 0.500 | 1.000 |
| Open-ended question | 22 | 16 | 0.841 | 0.241 | 0.625 | 1.194 |
| Direct question | 5 | 5 | 0.807 | 0.202 | 0.625 | 1.088 |
| Acoustic focus | 38 | 13 | 0.803 | 0.165 | 0.550 | 1.171 |
| Binary directive | 209 | 27 | 0.773 | 0.168 | 0.500 | 1.162 |
| Robustness focus | 4 | 4 | 0.735 | 0.165 | 0.625 | 0.974 |
| One-shot example | 27 | 2 | 0.720 | 0.159 | 0.500 | 1.000 |
| Conservative bias | 27 | 1 | 0.622 | 0.109 | 0.500 | 0.800 |
| Liberal bias | 30 | 1 | 0.523 | 0.054 | 0.500 | 0.700 |

### Variance Assessment

- Overall score range: 0.500 to 1.194
- Range of category means: 0.322
- **Conclusion:** Significant variation between prompt types. A summary table by category would be informative for the paper.
