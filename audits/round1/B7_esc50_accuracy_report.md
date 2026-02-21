# B.7 — ESC-50 Per-Category NONSPEECH Accuracy
**Date:** 2026-02-17

## Overview

NONSPEECH accuracy broken down by ESC-50 category for all 9 configurations.
Each category has 22 variants per base clip in the test set.

## Full Results by Acoustic Group

### Human vocalizations

| Category | Base Hand | Base OPRO-LLM | Base OPRO-Tmpl | LoRA Hand | LoRA OPRO-LLM | LoRA OPRO-Tmpl | Qwen3 Hand | Qwen3 OPRO-LLM | Qwen3 OPRO-Tmpl | Mean |
|----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|------|
| laughing             | **52.8% ** | **54.5% ** | **14.8% ** | **35.2% ** | **48.9% ** | **31.8% ** | **62.5% ** | **64.8% ** | **30.1% ** | 43.9% |
| coughing             | **54.5% ** | **60.6% ** | **43.9% ** | **35.9% ** | **23.2% ** | **56.6% ** | 90.9% | 94.4% | **47.5% ** | 56.4% |
| crying_baby          | **75.3% ** | 83.8% | **39.4% ** | **69.7% ** | **59.1% ** | **77.3% ** | **54.5% ** | **46.0% ** | **38.9% ** | 60.4% |
| sneezing             | 92.7% | **72.3% ** | 84.1% | **76.4% ** | **26.8% ** | 86.8% | 85.9% | 87.7% | **70.9% ** | 76.0% |
| drinking_sipping     | 100.0% | 81.8% | 89.1% | 84.5% | **17.3% ** | 97.3% | 100.0% | 100.0% | 87.3% | 84.1% |
| clapping             | 99.1% | 90.9% | 83.6% | 90.9% | 80.9% | 84.5% | **79.1% ** | 90.9% | 84.5% | 87.2% |
| breathing            | 96.2% | 85.0% | 91.3% | 90.6% | **67.5% ** | 97.2% | 98.3% | 95.5% | **64.3% ** | 87.3% |
| snoring              | 97.5% | 94.2% | **78.1% ** | 90.1% | **56.6% ** | 95.9% | 99.6% | 97.5% | 86.8% | 88.5% |
| brushing_teeth       | 100.0% | 97.2% | **75.6% ** | 98.9% | 91.5% | 99.4% | 100.0% | 97.7% | 90.3% | 94.5% |
| footsteps            | 100.0% | 99.6% | 99.2% | 100.0% | 81.4% | 100.0% | 100.0% | 100.0% | 98.9% | 97.7% |

### Animal vocalizations

| Category | Base Hand | Base OPRO-LLM | Base OPRO-Tmpl | LoRA Hand | LoRA OPRO-LLM | LoRA OPRO-Tmpl | Qwen3 Hand | Qwen3 OPRO-LLM | Qwen3 OPRO-Tmpl | Mean |
|----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|------|
| sheep                | 96.4% | 84.2% | **55.2% ** | 82.4% | **70.9% ** | 88.2% | **77.3% ** | **78.5% ** | **48.2% ** | 75.7% |
| cow                  | 91.9% | **77.6% ** | **64.0% ** | 80.2% | **52.9% ** | 87.7% | 93.2% | 90.6% | 80.8% | 79.9% |
| hen                  | 88.0% | 93.8% | **55.4% ** | 84.3% | 81.4% | 86.4% | 86.0% | **79.8% ** | **64.9% ** | 80.0% |
| dog                  | 95.0% | **77.7% ** | **55.0% ** | 82.3% | **45.5% ** | 82.7% | 97.3% | 97.3% | 89.1% | 80.2% |
| cat                  | 92.0% | 85.6% | **77.0% ** | 85.6% | **59.4% ** | 88.8% | 84.5% | 81.8% | **68.7% ** | 80.4% |
| pig                  | 96.8% | 87.7% | **62.3% ** | 84.1% | **64.1% ** | 95.0% | 93.6% | 91.8% | **79.5% ** | 83.9% |
| crow                 | 94.9% | 94.4% | **79.8% ** | 92.9% | 82.3% | 97.0% | 82.3% | **72.2% ** | **61.6% ** | 84.2% |
| rooster              | 94.3% | 89.8% | **68.9% ** | 91.3% | **76.1% ** | 96.2% | 87.5% | 82.6% | 80.3% | 85.2% |
| frog                 | 100.0% | 95.5% | 83.6% | 92.7% | **70.9% ** | 98.2% | 90.9% | 88.2% | **76.4% ** | 88.5% |
| insects              | 98.7% | 89.0% | 84.4% | 89.6% | **74.7% ** | 92.9% | 96.8% | 94.2% | 88.3% | 89.8% |

### Mechanical/domestic

| Category | Base Hand | Base OPRO-LLM | Base OPRO-Tmpl | LoRA Hand | LoRA OPRO-LLM | LoRA OPRO-Tmpl | Qwen3 Hand | Qwen3 OPRO-LLM | Qwen3 OPRO-Tmpl | Mean |
|----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|------|
| glass_breaking       | 100.0% | **77.3% ** | 94.5% | 85.5% | **40.5% ** | 91.4% | 95.5% | 98.6% | 90.9% | 86.0% |
| door_knock           | 100.0% | **76.9% ** | 96.2% | 83.7% | **34.5% ** | 95.8% | 100.0% | 100.0% | 92.8% | 86.7% |
| mouse_click          | 100.0% | 92.9% | 98.7% | 94.8% | **64.3% ** | 97.4% | 100.0% | 100.0% | 95.5% | 93.7% |
| can_opening          | 100.0% | 98.5% | 92.4% | 99.0% | **67.2% ** | 99.0% | 98.5% | 99.0% | 95.5% | 94.3% |
| clock_alarm          | 99.7% | 98.7% | **61.0% ** | 99.4% | 94.2% | 97.4% | 100.0% | 100.0% | 99.7% | 94.4% |
| washing_machine      | 99.6% | 93.8% | 82.6% | 97.5% | 88.4% | 99.6% | 98.8% | 97.9% | 93.8% | 94.7% |
| door_wood_creaks     | 99.2% | 94.7% | **77.3% ** | 93.2% | 88.6% | 100.0% | 100.0% | 100.0% | 100.0% | 94.8% |
| clock_tick           | 99.6% | 97.1% | 94.2% | 97.1% | **75.2% ** | 98.3% | 100.0% | 100.0% | 97.5% | 95.5% |
| keyboard             | 100.0% | 95.5% | 97.0% | 97.3% | **78.4% ** | 99.2% | 100.0% | 100.0% | 98.5% | 96.2% |
| vacuum_cleaner       | 100.0% | 99.2% | **79.3% ** | 99.6% | 97.9% | 100.0% | 100.0% | 98.3% | 98.3% | 97.0% |

### Natural/ambient

| Category | Base Hand | Base OPRO-LLM | Base OPRO-Tmpl | LoRA Hand | LoRA OPRO-LLM | LoRA OPRO-Tmpl | Qwen3 Hand | Qwen3 OPRO-LLM | Qwen3 OPRO-Tmpl | Mean |
|----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|------|
| toilet_flush         | 98.5% | 92.9% | **73.7% ** | 93.4% | **78.8% ** | 98.0% | 99.5% | 99.0% | 87.9% | 91.3% |
| crickets             | 97.7% | 96.6% | **61.9% ** | 97.2% | 89.2% | 90.3% | 100.0% | 97.2% | 96.6% | 91.9% |
| water_drops          | 100.0% | 93.6% | 94.5% | 94.5% | **60.9% ** | 97.7% | 100.0% | 100.0% | 95.9% | 93.0% |
| wind                 | 100.0% | 97.3% | 90.5% | 98.2% | 91.8% | 98.2% | 100.0% | 98.2% | 90.9% | 96.1% |
| pouring_water        | 100.0% | 98.7% | 96.8% | 98.1% | 82.5% | 98.7% | 100.0% | 100.0% | 98.1% | 97.0% |
| thunderstorm         | 100.0% | 98.1% | 96.8% | 98.7% | 85.7% | 99.4% | 100.0% | 100.0% | 96.8% | 97.3% |
| crackling_fire       | 100.0% | 98.1% | 96.2% | 99.2% | 93.6% | 99.2% | 100.0% | 99.2% | 92.8% | 97.6% |
| sea_waves            | 99.6% | 99.6% | 92.6% | 100.0% | 93.8% | 100.0% | 100.0% | 99.2% | 94.6% | 97.7% |
| chirping_birds       | 100.0% | 99.6% | 91.7% | 99.2% | 91.3% | 100.0% | 99.6% | 100.0% | 97.9% | 97.7% |
| rain                 | 100.0% | 100.0% | 96.2% | 100.0% | 99.2% | 100.0% | 100.0% | 100.0% | 98.5% | 99.3% |

### Machinery/transport

| Category | Base Hand | Base OPRO-LLM | Base OPRO-Tmpl | LoRA Hand | LoRA OPRO-LLM | LoRA OPRO-Tmpl | Qwen3 Hand | Qwen3 OPRO-LLM | Qwen3 OPRO-Tmpl | Mean |
|----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|------|
| chainsaw             | 98.3% | 89.8% | **43.2% ** | 86.4% | 88.6% | 95.5% | 98.3% | 96.6% | 92.6% | 87.7% |
| car_horn             | 94.9% | 91.4% | **59.6% ** | 91.9% | **74.7% ** | 97.5% | 98.0% | 95.5% | 91.4% | 88.3% |
| siren                | 98.3% | 94.8% | **66.4% ** | 95.8% | **78.3% ** | 98.3% | 97.9% | 92.0% | 97.6% | 91.0% |
| hand_saw             | 99.0% | 92.9% | 86.9% | 91.4% | **74.7% ** | 96.5% | 99.0% | 98.5% | 90.4% | 92.1% |
| fireworks            | 100.0% | 92.7% | 91.4% | 98.6% | **67.7% ** | 100.0% | 100.0% | 100.0% | 99.5% | 94.4% |
| church_bells         | 99.0% | 95.5% | 88.9% | 97.5% | 90.4% | 98.0% | 98.5% | 93.4% | 95.5% | 95.2% |
| engine               | 100.0% | 96.6% | 88.1% | 99.4% | 88.6% | 99.4% | 100.0% | 99.4% | 93.8% | 96.1% |
| helicopter           | 100.0% | 95.5% | 89.8% | 97.0% | 89.0% | 99.6% | 100.0% | 99.6% | 97.3% | 96.4% |
| airplane             | 100.0% | 97.7% | 90.2% | 97.0% | 95.5% | 100.0% | 100.0% | 100.0% | 94.7% | 97.2% |
| train                | 98.1% | 98.7% | 92.2% | 99.4% | 96.8% | 100.0% | 100.0% | 100.0% | 95.5% | 97.8% |

## Top 10 Hardest Categories (by mean accuracy)

| Rank | Category | Group | Mean Acc | Hardest Config | Easiest Config |
|------|----------|-------|----------|----------------|----------------|
| 1 | laughing | Human vocalizations | 43.9% | Base+OPRO-Tmpl (14.8%) | Qwen3+OPRO-LLM (64.8%) |
| 2 | coughing | Human vocalizations | 56.4% | LoRA+OPRO-LLM (23.2%) | Qwen3+OPRO-LLM (94.4%) |
| 3 | crying_baby | Human vocalizations | 60.4% | Qwen3+OPRO-Tmpl (38.9%) | Base+OPRO-LLM (83.8%) |
| 4 | sheep | Animal vocalizations | 75.7% | Qwen3+OPRO-Tmpl (48.2%) | Base+Hand (96.4%) |
| 5 | sneezing | Human vocalizations | 76.0% | LoRA+OPRO-LLM (26.8%) | Base+Hand (92.7%) |
| 6 | cow | Animal vocalizations | 79.9% | LoRA+OPRO-LLM (52.9%) | Qwen3+Hand (93.2%) |
| 7 | hen | Animal vocalizations | 80.0% | Base+OPRO-Tmpl (55.4%) | Base+OPRO-LLM (93.8%) |
| 8 | dog | Animal vocalizations | 80.2% | LoRA+OPRO-LLM (45.5%) | Qwen3+Hand (97.3%) |
| 9 | cat | Animal vocalizations | 80.4% | LoRA+OPRO-LLM (59.4%) | Base+Hand (92.0%) |
| 10 | pig | Animal vocalizations | 83.9% | Base+OPRO-Tmpl (62.3%) | Base+Hand (96.8%) |

## Verification of Paper Claims (Section 5.8)

Paper cites three hardest categories. The values match **LoRA+OPRO-Tmpl** exactly:

- **laughing:** paper=31.8%, LoRA+OPRO-Tmpl=31.8% [MATCH]
- **coughing:** paper=56.6%, LoRA+OPRO-Tmpl=56.6% [MATCH]
- **crying_baby:** paper=77.3%, LoRA+OPRO-Tmpl=77.3% [MATCH]

**Note:** The paper's Section 5.8 does not specify which configuration these values come from.
The script initially compared against Base+Hand (which gives different values: 52.8%, 54.5%, 75.3%).
Correct source is LoRA+OPRO-Tmpl, confirmed by exact match on all three categories.

## Group-Level Summary

| Group | Mean Acc (all configs) | Hardest Category | Easiest Category |
|-------|-----------------------|------------------|------------------|
| Human vocalizations | 77.6% | laughing (43.9%) | footsteps (97.7%) |
| Animal vocalizations | 82.8% | sheep (75.7%) | insects (89.8%) |
| Mechanical/domestic | 93.3% | glass_breaking (86.0%) | vacuum_cleaner (97.0%) |
| Natural/ambient | 95.9% | toilet_flush (91.3%) | rain (99.3%) |
| Machinery/transport | 93.6% | chainsaw (87.7%) | train (97.8%) |
