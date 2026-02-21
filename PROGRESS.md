# Progreso del Plan de Mejoras — Paper LALM VAD
## Archivo de referencia: paper_improvement_plan.md

---

## Registro de tareas

| Fecha | Tarea ID | Estado | Notas |
|-------|----------|--------|-------|
| 2026-02-20 | V.1-V.4 (PASO 9) | COMPLETADO | V.1: pyannote TODO confirmed (only in comment). V.2: No non-canonical names remain in .tex files. V.3: All new tables/figures (normalization_levels, multiseed_opro, esc50_heatmap) properly labeled and cross-referenced. V.4: All em-dashes (---) removed from main.tex (0 remaining). Full consistency pass complete. |
| 2026-02-20 | C.1-C.6 (PASO 8) | COMPLETADO | C.1: Silero in Table 2 + Table 4 + context. C.2: Normalization levels table. C.3: Cluster p-values in Table 5. C.4: Multi-seed OPRO table. C.5: Extended SNR thresholds + narrative + plot script. C.6: ESC-50 heatmap script + figure. Silero and seed limitations updated. |
| 2026-02-20 | A.6+A.15+A.18 (PASO 7) | COMPLETADO | DT90/SNR75 operational bounds disclaimer (A.6); "adaptation > scaling" claims hedged in Sec 5.1, 6, 7 (A.15); Qwen3+OPRO-LLM cluster-aware p=0.093 non-significance added in Sec 5.1 + 5.3 (A.18). Em-dashes fixed in Conclusion. |
| 2026-02-20 | A.7+A.8+A.9+A.10+A.14 (PASO 6) | COMPLETADO | 5 párrafos nuevos: baseline reframe (A.7), OPRO caveat (A.8), probing experiment (A.9), combined degradations (A.10), dataset defense (A.14). Limitations renumeradas (9 items + Additionally). |
| 2026-02-20 | A.5+A.11+A.12 (PASO 5) | COMPLETADO | Abstract simplificado; σ=10⁻⁴ justificado; λ=0.25 justificado. Em-dashes eliminados en abstract. |
| 2026-02-20 | A.2+A.3+A.4 (PASO 4) | COMPLETADO | Sec 5.9 reducida a transición; Sec 6.4 Qwen3 consolidada; Sec 2.4.1 párrafos separados. |
| 2026-02-20 | A.16+A.17 (PASO 3) | COMPLETADO | "seven"→"eight"; 320 responses corregido: eran UNKNOWN (texto chino), no "lower-level normalization". |
| 2026-02-20 | A.13 (PASO 2) | COMPLETADO | pyannote eliminado de 3 ubicaciones (lines 333, 338, 647). refs.bib no existe; dejado TODO para verificar. |
| 2026-02-20 | A.1 (PASO 1) | COMPLETADO | Nomenclatura unificada en main.tex, 5 tablas externas, plot script. 9 nombres canónicos aplicados. |
| 2026-02-17 | B.5 | COMPLETADO | 48/48 parámetros verificados. 0 discrepancias. |
| 2026-02-17 | B.2 | COMPLETADO | Parse rates 9/9 configs OK. Level breakdown completo para test set (3 configs re-evaluadas con --log_raw_text). 2 discrepancias menores con paper confirmadas. |
| 2026-02-17 | B.7 | COMPLETADO | 50 categorías × 9 configs extraídas. Top 3 difíciles: laughing, coughing, crying_baby. Valores del paper verificados (coinciden con LoRA+OPRO-Tmpl). |
| 2026-02-17 | B.8 | COMPLETADO | 435 evaluaciones de prompts extraídas (75 OPRO-LLM + 360 OPRO-Tmpl). 71 prompts únicos clasificados en 9 categorías funcionales. |
| 2026-02-17 | B.4 (parcial) | COMPLETADO | BA × SNR tabla extraída para 9 configs. 7/9 configs >90% BA a −10 dB. Extension a −15/−20 dB justificada. |
| 2026-02-17 | B.3 | COMPLETADO | Cluster-aware McNemar: Qwen3+Hand vs Qwen3+OPRO-LLM pasa de p=0.014 a p=0.093 (no significativo). Ver detalle abajo. |
| 2026-02-18 | B.2 (Ronda 2) | COMPLETADO | SLURM 2061638 (19h 21m). 3 configs re-evaluadas. 100% reproducibilidad. Level breakdown completo. 320 UNKNOWN en Base+OPRO-LLM son texto chino ("人类的语音。"). |
| 2026-02-18 | B.4 (Ronda 2) | COMPLETADO | SLURM 2061637 (2h 31m). 3 configs evaluadas a −15/−20 dB. Tabla combinada −20 a +20 dB generada. Solo LoRA+OPRO-Tmpl sobrevive a −15 dB (83.3% BA). |
| 2026-02-18 | B.6 | COMPLETADO | SLURM 2061636 (25m 53s). Silero VAD max-frame: BA=88.9% [88.4, 89.5]. Speech-ratio inadecuado (54.9% BA). Usar max-frame como resultado principal. |
| 2026-02-19 | B.1 (prep) | EN PROGRESO | Ronda 3: 15 SLURM jobs preparados (3 modelos × 5 seeds). Pendiente: lanzar jobs y post-procesar. |

---

## Detalle por tarea

### B.5 — Auditar consistencia hiperparámetros
- **Estado:** COMPLETADO
- **Fecha inicio:** 2026-02-17
- **Fecha fin:** 2026-02-17
- **Archivos generados:** `scripts/audit_hyperparams.py`, `audits/round1/B5_hyperparameter_audit.md`
- **Resumen de lo hecho:** Verificados 48 hiperparámetros (LoRA, OPRO-LLM, OPRO-Template, Evaluación, Audio, Data, Normalización) comparando main.tex vs código fuente, adapter_config.json y CSVs de datos. Script automatizado con extracción por regex de valores en código.
- **Problemas encontrados:** Ninguno bloqueante. config.yaml tiene valores prototipo obsoletos (LoRA r=8, lr=2e-4, batch=4) pero estos NO se usan en los scripts reales.
- **Discrepancias detectadas:** NINGUNA. Todos los 48 parámetros del paper coinciden con el código.

### B.2 — Normalization pathway stats
- **Estado:** PARCIAL (requiere Ronda 2 para completar)
- **Fecha inicio:** 2026-02-17
- **Fecha fin:** 2026-02-17 (parcial)
- **Archivos generados:** `scripts/extract_normalization_stats.py`, `audits/round1/B2_normalization_stats.md`
- **Resumen de lo hecho:** Parse rates extraídas para 9 configs desde predictions.csv del test set. Breakdown detallado por nivel de normalización extraído para 3 configs OPRO-Template desde iter*.csv del dev set. Documentada la modificación exacta a eval.py necesaria para Ronda 2.
- **Problemas encontrados:** predictions.csv del test set no contiene raw_text; solo se puede obtener parse rate (SPEECH/NONSPEECH/UNKNOWN). Breakdown detallado por nivel solo disponible desde iter*.csv del OPRO-Template (dev set).
- **Discrepancias detectadas:**
  1. Paper dice "7 of 9" configs con ≥99.7% parse rate, pero datos muestran **8 de 9** (LoRA+OPRO-LLM = 99.74%, justo por encima del umbral).
  2. Paper dice que 320 respuestas de Base+OPRO-LLM "required lower-level normalization" (implica resolución exitosa), pero predictions.csv las muestra como UNKNOWN (no resueltas). Wording del paper es misleading.

### B.7 — ESC-50 category accuracy
- **Estado:** COMPLETADO
- **Fecha inicio:** 2026-02-17
- **Fecha fin:** 2026-02-17
- **Archivos generados:** `scripts/extract_esc50_accuracy.py`, `audits/round1/B7_esc50_category_accuracy.csv`, `audits/round1/B7_esc50_accuracy_report.md`
- **Resumen de lo hecho:** Accuracy por categoría ESC-50 extraída para las 50 categorías × 9 configs. Categorías agrupadas en 5 tipos acústicos. Top 3 más difíciles: laughing (43.9% mean), coughing (56.4%), crying_baby (60.4%). Human vocalizations es el grupo más difícil (77.6% mean).
- **Problemas encontrados:** Ninguno.
- **Discrepancias detectadas:** Ninguna. Los valores del paper (laughing=31.8%, coughing=56.6%, crying_baby=77.3%) coinciden exactamente con LoRA+OPRO-Tmpl. El paper no especifica de qué config provienen estos valores.

### B.8 — OPRO prompts extraction
- **Estado:** COMPLETADO
- **Fecha inicio:** 2026-02-17
- **Fecha fin:** 2026-02-17
- **Archivos generados:** `scripts/extract_opro_prompts.py`, `audits/round1/B8_opro_all_prompts.csv`, `audits/round1/B8_opro_prompt_analysis.md`
- **Resumen de lo hecho:** 435 evaluaciones de prompts extraídas (75 de OPRO-LLM vía opro_prompts.jsonl, 360 de OPRO-Template vía optimization_history.json). 71 prompts únicos identificados y clasificados en 9 categorías funcionales (Binary directive, Direct question, Open-ended, Contrastive, One-shot, Acoustic focus, Robustness focus, Conservative bias, Other).
- **Problemas encontrados:** Ninguno.
- **Discrepancias detectadas:** Ninguna.

### B.4 Part 1 — SNR breakdown (Ronda 2)
- **Estado:** COMPLETADO
- **Fecha inicio:** 2026-02-17
- **Fecha fin:** 2026-02-17
- **Archivos generados:** `scripts/extract_snr_breakdown.py`, `audits/round2/B4_snr_breakdown.md`, `audits/round2/B4_snr_breakdown.csv`
- **Resumen de lo hecho:** BA extraída por nivel SNR (−10 a +20 dB) para las 9 configs. 970 muestras por nivel verificadas. 7 de 9 configs superan 90% BA a −10 dB (todos excepto Base+Hand y Base+OPRO-Tmpl). Extension a −15/−20 dB claramente justificada: LoRA+OPRO-Tmpl=96.3%, Qwen3+Hand=98.7%, Qwen3+OPRO-LLM=95.7% a −10 dB.
- **Hallazgo notable:** LoRA+OPRO-LLM tiene 100% speech recall a −10 dB pero solo 56.5% nonspeech recall — sesgo fuerte hacia SPEECH bajo ruido extremo. Contrasta con Qwen3+Hand que mantiene equilibrio (99.0%/98.4%).
- **Discrepancias detectadas:** Ninguna.

### B.3 — Cluster-aware McNemar (Ronda 2)
- **Estado:** COMPLETADO
- **Fecha inicio:** 2026-02-17
- **Fecha fin:** 2026-02-17
- **Archivos generados:** `scripts/cluster_mcnemar.py`, `audits/round2/B3_cluster_mcnemar.md`, `audits/round2/B3_cluster_mcnemar.csv`
- **Resumen:** 3 métodos McNemar comparados para 6 pares de configs:
  1. i.i.d. estándar (21,340 muestras)
  2. Cluster bootstrap (B=10,000, resampling 970 clips)
  3. Majority-vote colapsado (~970 clips independientes)
- **Hallazgos clave:**
  - **Qwen3+Hand vs Qwen3+OPRO-LLM (ΔBA=0.3pp):** i.i.d. p=0.014 → cluster p=0.093 → majority p=1.0. **La corrección cluster-aware elimina la significancia.** El paper debe actualizar esta comparación.
  - **LoRA+OPRO-Tmpl vs Qwen3+Hand:** i.i.d. p=1.5e-32 → cluster p=0.0001 → majority p=1.0. Significativo con cluster bootstrap pero no con majority-vote: la diferencia es real a nivel de condición pero no de clip.
  - Las 4 comparaciones restantes (con ΔBA >5pp) permanecen altamente significativas en los 3 métodos.
- **Implicación para el paper:** La afirmación de que OPRO-LLM mejora significativamente a Qwen3 Hand (Table 5) debe matizarse o eliminarse. Los p-values de las demás comparaciones son robustos al clustering.

### B.2 — Normalization level breakdown (Ronda 2, GPU)
- **Estado:** COMPLETADO
- **Fecha inicio:** 2026-02-17
- **Fecha fin:** 2026-02-18
- **SLURM Job:** 2061638 (19h 21m, A100)
- **Archivos generados:**
  - `src/qsm/utils/normalize.py` — `normalize_to_binary_with_level()`
  - `scripts/eval.py` — flag `--log_raw_text`
  - `scripts/analyze_normalization_levels.py` — post-procesamiento
  - `audits/round2/b2_normalization/{02,06,07}_*/predictions.csv` — predictions con raw_text y normalization_level
  - `audits/round2/B2_normalization_level_breakdown.md` — reporte final
- **Configs re-evaluadas:** 02 (Base+OPRO-LLM), 06 (LoRA+OPRO-Tmpl), 07 (Qwen3+Hand)
- **Reproducibilidad:** 100% match para las 3 configs (greedy decoding determinístico)
- **Resultados:**
  - **Base+OPRO-LLM:** L1=56.56%, L2=41.94%, L5=0.00% (1), L6b=1.50% (320). L1+L2=98.50%
  - **LoRA+OPRO-Tmpl:** L1=50.47%, L2=49.53%, L6b=0.00% (1 — "SNEEZE"). L1+L2=100.00%
  - **Qwen3+Hand:** L1=53.66%, L2=46.34%. L1+L2=100.00%. Cero unparseable.
- **Hallazgo sobre los 320 UNKNOWN:** Todas son respuestas en chino ("人类的语音。" = "human speech", "人类的说话声。" = "human speaking sound"). El modelo responde correctamente en contenido (identifica speech) pero el normalizer no reconoce chino. Son UNKNOWN genuinos, no errores L3–L5 como implica el paper.
- **Discrepancias confirmadas:** Las 2 discrepancias de Ronda 1 se confirman con datos completos.

### B.4 Part 3 — Extended SNR evaluation (Ronda 2, GPU)
- **Estado:** COMPLETADO
- **Fecha inicio:** 2026-02-17
- **Fecha fin:** 2026-02-18
- **SLURM Job:** 2061637 (2h 31m, A100)
- **Archivos generados:**
  - `scripts/generate_extended_snr.py` — generó 1,940 clips (970 × 2 SNR levels)
  - `scripts/combine_snr_tables.py` — combina Part 1 + Part 3
  - `audits/round2/b4_extended_snr/extended_snr_metadata.csv` — metadata
  - `audits/round2/b4_extended_snr/{06,07,08}_*/` — predictions + metrics por config
  - `audits/round2/B4_extended_snr_results.md` — tabla combinada −20 a +20 dB
  - `audits/round2/B4_snr_combined.csv` — CSV combinado (60 rows)
- **Configs evaluadas:** 06 (LoRA+OPRO-Tmpl), 07 (Qwen3+Hand), 08 (Qwen3+OPRO-LLM)
- **Resultados BA (extended):**

  | Config | −20 dB | −15 dB | −10 dB |
  |--------|--------|--------|--------|
  | LoRA+OPRO-Tmpl | 51.2% | **83.3%** | 96.3% |
  | Qwen3+Hand | 50.0% | 51.6% | 98.7% |
  | Qwen3+OPRO-LLM | 50.0% | 50.0% | 95.7% |

- **Hallazgos clave:**
  1. **−15 dB es el cliff:** LoRA+OPRO-Tmpl mantiene 83.3% BA; ambas Qwen3 colapsan a ~50% (puro sesgo NONSPEECH).
  2. **Fine-tuning > zero-shot bajo ruido extremo:** LoRA proporciona ~5 dB de ventaja en resiliencia al ruido.
  3. **−20 dB es el floor práctico:** Todos los sistemas degradan a chance.
  4. **Qwen3+OPRO-LLM peor que Qwen3+Hand a −15 dB:** 50.0% vs 51.6% — la optimización de prompt no ayuda bajo ruido extremo.

### B.6 — Silero VAD (Ronda 2, SLURM)
- **Estado:** COMPLETADO
- **Fecha inicio:** 2026-02-17
- **Fecha fin:** 2026-02-18
- **SLURM Job:** 2061636 (25m 53s, A100)
- **Archivos generados:**
  - `scripts/eval_silero.py` — evaluación con dual criteria
  - `scripts/analyze_silero.py` — métricas con bootstrap CIs (B=10,000)
  - `audits/round2/b6_silero/predictions_max_frame.csv` — predictions max-frame
  - `audits/round2/b6_silero/predictions_speech_ratio.csv` — predictions speech-ratio
  - `audits/round2/b6_silero/silero_analysis.json` — métricas JSON
  - `audits/round2/B6_silero_results.md` — reporte completo
- **Resultado principal — Max-frame criterion** (usar como fila Silero en Table 2):
  - **BA_clip:** 88.9% [88.4, 89.5]
  - **Recall(SPEECH):** 78.8% [78.0, 79.5]
  - **Recall(NONSPEECH):** 99.1% [98.9, 99.3]
  - **DT50:** 20 ms (below_range — ceiling para duraciones cortas)
  - **DT75:** 81 ms
  - **DT90:** 196 ms
  - **SNR75:** −7 dB
- **Speech-ratio criterion** (referencia solamente — inadecuado para este benchmark):
  - BA=54.9% [54.4, 55.4] — esencialmente chance
  - Recall(SPEECH)=9.8% — detecta <10% del speech
  - Razón: clips de 20–1000 ms son demasiado cortos para que la proporción de frames speech supere 0.5
- **Recomendación para el paper:** Usar max-frame como única fila de Silero en Table 2. Mencionar speech-ratio solo en texto para explicar por qué no se usa (clips cortos → speech-ratio degenerado).
- **Hallazgo contextual:** Silero max-frame (88.9% BA) se ubica entre Base+OPRO-LLM (86.0%) y LoRA+Hand (89.3%) en el ranking general. Supera a todos los Base configs pero es inferior a todos los LoRA y Qwen3 configs.

### B.1 — Multi-seed OPRO-Template (Ronda 3, preparación)
- **Estado:** EN PROGRESO (preparación completa, pendiente ejecución GPU)
- **Fecha inicio:** 2026-02-19
- **Fecha fin:** Pendiente (ejecución SLURM)
- **Archivos generados:**
  - `slurm/jobs/round3_b1/b1_opro_multiseed.job` — job SLURM parametrizado (MODEL_CONFIG × SEED)
  - `slurm/jobs/round3_b1/launch_all.sh` — lanza los 15 jobs con un comando
  - `scripts/analyze_multiseed_opro.py` — post-procesamiento: tablas, análisis de estabilidad, reporte
- **Diseño:**
  - 15 SLURM jobs independientes: 3 modelos (base, lora, qwen3) × 5 seeds (42, 123, 456, 789, 1024)
  - Un solo job template parametrizado (más mantenible que 15 archivos separados)
  - Cada job: Fase 1 (OPRO-Template search, 15 iter × 8 candidates × 20 samples) + Fase 2 (eval en 21,340 test samples)
  - Output: `audits/round3/b1_multiseed/{model}_seed{seed}/{optimization,evaluation}/`
- **Verificación del seed:**
  - `opro_template.py` ya acepta `--seed` (argparse line 457, default=42)
  - El seed afecta: `random.seed()`, `np.random.seed()`, `torch.manual_seed()`, `stratified_sample_df(seed+iter)`, `random.shuffle(templates)` (iter 2+)
  - Iteración 1 usa siempre los primeros 8 de 15 templates (sin shuffle); el seed solo afecta el sampling de mini-dev
  - Iteraciones 2–15: shuffle de templates restantes depende del seed
- **Para lanzar:**
  ```
  cd /mnt/fast/nobackup/users/gb0048/opro3_final
  bash slurm/jobs/round3_b1/launch_all.sh
  ```
- **Post-procesamiento (cuando terminen los 15 jobs):**
  ```
  python3 scripts/analyze_multiseed_opro.py
  ```
- **Verificación clave:** seed=42 + lora debe reproducir BA ≈ 93.3% (el valor del paper)

---

## Discrepancias encontradas (consolidado)

| Tarea | Parámetro | Valor en paper | Valor en datos | Severidad |
|-------|-----------|----------------|----------------|-----------|
| B.2 | Configs con ≥99.7% parse rate | "7 of 9" | 8 de 9 (LoRA+OPRO-LLM=99.74%) | MENOR — verificar con raw_text en Ronda 2 |
| B.2 | 320 respuestas Base+OPRO-LLM | "required lower-level normalization" | Son UNKNOWN (no resueltas) | MENOR — corregir wording en manuscript |
