# Guía de Edición de main.tex — Paper LALM VAD
## Para: Claude Code
## Fecha: 2026-02-20
## Referencia: paper_improvement_plan.md, PROGRESS.md, audits/round1/, audits/round2/, audits/round3/

---

## Instrucciones generales

Este documento contiene TODAS las ediciones que deben hacerse en `main.tex`. Lee completo antes de empezar. Las ediciones están organizadas en dos bloques:

- **Bloque 1 (A.1–A.18):** Cambios de redacción pura — no requieren datos externos
- **Bloque 2 (C.1–C.6):** Integración de resultados nuevos — requieren leer archivos de `audits/`

### Reglas generales de estilo
- **NUNCA uses guiones largos (em-dashes: —, –).** Reemplaza por comas, puntos, o reestructura la frase. Esto aplica a TODO texto nuevo que escribas. Si encuentras guiones existentes en el texto que estás modificando, reemplázalos también.
- Mantén el estilo LaTeX existente (mismos comandos, mismos entornos de tabla, mismas macros).
- No modifiques figuras existentes directamente — los cambios a figuras se hacen regenerando con scripts de Python (instrucciones en Bloque 2).
- Haz commits atómicos: un commit por cada edición (A.1, A.2, etc.) con mensaje descriptivo.
- Si no estás seguro de una edición, márcala con `% TODO: REVIEW` en el .tex y explica en PROGRESS.md.

---

## Bloque 1 — Ediciones de redacción (Phase A)

### A.1 — Unificar nomenclatura de sistemas

**Buscar y reemplazar en TODO el documento** (texto, tablas, figuras, captions):

| Variante incorrecta | Correcto |
|---|---|
| `OPRO_Classic` | `OPRO-Tmpl` |
| `OPRO_Open` | `OPRO-LLM` |
| `Base + OPRO` (sin especificar variante) | `Base+OPRO-LLM` o `Base+OPRO-Tmpl` según contexto |
| `LoRA + OPRO` (sin especificar variante) | `LoRA+OPRO-LLM` o `LoRA+OPRO-Tmpl` según contexto |
| `Qwen3 Baseline` | `Qwen3+Hand` |
| `Qwen3 + OPRO` (sin especificar) | `Qwen3+OPRO-LLM` o `Qwen3+OPRO-Tmpl` según contexto |

**Convención canónica (9 nombres):**
```
Base+Hand, Base+OPRO-LLM, Base+OPRO-Tmpl
LoRA+Hand, LoRA+OPRO-LLM, LoRA+OPRO-Tmpl
Qwen3+Hand, Qwen3+OPRO-LLM, Qwen3+OPRO-Tmpl
```

**Atención especial:**
- Table 5: tiene "OPRO_Classic" y "OPRO_Open" — corregir
- Figure 1 legend: dice "LoRA + OPRO" a secas — especificar cuál
- Verificar TODAS las tablas (2, 3, 4, 5, 6, 7) y TODAS las figuras (1-6)

### A.2 — Reducir redundancia Sections 5.9 / 6 / 7

**Section 5.9 (Summary of Findings):** Actualmente enumera 4 hallazgos detallados que se repiten casi verbatim en Section 7. Reemplazar todo el contenido de Section 5.9 por un párrafo de transición de ~4-5 líneas. Ejemplo:

```latex
\subsection{Summary of Findings}

The results across all nine cells and 22 degradation conditions converge on a consistent picture:
targeted parameter adaptation on a smaller dense model achieves the highest robustness,
surpassing both prompt-only optimization and a frozen MoE model with 4.4$\times$ more parameters.
The following section examines the mechanistic reasons for this hierarchy and its practical
implications.
```

**Section 7 (Conclusion):** Mantener los 4 hallazgos detallados aquí (es su lugar natural). No modificar Section 7 por A.2 — se modificará en A.15 y C.5.

### A.3 — Consolidar justificación no-PEFT en Qwen3-Omni

**Section 4.2.3:** Mantener texto completo tal cual está.

**Section 6.4 (Limitations):** Buscar el párrafo que repite la justificación de por qué no se aplica PEFT a Qwen3-Omni. Reemplazar por:

```latex
Third, Qwen3-Omni is evaluated only as a frozen model (see Section~\ref{sec:qwen3_setup}
for details on the architectural constraints that preclude PEFT).
The frozen evaluation thus establishes a lower bound on what the MoE architecture can achieve;
a complete comparison would require fine-tuning both models under equivalent conditions.
```

(Ajustar `\ref{sec:qwen3_setup}` al label real de Section 4.2.3)

### A.4 — Separar temas en Section 2.4.1

**Ubicación:** Section 2.4.1 ("Prompt Optimization and the Decoding Constraint Debate")

**Acción:** El primer párrafo actualmente mezcla adversarial injection attacks (Hou et al.) con constrained decoding. Dividir en dos párrafos:

**Párrafo 1** (adversarial → acoustic perturbations):
```latex
Hou et al.~\cite{hou2025evaluating} evaluated LALM robustness against audio injection attacks
and found that no model demonstrated consistent resistance. A negative correlation emerged
between instruction-following capability and injection robustness, suggesting that the
audio-text input channel is inherently sensitive to perturbations. We extend this concern
from adversarial to acoustic perturbations, hypothesizing that natural degradations (noise,
reverberation, short duration) may similarly destabilize model behavior.
```

**Párrafo 2** (decoding constraints):
```latex
A related but distinct question is whether constraining the model's output space can
mitigate this instability. Constrained decoding restricts outputs to a small token set
(e.g., ``A'' or ``B''), eliminating parsing ambiguity but potentially discarding useful
acoustic reasoning. Open decoding allows free-form generation, permitting richer responses
but introducing extraction noise under acoustic stress. The trade-off between these paradigms
for audio tasks is unclear; we test both strategies and compare their robustness curves
in Section~\ref{sec:results}.
```

### A.5 — Simplificar abstract

**Ubicación:** Abstract

**Buscar** la frase que lista rangos numéricos del banco de degradación (algo como "22 conditions across four axes: segment duration (20–1000 ms), signal-to-noise ratio (SNR), reverberation, and spectral filtering").

**Reemplazar** los rangos numéricos entre paréntesis con una versión más limpia:

```latex
We use a psychometric protocol with 22 degradation conditions spanning four acoustic axes:
segment duration, signal-to-noise ratio, reverberation, and spectral filtering.
```

### A.6 — DT90/SNR75 son operacionales

**Ubicación 1: Section 4.3.2** (donde se definen DT90 y SNR75). Añadir al final de la definición de SNR75:

```latex
Both thresholds are operational bounds specific to our evaluation protocol, including the
2000\,ms container, greedy decoding, and the VoxConverse/ESC-50 source material, and
should not be interpreted as universal architectural limits.
```

**Ubicación 2: Section 7 (Conclusion).** Añadir tras la primera mención de DT90/SNR75:

```latex
These thresholds characterize model behavior under our specific protocol conditions rather
than fundamental architectural limits.
```

### A.7 — Reframing DT90 del baseline

**Ubicación:** Section 5.2, primer párrafo tras Table 4.

**Buscar** la frase: "confirming that the zero-shot baseline lacks sufficient temporal resolution for reliable detection"

**Reemplazar** con:

```latex
The baseline model never crosses the 90\% criterion at any tested duration, but this
reflects a general failure of the zero-shot configuration rather than a temporal resolution
bottleneck specifically: its overall BA of 64.0\% indicates systematic misclassification
even at 1000\,ms. The concept of temporal integration limit is most meaningful for adapted
models that achieve ceiling performance at longer durations and degrade only as duration
decreases.
```

### A.8 — Advertencia OPRO gain como diagnóstico

**Ubicación:** Section 6.3, al final del párrafo que propone usar OPRO gain como proxy diagnóstico (el que dice "the magnitude of prompt-optimization gain serves as a proxy for how much of a model's failure is attributable to instruction misalignment versus representational limitations").

**Añadir después de ese párrafo:**

```latex
We note that the magnitude of OPRO gain is measured relative to the shared baseline prompt;
a prompt that happens to be better suited to one architecture than another would shift the
apparent gain independently of true latent capability. Controlled experiments with multiple
baseline prompts would be needed to disentangle prompt-architecture affinity from recoverable
capability, and we leave this investigation for future work.
```

### A.9 — Sugerir experimento concreto en Section 6.1

**Ubicación:** Section 6.1, final del último párrafo (después de la discusión sobre el bottleneck temporal).

**Añadir:**

```latex
Verifying this hypothesis would require probing intermediate representations before and
after LoRA adaptation. For instance, training linear classifiers on encoder outputs versus
post-attention representations for short-duration clips would determine whether temporal
evidence is present but underweighted in the frozen model (supporting our interpretation)
or genuinely absent at that processing stage.
```

### A.10 — Degradaciones combinadas como limitación

**Ubicación:** Section 6.4 (Limitations). Insertar como nuevo párrafo (idealmente segundo o tercer punto).

```latex
Our psychometric protocol varies each degradation axis independently while holding others
at neutral values. This one-factor-at-a-time design enables clear attribution of performance
changes to individual acoustic factors, following standard psychoacoustic methodology, but
does not capture interactions between degradation axes. In realistic deployment scenarios,
degradations co-occur (e.g., short segments in noisy, reverberant environments), and the
adaptation hierarchy established here may shift under combined stress. Extending the
evaluation to factorial or representative combinations of degradation conditions is a natural
direction for future work.
```

### A.11 — σ=10⁻⁴ como diseño unificado

**Ubicación:** Section 3.2.2 (SNR Manipulations), after the paragraph about near-silent segments (the one mentioning "RMS = 10⁻⁴").

**Añadir:**

```latex
We set the fallback noise RMS to match the container padding amplitude ($\sigma = 10^{-4}$),
ensuring that near-silent segments blend seamlessly into the noise floor of the container
and cannot be distinguished from the padding region by spectral characteristics alone.
```

### A.12 — Justificar λ=0.25

**Ubicación:** Section 3.4.1, immediately after Equation 2.

**Añadir:**

```latex
We set $\lambda = 0.25$ heuristically to give modest weight to per-axis uniformity without
dominating the global signal; this value was fixed throughout all experiments and was not
itself optimized. We acknowledge this as a design choice that could influence which prompts
are selected, though the final evaluation on the complete test set (21{,}340 samples) provides
the definitive performance comparison independently of the reward function used during
optimization.
```

### A.13 — Eliminar toda mención a pyannote

**Buscar** "pyannote" en todo el documento. Ubicaciones conocidas:
- ~Line 333 (Section 4.1): "dedicated lightweight VAD systems (Silero VAD, pyannote)"
- ~Line 647 (Section 6.4 Limitations): similar mention

**Acción en cada ubicación:**
- Eliminar "pyannote" y su cita bibliográfica de la frase
- Dejar Silero como único ejemplo de sistema VAD externo
- Verificar que la referencia bibliográfica de pyannote (Karan et al. 2024, ref [12]) no quede huérfana. Si se elimina de todas las citas, eliminar también de la sección de Referencias. PERO: verificar primero que [12] no se cite en otro contexto (e.g., como referencia a un método VAD en la introducción o related work). Si se cita en otro contexto, mantener la referencia.

### A.14 — Fortalecer defensa de datasets en limitaciones

**Ubicación:** Section 6.4, añadir nuevo párrafo (después del párrafo de degradaciones combinadas de A.10).

```latex
The NONSPEECH class is drawn exclusively from ESC-50, which comprises isolated environmental
sounds rather than the continuous background signals (sustained music, multi-talker babble,
ambient noise) encountered in typical VAD deployment. Similarly, the Silero-based quality
filter (speech ratio $\geq 0.8$) retains only clips with high speech occupancy, excluding the
onset/offset boundary regions that constitute the most challenging cases for practical VAD.
These design choices prioritize controlled evaluation and label integrity at the cost of
ecological representativeness. Extending the evaluation to continuous-background corpora
and to boundary-case speech clips would test whether the adaptation hierarchy generalizes
beyond the clean-segment regime.
```

### A.15 — Moderar claim "adaptation > scaling"

**Ubicaciones a modificar:** Sections 5.1, 5.9 (ya reducido por A.2), 6.2, y 7.

**Buscar** frases como:
- "parameter-efficient fine-tuning is more effective than architectural scaling"
- "challenges the hypothesis that MoE architectures exhibit stronger zero-shot generalization"
- cualquier claim que generalice el resultado como principio universal

**Reemplazar** con versiones moderadas. Ejemplo para Section 5.1:

```latex
Under our evaluation protocol, a 7B dense model with LoRA and OPRO optimization (93.3\% BA)
outperforms a frozen 30B MoE model (91.1\%), despite operating under 4-bit quantization.
While this suggests that targeted adaptation can compensate for the benefits of architectural
scaling, we note that these models differ along multiple dimensions (pretraining data, encoder
architecture, quantization regime), and the comparison addresses the practical question of
whether to adapt a smaller model or deploy a larger one frozen, rather than providing a
controlled ablation of model scale alone.
```

Aplicar moderación similar en cada ubicación, adaptando al contexto de cada sección.

### A.16 — Corregir "7 of 9" → "8 of 9"

**Ubicación:** Section 3.3 (Output Normalization).

**Buscar:** "over 99.7% of responses are resolved at levels 1–2 (direct substring match) for seven of the nine configurations"

**Reemplazar:** "seven" → "eight" (o equivalente numérico).

**Fuente:** audits/round1/B2_normalization_stats.md confirma que LoRA+OPRO-LLM tiene 99.74% parse rate, lo que sube el conteo de 7 a 8 configuraciones ≥ 99.7%.

### A.17 — Corregir descripción de 320 respuestas

**Ubicación:** Section 3.3, misma zona que A.16.

**Buscar:** "320 of 21,340 responses (1.5%) required lower-level normalization"

**Reemplazar con:**

```latex
320 of 21{,}340 responses (1.5\%) could not be resolved to a binary label by any
normalization level and were treated as invalid predictions. Qualitative inspection of
these responses revealed that the majority consisted of outputs in Chinese characters,
suggesting that the generative OPRO-LLM prompt occasionally triggered language-switching
behavior in the base model under acoustic degradation.
```

**Fuente:** audits/round2/B2_normalization_level_breakdown.md confirma que los 320 son UNKNOWN (texto chino).

### A.18 — Ajustar claim sobre OPRO en Qwen3

**Ubicación:** Section 5.1 y 5.3. El paper reporta Qwen3+Hand vs Qwen3+OPRO-LLM con ΔBA = 0.3 pp y p = 0.014.

**Problema:** Los cluster-aware p-values (audits/round2/B3_cluster_mcnemar.md) muestran que este efecto pierde significancia: p_cluster = 0.093, p_majority = 1.0.

**Acción en Section 5.1:** Donde dice "OPRO provides a modest improvement of 0.3 pp", reescribir:

```latex
For the frozen Qwen3-Omni model, OPRO-LLM yields a nominal improvement of 0.3\,pp
(91.1\% $\to$ 91.4\%); however, cluster-aware statistical tests that account for within-clip
dependence across degradation variants indicate that this difference is not statistically
significant ($p_{\mathrm{cluster}} = 0.093$; see Table~\ref{tab:mcnemar}), confirming that
the MoE architecture leaves minimal headroom for prompt-level gains.
```

**Acción en Table 5:** Actualizar los p-values usando los cluster-aware values de B3_cluster_mcnemar.csv. Añadir una columna `$p_{\mathrm{cluster}}$` o reemplazar los p-values originales con los cluster-aware y añadir una nota al pie explicando el método.

**Acción en Section 5.3:** Actualizar el texto que describe los paired comparisons para mencionar que se usa cluster bootstrap.

---

## Bloque 2 — Integración de resultados nuevos (Phase C)

### C.1 — Añadir Silero a Table 2 y texto

**Fuente de datos:** `audits/round2/B6_silero_results.md` y `audits/round2/b6_silero/silero_analysis.json`

**En Table 2:** Añadir fila al principio o al final (separada visualmente con una línea):

```latex
\midrule
Silero VAD v5 (reference) & 0.889 [0.884, 0.895] & 0.788 [0.780, 0.795] & 0.991 [0.989, 0.993] \\
```

(Verificar valores exactos en silero_analysis.json — usar el criterio max-frame, NO speech-ratio)

**Nota al pie de Table 2:**

```latex
\footnotesize{Silero VAD is a lightweight frame-level detector included as a non-LALM
reference. Its max-frame decision criterion (classify as SPEECH if any frame exceeds 0.5
probability) differs fundamentally from LALM clip-level classification. Computational cost
is orders of magnitude lower than any LALM configuration.}
```

**En Section 5.1 (Overall Performance):** Añadir un párrafo contextual:

```latex
For reference, Silero VAD~\cite{silero2024}, a lightweight dedicated VAD system operating
at frame level, achieves 88.9\% BA on the same test set. This positions Silero between
the prompt-optimized base model (Base+OPRO-LLM, 82.6\%) and the LoRA-adapted configuration
(LoRA+Hand, 86.4\%) in overall accuracy, though the comparison is asymmetric: Silero operates
at orders-of-magnitude lower computational cost with a fundamentally different architecture
and decision granularity.
```

**En Silero's psychometric thresholds:** Añadir a Table 4:

```latex
Silero VAD v5 (reference) & <20$^\dagger$ & 81 & 196 & $-7$ \\
```

(Verificar valores en silero_analysis.json)

### C.2 — Añadir tabla de normalization pathway stats

**Fuente:** `audits/round2/B2_normalization_level_breakdown.md`

**Ubicación:** Después de la discusión de normalización en Section 3.3, o como tabla nueva.

Solo tenemos breakdown detallado para 3 configs (02, 06, 07). Crear tabla con esas 3 más la información de parse rates (SPEECH/NONSPEECH/UNKNOWN) para las 9 configs de `audits/round1/B2_normalization_stats.md`.

**Formato sugerido:** Tabla con las 3 configs detalladas:

```latex
\begin{table}[t]
\centering
\caption{Normalization level breakdown for selected configurations on the test set
(21,340 samples each). Levels follow the priority hierarchy described in Section~\ref{sec:normalization}.}
\label{tab:normalization_levels}
\begin{tabular}{lrrr}
\toprule
Normalization Level & Base+OPRO-LLM & LoRA+OPRO-Tmpl & Qwen3+Hand \\
\midrule
L1: NONSPEECH substring & ... & ... & ... \\
L2: SPEECH substring & ... & ... & ... \\
L3: Letter mapping & ... & ... & ... \\
L4: YES/NO & ... & ... & ... \\
L5: Keywords & ... & ... & ... \\
L6: Heuristic fallback & ... & ... & ... \\
Unresolved (UNKNOWN) & 320 (1.5\%) & 1 (<0.01\%) & 0 (0\%) \\
\bottomrule
\end{tabular}
\end{table}
```

**Rellenar los valores exactos de** `B2_normalization_level_breakdown.md`.

### C.3 — Actualizar Table 5 con cluster-aware stats

**Fuente:** `audits/round2/B3_cluster_mcnemar.md` y `B3_cluster_mcnemar.csv`

**Acción:** Reemplazar los p-values en Table 5. Dos opciones (elegir la más limpia):

**Opción A (preferida):** Reemplazar columnas `p (raw)` y `$p_{\mathrm{Holm}}$` con los cluster-aware p-values. Añadir nota: "p-values computed via cluster bootstrap (B=10,000) resampling at the base-clip level to account for within-clip dependence across 22 degradation variants."

**Opción B:** Añadir columna `$p_{\mathrm{cluster}}$` manteniendo las originales. Más transparente pero tabla más ancha.

**CRÍTICO:** La comparación Qwen3+Hand vs Qwen3+OPRO-LLM debe mostrar p = 0.093 (no significativo), no p = 0.014.

### C.4 — Tabla multi-seed OPRO-Template

**Fuente:** `audits/round3/B1_multiseed_opro.md` y `B1_multiseed_opro.csv`

**Ubicación:** Nueva tabla en Section 5.6 (Prompting Strategies) o al final de Section 3.4.2.

```latex
\begin{table}[t]
\centering
\caption{OPRO-Template stability across five random seeds. BA (\%) evaluated on the
full test set (21,340 samples) using the winning template from each seed's optimization run.}
\label{tab:multiseed}
\begin{tabular}{lccccc|cc}
\toprule
Configuration & S=42 & S=123 & S=456 & S=789 & S=1024 & Mean & Std \\
\midrule
Base+OPRO-Tmpl & 75.1 & 75.1 & 75.1 & 61.4 & 75.1 & 72.3 & 6.1 \\
LoRA+OPRO-Tmpl & 93.3 & 91.5 & 87.7 & 93.3 & 93.3 & 91.8 & 2.4 \\
Qwen3+OPRO-Tmpl & 87.8 & 86.3 & 87.8 & 89.5 & 87.8 & 87.9 & 1.1 \\
\bottomrule
\end{tabular}
\end{table}
```

(Verificar valores exactos contra B1_multiseed_opro.csv)

**Texto acompañante** (insertar en Section 5.6 o 3.4.2):

```latex
To assess the stability of OPRO-Template selection, we repeated the optimization with
five random seeds (Table~\ref{tab:multiseed}). The LoRA-adapted model shows moderate
sensitivity (91.8\% $\pm$ 2.4\,pp), with three of five seeds selecting the same contrastive
template; seed 42, used throughout the main evaluation, achieves the highest BA (93.3\%),
indicating a mildly optimistic point estimate. The frozen Qwen3-Omni model is the most
stable across seeds (87.9\% $\pm$ 1.1\,pp). The base model exhibits the highest variance
(72.3\% $\pm$ 6.1\,pp), driven by one outlier seed. Importantly, the adaptation hierarchy
(LoRA+OPRO-Tmpl $>$ Qwen3+OPRO-Tmpl $>$ Base+OPRO-Tmpl) is preserved across all five
seeds, confirming that the ranking is robust to template selection variance.
```

**Nota:** Si el seed=42 es el mejor para LoRA, debemos reconocerlo honestamente. Añadir:

```latex
We note that seed 42, used in the primary evaluation, yields the maximum BA for the
LoRA model. The mean across seeds (91.8\%) remains above the frozen Qwen3-Omni baseline
(91.4\% with OPRO-LLM), preserving the central finding that targeted adaptation on a
smaller model matches or exceeds the frozen MoE architecture.
```

### C.5 — Tabla SNR extendida y actualización de narrativa

**Fuente:** `audits/round2/B4_snr_breakdown.csv` y `audits/round2/B4_extended_snr_results.md`

**Acción 1:** Actualizar Table 4 (psychometric thresholds). Los modelos que tenían SNR75 < −10 dB ahora pueden tener valores interpolados si bajan de 75% en el rango extendido. Leer B4_extended_snr_results.md para los valores exactos.

Del plan de Claude Code sabemos:
- LoRA+OPRO-Tmpl a −15 dB: 83.3% BA, a −20 dB: 51.2%
- Qwen3+Hand a −15 dB: 51.6%, a −20 dB: 50.0%
- Qwen3+OPRO-LLM a −15 dB: 50.0%, a −20 dB: 50.0%

Esto significa:
- LoRA+OPRO-Tmpl: SNR75 ≈ −17 dB (interpolar entre −15 y −20)
- Qwen3: SNR75 ≈ −11 dB (interpolar entre −10 y −15)

**Actualizar Table 4** con estos valores desensorados.

**Acción 2:** Actualizar texto en Section 5.4 (Noise Robustness):

```latex
Extending the SNR range to $-20$\,dB reveals a critical divergence between adaptation
strategies that was masked by the original $-10$\,dB floor. At $-15$\,dB, LoRA+OPRO-Tmpl
maintains 83.3\% BA, while both Qwen3-Omni configurations collapse to near-chance performance
($\leq$51.6\% BA). At $-20$\,dB, all configurations degrade severely. Interpolating the
extended curves, we estimate SNR75 $\approx -17$\,dB for LoRA+OPRO-Tmpl versus
$\approx -11$\,dB for the frozen Qwen3-Omni model, a 6\,dB advantage attributable to
parameter adaptation. This gap, invisible in the original evaluation range, demonstrates
that the noise invariance conferred by LoRA fine-tuning extends substantially deeper into
extreme noise conditions than the intrinsic robustness of the MoE architecture.
```

**Acción 3:** Actualizar Figure 3 para incluir los puntos −15 y −20 dB (solo para los 3 modelos evaluados). Esto requiere regenerar la figura con el script de plotting. Localizar el script que genera Figure 3 y añadir los data points.

### C.6 — Heatmap ESC-50 (nueva figura)

**Fuente:** `audits/round1/B7_esc50_category_accuracy.csv` y `B7_esc50_accuracy_report.md`

**Acción:** Crear un script Python que genere el heatmap y guardarlo como PDF. Luego incluirlo en el .tex.

**Script:** Crear `scripts/plot_esc50_heatmap.py` que:
1. Lee `audits/round1/B7_esc50_category_accuracy.csv`
2. Selecciona 4-5 configuraciones clave (Base+Hand, LoRA+OPRO-Tmpl, Qwen3+Hand, Qwen3+OPRO-LLM)
3. Agrupa categorías por tipo acústico (usar agrupación de B7_esc50_accuracy_report.md)
4. Genera heatmap con matplotlib/seaborn: filas = categorías ordenadas por dificultad dentro de cada grupo, columnas = configuraciones, color = accuracy (0-100%)
5. Guarda como `figures/esc50_heatmap.pdf`

**En main.tex:** Añadir en Section 5.8 (Non-Speech Category Difficulty):

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/esc50_heatmap.pdf}
\caption{NONSPEECH classification accuracy by ESC-50 category and model configuration.
Categories are grouped by acoustic type and ordered by mean difficulty (descending accuracy).
Human vocalizations (laughing, coughing, crying baby) constitute the hardest category group
across all configurations.}
\label{fig:esc50_heatmap}
\end{figure}
```

**Expandir texto de Section 5.8** usando datos de B7_esc50_accuracy_report.md. El texto actual menciona solo 3 categorías. Expandir con la visión del heatmap, mencionando patrones por grupo acústico y diferencias entre modelos.

---

## Bloque 3 — Verificaciones finales

### V.1 — Verificar referencias bibliográficas
- Si pyannote (ref [12], Karan et al. 2024) se eliminó de todas las citas tras A.13, eliminar de la bibliografía. Verificar que no se cita en otro contexto.
- Si se añadió Silero como baseline (C.1), verificar que la cita [26] (Silero Team, 2024) ya existe.

### V.2 — Verificar consistencia de nombres tras TODOS los cambios
- Hacer búsqueda global de todos los nombres de la convención A.1
- Verificar que no quedó ninguna variante incorrecta

### V.3 — Verificar que todas las tablas/figuras nuevas están referenciadas
- Table de normalization levels (C.2) → referenciada en Section 3.3
- Table multi-seed (C.4) → referenciada en Section 5.6 o 3.4.2
- Figure heatmap ESC-50 (C.6) → referenciada en Section 5.8
- Fila Silero en Table 2 (C.1) → mencionada en Section 5.1

### V.4 — Lectura final de flujo
- Leer el paper completo de principio a fin
- Verificar que las transiciones entre secciones son naturales
- Verificar que no hay contradicciones entre secciones (especialmente entre resultados nuevos y claims)
- Verificar que los números en el texto coinciden con los de las tablas

---

## Orden de ejecución recomendado

```
PASO 1: A.1 (nombres) — afecta todo el documento, hacer primero
PASO 2: A.13 (eliminar pyannote) — limpieza simple
PASO 3: A.16, A.17 (correcciones factuales) — fixes rápidos
PASO 4: A.2, A.3, A.4 (reorganización estructural)
PASO 5: A.5 (abstract), A.11, A.12 (aclaraciones)
PASO 6: A.7, A.8, A.9, A.10, A.14 (añadir párrafos nuevos)
PASO 7: A.6, A.15, A.18 (claims y moderación — requieren cuidado)
PASO 8: C.1 a C.6 (integración de resultados nuevos)
PASO 9: V.1 a V.4 (verificaciones finales)
```

---

## Archivos de referencia para datos

| Dato | Archivo |
|---|---|
| Parse rates (9 configs) | `audits/round1/B2_normalization_stats.md` |
| Normalization levels (3 configs) | `audits/round2/B2_normalization_level_breakdown.md` |
| Cluster McNemar p-values | `audits/round2/B3_cluster_mcnemar.csv` |
| SNR breakdown (9 configs) | `audits/round2/B4_snr_breakdown.csv` |
| Extended SNR (3 configs) | `audits/round2/B4_extended_snr_results.md` |
| Silero metrics | `audits/round2/b6_silero/silero_analysis.json` |
| ESC-50 accuracy (9 configs) | `audits/round1/B7_esc50_category_accuracy.csv` |
| Multi-seed OPRO | `audits/round3/B1_multiseed_opro.csv` |
| Hyperparameter audit | `audits/round1/B5_hyperparameter_audit.md` |
| OPRO prompts | `audits/round1/B8_opro_all_prompts.csv` |
