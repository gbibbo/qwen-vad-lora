# Plan Maestro de Mejoras — Paper "Audio-Language Models under Psychometric Degradations"
## Fecha de creación: 2026-02-17
## Autores del plan: Gabriel Bibbó + Claude Opus (revisión de pares)

---

## Instrucciones para Claude Code

Este plan describe todas las mejoras pendientes para el paper. Cada vez que trabajes en alguna tarea de este plan:

1. **Lee este archivo completo antes de comenzar** para entender el contexto general.
2. **Registra tu progreso** en un archivo separado llamado `PROGRESS.md` ubicado en la misma carpeta que este plan. Si el archivo no existe, créalo con la estructura indicada al final de este documento (ver sección "Plantilla de PROGRESS.md").
3. **No modifiques este archivo** (`paper_improvement_plan.md`). Solo modifica `PROGRESS.md`.
4. En `PROGRESS.md`, para cada tarea que completes o intentes, registra:
   - Fecha y hora
   - ID de la tarea (e.g., B.1, A.3)
   - Estado: COMPLETADO / EN PROGRESO / BLOQUEADO / REQUIERE DECISIÓN
   - Si BLOQUEADO o REQUIERE DECISIÓN: descripción del problema encontrado
   - Si COMPLETADO: resumen breve de qué se hizo y dónde están los outputs (archivos generados, tablas, etc.)
5. Si una tarea depende de otra que aún no está completada, indícalo y pasa a la siguiente tarea factible.
6. Si encuentras discrepancias entre lo que dice el paper y lo que muestra el código, repórtalas inmediatamente en PROGRESS.md con el nivel de severidad (CRÍTICO / ALTO / BAJO).

---

## Contexto del proyecto

Este paper evalúa Large Audio-Language Models (LALMs) para Voice Activity Detection (VAD) bajo un protocolo de degradaciones psicométricas. Se usa una matriz experimental 3×3: tres modelos (Qwen2-Audio base, Qwen2-Audio+LoRA, Qwen3-Omni frozen) × tres estrategias de prompting (Hand-crafted baseline, OPRO-LLM generativo, OPRO-Template determinista). El test set tiene 21,340 muestras (970 base clips × 22 condiciones de degradación). El paper fue revisado por dos revisores independientes (peer review estilo ICASSP). Este plan consolida todas las mejoras identificadas.

---

## FASE A — Modificaciones de redacción pura

> Estas tareas se realizan directamente sobre el manuscrito (main.tex o equivalente).
> No requieren acceso al código, datos, ni cómputo adicional.
> Pueden ejecutarse en paralelo con cualquier fase.

### A.1 Unificar nomenclatura de sistemas en todo el paper

**Problema:** Los nombres de las 9 configuraciones experimentales varían entre texto, tablas y figuras. Ejemplos de inconsistencia: Table 5 usa "OPRO_Classic" y "OPRO_Open" que no aparecen en ningún otro lugar. Figure 1 dice "LoRA + OPRO" sin especificar variante. En el texto a veces se dice "Base + OPRO" y otras "Base + OPRO-LLM" para la misma configuración.

**Convención a adoptar (9 nombres fijos):**
- `Base+Hand` | `Base+OPRO-LLM` | `Base+OPRO-Tmpl`
- `LoRA+Hand` | `LoRA+OPRO-LLM` | `LoRA+OPRO-Tmpl`
- `Qwen3+Hand` | `Qwen3+OPRO-LLM` | `Qwen3+OPRO-Tmpl`

**Alcance:** Revisar y corregir en: texto corrido, Table 2, Table 3, Table 4, Table 5, Table 6, Table 7, Figures 1-6, y todos los captions.

### A.2 Reducir redundancia entre Sections 5.9, 6 y 7

**Problema:** Los 4 hallazgos principales (adaptation hierarchy, temporal integration limit, noise invariance, error regime shift) se repiten casi verbatim en Section 5.9 (Summary of Findings), Section 6 (Discussion), y Section 7 (Conclusion).

**Acción:** Condensar Section 5.9 en un párrafo de transición de ~4-5 líneas que anticipe la discusión sin enumerar los 4 hallazgos en detalle. Reservar la enumeración detallada para Section 7 (Conclusion).

### A.3 Consolidar justificación de no aplicar PEFT a Qwen3-Omni

**Problema:** La explicación de por qué Qwen3-Omni se evalúa solo como modelo frozen (su arquitectura thinker-talker omni-modal carece de soporte estable para PEFT) aparece con texto casi idéntico en Section 4.2.3 y Section 6.4.

**Acción:** Mantener la explicación completa en Section 4.2.3. En Section 6.4, reemplazar por una referencia cruzada: "As noted in Section 4.2.3, stable PEFT support for Qwen3-Omni's architecture is not yet available."

### A.4 Separar temas en Section 2.4.1

**Problema:** Section 2.4.1 ("Prompt Optimization and the Decoding Constraint Debate") comienza discutiendo adversarial audio injection attacks (Hou et al.) y salta abruptamente a constrained vs open decoding. Son dos temas relacionados pero distintos.

**Acción:** Dividir en dos párrafos con transición explícita. Primer párrafo: vulnerabilidad de LALMs a perturbaciones en el canal audio-texto (Hou et al.), con extensión a perturbaciones acústicas (nuestra hipótesis). Segundo párrafo: constrained vs open decoding como estrategia para estabilizar outputs bajo incertidumbre de entrada.

### A.5 Simplificar abstract

**Problema:** El abstract lista valores numéricos específicos del banco de degradación (20-1000 ms, SNR, reverberation, spectral filtering) que ya están detallados en Table 1.

**Acción:** Reemplazar valores específicos por descripción conceptual. Ejemplo: "22 conditions across four psychoacoustic axes—segment duration, signal-to-noise ratio, reverberation, and spectral filtering" sin listar rangos.

### A.6 Explicitar que DT90/SNR75 son umbrales operacionales, no universales

**Problema:** Los umbrales psicométricos (DT90, SNR75) podrían interpretarse como límites fundamentales de las arquitecturas. En realidad son específicos a este protocolo (contenedor 2000 ms, greedy decoding, datasets VoxConverse/ESC-50).

**Acción:** Añadir frase en Section 4.3.2 (donde se definen) y en Section 7 (Conclusión). Texto sugerido: "These thresholds are operational bounds specific to our evaluation protocol—including the 2000 ms container, greedy decoding, and the VoxConverse/ESC-50 source material—and should not be interpreted as universal architectural limits."

### A.7 Reframing de DT90 para el baseline

**Problema:** Section 5.2 dice que el baseline "lacks sufficient temporal resolution" basándose en DT90 > 1000 ms. Pero con BA global de solo 64%, el problema no es primariamente temporal—el modelo falla incluso a 1000 ms. La narrativa de "temporal integration limit" solo es apropiada para modelos que alcanzan ceiling performance a duraciones largas.

**Acción:** Reformular en Section 5.2. Ejemplo: "The baseline model never crosses the 90% criterion at any tested duration, but this reflects a general failure of the zero-shot configuration rather than a temporal resolution bottleneck—its overall BA of 64.0% indicates systematic misclassification even at 1000 ms. The concept of temporal integration limit is most meaningful for adapted models that achieve ceiling performance at longer durations and degrade only as duration decreases."

### A.8 Advertencia sobre OPRO gain como proxy diagnóstico

**Problema:** Section 6.3 propone usar la magnitud del OPRO gain como proxy para "unlockable capability" de un modelo. Pero este gain es relativo al baseline prompt compartido, que podría estar desigualmente alineado con cada arquitectura.

**Acción:** Añadir al final del párrafo relevante en Section 6.3: "We note that the magnitude of OPRO gain is measured relative to the shared baseline prompt; a prompt that happens to be better-aligned with one architecture than another would shift the apparent gain independently of true latent capability. Controlled experiments with multiple baseline prompts would be needed to disentangle these factors."

### A.9 Sugerir experimento concreto en Section 6.1

**Problema:** La hipótesis de que el bottleneck temporal reside en el mapping entre features acústicos y vocabulario de decisión (no en el encoder) es plausible pero puramente especulativa.

**Acción:** Añadir al final de Section 6.1: "Verifying this hypothesis would require probing intermediate representations before and after LoRA adaptation—for instance, training linear classifiers on encoder outputs versus post-attention representations for short-duration clips—to determine whether temporal evidence is present but underweighted in the frozen model (supporting our interpretation) or genuinely absent at that processing stage."

### A.10 Añadir degradaciones combinadas como limitación en Section 6.4

**Problema:** El diseño one-axis-at-a-time (variar un eje manteniendo los demás en valores neutros) no captura interacciones entre degradaciones. En escenarios reales, un clip puede tener simultáneamente SNR bajo, reverberación alta y duración corta. Esta limitación no aparece actualmente en Section 6.4.

**Acción:** Añadir párrafo en Section 6.4 (idealmente como el segundo o tercer punto de limitaciones): "Our psychometric protocol varies each degradation axis independently while holding others at neutral values. This one-factor-at-a-time design enables clear attribution of performance changes to individual acoustic factors, following standard psychoacoustic methodology, but does not capture interactions between axes. In realistic deployment scenarios, degradations co-occur—short segments in noisy, reverberant environments—and the adaptation hierarchy established here may shift under combined stress. Extending the evaluation to factorial combinations of degradation axes is a natural direction for future work."

### A.11 Aclarar σ=10⁻⁴ como valor de diseño unificado

**Problema:** El valor σ = 10⁻⁴ aparece en dos contextos: (i) como amplitud del padding Gaussiano del contenedor 2000 ms (Section 3.2), y (ii) como RMS del noise fallback para segmentos near-silent donde no se puede calcular SNR (Section 3.2.2). No se explicita que sea el mismo valor por diseño.

**Acción:** Añadir en Section 3.2.2, tras describir el fallback: "We set the fallback noise RMS to match the container padding amplitude (σ = 10⁻⁴), ensuring that near-silent segments blend seamlessly into the noise floor of the container and cannot be distinguished from the padding region by spectral characteristics alone."

### A.12 Justificar λ=0.25 en la función de reward (Equation 2)

**Problema:** El peso λ=0.25 en R = BAclip + λ·BAcond no tiene justificación en el texto. No se reporta sensitivity analysis ni se explica por qué 0.25 y no otro valor.

**Acción:** Añadir tras Equation 2: "We set λ = 0.25 heuristically to give modest weight to per-axis uniformity without dominating the global signal; this value was fixed throughout all experiments and was not itself optimized. We acknowledge this as a design choice that could influence which prompts are selected, though the final evaluation on the complete test set (21,340 samples) provides the definitive performance comparison independently of the reward function used during optimization."

### A.13 Eliminar toda mención a pyannote

**Contexto:** Pyannote nunca se usó en el proyecto—ni como herramienta ni como sistema evaluado. Solo aparece citado en el manuscrito como ejemplo de sistema VAD ligero. Silero VAD sí se usó como herramienta de curación de datos (filtro speech ratio ≥ 0.8). Ahora que incluiremos Silero como baseline de referencia bajo el banco psicométrico (ver tarea B.6), las menciones a pyannote como "otro ejemplo" son innecesarias y potencialmente confusas.

**Ubicaciones conocidas en el .tex:**
- ~Línea 333 (Section 4.1): mención junto a Silero como sistemas contra los que no se compara
- ~Línea 647 (Limitations/Section 6.4): mención similar

**Acción:** Eliminar "or pyannote" / "pyannote [ref]" de ambas ubicaciones. Ajustar redacción para que Silero quede como el único sistema externo mencionado, con sus dos roles claramente separados: (a) herramienta de curación de datos y (b) baseline de referencia bajo el banco psicométrico.

### A.14 Fortalecer defensa de elección de datasets en limitaciones

**Problema:** ESC-50 contiene eventos ambientales "limpios y cortos", no background continuo típico de VAD real (música sostenida, babble, ruido ambiente). Además, el filtro Silero ≥ 0.8 para speech excluye el borde difícil de VAD (onsets/offsets, clips con baja ocupación vocal). Estas limitaciones de representatividad ecológica no están suficientemente discutidas.

**Acción:** Añadir párrafo en Section 6.4: "The NONSPEECH class is drawn exclusively from ESC-50, which comprises isolated environmental sounds rather than the continuous background signals (sustained music, multi-talker babble, ambient noise) encountered in VAD deployment. Similarly, the Silero-based quality filter (speech ratio ≥ 0.8) retains only clips with high speech occupancy, excluding the onset/offset boundary regions that constitute the most challenging cases for practical VAD. These design choices prioritize controlled evaluation and label integrity at the cost of ecological representativeness. Extending the evaluation to continuous-background corpora (e.g., MUSAN, AudioSet subsets) and to boundary-case speech clips would test whether the adaptation hierarchy generalizes beyond the clean-segment regime."

### A.15 Moderar claim "adaptation > scaling"

**Problema:** El paper generaliza el hallazgo de que LoRA+OPRO en Qwen2 supera a Qwen3-Omni frozen como evidencia de que "parameter-efficient fine-tuning is more effective than architectural scaling." Pero la comparación tiene múltiples confounds: los modelos difieren en pretraining data, arquitectura de encoder, instruction tuning, y cuantización (4-bit NF4 vs unquantized).

**Ubicaciones a modificar:** Sections 5.1, 5.9 (si queda tras A.2), 6.2, y 7.

**Acción:** Reformular como resultado específico del protocolo, no como principio general. Ejemplo: "Under our evaluation protocol, a 7B dense model with LoRA and OPRO optimization (93.3% BA) outperforms a frozen 30B MoE model (91.1%), despite operating under 4-bit quantization. While this suggests that targeted adaptation can compensate for—and exceed—the benefits of architectural scaling, we note that these models differ along multiple dimensions (pretraining data, encoder architecture, quantization regime), and the comparison isolates the practical question of 'adapt a smaller model vs. deploy a larger one frozen' rather than a controlled ablation of model scale."

---

## FASE B — Tareas que requieren Claude Code

> Estas tareas requieren acceso al código fuente, datos, logs, y/o cómputo.
> Se ejecutan en orden de prioridad (B.1 es la más crítica).
> Cada tarea incluye instrucciones detalladas para Claude Code.

### B.1 [PRIORIDAD: CRÍTICA] Multi-seed OPRO-Template

**Motivación:** El mejor sistema del paper (LoRA+OPRO-Tmpl, 93.3% BA) depende de una selección de template hecha con seed=42 y mini-dev de solo N=20 muestras por iteración (resolución 0.05). Múltiples templates empatan a ceiling (20/20). Un revisor puede argumentar que el resultado es un artefacto del seed.

**Tareas:**
1. Localizar el código de OPRO-Template en el repositorio
2. Identificar dónde entra el seed y cómo afecta: (a) orden de evaluación de templates, (b) sampling del mini-dev set
3. Modificar el script para aceptar seed como argumento CLI (si no lo acepta ya)
4. Correr OPRO-Template con seeds {42, 123, 456, 789, 1024} para las 3 configuraciones de modelo (Base, LoRA, Qwen3)
5. Para cada seed y modelo: registrar el template ganador, BA en mini-dev, y evaluar el template ganador en el test set completo (21,340 muestras)
6. Generar tabla resumen: filas = modelos, columnas = seeds, celdas = (template ganador, BA test)
7. Computar media ± std de BA test por modelo across seeds

**Output esperado:** Tabla para incluir en el paper + conclusión textual sobre estabilidad

**Estimación de cómputo:** 5 seeds × 3 modelos × 2,400 forward passes (OPRO search) + evaluación en test para cada template ganador. Este es el item más costoso del plan.

### B.2 [PRIORIDAD: CRÍTICA] Normalization pathway stats por celda

**Motivación:** El pipeline de evaluación normaliza las respuestas textuales de los modelos a labels binarias (SPEECH/NONSPEECH) mediante una jerarquía de 6 niveles (ver Section 3.3 del paper). Section 3.3 reporta parcialmente que ">99.7% de respuestas se resuelven en niveles 1-2 para 7 de 9 configuraciones," pero un revisor necesita ver el breakdown completo para descartar que el ranking dependa del parser.

**Tareas:**
1. Localizar la función de normalización en el código
2. Verificar si ya loguea el nivel de resolución para cada respuesta. Si no, añadir logging
3. Re-procesar los outputs guardados de las 9 configuraciones (o re-correr evaluación si los outputs no están guardados) para obtener: para cada muestra, en qué nivel de la jerarquía se resolvió
4. Generar tabla: 9 configuraciones × 7 columnas (Level 1: NONSPEECH substring, Level 2: SPEECH substring, Level 3: Letter mapping, Level 4: YES/NO, Level 5: Keywords, Level 6: Heuristic fallback, Invalid) con counts y porcentajes

**Output esperado:** Tabla formateada para incluir en el paper (Section 3.3 o como tabla nueva)

### B.3 [PRIORIDAD: CRÍTICA] Estadística cluster-aware para p-values

**Motivación:** Cada base clip genera 22 variantes degradadas, induciendo correlación intra-clip. El McNemar test actual podría tratar las 21,340 muestras como independientes, inflando p-values para efectos pequeños (especialmente Qwen3+Hand vs Qwen3+OPRO-LLM, ΔBA = 0.3 pp, p = 0.014).

**Tareas:**
1. Localizar el código que calcula los paired comparisons de Table 5
2. Verificar: ¿trata las 21,340 muestras como i.i.d., o ya agrupa por base clip?
3. Implementar cluster bootstrap para p-values: remuestrear por base clip (no por muestra individual), B = 10,000
4. Re-calcular p-values para todas las comparaciones de Table 5
5. Sanity check adicional: colapsar predicciones por base clip (majority vote sobre las 22 variantes de cada clip) → correr McNemar sobre los ~970 clips colapsados
6. Comparar p-values originales vs cluster-aware vs colapsados

**Output esperado:** Table 5 actualizada (o tabla complementaria). Si algún p-value cambia de significativo a no significativo (especialmente el ΔBA = 0.3 pp), reportar y ajustar el texto correspondiente.

### B.4 [PRIORIDAD: ALTA] Verificar rango de SNR y evaluar extensión

**Motivación:** Todos los modelos adaptados (LoRA variants + Qwen3) tienen SNR75 < −10 dB (censurado por debajo del rango testeado). Esto significa que el test bed no diferencia entre modelos en la dimensión de ruido—todos están en el piso.

**Tareas:**
1. Extraer y reportar BA por nivel de SNR (−10, −5, 0, +5, +10, +20) para las 9 configuraciones como tabla completa
2. Verificar: ¿cuánto baja LoRA+OPRO-Tmpl y Qwen3+Hand realmente a −10 dB? ¿Están ambos >90%?
3. Evaluar factibilidad: ¿puede el pipeline de degradación generar variantes a −15 dB y −20 dB?
4. Si factible y el cómputo es razonable: generar variantes a −15 y −20 dB para los base clips del test set, y evaluar al menos para los 3 mejores sistemas (LoRA+OPRO-Tmpl, Qwen3+Hand, Qwen3+OPRO-LLM)

**Output esperado:** Tabla completa de BA×SNR. Si se extiende el rango: datos adicionales y actualización de Figure 3 y Table 4.

### B.5 [PRIORIDAD: ALTA] Auditar consistencia hiperparámetros reportados vs código

**Motivación:** Verificar que lo que dice el paper coincide exactamente con lo que se implementó.

**Parámetros a verificar (buscar en el código cada uno):**
- **LoRA:** rank=64, α=16, dropout=0.05, lr=5e-5, warmup=100, weight_decay=0, epochs=3, per_device_batch=2, grad_accum=8 (effective batch=16), gradient checkpointing=True, quantization=NF4 4-bit
- **OPRO-LLM:** max_iterations=30, candidates_per_iter=3, top_k=10, early_stopping=5 iters, meta-LLM=Qwen2.5-7B-Instruct, temperature=0.7, top_p=0.9, max_new_tokens=2000
- **OPRO-Template:** iterations=15, K=8 candidates/iter, N=20 mini-dev samples, seed=42, 15 templates in library
- **Evaluación:** greedy decoding (temperature=0), max_new_tokens=128 (open-ended) / 1 (constrained A/B)
- **Audio preprocessing:** 16 kHz mono, 2000 ms container, padding σ=0.0001
- **Training set:** 3,072 clips (balanced SPEECH/NONSPEECH), Silero filter speech_ratio ≥ 0.8
- **OPRO dev set:** 30 base clips × 22 conditions = 660 samples
- **Test set:** 970 base clips × 22 conditions = 21,340 samples (485 SPEECH + 485 NONSPEECH base clips)

**Acción:** Para CADA parámetro, localizar el valor en el código y confirmar o reportar discrepancia.

**Output esperado:** Lista de confirmaciones y/o discrepancias. Cualquier discrepancia se clasifica como CRÍTICA (afecta resultados), ALTA (afecta reproducibilidad), o BAJA (cosmética).

### B.6 [PRIORIDAD: ALTA] Correr Silero VAD bajo el banco psicométrico como baseline de referencia

**Motivación:** Ambos revisores piden un baseline externo no-LALM. Silero VAD ya está integrado en el proyecto (se usa para curación de datos), así que la infraestructura existe. La comparación no es 1:1 (Silero es un modelo ligero frame-level, no un LALM), pero proporciona contexto valioso.

**Tareas:**
1. Localizar la integración actual de Silero en el proyecto (se usa para calcular speech_ratio en curación)
2. Adaptar para evaluación clip-level: pasar cada uno de los 21,340 clips del test set por Silero VAD
3. Para cada clip: obtener probabilidad frame-level → decisión clip-level. Usar el mismo threshold que se usó en curación (0.5) o el que ya esté configurado. Criterio: si speech_ratio > 0.5 → SPEECH, else → NONSPEECH (verificar si hay un criterio más apropiado)
4. Computar las mismas métricas que para los LALMs: BAclip, BAcond (por eje), DT90, SNR75, recall per class, con los mismos métodos de bootstrap y CI
5. Generar outputs en formato compatible para añadir una fila a Table 2 y puntos a Figures 1-4

**Output esperado:** Métricas de Silero en el mismo formato que las 9 configuraciones LALM.

**Nota para el paper:** Se añadirá con disclaimer: "Silero VAD is included as a non-LALM reference point; it operates at orders-of-magnitude lower computational cost with frame-level rather than clip-level granularity. This comparison contextualizes LALM performance but is not intended as a direct competition."

### B.7 [PRIORIDAD: MEDIA] Heatmap de accuracy NONSPEECH por categoría ESC-50 × configuración

**Motivación:** Section 5.8 menciona tres categorías difíciles (laughing 31.8%, coughing 56.6%, crying baby 77.3%) y una comparación puntual con Qwen3 en coughing (90.9%). Una visualización completa mostraría el gradiente de dificultad y las diferencias entre modelos de forma mucho más rica.

**Tareas:**
1. Verificar que las predictions del test set incluyen metadata de categoría ESC-50 para los clips NONSPEECH
2. Si no: cruzar clip IDs con el dataset ESC-50 para recuperar la categoría
3. Para cada configuración (al menos las 4-5 más relevantes: Base+Hand, LoRA+OPRO-Tmpl, Qwen3+Hand, Qwen3+OPRO-LLM, y quizás LoRA+Hand), calcular accuracy NONSPEECH por categoría ESC-50 (50 categorías)
4. Agrupar las 50 categorías en tipos acústicos: vocalizaciones humanas (laughing, coughing, crying baby, sneezing, clapping, breathing, snoring, drinking/sipping), vocalizaciones animales (cat, dog, rooster, crow, frog, chirping birds, insects, hen, pig, cow, sheep), mecánicos/domésticos (clock tick, door knock, mouse click, keyboard, washing machine, vacuum cleaner, clock alarm, can opening), naturales/ambiente (rain, sea waves, crackling fire, crickets, wind, pouring water, thunderstorm, water drops), música/instrumentos (si hay)
5. Generar heatmap (matplotlib/seaborn): filas = categorías ESC-50 agrupadas por tipo, columnas = configuraciones, color = accuracy. Ordenar categorías por dificultad (accuracy promedio)

**Output esperado:** Figura (heatmap) en formato PDF/PNG para inclusión en texto principal + datos numéricos para expandir la discusión en Section 5.8.

### B.8 [PRIORIDAD: MEDIA] Análisis de prompts perdedores agrupados por tipo

**Motivación:** Incluir una tabla de los prompts evaluados durante OPRO (no solo los ganadores) ayuda a otros investigadores a entender el espacio de búsqueda y la sensibilidad de la selección. Pero solo es informativo si hay variación significativa entre tipos de prompt.

**Tareas:**
1. Localizar logs de OPRO-LLM para los 3 modelos: extraer todos los prompts candidatos generados con sus scores en dev
2. Localizar la librería fija de 15 templates de OPRO-Template: extraer todos los templates con sus scores
3. Clasificar cada prompt en una categoría funcional:
   - Directiva binaria: "Answer SPEECH or NONSPEECH"
   - Pregunta abierta: "What type of sound is this?"
   - Multiple-choice A/B: "A) SPEECH B) NONSPEECH"
   - Contrastivo con definiciones: "SPEECH = human voice... NONSPEECH = ..."
   - One-shot con ejemplo: "Example: [audio with speech] → SPEECH"
   - Imperativo focalizado: "Focus on short/noisy clips..."
   - Conservative/Liberal bias framing
4. Para cada tipo: reportar cuántos prompts hay, rango de scores, y si algún tipo domina consistentemente
5. Evaluar: ¿hay variación informativa entre tipos? ¿O todos convergen al mismo score?

**Output esperado:** Si hay variación informativa → tabla para el paper agrupada por tipo de prompt, con score medio y rango. Si no hay variación → nota textual "prompt selection showed low variance across template types" y NO incluir tabla.

**Decisión final:** Gabriel decidirá tras ver los datos si la tabla aporta valor.

---

## FASE C — Integración final

> Se ejecuta después de completar Fases A y B.
> Consiste en incorporar todos los resultados nuevos al manuscrito y hacer revisión de flujo.

### C.1 Actualizar tablas y figuras existentes
- **Table 2:** Añadir fila con métricas de Silero VAD (de B.6)
- **Table 4:** Añadir fila Silero. Si se extiende rango SNR (B.4), actualizar valores censurados
- **Table 5:** Actualizar p-values si cambian con cluster-aware stats (de B.3)
- **Figures 1-4:** Añadir curva/punto de Silero VAD (de B.6). Si se extiende SNR (B.4), añadir puntos

### C.2 Añadir tablas y figuras nuevas
- **Tabla nueva:** Normalization pathway stats (de B.2) — en Section 3.3 o cercana
- **Tabla nueva:** Multi-seed OPRO stability (de B.1) — en Section 5.6 o nueva subsección
- **Tabla nueva (condicional):** Prompts por tipo (de B.8) — solo si es informativa
- **Figura nueva:** Heatmap ESC-50 × configuraciones (de B.7) — en Section 5.8

### C.3 Actualizar texto con nuevos resultados
- **Section 3.3:** Expandir con tabla de normalization pathways. Referenciar tabla nueva
- **Section 3.4.2 o 5.6:** Expandir con datos multi-seed. Añadir párrafo sobre estabilidad
- **Section 5.1 o nueva subsección:** Integrar resultados de Silero como referencia contextual
- **Section 5.4:** Si se extiende SNR, discutir diferenciación entre modelos adaptados
- **Section 5.8:** Expandir con heatmap y análisis por tipo acústico de categoría ESC-50

### C.4 Revisión final de prosa
- Verificar que TODOS los cambios de Fase A están integrados y son coherentes entre sí
- Verificar consistencia de nombres (A.1) tras todos los cambios
- Verificar que las figuras y tablas nuevas están referenciadas en el texto
- Verificar que las referencias bibliográficas son correctas (especialmente si se elimina alguna por la remoción de pyannote)
- Lectura completa de flujo del paper de principio a fin

---

## Orden de ejecución recomendado

```
RONDA 1 — Claude Code: Auditoría + extracción de datos existentes [NO requiere GPU]
├── B.5  Auditar hiperparámetros (rápido, puede revelar problemas bloqueantes)
├── B.2  Normalization pathway stats (re-procesar logs/outputs existentes)
├── B.7  Datos para heatmap ESC-50 (extraer de predictions existentes)
└── B.8  Prompts perdedores (extraer de logs OPRO)

RONDA 2 — Claude Code: Implementación + cómputo medio [requiere GPU]
├── B.3  Estadística cluster-aware (implementar bootstrap, correr)
├── B.4  Verificar/extender rango SNR
└── B.6  Silero bajo banco psicométrico

RONDA 3 — Claude Code: Cómputo alto [requiere GPU significativa]
└── B.1  Multi-seed OPRO-Template (5 seeds × 3 modelos)

EN PARALELO — Gabriel + Claude Opus: Redacción
└── FASE A completa (A.1 a A.15, no depende de ninguna tarea B)

FINAL — Gabriel + Claude Opus + Claude Code
└── FASE C — Integración de todo
```

---

## Decisiones explícitas de NO hacer

| Propuesta rechazada | Razón |
|---|---|
| Expandir con dataset MUSAN/AudioSet para NONSPEECH | Complejidad alta, valor marginal. Se defiende en limitaciones (A.14) |
| Incluir clips SPEECH con ratio Silero 0.2-0.8 (edge cases) | Mismo razonamiento. Se menciona como trabajo futuro en A.14 |
| Implementar degradaciones combinadas (noise+reverb+filter) | Se discute como limitación (A.10). El diseño one-axis-at-a-time sigue metodología psicoacústica estándar |
| Evaluar pyannote | Nunca se usó en el proyecto. Se eliminan menciones (A.13) |
| Verificar figuras en blanco y negro | No relevante para la conferencia destino |
| Crear apéndice | Todo va en texto principal |

---

## Plantilla de PROGRESS.md

Al crear el archivo `PROGRESS.md`, usar esta estructura:

```markdown
# Progreso del Plan de Mejoras — Paper LALM VAD
## Archivo de referencia: paper_improvement_plan.md

---

## Registro de tareas

| Fecha | Tarea ID | Estado | Notas |
|-------|----------|--------|-------|
| | | | |

---

## Detalle por tarea

### [Tarea ID] — [Nombre corto]
- **Estado:** PENDIENTE / EN PROGRESO / COMPLETADO / BLOQUEADO / REQUIERE DECISIÓN
- **Fecha inicio:**
- **Fecha fin:**
- **Archivos generados:**
- **Resumen de lo hecho:**
- **Problemas encontrados:**
- **Discrepancias detectadas:** (si aplica, con severidad: CRÍTICO / ALTO / BAJO)

---

## Discrepancias encontradas (consolidado)

| Tarea | Parámetro | Valor en paper | Valor en código | Severidad |
|-------|-----------|----------------|-----------------|-----------|
| | | | | |
```
