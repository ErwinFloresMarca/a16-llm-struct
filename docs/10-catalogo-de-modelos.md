# 10 — Catálogo de modelos para la A16

> **La clave: la misma arquitectura de Flash-Next existe en tamaños que sí te
> caben.** Lo que da velocidad no son los parámetros totales, son los **activos**.

## La fórmula que decide la velocidad

```
t/s ≈ ancho_de_banda_de_la_partición ÷ (params_activos × bytes_por_param)
```

### 🔴 El límite duro de esta tarjeta

| GPU | Ancho de banda | vs. A16 |
|---|---|---|
| **A16 (1 partición)** | **200 GB/s** | 1× |
| RTX 4080 | 736 GB/s | 3.7× |
| RTX 4090 | 1008 GB/s | 5.0× |
| A100 80GB | 2039 GB/s | 10.2× |

**Una partición de A16 tiene ~1/5 del ancho de banda de una RTX 4090.** Un modelo
que corre a 50 t/s en una 4090 corre a ~10 t/s acá. Esto no se arregla con
software: hay que elegir modelos con **muy pocos parámetros activos**.

## La familia Qwen3.8 comparte la arquitectura

| Modelo | Tipo | Total | Activos | ¿Cabe en 16 GB? |
|---|---|---|---|---|
| Qwen3.8-Flash-Next | MoE | ~180B | 6B | ❌ (mín. 72.5 GB) |
| Qwen3.8-2.4T-A95B | MoE | 2.4T | 95B | ❌ |
| **Qwen3.8-27B** | **Dense** | **27B** | **27B** | ✅ (13.8 GB @ IQ3_S) |

### Qwen3.8-27B: el Flash-Next que sí te cabe

Tiene **exactamente la misma arquitectura híbrida** que Flash-Next:

```
64 capas = 16 × ( 3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN) )

Gated DeltaNet:   48 heads lineales (V) / 16 (QK), head dim 128
Gated Attention:  24 Q heads / 4 KV heads, head dim 256
Contexto:         262,144 nativo → 1M con YaRN
Entrada:          texto, imagen, video (vision-language nativo)
```

**Y la ventaja del KV cache se hereda:** solo **16 de 64 capas** tienen KV
creciente; las otras 48 son Gated DeltaNet con estado fijo. Por eso puede
sostener 262K de contexto sin explotar memoria.

| Propiedad | Valor |
|---|---|
| Licencia | **Apache 2.0** ← clave para un producto con clientes |
| Liberado | 14-ago-2026 |
| Benchmarks | SWE-bench Pro **61.7** · Terminal-Bench 2.1 **73.0** · OSWorld 84.3 · GPQA-D 89.2 · LiveCodeBench v6 90.3 |
| Cuant para 16 GB | `AD-IQ3_S` = **13.8 GB** |
| Contexto práctico | ~8-32K según cuant y KV |

> **Ojo con la licencia.** Qwen3.8-27B es **Apache 2.0**. Flash-Next es
> **Qwen Community License 1.0**. Para un asistente que facturás a clientes,
> Apache 2.0 elimina riesgo legal.

**La contra:** es **denso**. Lee 27B por token → ~11-14 GB/token a IQ3 →
**~10-15 t/s**. Calidad alta, velocidad modesta.

## Candidatos ordenados por velocidad

Estimaciones desde la fórmula con 200 GB/s y eficiencia real del 35-50%.
**Medir siempre** — ningún cálculo reemplaza una medición.

| Modelo | Total | Activos | Cuant | Peso | t/s est. | ¿Cabe? | Licencia |
|---|---|---|---|---|---|---|---|
| **LFM2.5-8B-A1B** | 8.3B | **1.5B** | Q6 | 7.0 GB | **50-70** | ✅ holgado | — |
| **DeepSeek-Coder-V2-Lite** | 16B | **2.4B** | Q4 | 10 GB | **45-60** | ✅ | — |
| **Ornith 1.5 35B-A3B** | 34.7B | **3B** | IQ3_XXS | 13.7 GB | **35-50** | ✅ justo | **MIT** |
| **Qwen3.6-35B-A3B** | 35B | **3B** | IQ3_XXS | ~14 GB | **35-50** | ✅ justo | Apache 2.0 |
| **gpt-oss-20b** | 20.9B | **3.6B** | MXFP4 | 12.8 GB | **30-45** | ✅ | Apache 2.0 |
| **Qwen3-Coder-30B-A3B** | 30.5B | 3.3B | Q3 | ~16 GB | 30-45 | ⚠️ muy justo | Apache 2.0 |
| **Gemma 4 26B-A4B** | 26B | 4B | IQ4_XS | 13.9 GB | 25-40 | ✅ | — |
| **Qwen3.8-27B** (dense) | 27B | 27B | IQ3_S | 13.8 GB | **10-15** | ✅ | **Apache 2.0** |

## Las 3 recomendaciones

### 🥇 Mejor balance → `Ornith 1.5 35B-A3B`

- **Misma familia de arquitectura**: MoE con *hybrid linear + full attention*
  (el mismo truco que Flash-Next).
- 3B activos → **35-50 t/s**. 35B de capacidad total.
- Hecho específicamente para **coding agéntico y uso de terminal**.
- **Licencia MIT** — la más permisiva del lote.
- 13.7 GB en `AD-IQ3_XXS-IQ2_S` → entra con margen para contexto.
- ⚠️ Verificar disponibilidad y soporte en vLLM antes de comprometer.

### 🥈 Mejor calidad → `Qwen3.8-27B`

- **La misma arquitectura exacta que Flash-Next.** Es literalmente su hermano
  denso, y la evolución directa del linaje.
- **Apache 2.0**, vision-language nativo, 262K de contexto nativo.
- SWE-bench Pro 61.7 (por encima de Opus 4.6 Max en ese benchmark).
- Sacrificio: **~10-15 t/s**. Para asistente con prompts largos y respuestas
  pensadas, es tolerable. Para chat interactivo, se siente lento.
- **El pick si la calidad importa más que la velocidad**, o si el cliente hace
  tareas de análisis/documentos en vez de conversación rápida.

### 🥉 Máxima velocidad → `gpt-oss-20b` o `LFM2.5-8B-A1B`

- `gpt-oss-20b`: 3.6B activos, MXFP4 nativo (12.8 GB), **30-45 t/s**.
  Excelente en tools/function calling y razonamiento. Apache 2.0.
- `LFM2.5-8B-A1B`: 1.5B activos, 7 GB, **50-70 t/s**. El más rápido que corre
  con holgura. Para chat simple y tool use. Se puede subir a Q8 y sigue entrando.
- **El pick si el cliente espera respuestas instantáneas.**

## Estrategia para 2 particiones, 2 clientes

Si los clientes tienen perfiles distintos, **no los fuerces al mismo modelo**:

| Perfil del cliente | Partición | Modelo |
|---|---|---|
| Análisis, documentos, calidad | A | **Qwen3.8-27B** (IQ3_S) |
| Chat interactivo, agentes, velocidad | B | **Ornith 35B-A3B** o **gpt-oss-20b** |

## Contexto largo: cuánto KV te queda

La arquitectura híbrida hace que el KV crezca mucho más lento:

| Modelo | Capas con KV creciente | KV por token (FP16) | 32K | 262K |
|---|---|---|---|---|
| Qwen3.8-Flash-Next | 12 de 48 | ~24 KB | ~0.8 GB | ~6-8 GB |
| Qwen3.8-27B | 16 de 64 | ~64 KB | ~2 GB | ~17 GB |

**Cuantizá el KV** para estirar el contexto en 16 GB:

```bash
--kv-cache-dtype fp8        # vLLM
# o en llama.cpp:
-ctk q8_0 -ctv q8_0         # reduce el KV a la mitad
```

Con KV en q8_0, el 27B a 262K baja de ~17 GB a **~8.4 GB**. Ahí sí entra.

## Cómo elegir de verdad

Ninguna tabla reemplaza una prueba. Protocolo:

1. Bajá 2-3 candidatos.
2. Corré el **mismo set de tareas reales** de tus clientes con cada uno.
3. Medí por separado: **prefill t/s** y **decode t/s**.
4. Anotá: contexto usado, cuant, VRAM pico, si hubo swap.
5. Elegí por **tiempo total de tarea**, no por t/s en aislamiento.

> Un modelo a 15 t/s que resuelve la tarea en un intento le gana a uno a 45 t/s
> que necesita tres correcciones.
