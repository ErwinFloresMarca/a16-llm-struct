# 02 — Modelo: Qwen3.8-Flash-Next

> Open-weights de Alibaba. **Preview temprano de la arquitectura Qwen4.**
> Juega el mismo rol que Qwen3-Next jugó para la serie 3.5.

Lanzamiento: **26 de agosto de 2026**. Licencia propia (no estándar open-source,
revisar condiciones de uso comercial antes de producir con él).

## Parámetros

| Bloque | Params | Rol |
|---|---|---|
| Modelo principal MoE | **125B** (6B activos por token) | Cómputo |
| Tabla N-gram / PLE ("Engram") | **51B** | Lookup de contexto local |
| Head MTP | **4B** | Drafting especulativo |
| **Total** | **~180B** | |

**Advertencia de MoE:** a pesar de que solo 6B params se activan por token, **el
modelo entero debe estar en memoria en algún sitio** (VRAM o RAM). El ahorro
es de cómputo, no de memoria.

## Por qué se presta al tiering VRAM/RAM/SSD

La clave son tres bloques con **patrones de acceso distintos**:

### 1. Core de cómputo (~128B params)

MoE experts + attention + GDN + QSA + shared experts + gate.
Lectura **densa cada token** (los ~6B activos se eligen de entre todos).
→ **Debe estar residente. VRAM / unified memory.**

### 2. Tabla N-gram / PLE (51B params)

Un **gather disperso**: por token se toman ~16 filas de una tabla de
~320 millones, direccionadas por **hash determinista**.

- Tráfico por token: ~3-5 KB.
- A 30-40 t/s: **unos pocos MB/s de lecturas aleatorias**.

→ Eso es exactamente lo que un **page cache sobre NVMe** maneja bien. **No
necesita estar residente**; se puede servir desde RAM o mapear desde SSD.

⚠️ **Es acceso hasheado, o sea aleatorio puro** — no secuencial. De ahí que vaya
a RAM/SSD y a ningún lado más. También es la razón por la que no se puede
cuantizar agresivamente (el acceso aleatorio no tolera pérdida de resolución);
Unsloth la deja en Q8_0 o IQ4_NL mínimo.

### 3. Head MTP (4B params, 2.6 GB en Q8_0)

Drafter especulativo. Verificación exacta → **la salida no cambia, solo la
velocidad: 1.3× a 1.7×** en concurrencia baja. Ojo: a concurrencia 8 se midió
como **pérdida neta** (~0.81× a 0.87×). Para servir a mucha gente, apagar MTP.

## El detalle que hace barato el contexto largo

De las **48 capas**, solo **12** son de atención real (QSA) con KV cache
creciente. Las otras **36** son GDN (atención lineal) con estado recurrente de
**tamaño fijo**.

| | |
|---|---|
| KV por token | ~24 KiB |
| Contexto nativo completo (262,144 tokens) | **~6-8 GB** |
| Contexto extendido (1M con YaRN) | factible en disco, ~32 GB |

> Para un modelo denso de este tamaño, 262K de contexto costaría **~10× más**.
> El contexto largo no es el problema aquí; el problema es **meter los experts**.

## Tamaños por cuantización (Unsloth GGUF)

| Build | Descarga | Máquina típica |
|---|---|---|
| `UD-IQ1_S` | 72.5 GB | 96 GB — "para probar que bootea" |
| `UD-IQ1_M` | 74.5 GB | 96 GB |
| `UD-Q2_K_XL` | 78.9 GB | 128 GB unified |
| `UD-IQ3_XXS` | 82.0 GB | 128 GB, contexto corto |
| `UD-Q3_K_XL` | 90.0 GB | 128 GB, tight |
| `UD-IQ4_XS` | 93.7 GB | 128 GB / 4×3090 |
| `UD-Q4_K_XL` | 111.3 GB | 192 GB |
| `UD-Q5_K_XL` | 158.3 GB | 192 GB unified |
| `UD-Q6_K_XL` | 169.2 GB | 256 GB |
| `Q8_0` | 188.2 GB | servidor |
| `BF16` | 354.0 GB | nodo GPU |

**Conclusión para la A16:** 64 GB de VRAM **no alcanza para ningún cuant**. El
más chico (72.5 GB) ya es 12 GB mayor que toda la tarjeta, y con calidad
destruida. Los experts van siempre a la RAM del host.

## Formatos y runtimes

| Formato | Runtime | Caso de uso |
|---|---|---|
| GGUF | llama.cpp, Ollama, LM Studio, Unsloth Desktop | CPU, unified memory, híbrido |
| FP8 (172.78 GiB) | vLLM 0.29+, SGLang | Servir en serio, producción |
| AWQ/GPTQ | vLLM (CUDA/ROCm) | Servido en GPU |
| Transformers | HF | Investigación |

**Soporte de arquitectura:** `qwen4_exp` entró a llama.cpp master el
**27 de agosto de 2026** (PR #27742). Builds empaquetados anteriores **no
cargan el archivo**. Compilar desde master o usar un build más nuevo que
ese merge.

## Contexto de producción (API)

Para comparar, la API oficial `qwen3.8-flash`:

| Parámetro | Valor |
|---|---|
| Contexto | 1,000,000 tokens |
| Salida máx. | 131,072 tokens |
| CoT máx. | 262,144 tokens |
| Thinking | activo por defecto |
| `reasoning_effort` | `xhigh` (default) / `medium` / `low` |
| `preserve_thinking` | activo por defecto |

Siguiente: [03 — Arquitectura de despliegue](03-arquitectura-despliegue.md)
