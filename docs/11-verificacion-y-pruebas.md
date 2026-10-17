# 11 — Verificación y pruebas

> Los 2 perfiles de cliente ya están definidos. Este documento verifica qué
> candidato sirve para cada uno, y qué **no** pude verificar.

## Los 2 perfiles

| | Cliente A | Cliente B |
|---|---|---|
| **Uso** | Llamadas agénticas, chat con tareas, **tool calling** | **Coding** |
| **Prioridad** | Tool calling confiable + latencia baja | Calidad de código |
| **Partición** | A | B |

---

## ⚠️ Qué verifiqué y qué NO

**Sí verifiqué** (contra las fichas oficiales de los modelos): existencia,
tamaños reales, licencias, soporte de runtime y flags de tool calling.

**NO pude probarlo en la A16.** Esta máquina es una laptop con **RTX 2070 de
8 GB** (`hp-omen-15`); la A16 está en el server del trabajo y no es alcanzable
desde acá. Además acá no hay vLLM ni llama.cpp instalados.

Por eso el entregable no es un resultado, sino **un script que produce el
resultado en el server**. Los números de `docs/10` siguen siendo estimaciones
hasta que corras `30-smoke-tool-calling.py` allá.

---

## Hallazgo que cambia la recomendación

La restricción de 16 GB choca con una realidad de los runtimes:

> **vLLM sirve safetensors (bf16/fp8), no GGUF cuantizado. Y bf16 de un 30-35B
> pesa 60-71 GB.** En una partición de 16 GB no entra. La cuantización a 3-4
> bits, que es lo que lo hace entrar, hoy solo la da bien **llama.cpp**.

**Excepción:** `gpt-oss-20b` viene cuantizado **de fábrica** en MXFP4 (~12.8 GB),
así que vLLM lo sirve directo, sin GGUF, con su parser de tools.

### El trade-off

| | vLLM | llama.cpp |
|---|---|---|
| Tool calling | **Parsers dedicados** (`openai`, `qwen3_xml`, `qwen3_coder`) — robusto | `--jinja` aplica el chat template — más básico |
| Cuantización | Solo si el checkpoint ya viene cuantizado (FP8, AWQ, MXFP4) | **GGUF de 2 a 8 bits**, control total |
| Cabe un 35B en 16 GB | ❌ | ✅ (IQ3) |
| Concurrencia / prefix cache | **Excelente** | Correcto |

**Conclusión:** para el cliente de tool calling, elegir un modelo que vLLM
pueda servir. Para el de coding, aceptar llama.cpp a cambio de que entre.

---

## Cliente A → `gpt-oss-20b` (vLLM)

| Dato | Valor |
|---|---|
| Repo | `openai/gpt-oss-20b` |
| Arquitectura | MoE 20.9B totales / **3.6B activos** |
| Peso real | **~12.8 GB** (MXFP4 nativo) |
| Licencia | **Apache 2.0** |
| vLLM | ✅ Nativo. `--tool-call-parser openai` |
| t/s estimado | **30-45** |

**Por qué este:** es el **único** candidato que entra en 16 GB **sin cuantización
manual**, tiene el parser de tool calling más probado, y con 3.6B activos es el
más rápido del lote que cabe con margen para contexto. Entrenado específicamente
para function calling y operaciones agénticas.

```bash
bash scripts/10-serve-agentic-gptoss.sh 2 8001
```

### ⚠️ Gotchas de gpt-oss + tools (verificados)

1. **Tool calling dentro del chain-of-thought.** El modelo puede emitir la tool
   call *dentro* del razonamiento. En multi-turno hay que **devolver el campo
   `reasoning`** junto con el resultado de la tool, o la precisión cae.
2. **Python 3.12.** Hay validaciones documentadas que exigen Python 3.12 para
   function calling con gpt-oss. Si el parsing falla, mirar ahí primero.
3. **Necesita `--enable-auto-tool-choice` + `--tool-call-parser openai`.** Sin
   ambos, devuelve texto plano y el cliente no ve `tool_calls`.

---

## Cliente B → `Ornith-1.5-35B-A3B` (llama.cpp)

| Dato | Valor |
|---|---|
| Repo | `ornith-ai/Ornith-1.5-35B-A3B` |
| Arquitectura | MoE ~36B totales / **3B activos**, *hybrid linear + full attention* |
| Licencia | **MIT** (la más permisiva) |
| vLLM | ≥ 0.19.1 — pero **bf16 = ~71 GB**, no entra en 16 GB |
| Cuant que sí entra | `IQ3_XXS` = **~13.7 GB** |
| Terminal-Bench 2.1 | **67.8** |
| Extra | Multimodal, MTP speculative decoding, 256K de contexto |

**Por qué este:** tiene la **misma familia de arquitectura** que Flash-Next
(híbrida lineal + atención completa), y el mejor benchmark de terminal del lote:

| Modelo | Terminal-Bench 2.1 |
|---|---|
| **Ornith 1.5 35B-A3B** | **67.8** |
| Qwen3.5-397B (mucho más grande) | 53.5 |
| Qwen3.6-35B-A3B | 52.5 |
| Muse-Glimmer-30B | 51.7 |
| Gemma-4-31B | 42.1 |

```bash
bash scripts/20-serve-coding-ornith.sh 3 8002 IQ3_XXS
```

> **Ojo:** el README oficial de Ornith recomienda 2×80 GB en bf16. En 16 GB vas
> con GGUF IQ3 y con contexto de 32K. Es un uso no oficial — de ahí la
> importancia de medir antes de comprometer.

### Alternativas para el coding

| Modelo | Peso | Por qué |
|---|---|---|
| **Qwen3.8-27B** (IQ3_S) | 13.8 GB | Misma arquitectura exacta que Flash-Next, **Apache 2.0**, visión. Pero **denso** → 10-15 t/s |
| **Qwen3-Coder-30B-A3B** (UD-Q3_K_XL) | ~15 GB | Especialista en coding de Qwen, 3.3B activos. vLLM con `qwen3_coder` si hubiera checkpoint cuantizado |
| **`continuum-ai/qwen3-coder-30b-a3b-compacted-19b-256k`** | ~10.6 GB (Q4) | Poda a 19B con **256K de contexto** → entra cómodo y deja VRAM para contexto largo |

---

## Cómo correr la verificación

```bash
# 0. En el server, primero esto — responde los bloqueantes de docs/08
bash scripts/00-preflight.sh

# 1. Levantar el cliente A (tool calling) en la partición libre
bash scripts/10-serve-agentic-gptoss.sh 2 8001

# 2. Verificar que realmente hace tool calling
python3 scripts/30-smoke-tool-calling.py http://localhost:8001/v1 agentic

# 3. Levantar el cliente B (coding)
bash scripts/20-serve-coding-ornith.sh 3 8002 IQ3_XXS
python3 scripts/30-smoke-tool-calling.py http://localhost:8002/v1 coding
```

### Qué devuelve el smoke test

1. ¿Responde? → lista de modelos
2. ¿Emite `tool_calls`? → nombre + argumentos, y valida que el JSON sea válido
3. ¿Cierra el ciclo? → manda el resultado de la tool y verifica la respuesta final
4. **TTFT y decode t/s** (streaming)
5. **Prefill t/s** con prompt largo

Pasa solo si las 3 primeras salen bien **y** los argumentos son JSON válido.

---

## Cómo interpretar los resultados

### Si `tool_calls` no aparece

En orden de probabilidad:

```bash
# vLLM: faltan los dos flags
--enable-auto-tool-choice --tool-call-parser openai   # gpt-oss
--enable-auto-tool-choice --tool-call-parser qwen3_xml # Ornith/Qwen

# llama.cpp: falta el chat template
--jinja

# ambos: el alias no coincide
--served-model-name agentic   # vLLM   -> usar "agentic" en las requests
--alias coding                # llama.cpp -> usar "coding"
```

### Si el decode está muy por debajo de lo estimado

Sospechosos, en orden:

1. **Throttling.** `nvidia-smi -q -d POWER,PERFORMANCE | grep -i Active`
2. **El modelo no entró entero en VRAM.** Si hay offload parcial, cada token
   cruza PCIe. Confirmar con `nvidia-smi` que la VRAM está al tope durante la
   generación.
3. **El KV cuantizado se está des-cuantizando por token.** Probar sin
   `-ctk q8_0 -ctv q8_0`.
4. **La partición está compartida.** La A16 es una tarjeta VDI: si las otras 2
   particiones hacen video, el presupuesto de 250 W se reparte.

### Si el prefill es lento pero el decode está bien

Es **esperado** en esta tarjeta: el prefill es cómputo puro y la A16 tiene solo
**1280 CUDA cores por partición**. El decode depende de ancho de banda y sufre
menos. Mitigación: **prefix caching** (ya está en los scripts). El system prompt
se procesa una vez y después es gratis.

---

## Tabla de veredicto

| Pregunta | ¿Verificado? | Cómo |
|---|---|---|
| Ornith existe y es MIT | ✅ | repo HF `ornith-ai/Ornith-1.5-35B-A3B` |
| Ornith soporta vLLM | ✅ | ≥ 0.19.1 (pero bf16 71 GB → no entra) |
| Ornith hace tool calling | ✅ documentado, ❌ no probado | `--tool-call-parser qwen3_xml` |
| gpt-oss-20b pesa 12.8 GB | ✅ | MXFP4 nativo, ficha de OpenAI |
| gpt-oss-20b cabe en 16 GB | ⚠️ **justo** | 12.8 GB de pesos + KV → medir |
| Qwen3-Coder-30B-A3B soporta vLLM | ✅ | parser `qwen3_coder` |
| Qwen3-Coder cabe en 16 GB | ❌ a Q4 (~18.6 GB) | requiere Q3/IQ3 |
| Qwen3.8-27B comparte arquitectura con Flash-Next | ✅ | 16 × (3×GDN → FFN) + 1×GA → FFN |
| Velocidades de docs/10 | ❌ **estimadas** | correr el smoke test |

Ver: [10 — Catálogo de modelos](10-catalogo-de-modelos.md) ·
[09 — Escenario real](09-escenario-real-a16.md)
