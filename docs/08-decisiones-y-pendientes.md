# 08 — Decisiones y pendientes

## Registro de decisiones (ADR)

### ADR-001 — No correr Flash-Next en producción sobre la A16

**Estado:** Propuesta sólida, pendiente de validación con datos del server.

**Contexto:** Se quiere un asistente de cliente final con ~200 usuarios
simultáneos y un asistente para un equipo de 20 devs, corriendo localmente en
una NVIDIA A16.

**Decisión:** No usar Flash-Next en la A16 para ninguna carga de producción.

**Razón:**

1. **Capacidad.** La A16 tiene 64 GB de VRAM. El quant más chico de
   Flash-Next es 72.5 GB. No cabe.
2. **Consecuencia.** Los experts MoE van siempre a la RAM del host, y el decode
   queda atado al ancho de banda de la **RAM**, no al de la GPU. Los 200 GB/s
   de la A16 quedan inutilizados.
3. **Resultado.** 1-3 usuarios concurrentes. No 20, no 200.
4. **Alternativa más barata.** Para 200 usuarios concurrentes se necesitan
   ~4-8 GPU clase H200 (decenas a cientos de miles de USD). La misma carga en
   Qwen Cloud cuesta ~USD 30-60/mes.

**Consecuencias:**
- La A16 se reorienta a Capa 2 (equipo de dev) en data-parallel con modelo chico.
- La producción va a API detrás de un gateway.
- Flash-Next queda como laboratorio y experimentación.

**Cómo revertir:** Si aparece requisito de compliance que prohíba la nube,
retomar el diseño desde "qué modelo chico corremos on-prem". No desde
"¿cómo entro Flash-Next en la A16?".

---

### ADR-002 — Data-parallel, no tensor-parallel, sobre la A16

**Decisión:** Las 4 particiones de la A16 corren **4 réplicas independientes**
con balanceo de carga, no un modelo sharded.

**Razón:** La A16 no tiene NVLink. Tensor-parallel exige un *allreduce* por
capa sobre PCIe Gen4 x16 (~32 GB/s) → overhead fijo que destruye la latencia.
Data-parallel tiene cero comunicación entre GPUs y escalado lineal en
throughput.

---

### ADR-003 — vLLM para la capa de dev, llama.cpp solo para Flash-Next

**Decisión:** La Capa 2 (réclicas) usa **vLLM**. La Capa 3 (Flash-Next
experimental) usa **llama.cpp** con GGUF.

**Razón:**

| | vLLM | llama.cpp |
|---|---|---|
| Continuous batching | **Sí** | Limitado |
| PagedAttention | **Sí** | No |
| Prefix caching | **Sí** | No |
| Métricas de serving | **Sí** | Básicas |
| Soporte de MoE gigante en CPU/RAM | Limitado | **Sí** (`--cpu-moe`) |
| GGUF y tiering SSD | No | **Sí** (mmap lazy) |

Servir a varias personas → vLLM. Cargar un modelo que no cabe en VRAM → llama.cpp.

---

### ADR-004 — Apagar MTP en cualquier escenario de concurrencia

**Decisión:** MTP (`--spec-type draft-mtp`) **solo** a concurrencia 1.

**Razón:** Medido. 1.3-1.7× de ganancia a concurrencia 1, pero **~0.81-0.87×
de pérdida neta** a concurrencia 8. Un modelo ocupado no tiene capacidad ociosa
para que un drafter la aproveche.

---

### ADR-005 — Dos réplicas de modelo chico, no Flash-Next

**Estado:** Aceptada (con datos reales del server).

**Contexto:** El server tiene **2 de 4 particiones de la A16 libres** (32 GB
VRAM) y **16 GB de RAM libres** (ampliable). El alcance actual es de **1 a 2
clientes frecuentes**, no 200 ni 20.

**Decisión:** Correr **2 réplicas vLLM data-parallel**, una por partición libre,
con un modelo chico que entre en 16 GB. **No** intentar Flash-Next.

**Razón:**

1. Flash-Next IQ3 necesita ~55 GB residentes (82 GB − 27 GB de tabla en SSD)
   contra 48 GB disponibles (32 VRAM + 16 RAM). No entra.
2. Con 16 GB de RAM, el mmap de la tabla N-gram hace *thrash*: el lazy-mode
   depende del page cache, y no hay suficiente.
3. El setup de 2 réplicas requiere **0 GB de RAM extra** (todo en VRAM) y corre hoy.
4. El prefill es mucho mejor: modelo chico + prefix caching, contra Flash-Next
   atado a los 5120 CUDA cores de la A16.

**Consecuencias:**
- 0 inversión para arrancar.
- Flash-Next queda como laboratorio, y solo después de subir la RAM a 64 GB.
- Cada cliente tiene su propia réplica → sin competencia entre ellos.

**Cómo revertir:** Si se amplía la RAM a 64-96 GB y se quiere capacidad frontier
para un caso puntual, ver [05](05-runbook-llamacpp-en-la-a16.md).

---

### ADR-006 — Modelo: MoE de ~3B activos, o Qwen3.8-27B si prima la calidad

**Estado:** Superada por **ADR-007** (una vez conocidas las tareas de los clientes).

**Contexto:** Una partición de A16 tiene **200 GB/s**, ~1/5 de una RTX 4090
(1008 GB/s). La velocidad en decode es puramente ancho de banda sobre los
parámetros **activos**, no sobre los totales.

**Decisión:** Priorizar modelos con **muy pocos parámetros activos**:

| Perfil | Modelo | Activos | t/s est. |
|---|---|---|---|
| Máxima velocidad | `LFM2.5-8B-A1B` | 1.5B | 50-70 |
| Mejor balance | `Ornith 1.5 35B-A3B` (MIT) | 3B | 35-50 |
| Alternativa | `gpt-oss-20b` | 3.6B | 30-45 |
| Mejor calidad | **`Qwen3.8-27B`** (Apache 2.0) | 27B (denso) | 10-15 |

**Razón:** El descubrimiento clave es que **Qwen3.8-27B tiene exactamente la
misma arquitectura híbrida que Flash-Next** (Gated DeltaNet + Gated Attention,
layout `16 × (3 × GDN → FFN) + 1 × GA → FFN`), y solo 16 de 64 capas tienen KV
creciente. Es "el Flash-Next que sí entra", con licencia **Apache 2.0** en vez de
la Qwen Community License 1.0 — relevante para un producto que se factura a
clientes.

**Consecuencias:**
- Con 2 particiones y 2 clientes de perfiles distintos, se puede dar **un modelo
  distinto a cada uno**.
- Cuantizar el KV (`--kv-cache-dtype fp8`) es clave para estirar el contexto
  en 16 GB.

Ver: [10 — Catálogo de modelos](10-catalogo-de-modelos.md)

---

### ADR-007 — Un modelo distinto por cliente

**Estado:** Aceptada.

**Contexto:** Las tareas de los 2 clientes están definidas y son **distintas**:

| Cliente | Uso |
|---|---|
| **A** | Llamadas agénticas, chat con tareas, **tool calling** |
| **B** | **Coding** |

**Decisión:** Darle a cada uno el modelo que corresponde a su perfil, en su
propia partición:

| Cliente | Modelo | Runtime | Peso | t/s est. | Licencia |
|---|---|---|---|---|---|
| **A** — agéntico | `gpt-oss-20b` | **vLLM** | 12.8 GB (MXFP4 nativo) | 30-45 | Apache 2.0 |
| **B** — coding | `Ornith-1.5-35B-A3B` | **llama.cpp** | 13.7 GB (IQ3_XXS) | 35-50 | **MIT** |

**Razón — la restricción que decide todo:**

> **vLLM sirve safetensors, no GGUF.** El checkpoint bf16 de Ornith pesa ~71 GB
> y el de Qwen3-Coder-30B ~61 GB. No entran en 16 GB. La cuantización de 3-4 bits,
> que es lo único que los hace entrar, hoy solo la da bien **llama.cpp**.

Por eso:

- **Cliente A** necesita tool calling **confiable** → vLLM, por sus parsers
  dedicados. Y el único de los candidatos que vLLM puede servir **directo** es
  `gpt-oss-20b`, porque viene **cuantizado de fábrica en MXFP4** (~12.8 GB).
- **Cliente B** prioriza capacidad de código → `Ornith-1.5-35B-A3B`, que tiene
  **Terminal-Bench 2.1 = 67.8** (el mejor del lote: Qwen3.6-35B-A3B 52.5,
  Gemma-4-31B 42.1) y **licencia MIT**. Se sirve con GGUF IQ3 vía llama.cpp.

**Consecuencias:**
- Dos runtimes distintos en el server (vLLM + llama.cpp). Es más superficie
  operativa, pero cada uno está en su propia partición y no se estorban.
- El tool calling del Cliente B es vía `--jinja` (chat template), **más básico**
  que los parsers de vLLM. Si el cliente exige robustez total, cambiar el modelo
  de coding por uno que vLLM pueda servir cuantizado.
- **Ninguna velocidad está medida.** Ver [11](11-verificacion-y-pruebas.md) para
  el script de verificación.

Ver: [11 — Verificación y pruebas](11-verificacion-y-pruebas.md)

---

## Pendientes — datos a confirmar

### ✅ Resueltos

| # | Pregunta | Respuesta |
|---|---|---|
| 1 | ¿Las 4 particiones están libres? | **No: 2 libres, 2 ocupadas** → 32 GB VRAM |
| 2 | ¿Cuánta RAM? | **16 GB libres**, ampliable (canales sin confirmar) |
| 3 | ¿200 usuarios concurrentes o registrados? | **Ninguno todavía**: el alcance real son **1-2 clientes**. El escenario de 200 pasa a ser horizonte futuro, no alcance actual. |
| 4 | **¿Qué tareas hacen los clientes?** | **Cliente A: agéntico + chat con tareas + tool calling. Cliente B: coding.** → ver [ADR-007](#adr-007--un-modelo-distinto-por-cliente) y [11](11-verificacion-y-pruebas.md) |

### 🔴 Bloqueantes

| # | Pregunta | Impacto |
|---|---|---|
| 5 | **¿Cuántos canales de RAM tiene la placa y cuánto soporta?** | Decide si Flash-Next es alguna vez viable y si conviene ampliar. |

### 🟡 Importantes

| # | Pregunta | Impacto |
|---|---|---|
| 6 | **¿Hay compliance que prohíba mandar datos a la nube?** | Si sí, la Capa 1 se cae y el diseño se reorienta por completo. |
| 7 | **¿Qué CPU y cuántos núcleos?** | Afecta el prefill con experts en RAM. |
| 8 | **¿NVMe disponible y libre?** | Espacio para cuants + page cache de la tabla N-gram. Velocidad del NVMe afecta directamente. |
| 9 | **¿Las 2 particiones ocupadas comparten los 250 W de la tarjeta?** | Riesgo de throttling si se suma inferencia. Verificar con `nvidia-smi -q -d POWER,PERFORMANCE`. |

### 🟢 Deseables

| # | Pregunta |
|---|---|
| 10 | ¿Tamaño del presupuesto operativo mensual para API? |
| 11 | ¿Requisitos de latencia (primer token) para los clientes? |
| 12 | ¿Idioma y tipo de tareas dominantes? (afecta el modelo elegido) |

---

## Próximos pasos sugeridos

**Inmediato — correr el preflight en el server:**
1. `bash scripts/00-preflight.sh` → responde los pendientes #5 a #9 de una sola vez
   (particiones libres, NVLink, throttling, RAM y canales, NVMe, runtimes).

**Corto plazo — levantar y medir (ADR-007):**
2. `bash scripts/10-serve-agentic-gptoss.sh <partición_libre> 8001` (Cliente A).
3. `python3 scripts/30-smoke-tool-calling.py http://localhost:8001/v1 agentic`
   → **confirmar que emite `tool_calls` válidas** y medir t/s.
4. `bash scripts/20-serve-coding-ornith.sh <otra_partición> 8002 IQ3_XXS` (Cliente B).
5. `python3 scripts/30-smoke-tool-calling.py http://localhost:8002/v1 coding`.
6. Anotar los números reales en [11](11-verificacion-y-pruebas.md) y corregir las
   estimaciones de [10](10-catalogo-de-modelos.md).

**Medio plazo:**
7. Cargar el gateway LiteLLM (aunque sea solo local, para tener logging y keys)
   ([07](07-runbook-gateway.md)).
8. Comparar los 2 modelos contra el set de tareas reales de cada cliente y
   ajustar la elección.

**Horizonte:**
9. Si el alcance crece a muchos usuarios → evaluar la ruta API
   ([03](03-arquitectura-despliegue.md), [04](04-costos-y-escala.md)).
10. Si aparece necesidad de capacidad frontier local → ampliar RAM a 64-96 GB y
    recién ahí evaluar Flash-Next ([05](05-runbook-llamacpp-en-la-a16.md)).
