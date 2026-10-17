# a16-struct

Diseño de despliegue de IA local sobre el **NVIDIA A16** del server de trabajo.

**Alcance real (actualizado):** 2 particiones libres de la A16 y **2 clientes**:

| | Cliente A | Cliente B |
|---|---|---|
| **Uso** | Llamadas agénticas, chat con tareas, **tool calling** | **Coding** |

El horizonte de 200 usuarios + 20 devs sigue documentado como escenario futuro
(ver [04](docs/04-costos-y-escala.md)), pero **no es el alcance actual**.

El modelo que motivó el análisis es **Qwen3.8-Flash-Next** (open-weights de
Alibaba, arquitectura Qwen4-preview), que corre localmente con tiering de memoria
entre **VRAM / RAM / SSD**.

> ### 🚀 Para configurar el server ahora
> Andá directo a **[QUICKSTART.md](QUICKSTART.md)** — los 4 pasos para levantar
> los 2 servicios en la A16 y medir el rendimiento real.

---

## TL;DR — Veredicto

**Qwen3.8-Flash-Next no es viable en producción sobre la A16 para 200 usuarios.**
La tarjeta tiene 64 GB de VRAM y el quant más chico del modelo pesa 72.5 GB.
Los experts van siempre a la RAM del host y el decode queda atado al ancho de
banda de la RAM, no al de la GPU. Resultado real: **1 a 3 usuarios concurrentes**.

Para los 200 usuarios, la respuesta correcta es **la API de Qwen Cloud** (~USD 30-60
al mes para ese volumen, vs. 4-8 GPU clase H200 si lo quisieras local). La A16 se
aprovecha bien para el equipo de desarrollo con **4 réplicas data-parallel de un
modelo chico** vía vLLM.

---

## Documentos

| # | Documento | Contenido |
|---|---|---|
| 01 | [Hardware: la A16](docs/01-hardware-a16.md) | Qué es realmente la tarjeta y por qué no es una GPU de inferencia |
| 02 | [Modelo: Qwen3.8-Flash-Next](docs/02-modelo-qwen38-flash-next.md) | Arquitectura, por qué se presta al tiering, tamaños por cuant |
| 03 | [Arquitectura de despliegue](docs/03-arquitectura-despliegue.md) | El diseño híbrido en 3 capas: API + réplicas A16 + laboratorio |
| 04 | [Costos y escala](docs/04-costos-y-escala.md) | La cuenta de cuántos GPU necesitarías vs. cuánto cuesta la API |
| 05 | [Runbook: Flash-Next en la A16](docs/05-runbook-llamacpp-en-la-a16.md) | Flags exactos de llama.cpp y los gotchas que rompen todo |
| 06 | [Runbook: réplicas vLLM en la A16](docs/06-runbook-replicas-vllm.md) | Patrón data-parallel por partición, por qué no tensor-parallel |
| 07 | [Runbook: gateway LiteLLM](docs/07-runbook-gateway.md) | El punto único de salida para prod + local |
| 08 | [Decisiones y pendientes](docs/08-decisiones-y-pendientes.md) | Registro de decisiones (ADR) y datos que faltan confirmar |
| 09 | **[Escenario real (A16 del trabajo)](docs/09-escenario-real-a16.md)** | **Los datos reales: 2 particiones libres, 16 GB RAM, 1-2 clientes** |
| 10 | **[Catálogo de modelos](docs/10-catalogo-de-modelos.md)** | **Qué modelos correr y a qué velocidad en la A16** |
| 11 | **[Verificación y pruebas](docs/11-verificacion-y-pruebas.md)** | **Modelo elegido por cliente, qué se verificó y qué no, cómo probar** |

---

## Scripts

Para correr **en el server del trabajo**, en este orden:

| Script | Qué hace |
|---|---|
| `scripts/00-preflight.sh` | Chequeo de realidad: particiones libres, NVLink, throttling, RAM y canales, NVMe, runtimes. **Sin sudo no obtiene los canales de RAM.** |
| `scripts/10-serve-agentic-gptoss.sh` | Levanta el **Cliente A** (`gpt-oss-20b`, tool calling) en una partición |
| `scripts/20-serve-coding-ornith.sh` | Levanta el **Cliente B** (`Ornith-1.5-35B-A3B`, coding) vía llama.cpp |
| `scripts/30-smoke-tool-calling.py` | Prueba de punta a punta: ¿emite `tool_calls` válidas?, ¿cierra el ciclo?, **TTFT, decode t/s y prefill t/s** |

```bash
bash scripts/00-preflight.sh
bash scripts/10-serve-agentic-gptoss.sh 2 8001
python3 scripts/30-smoke-tool-calling.py http://localhost:8001/v1 agentic
```

> ⚠️ Nada de esto se probó todavía en la A16: la máquina desde donde se escribió
> es una laptop con una RTX 2070 de 8 GB. Ver [11](docs/11-verificacion-y-pruebas.md).

---

## Estado

**Fase: diseño cerrado, pendiente de implementación.**

### Situación real (actualizada)

| Recurso | Disponible |
|---|---|
| Particiones A16 | **2 libres** (2 ocupadas) = 32 GB VRAM |
| RAM | **16 GB** libres, ampliable |
| Alcance | **1 a 2 clientes frecuentes** |

### Decisión actual

1. **No usar Flash-Next** para estos clientes. Necesita ~55 GB residentes contra
   48 disponibles, y 16 GB de RAM hacen *thrash* con el mmap de la tabla N-gram.
2. **Usar 2 réplicas**, una por partición libre → 0 GB de RAM extra.
3. **Un modelo distinto por cliente** (ADR-007):

| Cliente | Modelo | Runtime | Peso | t/s est. |
|---|---|---|---|---|
| **A** — agéntico + tool calling | `gpt-oss-20b` (Apache 2.0) | vLLM | 12.8 GB (MXFP4 nativo) | 30-45 |
| **B** — coding | `Ornith-1.5-35B-A3B` (**MIT**) | llama.cpp | 13.7 GB (IQ3_XXS) | 35-50 |

Ver [09](docs/09-escenario-real-a16.md), [10](docs/10-catalogo-de-modelos.md) y
[11](docs/11-verificacion-y-pruebas.md).

### ⚠️ Dato crítico descubierto

**Una partición de A16 tiene ~1/5 del ancho de banda de una RTX 4090.**
No se arregla con software: hay que elegir modelos con **muy pocos parámetros
activos**. Es el factor que domina toda decisión de modelo en esta tarjeta.

**Segundo dato crítico: vLLM sirve safetensors, no GGUF.** Un 30-35B en bf16 pesa
60-71 GB. Por eso el modelo de coding va por llama.cpp con GGUF, y el de tool
calling se eligió entre los que vLLM puede servir directo.

### Pendientes

Los datos que todavía faltan están en [08](docs/08-decisiones-y-pendientes.md).
Los más relevantes ahora:

- **¿Cuánta RAM tiene el server y cuántos canales?** → habilita o no Flash-Next
- **¿Las 2 particiones ocupadas comparten presupuesto de 250 W?** → riesgo de throttling
- **¿Hay compliance que prohíba la nube?** → decide si el escenario de 200 usuarios
  va por API o por on-prem
- **Las velocidades de [10](docs/10-catalogo-de-modelos.md) son estimadas** →
  correr `scripts/30-smoke-tool-calling.py` en el server para confirmarlas
