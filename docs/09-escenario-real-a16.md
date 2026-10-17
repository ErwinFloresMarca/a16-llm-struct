# 09 — Escenario real (A16 del trabajo)

> Este documento reemplaza las suposiciones del diseño inicial con los **datos
> reales del server**.

## Situación

| Recurso | Disponible |
|---|---|
| Particiones A16 | **2 libres** (2 ocupadas) = **32 GB VRAM** |
| RAM | **16 GB libres**, ampliable |
| Alcance actual | **1 a 2 clientes frecuentes** |

## Por qué Flash-Next sigue fuera de alcance

```
UD-IQ3_XXS                          82 GB
  − tabla N-gram (a SSD)          − 27 GB
  ───────────────────────────────────────
  residentes necesarios             ≈55 GB

Disponible: 32 (VRAM) + 16 (RAM)  = 48 GB   ← faltan ~7 GB
```

Y hay un problema **peor que la cuenta**: con 16 GB de RAM, el mmap de la tabla
N-gram hace *thrash*. El `lazy-mode` depende del page cache para servir las ~16
filas por token. Con tan poca RAM, cada lookup dispara lecturas a disco y el
prefill se vuelve inmanejable.

> **Conclusión:** Flash-Next necesita RAM de 64 GB mínimo (96 GB ideal) **y**
> sigue siendo lento en la A16. No es una opción para estos 1-2 clientes.

## La decisión: 2 réplicas, una por partición

```
Partición 2 (16 GB) ──▶ vLLM réplica A ──▶ Cliente 1
Partición 3 (16 GB) ──▶ vLLM réplica B ──▶ Cliente 2

     ✦ sin competencia entre clientes
     ✦ 0 GB de RAM extra requeridos (todo en VRAM)
     ✦ corre hoy, con lo que hay
```

**Ventajas:**

| | 2 réplicas | Flash-Next IQ3 |
|---|---|---|
| RAM extra | **0 GB** | 64-96 GB |
| ¿Corre hoy? | **Sí** | No |
| Prefill | **Rápido** (modelo chico + prefix cache) | Lento |
| Un cliente afecta al otro | **No** | Sí |
| Licencia | Apache 2.0 / MIT (ver [10](10-catalogo-de-modelos.md)) | Qwen Community License |

**El prefill es el argumento decisivo.** Con vLLM y prefix caching, el system
prompt se procesa una vez y después es gratis. Flash-Next procesa cada token
contra los 5120 CUDA cores de la A16, y ahí es donde la tarjeta se arrastra.

### Variante: un modelo distinto por partición

Con 2 particiones y 2 clientes podés dar **perfiles distintos**:

- **Partición A → el modelo de mejor calidad** (Qwen3.8-27B, mismo arquitectura
  que Flash-Next, Apache 2.0).
- **Partición B → el modelo más rápido** (un MoE de ~3B activos).

Si los dos clientes hacen cosas distintas, esto es mejor que la simetría.

## Setup

```bash
# 1. Ver qué particiones están libres
nvidia-smi --query-gpu=index,name,memory.used --format=csv

# 2. Réplica A → Cliente 1
CUDA_VISIBLE_DEVICES=2 vllm serve <modelo> \
  --port 8001 --tensor-parallel-size 1 \
  --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --served-model-name qwen-local --enable-prefix-caching

# 3. Réplica B → Cliente 2
CUDA_VISIBLE_DEVICES=3 vllm serve <modelo> \
  --port 8002 --tensor-parallel-size 1 \
  --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --served-model-name qwen-local --enable-prefix-caching

# 4. Balanceador (LiteLLM) — ver 07-runbook-gateway.md
```

**No usar tensor-parallel entre las 2 particiones.** Sin NVLink, el allreduce por
capa sobre PCIe es más caro que la ganancia. (Ver [ADR-002](08-decisiones-y-pendientes.md).)

## ⚠️ Advertencia de potencia

La A16 entrega **250 W para las 4 GPUs juntas**. Si las 2 particiones ocupadas
están haciendo VDI o video, sumarle inferencia a las otras 2 puede topar el
límite. Verificar:

```bash
nvidia-smi -q -d POWER,PERFORMANCE | grep -iE 'throttle|power limit|slowdown'
```

Si aparece `HW Slowdown` o `SW Power Cap` activo, la velocidad real va a estar
por debajo de las estimaciones.

## Sobre ampliar la RAM

**Para esta ruta: no hace falta.** Pero si querés dejar la puerta abierta a
Flash-Next:

| RAM | Qué habilita |
|---|---|
| 32 GB | Nada nuevo. Flash-Next sigue fuera. |
| **64 GB** | Mínimo realista para Flash-Next IQ3 con tabla en SSD |
| 96 GB | Flash-Next cómodo |
| 128 GB | Flash-Next Q4 |

⚠️ **Más importante que la capacidad:** llenar **todos los canales** con DIMMs
iguales. El decode de Flash-Next con experts en RAM es puro ancho de banda, y
4×16 GB en 4 canales rinde mucho más que 2×32 GB en 2 canales.

Verificar con: `sudo dmidecode -t memory`

Ver: [10 — Catálogo de modelos](10-catalogo-de-modelos.md)
