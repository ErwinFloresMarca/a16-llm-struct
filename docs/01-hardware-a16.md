# 01 — Hardware: la NVIDIA A16

> Es una tarjeta de **virtualización de escritorio (VDI/vGPU)**, no de inferencia.
> Entender eso es la mitad del diseño.

## Especificaciones oficiales

| Spec | Valor |
|---|---|
| Memoria | **4× 16 GB GDDR6 con ECC = 64 GB** (4 GPUs separadas en una tarjeta física) |
| Ancho de banda | **4× 200 GB/s** |
| CUDA cores | **4× 1280** (total 5120) — Ampere (GA107) |
| Bus de memoria | 128-bit por GPU |
| Interconexión | **PCIe Gen4 x16 — sin NVLink** |
| Consumo | 250 W |
| Uso diseñado | VDI, vGPU, decode/encode de video |

Cada partición tiene ~200 GB/s de ancho de banda y 1280 CUDA cores. En términos
de potencia bruta de cómputo, **cada partición es comparable a una GTX 1650**.
Las cuatro juntas suman 800 GB/s "en teoría" y 5120 cores.

## Por qué esto importa para inferencia

Hay **dos cuellos de botella** en una GPU corriendo un LLM, y la A16 es lenta
en ambos:

### 1. Capacidad (VRAM)

Para inferencia con MoE, **todos los pesos deben estar residentes en algún sitio
de memoria**, aunque solo se computen ~6B de parámetros por token. El "solo 6B
activos" te ahorra **cómputo**, no **memoria**.

64 GB de VRAM suena a mucho, pero el modelo que queremos correr no cabe.

### 2. Ancho de banda (decode)

El paso de *decode* (generar token a token) lee los pesos activados de cada
token desde la memoria más rápida disponible. Es un proceso **atado a ancho de
banda, no a cómputo**: no importa qué tan rápido sea el CUDA core si los datos
no llegan.

> Regla práctica: `t/s ≈ ancho_de_bandwidth_de_memoria / bytes_leídos_por_token`

Esto es lo que decide si el modelo corre rápido o lento, y por eso la A16
pierde: su ancho de banda está fragmentado en 4 canales de 200 GB/s y además
no le cabe el modelo.

### 3. Ancho de banda (prefill)

El paso de *prefill* (procesar el prompt de entrada) **sí es atado a cómputo**,
y ahí los 5120 CUDA cores de la A16 son el límite. Para un asistente de cliente
con prompts largos o RAG, el prefill es justamente el punto que duele.

## La A16 con tensor-parallel: no

Sin NVLink, cualquier sharding del modelo entre las 4 particiones obliga a pasar
activaciones y gradients por **PCIe Gen4 x16 (~32 GB/s)**. Cada capa del modelo
requiere un *allreduce*, así que el tensor-parallel sobre la A16 tiene un overhead
fijo alto.

**Regla de diseño:** usar la A16 en **data-parallel** (una réplica completa por
partición), no en tensor-parallel.

## Resumen

- VRAM: 64 GB, fragmentada en 4 × 16 GB.
- Ancho de banda: 4 × 200 GB/s, sin NVLink.
- Cómputo: 5120 CUDA cores ≈ 4 × GTX 1650.
- Diseñada para VDI, no para inferencia.
- **No usar tensor-parallel; sí data-parallel.**

Veriguiente: [02 — Modelo Qwen3.8-Flash-Next](02-modelo-qwen38-flash-next.md)
