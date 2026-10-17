# 05 — Runbook: Flash-Next en la A16 (Capa 3, laboratorio)

> **No para producción.** Esperado: **1-3 usuarios concurrentes.**
> Propósito: experimentar, medir, aprender el tiering.

## Por qué este techo

```
VRAM A16:                              64 GB
Quant mínimo (UD-IQ1_S):              72.5 GB   ← no cabe
Quant razonable (UD-IQ3_XXS):          82 GB
Quant bueno (UD-Q4_K_XL):             111.3 GB

→ experts SIEMPRE a la RAM del host
→ decode atado a ancho de banda de la RAM, no de la GPU
```

### Estimación de t/s según dónde vivan los experts

`t/s ≈ BW / 3.4 GB_por_token` (6B activos a 4-bit ≈ 3.4 GB/token)

| Dónde viven los experts | BW | teórico | real (eficiencia ~40-60%) |
|---|---|---|---|
| Una partición A16 (200 GB/s) | 200 GB/s | ~60 | **25-35** |
| DDR5 dual-channel | 76 GB/s | ~22 | **8-15** |
| DDR5 8 canales | 300 GB/s | ~90 | **30-50** |

> La A16 no puede aportar sus 200 GB/s al modelo **porque no le cabe**. El
> cuello de botella real es la **RAM del host**. Un server de 8 canales puede
> ser **mejor** que la propia tarjeta para este modelo.

⚠️ **Prefill:** los 5120 CUDA cores de la A16 ≈ 4 × GTX 1650. Con prompts largos
o RAG, el prefill va a doler. MTP acelera decode, no prefill.

## Pre-requisitos

| Qué | Detalle |
|---|---|
| llama.cpp | **compilado desde master** (arquitectura `qwen4_exp`, PR #27742, 27-ago-2026). Builds empaquetados anteriores no cargan el archivo. |
| RAM host | ≥ 96 GB ideal; 128 GB para cuants ≥ 4-bit |
| NVMe | ≥ 150 GB libres para el cuant + page cache |
| `CUDA_VISIBLE_DEVICES` | asignar la partición a usar |

## Paso 1 — Compilar llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
  -DGGML_CUDA=ON \
  -DBUILD_SHARED_LIBS=OFF

cmake --build llama.cpp/build --config Release -j \
  --clean-first --target llama-cli llama-server
```

Verificar que el soporte de la arquitectura esté:
```bash
grep -r "qwen4_exp" llama.cpp/ggml/src/ && echo "OK: arquitectura soportada"
```

## Paso 2 — Descargar

```bash
pip install -U "huggingface_hub[cli]"

hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --local-dir /mnt/ssdPart1/models/qwen3.8-flash-next \
  --include "*UD-IQ3_XXS*"
```

> ⚠️ La documentación de Unsloth ha tenido inconsistencias en los ejemplos de
> descarga (rutas mezclando directorio y nombre de shard). **Verificá que
> hayan llegado los archivos reales** antes de apuntar el runtime. No reconstruyas
> el path de memoria.

## Paso 3 — Lanzar

### Opción A: el camino simple (para verificar que bootea)

```bash
./llama.cpp/llama-server \
  -hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ3_XXS \
  -c 8192 -ngl 99 \
  --jinja \
  --port 8080
```

### Opción B: el tiering completo (VRAM + RAM + SSD)

```bash
./llama.cpp/llama-server \
  -hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL \
  --jinja \
  -c 32768 \
  -ngl 99 \
  --n-cpu-moe 30 \                        # experts a RAM (ajustar hasta llenar 64GB VRAM)
  -ot "per_layer_token_embd=CPU" \        # tabla N-gram al host
  -lm mmap \                              # mmap ON → tabla pageada desde NVMe
  -fit off \                              # auto-fit calcula MAL esta arquitectura
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --host 127.0.0.1 --port 8080
```

### Opción C: con MTP (1.3-1.7× más rápido, concurrencia 1)

```bash
./llama.cpp/llama-server \
  -hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS \
  --jinja \
  -c 32768 -ngl 99 --cpu-moe \
  -ot "per_layer_token_embd=CPU" -lm mmap -fit off \
  -md /mnt/ssdPart1/models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --port 8080
```

> ⚠️ **Siempre pasá `-md` explícitamente.** Los heads MTP viven en una subcarpeta
> `MTP/` que el auto-descubrimiento no busca. `--spec-type draft-mtp` solo
> **no encuentra nada y no lanza error**: te rinde igual que sin MTP.
>
> ⚠️ **Apagar MTP si vas a servir a varias personas.** Medido: 1.3-1.7× ganancia
> a concurrencia 1, pero **~0.81-0.87× pérdida neta** a concurrencia 8.

### Opción D: contexto deshabilitado

Para un endpoint instruct puro sin thinking:
```bash
  --chat-template-kwargs '{"enable_thinking": false}' \
  --temp 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5 \
```

## Gotchas que rompen todo

Todos reales y documentados:

| # | Gotcha | Consecuencia |
|---|---|---|
| 1 | **Desactivar mmap** (`--load-mode none`, `--lazy-mode off`) | Fuerza los ~50 GB de la tabla N-gram a RAM residente. En un DGX Spark crasheó el driver CUDA. |
| 2 | **No pasar `-fit off`** | El auto-fit de llama.cpp calcula mal esta arquitectura y falla la allocación. |
| 3 | **Contexto > 131072 en CUDA** | Abort en `rms_norm` (grid `n_ctx/4` excede el límite de launch). No es OOM. **Tratar 131072 como tope.** |
| 4 | **Mandar la tabla N-gram a la GPU** con `-ot` a GPU | Issue #28201. Usar siempre `per_layer_token_embd=CPU`. |
| 5 | **Build anterior al 27-ago-2026** | No carga el archivo, sin mensaje útil. |
| 6 | **Cuantizar agresivamente la tabla N-gram** | Destruye la calidad. Es un acceso aleatorio con hash; no tolera poca resolución. Mantener ≥ 4-bit. |
| 7 | **Omitir `--jinja`** | El chat template no se aplica. |

## Tiers de cuant para este server

| RAM disponible | Cuant recomendado | Notas |
|---|---|---|
| 96 GB | `UD-IQ3_XXS` (82 GB) | contexto 16-32K |
| 128 GB | `UD-IQ4_XS` (93.7 GB) | tabla desde SSD, contexto 32K |
| 128 GB + NVMe rápido | `UD-Q4_K_XL` (111 GB) | tabla en page cache, contexto 32K |
| 192 GB+ | `UD-Q4_K_XL` | sin restricción práctica |

## Qué medir

```bash
# 1. Confirmar que cargó el cuant correcto (no otro modelo)
curl http://localhost:8080/v1/models

# 2. Memoria tras la carga
free -h
nvidia-smi

# 3. Tiempo a primer token y t/s sostenidos
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Explica el patrón Observer"}]}'

# 4. Subir el contexto de a poco, vigilando:
watch -n1 'free -h; nvidia-smi --query-gpu=memory.used --format=csv'
```

**Pará a la señal de swap o a degradación de latencia.** Esas son señales de
que te pasaste, no de que hay que tunear más.

Siguiente: [06 — Runbook: réplicas vLLM en la A16](06-runbook-replicas-vllm.md)
