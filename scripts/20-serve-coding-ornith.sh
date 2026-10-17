#!/usr/bin/env bash
# 20-serve-coding-ornith.sh
#
# Cliente B: asistente de CODING (con tools, porque los agentes de código usan tools).
# Modelo: Ornith-1.5-35B-A3B  (DeepReinforce, MIT, MoE 36B con 3B activos)
#
# Por qué este modelo:
#   - MoE con *hybrid linear + full attention*: la misma familia arquitectónica
#     que Flash-Next y Qwen3.8-27B.
#   - 3B activos sobre 36B totales -> mucha capacidad a velocidad de modelo chico.
#   - Terminal-Bench 2.1: 67.8  (vs 52.5 de Qwen3.6-35B-A3B, 42.1 de Gemma-4-31B)
#   - Licencia MIT, multimodal (imagen), MTP speculative decoding, 256K contexto.
#
# ⚠️ POR QUÉ NO vLLM ACÁ:
#   El checkpoint oficial es bf16 = ~71 GB. vLLM no puede servirlo en 16 GB.
#   No hay release oficial de FP8/AWQ. Así que la ruta es llama.cpp + GGUF,
#   que es lo que permite bajar a ~14 GB con IQ3.
#
#   Trade-off honesto: llama.cpp tiene tool calling vía --jinja (chat template),
#   más básico que los parsers dedicados de vLLM. Funciona, pero si el cliente
#   exige tool calling robusto, usar la partición para un modelo con vLLM nativo.
#
# USO:  bash scripts/20-serve-coding-ornith.sh <INDICE_PARTICION> [PUERTO] [CUANT]
#       bash scripts/20-serve-coding-ornith.sh 3 8002 IQ3_XXS

set -euo pipefail

GPU="${1:?Falta el índice de partición. Ej: 3}"
PORT="${2:-8002}"
QUANT="${3:-IQ3_XXS}"     # IQ3_XXS ~13.7 GB | IQ3_M ~16 GB | Q4_K_M ~20 GB (NO entra)

MODEL_DIR="${MODEL_DIR:-/mnt/models/ornith-1.5-35b-a3b}"
LLAMA_BIN="${LLAMA_BIN:-llama-server}"
CTX="${CTX:-32768}"

echo ">>> Partición $GPU | puerto $PORT | cuant $QUANT | contexto $CTX"

# ─── 1. Chequear que la partición esté libre ───────────────────────────
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
if [ "$USED" -gt 2000 ]; then
  echo "!! ADVERTENCIA: partición $GPU con ${USED} MiB en uso. Abortando."
  exit 1
fi

# ─── 2. Verificar que llama.cpp soporte la arquitectura ───────────────
if ! command -v "$LLAMA_BIN" >/dev/null 2>&1; then
  echo "!! $LLAMA_BIN no está en PATH."
  echo "!! Compilar desde master: ver docs/05-runbook-llamacpp-en-la-a16.md"
  exit 1
fi

# ─── 3. Descargar si no está ──────────────────────────────────────────
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
  echo ">>> Descargando $QUANT..."
  mkdir -p "$MODEL_DIR"
  pip install -U "huggingface_hub[cli]"
  hf download bartowski/Ornith-1.5-35B-A3B-GGUF \
    --local-dir "$MODEL_DIR" \
    --include "*${QUANT}*"
  echo ">>> Verificá los archivos que llegaron:"
  ls -lh "$MODEL_DIR"
  echo ">>> Si hay varios shards, usar el -00001-of-* como -m"
fi

# ─── 4. Levantar ──────────────────────────────────────────────────────
# --jinja           : aplica el chat template (necesario para tool calling)
# -ngl 99           : todas las capas en GPU (todo debe entrar en 16 GB)
# -c $CTX           : contexto
# -fa on            : flash attention
# -ctk/-ctv q8_0    : KV cuantizado -> estira el contexto en 16 GB
# --no-mmap         : NO acá. En este modelo NO desactivar mmap.
CUDA_VISIBLE_DEVICES="$GPU" "$LLAMA_BIN" \
  -hf "bartowski/Ornith-1.5-35B-A3B-GGUF:${QUANT}" \
  --alias coding \
  --jinja \
  -ngl 99 \
  -c "$CTX" \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  --host 0.0.0.0 \
  --port "$PORT"

# ─── ALTERNATIVA: Qwen3.8-27B (misma arquitectura que Flash-Next, Apache 2.0) ───
#
# Si preferís el linaje Qwen + licencia Apache 2.0 + vision, y podés aceptar
# ~10-15 t/s por ser DENSO (lee 27B por token, no 3B):
#
#   CUDA_VISIBLE_DEVICES="$GPU" llama-server \
#     -hf Qwen/Qwen3.8-27B-GGUF:Q3_K_M \
#     --alias coding --jinja -ngl 99 -c 32768 -fa on \
#     -ctk q8_0 -ctv q8_0 --host 0.0.0.0 --port "$PORT"
#
# ─── ALTERNATIVA: Qwen3-Coder-30B-A3B (especialista en coding, Qwen) ───
#
# 30.5B total / 3.3B activos. Excelente tool calling en vLLM con
# --tool-call-parser qwen3_coder, pero bf16 son 61 GB -> en 16 GB va por GGUF:
#
#   llama-server -hf unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:UD-Q3_K_XL ...
#
# ─── ALTERNATIVA: versión COMPACTADA de 19B con 256K (entra cómoda) ───
#
#   continuum-ai/qwen3-coder-30b-a3b-compacted-19b-256k  (~19B, Q4 ~10.6 GB)
#   Más contexto y más margen de VRAM, a costa de capacidad podada.
#
# ─── Verificar ────────────────────────────────────────────────────────
#   curl -s http://localhost:8002/v1/models | python3 -m json.tool
#   python3 scripts/30-smoke-tool-calling.py http://localhost:8002/v1 coding
