#!/usr/bin/env bash
# 10-serve-agentic-gptoss.sh
#
# Cliente A: asistente AGÉNTICO con tool calling.
# Modelo: gpt-oss-20b  (OpenAI, Apache 2.0, MoE 20.9B con 3.6B activos)
#
# Por qué este modelo para este cliente:
#   - Es el ÚNICO de la lista que viene cuantizado NATIVAMENTE (MXFP4, ~12.8 GB),
#     así que entra en una partición de 16 GB sin recurrir a GGUF.
#   - vLLM lo sirve directo con el parser de tool calling más confiable
#     (--tool-call-parser openai).
#   - Entrenado explícitamente para function calling, browsing y agentic ops.
#   - 3.6B activos -> el más rápido del lote que cabe cómodo (~30-45 t/s est.).
#
# USO:  bash scripts/10-serve-agentic-gptoss.sh <INDICE_PARTICION> [PUERTO]
#       bash scripts/10-serve-agentic-gptoss.sh 2 8001

set -euo pipefail

GPU="${1:?Falta el índice de partición. Ej: 2}"
PORT="${2:-8001}"

MODEL="openai/gpt-oss-20b"
NAME="agentic"
CTX="${CTX:-32768}"     # subir a 65536 o 131072 si hay headroom; bajar si falta VRAM

echo ">>> Partición $GPU  |  puerto $PORT  |  contexto $CTX"
echo ">>> Modelo: $MODEL"

# Verificar que la partición esté libre antes de ocuparla
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
if [ "$USED" -gt 2000 ]; then
  echo "!! ADVERTENCIA: la partición $GPU ya tiene ${USED} MiB en uso."
  echo "!! Puede estar asignada a VDI. Abortando para no afectar a nadie."
  exit 1
fi

CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL" \
  --served-model-name "$NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --max-model-len "$CTX" \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser openai

# ─── NOTAS DE OPERACIÓN ────────────────────────────────────────────────
#
# ¿Versión de vLLM? gpt-oss necesita soporte específico; usar la última estable.
#   python3 -c "import vllm; print(vllm.__version__)"
#   pip install -U vllm
#
# OJO con el contexto: gpt-oss-20b tiene 128K nativo, pero cada token de
# contexto consume VRAM. Empezar en 32768 y subir midiendo.
#
# OJO con el razonamiento + tools: el modelo puede hacer tool calling DENTRO
# del chain-of-thought. En multi-turno hay que devolver el `reasoning` que
# devuelve la API junto con el resultado de la tool, o baja la precisión.
#
# OJO con Python: algunas validaciones documentadas de function calling para
# gpt-oss requieren Python 3.12. Si falla el parsing, verificar la versión.
#
# Verificar que levantó:
#   curl -s http://localhost:8001/v1/models | python3 -m json.tool
#
# Probar tool calling:
#   python3 scripts/30-smoke-tool-calling.py http://localhost:8001/v1 agentic
