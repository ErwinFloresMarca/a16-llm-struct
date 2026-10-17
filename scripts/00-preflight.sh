#!/usr/bin/env bash
# 00-preflight.sh — Chequeo de realidad del server con la A16.
#
# Correr ESTO PRIMERO, en el server del trabajo:
#     bash scripts/00-preflight.sh
#
# Responde las preguntas bloqueantes de docs/08-decisiones-y-pendientes.md.
# No instala nada ni modifica nada. Solo lee y reporta.

set -uo pipefail

hr() { printf '\n=== %s ===\n' "$1"; }

hr "1. GPUs y particiones (¿cuáles están libres?)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
           --format=csv

hr "2. Topología (¿hay NVLink entre particiones?)"
echo "Si NO aparece NVLink en la matriz, tensor-parallel está descartado."
nvidia-smi topo -m 2>/dev/null || echo "(topo no disponible)"

hr "3. Throttling y límite de potencia"
echo "Buscar 'HW Slowdown', 'SW Power Cap' o 'HW Thermal' en Active."
nvidia-smi -q -d POWER,PERFORMANCE 2>/dev/null \
  | grep -iE 'Power Limit|Slowdown|Throttle|Active' | head -20

hr "4. RAM total y disponible"
free -h

hr "5. Canales de RAM, slots y velocidad  (NECESITA sudo)"
if command -v dmidecode >/dev/null 2>&1; then
  echo "Slots ocupados / tipo / velocidad:"
  sudo dmidecode -t memory 2>/dev/null \
    | grep -E 'Locator:|Size:|Type:|Speed:|Rank:|Configured Memory Speed' \
    | grep -v 'No Module Installed' \
    | sed 's/^[[:space:]]*//'
  echo
  echo "Interpretación: contá cuántos 'Size:' distintos de 0 aparecen."
  echo "  4+ módulos en 4+ canales  -> ancho de banda alto (bueno)"
  echo "  2 módulos en 2 canales    -> cuello de botella (malo)"
else
  echo "dmidecode no instalado. Instalar:  sudo apt install dmidecode"
fi

hr "6. Almacenamiento (¿hay NVMe libre?)"
lsblk -d -o NAME,SIZE,ROTA,MODEL 2>/dev/null
echo
df -h / /mnt/* 2>/dev/null | grep -v '^tmpfs'

hr "7. CPU (afecta el prefill)"
lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|NUMA node\(s\)'

hr "8. Runtimes de inferencia instalados"
python3 -c "import vllm, sys; print('vLLM        ', vllm.__version__)" 2>/dev/null \
  || echo "vLLM         : NO instalado"
python3 -c "import torch; print('PyTorch     ', torch.__version__, '| CUDA', torch.version.cuda)" 2>/dev/null \
  || echo "PyTorch      : NO instalado"
command -v llama-server >/dev/null 2>&1 \
  && echo "llama.cpp    : $(command -v llama-server)" \
  || echo "llama.cpp    : NO en PATH (compilar desde master, ver docs/05)"
command -v python3 >/dev/null 2>&1 \
  && echo "Python       : $(python3 --version)" \
  || echo "Python       : NO instalado"

hr "RESUMEN — qué anotar"
cat <<'EOF'
  [ ] ¿Cuántas particiones libres y cuál índice?         -> CUDA_VISIBLE_DEVICES=?
  [ ] ¿Hay NVLink? (sí/no)                               -> habilita o no tensor-parallel
  [ ] ¿Hay throttling activo? (sí/no)                    -> ajusta expectativas de t/s
  [ ] RAM total: ____ GB en ____ canales                 -> habilita o no Flash-Next
  [ ] NVMe: ____ GB libres en ____ (ruta)                -> espacio para modelos
  [ ] vLLM instalado? versión ____                       -> ruta agéntica
  [ ] llama.cpp compilado? (sí/no)                       -> ruta de cuants densos
EOF
