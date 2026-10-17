# Makefile — atajos para configurar la A16 y medir rendimiento.
#
# Correr `make` o `make help` para ver todos los targets.
#
# Los 2 servicios:
#   agentic -> gpt-oss-20b        en vLLM       (Cliente A: tool calling)
#   coding  -> Ornith-1.5-35B-A3B en llama.cpp  (Cliente B: coding)

.DEFAULT_GOAL := help
SHELL := /bin/bash

# ─── Configuración (override desde la línea de comandos) ───────────────
#   make start GPU_A=2 GPU_B=3 CTX=16384
#
# OJO: no alinear comentarios con espacios a la derecha de un valor.
# Make conserva el espacio final en la variable y rompe los argumentos.

# Partición libre para el servicio agentic
GPU_A  ?= 2
# Partición libre para el servicio coding
GPU_B  ?= 3
PORT_A ?= 8001
PORT_B ?= 8002
# Contexto. Bajar a 16384 si no entra en VRAM
CTX    ?= 32768
# IQ3_XXS ~13.7 GB | IQ3_M ~16 GB | Q4_K_M ~20 GB (NO entra en 16 GB)
QUANT  ?= IQ3_XXS
VENV   ?= $(HOME)/venv-vllm

.PHONY: help preflight status \
        install-vllm install-llamacpp \
        serve-agentic serve-coding \
        start start-agentic start-coding \
        stop stop-agentic stop-coding restart \
        test test-agentic test-coding \
        logs-agentic logs-coding report clean

# ─── Ayuda ─────────────────────────────────────────────────────────────
help: ## Muestra esta ayuda
	@echo "a16-llm-struct — atajos para el server con la A16"
	@echo
	@echo "Uso: make <target> [GPU_A=2] [GPU_B=3] [CTX=32768] [QUANT=IQ3_XXS]"
	@echo
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  Config:  GPU_A=$(GPU_A)  GPU_B=$(GPU_B)  CTX=$(CTX)  QUANT=$(QUANT)"
	@echo "  Puertos: agentic=$(PORT_A)  coding=$(PORT_B)"
	@echo
	@echo "Primera vez:  make preflight  ->  make install-vllm  ->  make install-llamacpp"
	@echo "Uso normal:   make start      ->  make report"

# ─── Diagnóstico e instalación ─────────────────────────────────────────
preflight: ## Diagnóstico del server: particiones, NVLink, throttling, RAM, NVMe. No instala nada
	bash scripts/00-preflight.sh

status: ## Estado de GPU, puertos y procesos de los 2 servicios
	@echo "── GPU ──"
	@nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1
	@echo
	@echo "── Servicios ──"
	@for p in $(PORT_A):agentic $(PORT_B):coding; do \
	  port=$${p%%:*}; name=$${p##*:}; \
	  if curl -sf --max-time 3 "http://localhost:$$port/v1/models" >/dev/null 2>&1; then \
	    printf "  \033[32m●\033[0m %-8s :%s  respondiendo\n" "$$name" "$$port"; \
	  else \
	    printf "  \033[31m○\033[0m %-8s :%s  no responde\n" "$$name" "$$port"; \
	  fi; \
	done
	@echo
	@echo "── Procesos ──"
	@pgrep -af "[v]llm serve|[l]lama-server" 2>/dev/null || echo "  (ninguno)"

install-vllm: ## Crea un venv (~/venv-vllm) e instala vLLM
	@python3 -m venv $(VENV)
	@$(VENV)/bin/pip install -q -U pip
	@$(VENV)/bin/pip install -U vllm
	@$(VENV)/bin/python -c "import vllm; print('  vLLM', vllm.__version__, 'listo en $(VENV)')"

install-llamacpp: ## Compila llama.cpp con CUDA desde master (necesario: kernels nuevos)
	@if [ ! -d llama.cpp ]; then \
	  git clone --depth 1 https://github.com/ggml-org/llama.cpp; \
	fi
	cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
	cmake --build llama.cpp/build --config Release -j"$$(nproc)"
	@echo "  compilado: llama.cpp/build/bin/llama-server"
	@echo "  instalalo en el PATH con:  sudo install -m755 llama.cpp/build/bin/llama-server /usr/local/bin/"

# ─── Levantar en primer plano (bloqueante, Ctrl-C para cortar) ─────────
serve-agentic: ## Levanta el servicio agentic en primer plano (Ctrl-C para cortar)
	CTX=$(CTX) bash scripts/10-serve-agentic-gptoss.sh $(GPU_A) $(PORT_A)

serve-coding: ## Levanta el servicio coding en primer plano (Ctrl-C para cortar)
	CTX=$(CTX) bash scripts/20-serve-coding-ornith.sh $(GPU_B) $(PORT_B) $(QUANT)

# ─── Levantar en background ────────────────────────────────────────────
start: start-agentic start-coding ## Levanta los 2 servicios en background
	@echo
	@echo "  Esperá a que carguen (la 1.ª vez descarga los modelos) y después:"
	@echo "    make status     # ver si ya responden"
	@echo "    make report     # medir rendimiento"

start-agentic: ## Levanta agentic en background (log en logs/agentic.log)
	@mkdir -p logs
	@if [ -f logs/agentic.pid ] && kill -0 $$(cat logs/agentic.pid) 2>/dev/null; then \
	  echo "  agentic ya está corriendo (grupo $$(cat logs/agentic.pid))"; \
	else \
	  PATH=$(VENV)/bin:$$PATH CTX=$(CTX) setsid bash scripts/10-serve-agentic-gptoss.sh \
	    $(GPU_A) $(PORT_A) > logs/agentic.log 2>&1 < /dev/null & \
	  echo $$! > logs/agentic.pid; \
	  echo "  agentic arrancando en :$(PORT_A) (partición $(GPU_A)) -> make logs-agentic"; \
	fi

start-coding: ## Levanta coding en background (log en logs/coding.log)
	@mkdir -p logs
	@if [ -f logs/coding.pid ] && kill -0 $$(cat logs/coding.pid) 2>/dev/null; then \
	  echo "  coding ya está corriendo (grupo $$(cat logs/coding.pid))"; \
	else \
	  CTX=$(CTX) setsid bash scripts/20-serve-coding-ornith.sh \
	    $(GPU_B) $(PORT_B) $(QUANT) > logs/coding.log 2>&1 < /dev/null & \
	  echo $$! > logs/coding.pid; \
	  echo "  coding arrancando en :$(PORT_B) (partición $(GPU_B), $(QUANT)) -> make logs-coding"; \
	fi

# ─── Detener ───────────────────────────────────────────────────────────
stop: stop-agentic stop-coding ## Detiene los 2 servicios

stop-agentic: ## Detiene el servicio agentic
	@if [ -f logs/agentic.pid ]; then \
	  G=$$(cat logs/agentic.pid); \
	  if kill -TERM -$$G 2>/dev/null; then echo "  agentic detenido (grupo $$G)"; \
	  else echo "  agentic no estaba corriendo"; fi; \
	  rm -f logs/agentic.pid; \
	else echo "  agentic no estaba corriendo"; fi

stop-coding: ## Detiene el servicio coding
	@if [ -f logs/coding.pid ]; then \
	  G=$$(cat logs/coding.pid); \
	  if kill -TERM -$$G 2>/dev/null; then echo "  coding detenido (grupo $$G)"; \
	  else echo "  coding no estaba corriendo"; fi; \
	  rm -f logs/coding.pid; \
	else echo "  coding no estaba corriendo"; fi

restart: stop start ## Reinicia los 2 servicios

# ─── Medir ─────────────────────────────────────────────────────────────
test-agentic: ## Smoke test del servicio agentic (tool calling + t/s)
	python3 scripts/30-smoke-tool-calling.py http://localhost:$(PORT_A)/v1 agentic || true

test-coding: ## Smoke test del servicio coding (tool calling + t/s)
	python3 scripts/30-smoke-tool-calling.py http://localhost:$(PORT_B)/v1 coding || true

test: test-agentic test-coding ## Smoke test de los 2 servicios

report: ## Corre status + los 2 smoke tests y guarda todo en logs/report-<fecha>.md
	@mkdir -p logs
	@F=logs/report-$$(date +%Y%m%d-%H%M).md; \
	{ \
	  echo "# Reporte de rendimiento — $$(date -Iseconds)"; \
	  echo; \
	  echo "Config: GPU_A=$(GPU_A) GPU_B=$(GPU_B) CTX=$(CTX) QUANT=$(QUANT)"; \
	  echo; \
	  echo '## Hardware'; echo; echo '```'; \
	  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1; \
	  echo '```'; echo; \
	  echo '## Topología (¿NVLink?)'; echo; echo '```'; \
	  nvidia-smi topo -m 2>&1 | head -12; \
	  echo '```'; echo; \
	  echo '## Throttling'; echo; echo '```'; \
	  nvidia-smi -q -d POWER,PERFORMANCE 2>/dev/null | grep -iE 'Power Limit|Slowdown|Throttle|Active' | head -12; \
	  echo '```'; echo; \
	  echo '## Servicio agentic (:$(PORT_A))'; echo; echo '```'; \
	  python3 scripts/30-smoke-tool-calling.py http://localhost:$(PORT_A)/v1 agentic 2>&1 || true; \
	  echo '```'; echo; \
	  echo '## Servicio coding (:$(PORT_B))'; echo; echo '```'; \
	  python3 scripts/30-smoke-tool-calling.py http://localhost:$(PORT_B)/v1 coding 2>&1 || true; \
	  echo '```'; \
	} > $$F; \
	echo "  reporte guardado en $$F"

# ─── Logs y limpieza ───────────────────────────────────────────────────
logs-agentic: ## Sigue el log del servicio agentic
	@tail -f logs/agentic.log

logs-coding: ## Sigue el log del servicio coding
	@tail -f logs/coding.log

clean: ## Borra logs y archivos de PID (no toca los modelos descargados)
	@rm -rf logs
	@echo "  logs borrados"
