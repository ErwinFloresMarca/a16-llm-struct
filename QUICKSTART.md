# QUICKSTART — configurar el server y ver el rendimiento

> **Objetivo:** dejar los 2 servicios corriendo en la A16 y **medir** el
> rendimiento real. Lucas de los pasos son de diagnóstico; no hay que instalar
> nada para los primeros 2.

| Servicio | Cliente | Modelo | Runtime | Partición |
|---|---|---|---|---|
| `agentic` | A — llamadas agénticas, chat con tareas, tool calling | `gpt-oss-20b` | vLLM | una libre |
| `coding` | B — coding | `Ornith-1.5-35B-A3B` | llama.cpp | la otra libre |

Razonamiento completo en [docs/11-verificacion-y-pruebas.md](docs/11-verificacion-y-pruebas.md)
y [docs/08-decisiones-y-pendientes.md](docs/08-decisiones-y-pendientes.md) (ADR-007).

---

## Paso 0 — Diagnóstico (no instala nada)

```bash
git clone git@github.com:ErwinFloresMarca/a16-llm-struct.git
cd a16-llm-struct
bash scripts/00-preflight.sh
```

**Anotá de la salida:**

| Dato | Para qué sirve |
|---|---|
| Índices de las **particiones libres** | van en `CUDA_VISIBLE_DEVICES` |
| **NVLink**: sí/no | confirma que no conviene tensor-parallel |
| **Throttling** activo | ajusta las expectativas de velocidad |
| **RAM y canales** | habilita o no Flash-Next a futuro |
| **vLLM / llama.cpp** instalados | qué falta instalar |

> Los canales de RAM requieren `sudo dmidecode -t memory`. Sin sudo, ese paso
> queda afuera.

---

## Paso 1 — Instalar los runtimes

**vLLM** (para el servicio `agentic`):

```bash
python3 -m venv ~/venv-vllm && source ~/venv-vllm/bin/activate
pip install -U vllm
python3 -c "import vllm; print('vLLM', vllm.__version__)"
```

**llama.cpp** (para el servicio `coding`). Compilar desde master: la arquitectura
de estos modelos necesita los kernels más nuevos.

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)"
sudo install -m755 build/bin/llama-server /usr/local/bin/llama-server
llama-server --version
```

---

## Paso 2 — Levantar el servicio `agentic` (Cliente A)

```bash
cd ~/a16-llm-struct
bash scripts/10-serve-agentic-gptoss.sh <PARTICION_LIBRE> 8001
```

Reemplazá `<PARTICION_LIBRE>` por el índice que salió del paso 0 (por ejemplo `2`).
La primera vez descarga ~13 GB. Cuando vea `Application startup complete`, queda
escuchando.

**En otra terminal:**

```bash
python3 scripts/30-smoke-tool-calling.py http://localhost:8001/v1 agentic
```

---

## Paso 3 — Levantar el servicio `coding` (Cliente B)

```bash
bash scripts/20-serve-coding-ornith.sh <OTRA_PARTICION_LIBRE> 8002 IQ3_XXS
python3 scripts/30-smoke-tool-calling.py http://localhost:8002/v1 coding
```

---

## Paso 4 — Anotar los números

El smoke test devuelve 5 cosas. Cargalas acá:

| Métrica | `agentic` | `coding` | Estimado ([10](docs/10-catalogo-de-modelos.md)) |
|---|---|---|---|
| ¿Responde? | | | sí |
| ¿Emite `tool_calls` válidas? | | | sí |
| ¿Cierra el ciclo tool→respuesta? | | | sí |
| **TTFT** (primer token) | | | — |
| **Decode** t/s | | | 30-45 / 35-50 |
| **Prefill** t/s | | | — |
| VRAM pico | | | <16 GB |

Y después:

```bash
nvidia-smi -q -d POWER,PERFORMANCE | grep -iE 'Active|Slowdown|Throttle'
```

Si aparece throttling activo, ese es el techo real. Ver
[docs/11](docs/11-verificacion-y-pruebas.md) para interpretar cada resultado.

---

## Si algo falla

| Síntoma | Causa probable | Arreglo |
|---|---|---|
| No aparecen `tool_calls` | Faltan flags de tool calling | Ver tabla en [docs/11](docs/11-verificacion-y-pruebas.md#si-tool_calls-no-aparece) |
| `CUDA out of memory` | Modelo + contexto no entran | Bajar `CTX`: `CTX=16384 bash scripts/...` |
| Decode muy por debajo | Offload parcial a RAM o throttling | Ver [docs/11](docs/11-verificacion-y-pruebas.md#si-el-decode-está-muy-por-debajo-de-lo-estimado) |
| El proceso muere al arrancar | Modelo no soportado por la versión del runtime | Actualizar vLLM / recompilar llama.cpp desde master |
| El script aborta diciendo que la partición está en uso | Ocupaste una partición en VDI | Elegir otra con el `nvidia-smi` del paso 0 |

---

## Advertencias

- **La A16 es una tarjeta VDI.** 2 de las 4 particiones sirven a otra cosa. Los
  scripts **abortan si la partición ya tiene VRAM en uso** — es a propósito.
- **250 W para las 4 particiones juntas.** Sumar inferencia a las 2 ocupadas
  puede provocar throttling. Medir antes de prometer velocidades.
- **Nada de esto fue probado en la A16 todavía.** Todos los t/s son estimaciones
  calculadas desde el ancho de banda. El paso 4 existe para reemplazarlas con
  números reales.
