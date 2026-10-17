# QUICKSTART — configurar el server y ver el rendimiento

> **Objetivo:** dejar los 2 servicios corriendo en la A16 y **medir** el
> rendimiento real. Los primeros pasos son de diagnóstico; no instalan nada.

| Servicio | Cliente | Modelo | Runtime | Puerto |
|---|---|---|---|---|
| `agentic` | A — llamadas agénticas, chat con tareas, tool calling | `gpt-oss-20b` | vLLM | 8001 |
| `coding` | B — coding | `Ornith-1.5-35B-A3B` | llama.cpp | 8002 |

Razonamiento completo en [docs/11-verificacion-y-pruebas.md](docs/11-verificacion-y-pruebas.md)
y [docs/08-decisiones-y-pendientes.md](docs/08-decisiones-y-pendientes.md) (ADR-007).

---

## Atajos

Todo se maneja con `make`. Corré `make` solo para ver la lista completa.

```bash
git clone git@github.com:ErwinFloresMarca/a16-llm-struct.git
cd a16-llm-struct
make
```

| Comando | Qué hace |
|---|---|
| `make preflight` | Diagnóstico del server. **No instala nada** |
| `make install-vllm` | Crea el venv e instala vLLM |
| `make install-llamacpp` | Compila llama.cpp con CUDA desde master |
| `make start` | Levanta los 2 servicios en background |
| `make status` | GPU + puertos + procesos |
| `make test` | Smoke test de los 2 |
| `make report` | **Mide y guarda el reporte en `logs/`** |
| `make stop` | Detiene los 2 |
| `make logs-agentic` / `make logs-coding` | Sigue los logs |

Podés sobreescribir la config en la línea de comandos:

```bash
make start GPU_A=2 GPU_B=3 CTX=16384
make serve-coding QUANT=IQ3_M
```

---

## Paso 0 — Diagnóstico

```bash
make preflight
```

**Anotá de la salida:**

| Dato | Para qué sirve |
|---|---|
| Índices de las **particiones libres** | van en `GPU_A=` / `GPU_B=` |
| **NVLink**: sí/no | confirma que no conviene tensor-parallel |
| **Throttling** activo | ajusta las expectativas de velocidad |
| **RAM y canales** | habilita o no Flash-Next a futuro |
| **vLLM / llama.cpp** instalados | qué falta instalar |

> Los canales de RAM requieren `sudo dmidecode -t memory`. Sin sudo, ese paso
> queda afuera.

Si las particiones libres no son la 2 y la 3:

```bash
make start GPU_A=<libre> GPU_B=<otra_libre>
```

---

## Paso 1 — Instalar los runtimes

```bash
make install-vllm       # venv en ~/venv-vllm (make lo usa solo, no hay que activarlo)
make install-llamacpp   # compila desde master: estos modelos necesitan los kernels nuevos
```

`install-llamacpp` deja el binario en `llama.cpp/build/bin/llama-server`. Para
que el script lo encuentre:

```bash
sudo install -m755 llama.cpp/build/bin/llama-server /usr/local/bin/
```

---

## Paso 2 — Levantar los 2 servicios

```bash
make start
```

La primera vez descarga ~13 GB (agentic) y ~14 GB (coding). Podés seguir el
progreso:

```bash
make logs-agentic     # Ctrl-C para salir del log, el servicio sigue
make status          # cuando diga "respondiendo", está listo
```

> Los scripts **abortan si la partición ya tiene VRAM en uso**. Es a propósito:
> la A16 es una tarjeta VDI y 2 de las 4 particiones sirven a otra cosa.

Si preferís verlo en primer plano (por ejemplo, para leer los warnings al
arrancar):

```bash
make serve-agentic    # Ctrl-C para cortar
```

---

## Paso 3 — Medir el rendimiento

```bash
make report
```

Eso genera `logs/report-<fecha>.md` con: hardware, topología, throttling, y el
smoke test de los 2 servicios. **Ese archivo es el entregable.**

Para verlo suelto en pantalla:

```bash
make test
```

El smoke test devuelve 5 cosas por servicio:

| Métrica | `agentic` | `coding` | Estimado ([10](docs/10-catalogo-de-modelos.md)) |
|---|---|---|---|
| ¿Responde? | | | sí |
| ¿Emite `tool_calls` válidas? | | | sí |
| ¿Cierra el ciclo tool→respuesta? | | | sí |
| **TTFT** (primer token) | | | — |
| **Decode** t/s | | | 30-45 / 35-50 |
| **Prefill** t/s | | | — |
| VRAM pico | | | <16 GB |

---

## Si algo falla

| Síntoma | Causa probable | Arreglo |
|---|---|---|
| No aparecen `tool_calls` | Faltan flags de tool calling | [docs/11](docs/11-verificacion-y-pruebas.md#si-tool_calls-no-aparece) |
| `CUDA out of memory` | Modelo + contexto no entran | `make restart CTX=16384` |
| Decode muy por debajo | Offload parcial a RAM o throttling | [docs/11](docs/11-verificacion-y-pruebas.md#si-el-decode-está-muy-por-debajo-de-lo-estimado) |
| El script aborta: partición en uso | Ocupaste una partición en VDI | `make start GPU_A=<otra>` |
| `make status` dice que no responde pero el log avanza | Todavía está cargando o descargando | `make logs-agentic` |
| Sin espacio en disco | Los 2 modelos suman ~27 GB | Ver la ruta de `MODEL_DIR` en [docs/11](docs/11-verificacion-y-pruebas.md) |

---

## Advertencias

- **La A16 es una tarjeta VDI.** 2 de las 4 particiones sirven a otra cosa.
- **250 W para las 4 particiones juntas.** Sumar inferencia a las 2 ocupadas
  puede provocar throttling. Medir antes de prometer velocidades.
- **Nada de esto fue probado en la A16 todavía.** Todos los t/s son estimaciones
  calculadas desde el ancho de banda (200 GB/s por partición ≈ 1/5 de una RTX
  4090). El paso 3 existe para reemplazarlas con números reales.
