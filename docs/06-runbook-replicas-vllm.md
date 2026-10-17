# 06 — Runbook: réplicas vLLM en la A16 (Capa 2, equipo de dev)

> **Patrón: data-parallel por partición.** 4 réplicas de un modelo chico,
> una copia por cada GPU de la A16, detrás de un balanceador.

## Por qué data-parallel y no tensor-parallel

La A16 es **4 GPUs separadas sin NVLink**. Con tensor-parallel, cada capa
requiere un *allreduce* que pasa por **PCIe Gen4 x16 (~32 GB/s)**. Eso agrega
un overhead fijo por capa que destruye la latencia.

Data-parallel: cada partición corre una **copia completa e independiente** del
modelo. Cero comunicación entre GPUs. Escalado lineal en throughput.

```
tensor-parallel (NO): 4 GPUs × 1 modelo sharded  → allreduce por capa por PCIe
data-parallel  (SÍ):  4 GPUs × 4 copias completas → router distribuye requests
```

## Elegir el modelo

Debe caber en **16 GB por partición** con espacio para KV cache:

```
16 GB (VRAM) − KV cache (8K-32K) − overhead CUDA (~1 GB) ≈ 12-14 GB para pesos
```

| Modelo | Cuant Q4 | ¿Cabe? |
|---|---|---|
| Qwen3.8-27B | ~16 GB | No (Q3 ~12 GB sí) |
| ~12-14B densos | ~7-9 GB | Cómodo, con contexto largo |
| ~8-14B | Q8 ~8-14 GB | Cómodo |

> **Calidad vs. contexto:** preferí un modelo más chico con **más contexto** a
> uno más grande con poco contexto. Para asistente de dev, el contexto es
> más valioso que unos puntos extra de benchmarks.

## Instalar vLLM

```bash
pip install vllm
```

## Lanzar las 4 réplicas

**Nodo 0 (GPU 0):**
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.8-27B-Chat \
  --host 0.0.0.0 --port 8001 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --served-model-name qwen-local \
  --enable-prefix-caching
```

Repetir en puertos **8001, 8002, 8003, 8004** para GPUs 0-3.

### Verificar

```bash
for p in 8001 8002 8003 8004; do
  echo "=== GPU/puerto $p ==="
  curl -s http://localhost:$p/v1/models | python3 -m json.tool
done

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

Cada GPU debe mostrar ~14-15 GB usados y el mismo `served-model-name`.

## Balanceador simple (hasta montar el LiteLLM de verdad)

**Opción A — LiteLLM (recomendada, misma herramienta que la Capa 1):**
```yaml
# litellm_config.yaml
model_list:
  - model_name: qwen-local
    litellm_params:
      model: openai/qwen-local
      api_base: http://localhost:8001/v1
      api_key: local
  - model_name: qwen-local
    litellm_params:
      model: openai/qwen-local
      api_base: http://localhost:8002/v1
      api_key: local
  - model_name: qwen-local
    litellm_params:
      model: openai/qwen-local
      api_base: http://localhost:8003/v1
      api_key: local
  - model_name: qwen-local
    litellm_params:
      model: openai/qwen-local
      api_base: http://localhost:8004/v1
      api_key: local

router:
  strategy: simple-shuffle
  num_retries: 3
  cooldown_time: 60

litellm_settings:
  drop_params: true
```
```bash
litellm --config litellm_config.yaml --port 4000
```

**Opción B — nginx (más simple, sin lógica de reintentos):**
```nginx
upstream qwen_local {
  least_conn;
  server 127.0.0.1:8001;
  server 127.0.0.1:8002;
  server 127.0.0.1:8003;
  server 127.0.0.1:8004;
}

server {
  listen 4000;
  location /v1/ {
    proxy_pass http://qwen_local;
    proxy_read_timeout 600s;
  }
}
```

## Uso desde OpenCode

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "qwen-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Qwen Local (A16)",
      "options": {
        "baseURL": "http://localhost:4000/v1",
        "apiKey": "local"
      },
      "models": {
        "qwen-local": { "name": "Qwen 27B local (A16)" }
      }
    }
  }
}
```

## Variables a ajustar

| Variable | Qué hace | Cuándo tocarla |
|---|---|---|
| `gpu-memory-utilization` | Cuánta VRAM usa | Subir a 0.95 si no comparte GPU |
| `max-model-len` | Contexto | Es lo primero que bajás si falta memoria |
| `max-num-seqs` | Requests simultáneos por réplica | Subir si la GPU está ociosa |
| `enable-prefix-caching` | Cachea el system prompt | **Mantener ON**, ahorra mucho |
| `max-num-batched-tokens` | Tokens por batch | Subir = más throughput, más memoria |

## Cargar a un equipo de 20 devs

Estimación gruesa. Con ~10B activos por token a Q4 (~6 GB/token) y 200 GB/s
por partición: **~25-30 t/s por réplica, ~100-120 t/s agregados**.

Para 20 devs donde **no todos generan a la vez**, con respuestas de ~400 tokens:

| Concurrentes reales | ¿100-120 t/s alcanza? |
|---|---|
| 1-5 | Cómodo |
| 10 | Justo, aceptable |
| 20 | Cada uno recibe ~5-6 t/s → lento |

**Señales de que hay que escalar:**
- `queue_depth` sostenido > 0 en las métricas de vLLM
- Primer token > 5 s
- VRAM al 95% con `max-model-len` ya bajado

**Opciones de escala en orden de costo:**
1. Bajar `max-model-len` (ganas throughput, pierdes contexto)
2. Subir `gpu-memory-utilization` a 0.95
3. Modelo más chico
4. Añadir una segunda A16
5. Mover parte del tráfico a la API (Capa 1)

## Con MTP no

MTP es ganancia a **concurrencia 1** y **pérdida neta a concurrencia 8**
(~0.81-0.87×). Para servir a un equipo, mantenerlo **apagado**.

Siguiente: [07 — Runbook: gateway LiteLLM](07-runbook-gateway.md)
