# 07 — Runbook: gateway LiteLLM (Capa 1)

> **Un solo punto de salida** para producción (API) y local (Capa 2).
> Es lo que hace que el diseño sea mantenible.

## Qué resuelve

- **Auth**: API keys por cliente/tenant.
- **Rate limiting**: protege el costo contra picos y abuso.
- **Logging**: prompt, respuesta, tokens, latencia, costo.
- **Fallback**: si el API decae, rutea a local (degradado).
- **Cost control**: alertas y caps por tenant.
- **Ruteo por modelo**: clientes premium → mejor modelo; el resto → estándar.

## Config

```yaml
# litellm_config.yaml

model_list:
  # ── Capa 1: producción (API) ──────────────────────────
  - model_name: qwen-flash
    litellm_params:
      model: openai/qwen3.8-flash
      api_key: os.environ/DASHSCOPE_API_KEY
      api_base: https://maas.qwencloud.com/compatible-mode/v1

  # ── Capa 2: local (A16) ───────────────────────────────
  - model_name: qwen-local
    litellm_params:
      model: openai/qwen-local
      api_base: http://localhost:4001/v1   # el balanceador de las 4 réplicas
      api_key: local

# ── Ruteo con fallback ─────────────────────────────────
router_strategy: simple-shuffle
router:
  fallbacks:
    - qwen-flash: [qwen-local]   # si la API falla → local
  num_retries: 2
  cooldown_time: 120
  enable_pre_call_checks: true

litellm_settings:
  drop_params: true
  set_verbose: false
  # ── Cost control ─────────────────────────────────────
  max_budget: 500                       # USD por mes
  budget_duration: 30d
  alerting_threshold: 2.0               # USD en una sola llamada

# ── Keys por tenant ────────────────────────────────────
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/LITELLM_DB_URL     # Postgres, opcional pero recomendado
  store_model_in_db: true

  # ── Rate limiting ────────────────────────────────────
  default_max_tps: 10          # requests por segundo por key
  default_max_budget: 100      # USD por key
  max_parallel_request_limit: 5
```

## Levantar

```bash
pip install 'litellm[proxy]'

export DASHSCOPE_API_KEY="sk-..."
export LITELLM_MASTER_KEY="sk-master-..."
export LITELLM_DB_URL="postgresql://..."

litellm --config litellm_config.yaml --port 4000
```

Dashboard en `http://localhost:4000/` (requiere `store_model_in_db: true`).

## Generar keys por cliente/tenant

```bash
# Cliente A, 100 USD/mes, 5 req/s
curl -X POST http://localhost:4000/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "models": ["qwen-flash"],
    "max_budget": 100,
    "budget_duration": "30d",
    "max_parallel_requests": 5,
    "tpm_limit": 50000,
    "rpm_limit": 300
  }'
```

## Métricas a vigilar

| Métrica | Umbral de alerta |
|---|---|
| Latencia primer token (p95) | > 3 s |
| Error rate (5xx) | > 1% |
| Costo diario | > $50 |
| Budget consumido (mes) | > 80% |
| Fallback a local | > 10% de requests (la API decae) |

## Por qué este punto es no negociable

1. **Un solo lugar** para cambiar de proveedor (Qwen Cloud → OpenRouter →
   self-hosted) sin tocar los clientes.
2. **Control de costo** — sin esto, un bug en un cliente puede generar
   miles de USD de consumo.
3. **Auditoría** — saber qué se le envió a qué cliente y cuándo.
4. **Fuga de provider** — la Capa 2 local ya está cableada como fallback, así
   que un outage del proveedor degrada en vez de romper.

Siguiente: [08 — Decisiones y pendientes](08-decisiones-y-pendientes.md)
