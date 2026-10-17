# 03 — Arquitectura de despliegue

> **Diseño híbrido en 3 capas.** No intentar hacer todo con una sola A16.

Los dos objetivos (200 usuarios de cliente final + 20 devs) son **cargas
distintísimas** y cada uno merece su propia capa.

## Diagrama

```
                          ┌──────────────────────────────────┐
                          │   Gateway único (LiteLLM)        │
   200 usuarios ─────────▶│   auth · rate limit · logging    │
   (cliente final)        │   cache prefix · cost control    │──▶ CAPA 1
                          │   fallback / routing por modelo  │    Qwen Cloud API
                          └────────────────┬─────────────────┘    (producción)
                                           │
                          ┌────────────────┴─────────────────┐
   20 devs ──────────────▶│   4 réplicas vLLM en la A16      │──▶ CAPA 2
   (uso interno)          │   data-parallel, 1 por GPU      │    equipo de dev
                          │   modelo chico que cabe en 16 GB │
                          └────────────────┬─────────────────┘
                                           │
                          ┌────────────────┴─────────────────┐
                          │   Flash-Next en la A16 + RAM     │──▶ CAPA 3
                          │   experts en host, tabla en SSD  │    laboratorio
                          │   1-3 usuarios, experimentación  │
                          └──────────────────────────────────┘
```

## Capa 1 — Producción (200 usuarios) → API

**Qué:** Qwen Cloud API (o un agregador tipo OpenRouter) vía el gateway.

**Por qué no local:**

| | Local en A16 | API Qwen Cloud |
|---|---|---|
| 200 usuarios concurrentes | imposible (4-8 GPU H200) | trivial |
| Costo mensual | hardware de 6 cifras + ops 24/7 | ~USD 30-60 |
| Escalar a 500 | más hardware | más presupuesto |
| On-call y SLA | tu equipo | ellos |

**Funciones del gateway (no negociables):**

- **Auth**: API keys por cliente/tenant.
- **Rate limiting**: por usuario, por minuto/día. Protege el costo.
- **Logging**: prompt + respuesta + tokens. Necesario para debugging y facturación.
- **Prefix caching**: un system prompt compartido entre 200 usuarios se cachea y
  baja el costo de lectura de $0.15/M a **$0.016/M** (~10×).
- **Fallback**: si el API decae, rutea a la Capa 2 (local) degradada.
- **Control de costo**: alertas y caps por tenant.

## Capa 2 — Equipo de dev (20 integrantes) → A16 con vLLM

**Qué:** las **4 particiones de la A16** corriendo **4 réplicas data-parallel**
de un modelo denso chico.

**Modelo candidato:** algo que entre en 16 GB por partición.
Ejemplos: `Qwen3.8-27B` a Q3/Q4, o un ~12-14B a Q8. La referencia es:

```
pesos (16 GB VRAM) − KV cache − overhead ≈ 12-14 GB disponibles
```

**Por qué data-parallel y no tensor-parallel:**

- Sin NVLink → todo el allreduce de cada capa pasa por **PCIe Gen4 x16 (~32 GB/s)**.
- Tensor-parallel sobre la A16 mete un overhead fijo por capa que destruye la latencia.
- Data-parallel: cada partición corre una **copia completa** del modelo; un router
  distribuye requests entre ellas. Escalado lineal, sin comunicación entre GPUs.

**Por qué vLLM y no llama.cpp:**

- **Continuous batching** — es lo que te permite servir a varios usuarios a la vez.
- **PagedAttention** — gestiona el KV cache eficiente.
- **Prefix caching** — el system prompt compartido del asistente se cachea una vez.
- **API OpenAI-compatible** — el equipo de dev la consume directo desde OpenCode.
- **Métricas** — GPU utilization, queue depth, t/s, para saber cuándo escalar.

## Capa 3 — Laboratorio → Flash-Next en la A16

**Qué:** Flash-Next corriendo con experts en RAM del host, para **aprender el
tiering y experimentar**. Ver [05 — Runbook](05-runbook-llamacpp-en-la-a16.md).

**Límites esperados: 1 a 3 usuarios concurrentes.** No sirve para producción ni
para el equipo completo.

**Para qué sí sirve:**
- Medir t/s y entender el modelo.
- Probar prompts/tareas antes de pagar API.
- Estudiar la arquitectura Qwen4-preview.
- Offline de emergencia (un usuario, sin dependencia de nube).

## La pregunta que puede cambiar todo

**¿Hay requisito de compliance que impida mandar datos a la nube?**

Si la respuesta es **sí**, la Capa 1 se cae y el diseño completo se reorienta
a "qué modelo chico corremos on-prem" — que es un problema **más fácil** que
correr Flash-Next, pero donde la A16 se vuelve el recurso principal en vez de
un accesorio.

Siguiente: [04 — Costos y escala](04-costos-y-escala.md)
