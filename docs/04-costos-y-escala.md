# 04 — Costos y escala

> La cuenta que decide si 200 usuarios es un problema de ingeniería o un
> problema de presupuesto.

## Cuántos GPU necesitarías localmente

Para servir **200 usuarios realmente concurrentes** a calidad de asistente de
código, se necesitan **~4000 t/s agregados**.

La regla de ancho de banda: `t/s ≈ BW / bytes_por_token`.

Con un quant FP8 (~6 GB de pesos activos por token):

| GPU | Ancho de banda | t/s por GPU | GPUs para 4000 t/s |
|---|---|---|---|
| RTX 4090 | 1008 GB/s | ~170 | ~24 |
| A100 80GB | 2039 GB/s | ~340 | ~12 |
| H100 80GB | 3350 GB/s | ~560 | ~8 |
| **H200 141GB** | **4800 GB/s** | **~800** | **~5** |

Esto coincide con el layout oficial de vLLM de Qwen: **4 GPU para FP8 en
production**, o **8× H200 con expert parallel** para mayor escala.

> **Aproximado:** 4-8 GPU clase H200. Eso es **decenas a cientos de miles de
> USD** de hardware, más operación 24/7, más cooling, más rack.

## La misma carga en la API

Precios de `qwen3.8-flash` (Qwen Cloud / Alibaba):

| Concepto | USD / 1M tokens |
|---|---|
| Input | **$0.15** |
| Output | **$0.47** |
| Cache read | **$0.016** (~10× más barato) |

### Escenario: 200 usuarios de cliente final

Asumiendo:
- 200 usuarios
- 10 interacciones/día por usuario
- 2,000 tokens de input + 500 tokens de output por interacción
- **Sin** prefix caching

```
input:  200 × 10 × 2,000 × $0.15/M = $0.60 / día
output: 200 × 10 ×   500 × $0.47/M = $0.47 / día
─────────────────────────────────────────────────
                                        $1.07 / día
                                        ≈ $32 / mes
```

Con **prefix caching** (system prompt compartido cacheado a $0.016/M en lugar
de $0.15/M), el input baja un orden de magnitud → **~USD 15-20/mes**.

### Escenarios altos

| Escenario | USD / mes |
|---|---|
| 200 usuarios, 10 inter/día (base) | ~$32 |
| 200 usuarios, 100 inter/día | ~$320 |
| 500 usuarios, 50 inter/día | ~$1,000 |
| 20 devs, prompts de código largos (50K in / 5K out, 40 inter/día) | ~$370 |

Comparado con: **hardware de 4-8 GPU + operación 24/7**.

## Conclusión

> **Para los 200 usuarios de cliente final, la API no es el plan B — es la
> respuesta correcta.**

El local no compite en costo ni en esfuerzo operativo, y te ahorra cargar con
SLA, escalado, cooling, y on-call de un servicio que factura a clientes.

**La A16 tiene valor en la Capa 2** (equipo de dev, 20 integrantes): ya la tienes,
no cuesta más, y con 4 réplicas de un modelo chico sirve perfectamente.

## Cuándo la API deja de ser la respuesta

Ajustar hacia local cuando:

1. **Compliance**: no se puede sacar data de la empresa. → reorientar el diseño
   a un modelo chico on-prem (ver [03](03-arquitectura-despliegue.md)).
2. **Volumen**: el costo API supera ~USD 2,000-5,000/mes estable. Ahí un nodo
   propio empieza a tener ROI.
3. **Latencia**: necesitas <100 ms de primer token y el API no lo da.
4. **Disponibilidad**: el SLA de terceros no te sirve para facturar a clientes.

En cualquiera de esos casos, **el primer paso no es Flash-Next**, es:
- calcular cuántos tokens/s reales necesitas,
- elegir un modelo que cumpla la calidad mínima,
- dimensionar GPU para ese número.

Siguiente: [05 — Runbook: Flash-Next en la A16](05-runbook-llamacpp-en-la-a16.md)
