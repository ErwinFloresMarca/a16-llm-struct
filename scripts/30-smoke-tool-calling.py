#!/usr/bin/env python3
"""
30-smoke-tool-calling.py

Verifica de punta a punta que un endpoint local (vLLM o llama.cpp):

  1. responde y lista modelos
  2. emite tool_calls bien formadas (nombre + argumentos JSON válido)
  3. acepta el resultado de la tool y cierra la respuesta
  4. mide TTFT (primer token) y tokens/s reales
  5. mide prefill con un prompt largo

USO:
    python3 scripts/30-smoke-tool-calling.py [BASE_URL] [MODELO]

EJEMPLOS:
    python3 scripts/30-smoke-tool-calling.py http://localhost:8001/v1 agentic
    python3 scripts/30-smoke-tool-calling.py http://localhost:8002/v1 coding
"""

import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "agentic"

OK = "\u2705"
WARN = "\u26a0\ufe0f "
FAIL = "\u274c"


def _req(url, payload=None, timeout=300, stream=False):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(url, timeout=20):
    with _req(url, None, timeout) as r:
        return json.loads(r.read())


def post_json(url, payload, timeout=300):
    with _req(url, payload, timeout) as r:
        return json.loads(r.read())


def header(title):
    print()
    print("=" * 58)
    print(f" {title}")
    print("=" * 58)


# ─────────────────────────────────────────────────────────────
header("Smoke test de tool calling")
print(f" Endpoint: {BASE}")
print(f" Modelo:   {MODEL}")

# ── 1. ¿Responde? ────────────────────────────────────────────
print()
print("--- 1. Modelos disponibles ---")
try:
    models = get_json(f"{BASE}/models")
    names = [m.get("id") for m in models.get("data", [])]
    print(f"{OK} El endpoint responde. Modelos: {names}")
    if MODEL not in names:
        print(
            f"{WARN}OJO: '{MODEL}' no está en la lista. "
            "Revisá --served-model-name (vLLM) o --alias (llama.cpp)."
        )
except urllib.error.URLError as e:
    print(f"{FAIL} El endpoint no responde en {BASE}: {e}")
    sys.exit(1)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

PROMPT = "What is the weather in Buenos Aires right now? Use the tool."

# ── 2. Tool calling ──────────────────────────────────────────
print()
print("--- 2. Tool calling (prompt que fuerza el uso de una tool) ---")

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant. Use the provided tools when needed."},
        {"role": "user", "content": PROMPT},
    ],
    "tools": TOOLS,
    "tool_choice": "auto",
    "temperature": 0.0,
    "max_tokens": 512,
}

t0 = time.time()
try:
    resp = post_json(f"{BASE}/chat/completions", payload)
except urllib.error.HTTPError as e:
    print(f"{FAIL} HTTP {e.code}: {e.read().decode()[:600]}")
    sys.exit(1)
elapsed = time.time() - t0

if "error" in resp:
    print(f"{FAIL} El servidor devolvió un error:")
    print(json.dumps(resp["error"], indent=2)[:800])
    sys.exit(1)

msg = resp.get("choices", [{}])[0].get("message", {})
calls = msg.get("tool_calls") or []
usage = resp.get("usage", {})

print(f"Latencia total: {elapsed:.2f} s")
if usage:
    ct = usage.get("completion_tokens", 0)
    print(f"Tokens generados: {ct}  ->  ~{ct / max(elapsed, 1e-9):.1f} tok/s (incluye TTFT)")

tool_ok = False
if not calls:
    print()
    print(f"{WARN}SIN TOOL CALLS. El modelo respondió en texto plano.")
    print("     Causas típicas:")
    print("       vLLM      -> falta --enable-auto-tool-choice")
    print("       vLLM      -> falta --tool-call-parser <nombre>")
    print("       llama.cpp -> falta --jinja")
    print("       ambos     -> el nombre del modelo no coincide con el alias")
    print()
    print("     Contenido devuelto (primeros 500 chars):")
    print("      ", (msg.get("content") or "(vacío)")[:500])
else:
    print()
    print(f"{OK} TOOL CALLS DETECTADAS")
    all_json = True
    for c in calls:
        fn = c.get("function", {})
        print(f"    id:   {c.get('id')}")
        print(f"    name: {fn.get('name')}")
        args = fn.get("arguments")
        try:
            print(f"    args: {json.dumps(json.loads(args), ensure_ascii=False)}")
            print(f"    args: JSON válido {OK}")
        except Exception:
            all_json = False
            print(f"    args: {args!r}  <- NO es JSON válido {FAIL}")
    tool_ok = all_json

# ── 3. Cierre del ciclo ──────────────────────────────────────
print()
print("--- 3. Cierre del ciclo (tool result -> respuesta final) ---")
if not calls:
    print("Omitido: no hubo tool call que cerrar.")
else:
    followup = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "content": msg.get("content"), "tool_calls": calls},
            {
                "role": "tool",
                "tool_call_id": calls[0].get("id"),
                "content": '{"temp_c": 22, "conditions": "sunny"}',
            },
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    try:
        out = post_json(f"{BASE}/chat/completions", followup)
        if "error" in out:
            print(f"{FAIL} {json.dumps(out['error'])[:400]}")
        else:
            final = out["choices"][0]["message"].get("content") or "(vacío)"
            print(f"Respuesta final: {final[:600]}")
            print()
            print(f"{OK} CICLO COMPLETO: tool call -> resultado -> respuesta final")
    except urllib.error.HTTPError as e:
        print(f"{FAIL} HTTP {e.code} al cerrar el ciclo: {e.read().decode()[:400]}")

# ── 4. TTFT + decode con streaming ───────────────────────────
print()
print("--- 4. Velocidad real (streaming: TTFT + decode) ---")
stream_payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Count from 1 to 100, comma separated."}],
    "temperature": 0.0,
    "max_tokens": 300,
    "stream": True,
}
try:
    ttft = None
    ntok = 0
    t0 = time.time()
    with _req(f"{BASE}/chat/completions", stream_payload, stream=True) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                d = json.loads(chunk)
            except Exception:
                continue
            delta = (d.get("choices") or [{}])[0].get("delta", {})
            if delta.get("content"):
                if ttft is None:
                    ttft = time.time() - t0
                ntok += 1
    total = time.time() - t0
    decode_s = max(total - (ttft or 0), 1e-9)
    print(f"TTFT:  {(ttft or 0):.2f} s")
    print(f"Decode: {ntok} chunks en {decode_s:.2f} s  ->  ~{ntok / decode_s:.1f} tok/s")
    print("(comparar contra lo estimado en docs/10-catalogo-de-modelos.md)")
except Exception as e:
    print(f"No se pudo medir streaming: {e}")

# ── 5. Prefill ───────────────────────────────────────────────
print()
print("--- 5. Prefill (prompt largo) ---")
long_prompt = (
    "Analiza el siguiente texto y resúmelo en una línea. "
    + "lorem ipsum dolor sit amet consectetur adipiscing elit " * 400
)
pf = {
    "model": MODEL,
    "messages": [{"role": "user", "content": long_prompt}],
    "max_tokens": 32,
    "temperature": 0.0,
}
try:
    t0 = time.time()
    pr = post_json(f"{BASE}/chat/completions", pf)
    el = time.time() - t0
    pt = pr.get("usage", {}).get("prompt_tokens", 0)
    print(f"Prompt: {pt} tokens en {el:.2f} s  ->  ~{pt / max(el, 1e-9):.0f} tok/s de PREFILL")
    print("(el prefill es lo que más sufre en la A16: solo 5120 CUDA cores)")
except Exception as e:
    print(f"No se pudo medir el prefill: {e}")

# ── Resultado ────────────────────────────────────────────────
header(f"RESULTADO tool calling: {'OK ' + OK if tool_ok else 'CON PROBLEMAS ' + WARN}")
print(" Próximo paso: anotar estos números en docs/11-verificacion-y-pruebas.md")
print()
