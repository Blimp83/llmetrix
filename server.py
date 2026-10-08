#!/usr/bin/env python3
"""LLMetrix Backend — Aggregiert GPU-Metriken (Prometheus/DCGM) + Ollama-Status
und serviert das Dashboard. Läuft auf .22, Daten kommen von .40.

Endpoints:
  GET  /            -> Dashboard (web/index.html)
  GET  /api/metrics -> GPU + Ollama + Server-Status
  GET  /api/history -> Token-Rate-Historie (Sparkline)
  POST /api/inference {target: gpu0|gpu1|dual} -> echtes Benchmark, liefert tk/s
"""
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROM_URL = os.environ.get("LMPROM_URL", "http://192.168.178.22:9090")
OLLAMA_EP = {
    "gpu0": os.environ.get("LM_OLLAMA_GPU0", "https://rtx-gpu0.obsy.icu"),
    "gpu1": os.environ.get("LM_OLLAMA_GPU1", "https://rtx-gpu1.obsy.icu"),
    "dual": os.environ.get("LM_OLLAMA_DUAL", "https://rtx-dual.obsy.icu"),
}
TIMEOUT = float(os.environ.get("LM_TIMEOUT", "6"))
LOKI_URL = os.environ.get("LM_LOKI_URL", "http://localhost:3100")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

# ── History (in-memory, optional persist) ────────────────────────────────
HIST_FILE = os.environ.get("LM_HIST_FILE", "/data/history.json")
MAX_HIST = 240
history = {"gen": [], "prompt": []}
hist_lock = threading.Lock()


def _load_history():
    try:
        with open(HIST_FILE) as f:
            d = json.load(f)
            history["gen"] = d.get("gen", [])[-MAX_HIST:]
            history["prompt"] = d.get("prompt", [])[-MAX_HIST:]
    except Exception:
        pass


def _save_history():
    try:
        os.makedirs(os.path.dirname(HIST_FILE), exist_ok=True)
        with open(HIST_FILE, "w") as f:
            json.dump({"gen": history["gen"][-MAX_HIST:], "prompt": history["prompt"][-MAX_HIST:]}, f)
    except Exception:
        pass


def _push_history(gen, prompt):
    now = time.time()
    with hist_lock:
        history["gen"].append([round(now, 2), round(gen, 1)])
        history["prompt"].append([round(now, 2), round(prompt, 1)])
        if len(history["gen"]) > MAX_HIST:
            history["gen"] = history["gen"][-MAX_HIST:]
            history["prompt"] = history["prompt"][-MAX_HIST:]
    _save_history()


# ── Prometheus (DCGM) ─────────────────────────────────────────────────────
def prom_query(query):
    url = f"{PROM_URL}/api/v1/query"
    data = ("query=" + urllib.parse.quote(query)).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)["data"]["result"]


def get_gpus():
    gpus = {}
    try:
        for m in prom_query('DCGM_FI_DEV_GPU_UTIL{instance="192.168.178.40:9400"}'):
            gpus[m["metric"].get("gpu", "0")] = {"load": float(m["value"][1])}
        for m in prom_query('DCGM_FI_DEV_FB_USED{instance="192.168.178.40:9400"}'):
            gpus.setdefault(m["metric"].get("gpu", "0"), {})["vram_used_mib"] = float(m["value"][1])
        for m in prom_query('DCGM_FI_DEV_FB_FREE{instance="192.168.178.40:9400"}'):
            gpus.setdefault(m["metric"].get("gpu", "0"), {})["vram_free_mib"] = float(m["value"][1])
        for m in prom_query('DCGM_FI_DEV_GPU_TEMP{instance="192.168.178.40:9400"}'):
            gpus.setdefault(m["metric"].get("gpu", "0"), {})["temp_c"] = float(m["value"][1])
        for m in prom_query('DCGM_FI_DEV_POWER_USAGE{instance="192.168.178.40:9400"}'):
            gpus.setdefault(m["metric"].get("gpu", "0"), {})["power_w"] = float(m["value"][1])
    except Exception as e:
        return {"error": str(e)}
    for g in gpus.values():
        if "vram_used_mib" in g and "vram_free_mib" in g:
            g["vram_total_mib"] = round(g.pop("vram_free_mib") + g["vram_used_mib"])
    return gpus


def get_ollama():
    out = {}
    for key, base in OLLAMA_EP.items():
        info = {"online": False}
        try:
            with urllib.request.urlopen(base + "/api/version", timeout=TIMEOUT) as r:
                info["version"] = json.load(r).get("version")
                info["online"] = True
            with urllib.request.urlopen(base + "/api/ps", timeout=TIMEOUT) as r:
                ps = json.load(r)
            sysinfo = ps.get("system_info") or {}
            info["vram_total_mib"] = (sysinfo.get("vram_total") or [0])[0]
            models = []
            for m in ps.get("models", []):
                d = m.get("details", {})
                models.append({
                    "name": m.get("name"),
                    "params": d.get("parameter_size"),
                    "quant": d.get("quantization_level"),
                    "ctx": m.get("context_length"),
                    "vram_gb": round(m.get("size_vram", 0) / 1e9, 1),
                })
            info["models"] = models
        except Exception:
            pass
        out[key] = info
    return out


def build_metrics():
    gpus = get_gpus()
    ollama = get_ollama()
    ai_online = any(g.get("load") is not None for g in gpus.values()) if isinstance(gpus, dict) and "error" not in gpus else False
    return {
        "ts": round(time.time(), 2),
        "ai_server_online": ai_online,
        "gpus": gpus,
        "ollama": ollama,
    }


# ── Loki (Ollama-Docker-Logs) ─────────────────────────────────────────────
def loki_query_range(query, limit=150, seconds=3600):
    now = int(time.time() * 1e9)
    params = urllib.parse.urlencode({
        "query": query,
        "start": str(now - seconds * 1_000_000_000),   # ganzzahlige ns!
        "end": str(now),
        "limit": str(limit),
    })
    url = f"{LOKI_URL}/loki/api/v1/query_range?{params}"
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
        return json.load(r)["data"]["result"]


def get_logs(gpu, limit=80):
    """Neueste Logs eines Ollama-Containers (gpu: gpu0|gpu1|dual) aus Loki."""
    try:
        series = loki_query_range(f'{{job="ollama",gpu="{gpu}"}}', limit=limit)
        entries = []
        for s in series:
            for ts_ns, line in s.get("values", []):
                entries.append({"ts": int(ts_ns) / 1e9, "line": line})
        entries.sort(key=lambda e: e["ts"])
        return {"gpu": gpu, "entries": entries[-limit:]}
    except Exception as e:
        return {"gpu": gpu, "error": str(e), "entries": []}


# ── Live t/s aus den Ollama-Logzeilen (llama.cpp print_timing) ─────────────
_TG_RE = re.compile(r"n_gen\s*=\s*(\d+),\s*tg\s*=\s*([\d.]+)\s*t/s,\s*tg_3s\s*=\s*([\d.]+)\s*t/s")
_PROMPT_RE = re.compile(r"prompt eval time\s*=\s*[\d.\s]+ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second")


def get_live_tps(gpu, window=900):
    """Kontinuierliche t/s-Punkte aus den Live-Logs (letzte `window` Sekunden).

    Punkte:
      kind=gen    → tg (Ø t/s der laufenden Generation), tg3s (Rolling 3s)
      kind=prompt → tps des Prompt-Evals (eigener Punkt, wird nicht gemischt)
    """
    try:
        pts = []
        for s in loki_query_range(f'{{job="ollama",gpu="{gpu}"}}', limit=500, seconds=window):
            for ts_ns, line in s.get("values", []):
                m = _TG_RE.search(line)
                if m:
                    pts.append({"t": int(ts_ns) // 1_000_000_000, "kind": "gen",
                                "tg": float(m.group(2)), "tg3s": float(m.group(3))})
                    continue
                p = _PROMPT_RE.search(line)
                if p:
                    pts.append({"t": int(ts_ns) // 1_000_000_000, "kind": "prompt",
                                "tps": float(p.group(2))})
        pts.sort(key=lambda x: x["t"])
        now = int(time.time())
        last_gen = next((p for p in reversed(pts) if p["kind"] == "gen"), None)
        return {
            "gpu": gpu,
            "window_s": window,
            "count": len(pts),
            "points": pts,
            "active": bool(last_gen and now - last_gen["t"] < 90),
            "current_tps": round(last_gen["tg"], 1) if last_gen else None,
        }
    except Exception as e:
        return {"gpu": gpu, "error": str(e), "points": [], "count": 0}


# ── Inference Benchmark (echte tk/s) ──────────────────────────────────────
def run_benchmark(target):
    base = OLLAMA_EP.get(target)
    if not base:
        return {"error": f"unknown target {target}"}
    model = None
    cold_start = False
    try:
        with urllib.request.urlopen(base + "/api/ps", timeout=TIMEOUT) as r:
            models = json.load(r).get("models", [])
        if models:
            model = models[0]["name"]
        else:
            # Fallback: erstes verfügbares Modell (Cold-Start, dauert länger)
            with urllib.request.urlopen(base + "/api/tags", timeout=TIMEOUT) as r:
                tags = json.load(r).get("models", [])
            if tags:
                model = tags[0]["name"]
                cold_start = True
    except Exception:
        pass
    if not model:
        return {"error": f"kein Modell auf {target} (ps + tags leer)"}

    payload = json.dumps({
        "model": model,
        "prompt": "Count from 1 to 20.",
        "stream": False,
        "options": {"num_predict": 24},
    }).encode()
    req = urllib.request.Request(base + "/api/generate", data=payload,
                                  headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            d = json.load(r)
    except Exception as e:
        return {"error": str(e)}
    wall = round(time.time() - t0, 2)

    pec = d.get("prompt_eval_count", 0)
    ped = d.get("prompt_eval_duration", 0) / 1e9
    ec = d.get("eval_count", 0)
    ed = d.get("eval_duration", 0) / 1e9
    prompt_tps = round(pec / ped, 1) if ped > 0.01 else None
    gen_tps = round(ec / ed, 1) if ed > 0.01 else None
    _push_history(gen_tps or 0, prompt_tps or 0)
    return {
        "target": target,
        "model": model,
        "cold_start": cold_start,
        "wall_s": wall,
        "prompt_tokens": pec,
        "prompt_tps": prompt_tps,
        "gen_tokens": ec,
        "gen_tps": gen_tps,
    }


# ── HTTP ──────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    server_version = "LLMetrix/1.0"

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/metrics":
            return self._send(200, build_metrics())
        if path == "/api/history":
            with hist_lock:
                h = {"gen": history["gen"][-MAX_HIST:], "prompt": history["prompt"][-MAX_HIST:]}
            return self._send(200, h)
        if path == "/api/logs":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            gpu = (q.get("gpu", ["dual"])[0] or "dual")
            if gpu not in ("gpu0", "gpu1", "dual"):
                gpu = "dual"
            limit_s = q.get("limit", ["80"])[0]
            try:
                limit = max(1, min(int(limit_s), 300))
            except ValueError:
                limit = 80
            return self._send(200, get_logs(gpu, limit))
        if path == "/api/live_tps":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            gpu = (q.get("gpu", ["dual"])[0] or "dual")
            if gpu not in ("gpu0", "gpu1", "dual"):
                gpu = "dual"
            win_s = q.get("window", ["900"])[0]
            try:
                window = max(60, min(int(win_s), 3600))
            except ValueError:
                window = 900
            return self._send(200, get_live_tps(gpu, window))
        # static
        rel = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
        fp = os.path.normpath(os.path.join(WEB_DIR, rel))
        if not fp.startswith(WEB_DIR) or not os.path.isfile(fp):
            return self._send(404, {"error": "not found"})
        ctype = "text/html" if fp.endswith(".html") else "application/octet-stream"
        with open(fp, "rb") as f:
            return self._send(200, f.read(), ctype)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path != "/api/inference":
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._send(400, {"error": "bad json"})
        target = body.get("target", "dual")
        result = run_benchmark(target)
        code = 200 if "error" not in result else 502
        return self._send(code, result)

    def log_message(self, fmt, *args):
        print("[llmetrix]", fmt % args, flush=True)


_load_history()
if __name__ == "__main__":
    port = int(os.environ.get("LM_PORT", "8790"))
    print(f"[llmetrix] starting on :{port} (prometheus={PROM_URL})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
