#!/usr/bin/env python3
"""LLMetrix Backend — Aggregiert GPU-Metriken (DCGM/Prometheus) + Ollama-Status
und serviert das Dashboard. Konfiguration komplett über Env-Vars (LM_*).

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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROM_URL = os.environ.get("LMPROM_URL", "http://127.0.0.1:9090")
DCGM_URL = os.environ.get("LM_DCGM_URL", "http://127.0.0.1:9400")   # direkter Exporter-Scrape (~17 ms) statt Prometheus (5–15 s Latenz)
POWER_LIMIT_W = float(os.environ.get("LM_POWER_LIMIT_W", "180"))   # Max-Power der Karte (RTX 5060 Ti = 180 W) — für den Power-Anzeiger
SERVER_NAME = os.environ.get("LM_SERVER_NAME", "")   # optionale freundliche Server-Bezeichnung fürs Dashboard
OLLAMA_EP = {
    "gpu0": os.environ.get("LM_OLLAMA_GPU0", "http://127.0.0.1:11434"),
    "gpu1": os.environ.get("LM_OLLAMA_GPU1", "http://127.0.0.1:11435"),
    "dual": os.environ.get("LM_OLLAMA_DUAL", "http://127.0.0.1:11436"),
}
TIMEOUT = float(os.environ.get("LM_TIMEOUT", "8"))   # Ollama antwortet unter Inferenzlast teils 2–4 s
LOKI_URL = os.environ.get("LM_LOKI_URL", "http://localhost:3100")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

# ── Generische Inference-Targets (ollama oder openai-kompatibel) ─────────────
# Instanzen sind reine Konfiguration: LM_TARGETS=name=url,name2=url2. Der Typ wird
# pro Target entschieden — ollama-Typ nutzt /api/ps + /api/show + /api/generate,
# openai-Typ (/v1) liest seine Metriken aus dem Prometheus-Textformat von /metrics,
# braucht also kein Loki und keine Docker-Container-Namen. Key pro Target über
# LM_KEY_<NAME> (oder global LM_OPENAI_KEY). Nichts ist an einen bestimmten Host
# gebunden: jeder vLLM/Strata/LlamaIndex-kompatible Server funktioniert.
TARGETS = {}
for _spec in os.environ.get("LM_TARGETS", "").split(","):
    _spec = _spec.strip()
    if not _spec or "=" not in _spec:
        continue
    _n, _u = _spec.split("=", 1)
    TARGETS[_n] = {"url": _u.strip().rstrip("/"), "key": os.environ.get("LM_KEY_" + _n.upper()) or os.environ.get("LM_OPENAI_KEY", "")}
for _k, _v in OLLAMA_EP.items():          # Legacy-Env bleibt erhalten
    TARGETS.setdefault(_k, {"url": _v.rstrip("/"), "key": os.environ.get("LM_OPENAI_KEY", "")})

DEFAULT_TARGET = os.environ.get("LM_DEFAULT_TARGET", "")

def resolve_target(gpu):
    """Target-Namen akzeptieren (ollama- oder openai-Typ), sonst konfigurierbarer Default."""
    return gpu if gpu in TARGETS else (DEFAULT_TARGET or next(iter(TARGETS), ""))

def target_type(name):
    return "ollama" if name in OLLAMA_EP else "openai"

def auth_headers(key, accept=None):
    h = {"User-Agent": "llmetrix"}
    if key:
        h["Authorization"] = "Bearer " + key
    if accept:
        h["Accept"] = accept
    return h

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
# ── Prometheus-Textformat: direkter Scrape (kein Prometheus nötig) ───────────
# Wert muss auch Dezimalzahlen und Exponenten erfassen (104.1, 0.985) — sonst fallen
# alle Gauge-Werte mit Komma durch den Parser und die Instanz zeigt null.
_PLINE_RE = re.compile(r'^([a-zA-Z_:][\w:]*)(?:\{([^}]*)\})?\s+(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$')

def parse_prom_text(text):
    """{metric: [(labels, value), ...]} aus Prometheus-Text (Exposition Format)."""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _PLINE_RE.match(line)
        if not m:
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', m.group(2) or ""))
        try:
            out.setdefault(m.group(1), []).append((labels, float(m.group(3))))
        except ValueError:
            pass
    return out

def scrape_metrics(url, key="", timeout=None):
    req = urllib.request.Request(url + "/metrics", headers=auth_headers(key, "text/plain"))
    with urllib.request.urlopen(req, timeout=timeout or TIMEOUT) as r:
        return parse_prom_text(r.read().decode("utf-8", "replace"))

def pick(series, label, want):
    for labels, v in series:
        if labels.get(label) == want:
            return v
    return None

def prom_query(query):
    url = f"{PROM_URL}/api/v1/query"
    data = ("query=" + urllib.parse.quote(query)).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)["data"]["result"]


def get_gpus():
    gpus = {}
    try:
        # Alle Metriken parallel (statt sequenziell) → /api/metrics bleibt schnell, auch unter Inferenzlast.
        # Instance aus DCGM_URL ableiten (universell), statt hartkodeter IP.
        dcgm_host = urllib.parse.urlparse(DCGM_URL).netloc
        def one(metric, key):
            out = {}
            try:
                for m in prom_query(f'{metric}{{instance="{dcgm_host}"}}'):
                    out[m["metric"].get("gpu", "0")] = float(m["value"][1])
            except Exception:
                pass
            return key, out
        with ThreadPoolExecutor(max_workers=5) as ex:
            for key, vals in ex.map(lambda kv: one(*kv), [
                ('DCGM_FI_DEV_GPU_UTIL', 'load'),
                ('DCGM_FI_DEV_FB_USED', 'vram_used_mib'),
                ('DCGM_FI_DEV_FB_FREE', 'vram_free_mib'),
                ('DCGM_FI_DEV_GPU_TEMP', 'temp_c'),
                ('DCGM_FI_DEV_POWER_USAGE', 'power_w'),
            ]):
                for gpu, v in vals.items():
                    gpus.setdefault(gpu, {})[key] = v
    except Exception as e:
        return {"error": str(e)}
    for g in gpus.values():
        if "vram_used_mib" in g and "vram_free_mib" in g:
            g["vram_total_mib"] = round(g.pop("vram_free_mib") + g["vram_used_mib"])
    return gpus


# ── Ollama /api/ps-Cache: unter Inferenz-Lust antwortet /api/ps teils leer/zeitüberschritten ──
# → letzter bekannt guter Stand wird vorgehalten (bis 60 s), damit das Dashboard nicht flackert.
_ps_cache = {}
_ps_cache_lock = threading.Lock()
PS_CACHE_TTL = 60.0
_online_cache = {}   # letzter guter /api/version-Check pro Endpoint (unter Last teils Timeout → "online" flackert)
ONLINE_TTL = 60.0


def _ps_models_fresh(base, gpu, key=""):
    """/api/ps mit Retry + Cache-Fallback (unter Inferenz-Lust teils leer). Liefert (models_raw, sysinfo, stale)."""
    models_raw, sysinfo = [], {}
    try:
        for attempt in range(2):
            with urllib.request.urlopen(urllib.request.Request(base + "/api/ps", headers=auth_headers(key)), timeout=TIMEOUT) as r:
                ps = json.load(r)
            models_raw = ps.get("models") or []
            sysinfo = ps.get("system_info") or {}
            if models_raw:
                break
            time.sleep(0.4)
    except Exception:
        pass
    if models_raw:
        with _ps_cache_lock:
            _ps_cache[gpu] = (time.time(), (models_raw, sysinfo))
    else:
        with _ps_cache_lock:
            hit = _ps_cache.get(gpu)
        if hit and time.time() - hit[0] < PS_CACHE_TTL:
            models_raw, sysinfo = hit[1]
            return models_raw, sysinfo, True
    return models_raw, sysinfo, False


def _ollama_one(key, base):
    apikey = TARGETS.get(key, {}).get("key", "")
    info = {"online": False, "type": "ollama"}
    try:
        with urllib.request.urlopen(urllib.request.Request(base + "/api/version", headers=auth_headers(apikey)), timeout=TIMEOUT) as r:
            info["version"] = json.load(r).get("version")
            info["online"] = True
        _online_cache[key] = time.time()
    except Exception:
        # unter Inferenz-Lust teils Timeout → letzter guter Stand vorgehalten (bis 60 s), damit "Modell: none" nicht flackert
        if time.time() - _online_cache.get(key, 0) < ONLINE_TTL:
            info["online"] = True
    models_raw, sysinfo, stale = _ps_models_fresh(base, key, TARGETS.get(key, {}).get("key", ""))
    info["vram_total_mib"] = (sysinfo.get("vram_total") or [0])[0]
    models = []
    for m in models_raw:
        d = m.get("details", {})
        models.append({
            "name": m.get("name"),
            "params": d.get("parameter_size"),
            "quant": d.get("quantization_level"),
            "ctx": m.get("context_length"),
            "vram_gb": round(m.get("size_vram", 0) / 1e9, 1),
            # Offload-Rate: Anteil des Modells im VRAM (1.0 = voller GPU-Offload, kleiner = CPU-Spill-over)
            "offload_pct": round(100 * m.get("size_vram", 0) / m["size"], 1) if m.get("size") else None,
        })
    if models:
        info["models"] = models
    if stale and models:
        info["models_stale"] = True
    return key, info


def _openai_one(name):
    """Eine openai-kompatible Instanz (Strata, vLLM, llama.cpp server): Modell + Metriken
    aus /health und dem /metrics-Scrape. Rein numerisch → kein Loki, keine Log-Regex."""
    t = TARGETS[name]
    info = {"online": False, "type": "openai", "url": t["url"]}
    try:
        req = urllib.request.Request(t["url"] + "/health", headers=auth_headers(t["key"]))
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            health = json.load(r)
        info["online"] = True
        _online_cache[name] = time.time()
    except Exception as e:
        if time.time() - _online_cache.get(name, 0) < ONLINE_TTL:
            info["online"] = True
        else:
            info["error"] = str(e)
            _activity(name, info)
            return name, info
    try:
        m = scrape_metrics(t["url"], t["key"])
    except Exception as e:
        info["metrics_error"] = str(e)
        _activity(name, info)
        return name, info
    model = health.get("model") if isinstance(health, dict) else None
    model = model or next((l.get("model_name") for l, _ in m.get("vllm:num_requests_running", [])), None)
    g = lambda metric: pick(m.get(metric, []), "model_name", model)
    info["model"] = model
    info["max_context"] = g("strata:engine_max_context")
    info["running"] = int(g("vllm:num_requests_running") or 0)
    info["waiting"] = int(g("vllm:num_requests_waiting") or 0)
    info["kv_usage_pct"] = round(100 * (g("vllm:kv_cache_usage_perc") or 0), 1)
    info["state"] = next((l.get("state") for l, v in m.get("strata:live_state", []) if v == 1), None)
    info["live_tok_s"] = g("strata:live_tok_s")
    info["live_prefill_tok_s"] = g("strata:live_prefill_tok_s_mean")
    q, hit = g("vllm:prefix_cache_queries_total") or 0, g("vllm:prefix_cache_hits_total") or 0
    info["cache_hit_pct"] = round(100 * hit / q, 1) if q else None
    info["last_decode_tok_s"] = g("strata:last_decode_tok_s")
    d, a, n = int(g("vllm:spec_decode_num_draft_tokens_total") or 0), int(g("vllm:spec_decode_num_accepted_tokens_total") or 0), int(g("vllm:request_success_total") or 0)
    info["mtp"] = {"acc_rate": round(a / d, 3) if d else None, "mean_acc_len": round(a / d, 2) if d else None,
                   "draft_tokens": d, "accepted_tokens": a}
    gpus = {}
    for metric, key_ in (("strata:gpu_util", "load"), ("strata:gpu_mem_used_bytes", "vram_used_mib"),
                         ("strata:gpu_mem_total_bytes", "vram_total_mib"), ("strata:gpu_temp_celsius", "temp_c"),
                         ("strata:gpu_power_watts", "power_w")):
        for labels, v in m.get(metric, []):
            gpu = labels.get("gpu")
            if gpu is None:
                continue
            cell = gpus.setdefault(gpu, {})
            cell[key_] = round(v / 1e6) if "bytes" in metric else v
    info["gpus"] = gpus
    # Modelle im selben Format wie der ollama-Record (name/ctx/vram_gb), damit das
    # Frontend beide Instanztypen gleich behandeln kann. params/quant bleiben leer,
    # wenn der Server sie nicht nennt.
    vram_gb = round(sum(c.get("vram_used_mib", 0) for c in gpus.values()) / 1024, 1)
    info["models"] = [{"name": model, "params": None, "quant": None, "ctx": info["max_context"],
                       "vram_gb": vram_gb, "offload_pct": None}] if model else []
    _activity(name, info)
    return name, info

def get_openai():
    names = [n for n in TARGETS if target_type(n) == "openai"]
    if not names:
        return {}
    with ThreadPoolExecutor(max_workers=min(4, len(names))) as ex:
        return dict(ex.map(lambda n: _openai_one(n), names))

def get_ollama():
    # Parallel: Ollama-Server unter Inferenzlast antworten teils langsam (5–12 s) →
    # sequenziell würde /api/metrics auf 10+ s laufen und der Frontend-Poll (2 s) würde timeouten.
    with ThreadPoolExecutor(max_workers=3) as ex:
        return dict(ex.map(lambda kv: _ollama_one(*kv), [(n, TARGETS[n]["url"]) for n in OLLAMA_EP]))


# ── Ollama-Background-Refresher: hält den Stand immer frisch, damit /api/metrics sofort antwortet ──
# Unter Inferenzlast brauchen die Ollama-Calls 5–12 s. Statt dass jeder Poll wartet,
# aktualisiert ein Daemon-Thread alle 3 s im Hintergrund; der Endpoint liefert den letzten guten Stand.
_oll_cache = {"ts": 0, "data": None}

# ── DCGM-Live-Cache: direkter Scrape des Exporters alle 1 s → GPU-Last kommt ~live ins Dashboard ──
_dcg_cache = {"ts": 0, "gpus": None}
_DCGM_RE = re.compile(r'^(DCGM_FI_DEV_(?:GPU_UTIL|FB_USED|FB_FREE|GPU_TEMP|POWER_USAGE))\{gpu="(\d+)"[^}]*\}\s+([\d.]+)\s*$')
_DCGM_LABEL_RE = re.compile(r'^DCGM_FI_DEV_\w+\{([^}]*)\}')

# node-exporter (AI-Server-System: CPU/RAM)
NODE_URL = os.environ.get("LM_NODE_URL", "http://127.0.0.1:9100")
_NODE_RE = re.compile(r'^(node_cpu_seconds_total|node_memory_MemTotal_bytes|node_memory_MemAvailable_bytes)\{?([^}]*)\}?\s+([0-9.]+(?:[eE][+-]?[0-9]+)?)\s*$')
_sys_cache = {"ts": 0, "data": None}


def _node_scrape():
    """node-exporter scrapen → System-CPU (via idle-Delta) + RAM-Füllstand."""
    req = urllib.request.Request(f"{NODE_URL}/metrics", headers={"User-Agent": "llmetrix"})
    with urllib.request.urlopen(req, timeout=3) as r:
        text = r.read().decode("utf-8", "replace")
    mem_total = mem_avail = None
    idle_sum = 0.0
    ncpu = 0
    for line in text.splitlines():
        m = _NODE_RE.match(line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2), float(m.group(3))
        if name == "node_cpu_seconds_total" and 'mode="idle"' in labels:
            idle_sum += val
            ncpu += 1
        elif name == "node_memory_MemTotal_bytes":
            mem_total = val
        elif name == "node_memory_MemAvailable_bytes":
            mem_avail = val
    cpu_pct = None
    now = time.time()
    prev_idle, prev_ts = _sys_cache.get("idle_sum"), _sys_cache.get("ts", 0)
    if prev_idle is not None and idle_sum > prev_idle and ncpu:
        dt = max(1e-6, now - prev_ts)
        idle_frac = (idle_sum - prev_idle) / (ncpu * dt)
        cpu_pct = round(max(0.0, min(100.0, 100.0 * (1.0 - idle_frac))), 1)
    _sys_cache["idle_sum"] = idle_sum
    _sys_cache["ts"] = now
    return {
        "cpu_pct": cpu_pct,
        "ncpu": ncpu or None,
        "ram_total_mb": round(mem_total / 1e6) if mem_total else None,
        "ram_used_mb": round((mem_total - mem_avail) / 1e6) if (mem_total and mem_avail is not None) else None,
    }


def _node_refresher():
    while True:
        try:
            _sys_cache["data"] = _node_scrape()
        except Exception:
            pass   # letzter guter Stand bleibt
        time.sleep(2.0)


def _dcgm_scrape():
    """DCGM-Exporter direkt scrapen (Prometheus-Textformat). Liefert {gpu: {load, vram_used_mib, ...}} + gpu_name/hostname."""
    req = urllib.request.Request(f"{DCGM_URL}/metrics", headers={"User-Agent": "llmetrix"})
    with urllib.request.urlopen(req, timeout=3) as r:
        text = r.read().decode("utf-8", "replace")
    gpus = {}
    gpu_name = hostname = None
    for line in text.splitlines():
        if gpu_name is None and line.startswith("DCGM_FI_DEV_GPU_UTIL{"):
            # modelName/Hostname aus den Labels der ersten Zeile (universell: keine hartkodierten Namen)
            lm = _DCGM_LABEL_RE.match(line)
            if lm:
                labels = dict(re.findall(r'(\w+)="([^"]*)"', lm.group(1)))
                gpu_name = labels.get("modelName")
                hostname = labels.get("Hostname")
        m = _DCGM_RE.match(line)
        if m:
            name, gpu, val = m.group(1), int(m.group(2)), float(m.group(3))
            key = {"GPU_UTIL": "load", "FB_USED": "vram_used_mib",
                   "FB_FREE": "vram_free_mib", "GPU_TEMP": "temp_c",
                   "POWER_USAGE": "power_w"}[name.split("_DEV_")[1]]
            gpus.setdefault(gpu, {})[key] = val
    if not gpus:
        raise ValueError("keine DCGM-Werte")
    for g in gpus.values():
        if "vram_used_mib" in g and "vram_free_mib" in g:
            g["vram_total_mib"] = round(g.pop("vram_free_mib") + g["vram_used_mib"])
    if gpu_name:
        for g in gpus.values():
            g["gpu_name"] = gpu_name
    return {"gpus": gpus, "gpu_name": gpu_name, "hostname": hostname}


def _dcgm_refresher():
    while True:
        try:
            data = _dcgm_scrape()
            _dcg_cache["gpus"] = data["gpus"]
            _dcg_cache["gpu_name"] = data.get("gpu_name")
            _dcg_cache["hostname"] = data.get("hostname")
            _dcg_cache["ts"] = time.time()
        except Exception:
            pass   # letzter guter Stand bleibt (bis 10 s alt → Fallback auf Prometheus)
        time.sleep(1.0)


def get_gpus_live():
    """GPU-Daten: frisch aus dem DCGM-Cache (≤10 s alt), sonst Fallback auf Prometheus."""
    if _dcg_cache["gpus"] and time.time() - _dcg_cache["ts"] < 10:
        return {str(k): v for k, v in _dcg_cache["gpus"].items()}


def _ollama_refresher():
    while True:
        try:
            data = get_ollama()
            _oll_cache["data"] = data
            _oll_cache["ts"] = time.time()
        except Exception:
            pass
        time.sleep(3)


_openai_cache = {"ts": 0, "data": None}
_openai_tps = {}          # target -> Punkte. Die Gauges sind Momentwerte, ohne Puffer wäre
                          # pro Poll genau ein Punkt da → Sparkline wäre leer.

def _openai_refresher():
    while True:
        try:
            data = get_openai()
            _openai_cache["data"] = data
            _openai_cache["ts"] = time.time()
            now = int(time.time())
            for name, inst in data.items():
                pts = _openai_tps.setdefault(name, [])
                if inst.get("live_tok_s"):
                    pts.append({"t": now, "kind": "gen", "tg": inst["live_tok_s"]})
                if inst.get("live_prefill_tok_s"):
                    pts.append({"t": now, "kind": "prompt", "tps": inst["live_prefill_tok_s"]})
                del pts[:-MAX_HIST]
        except Exception:
            pass   # letzter guter Stand bleibt
        time.sleep(3)

def get_openai_cached():
    if _openai_cache["data"] is None:
        try:
            _openai_cache["data"] = get_openai()
        except Exception:
            pass
    return _openai_cache["data"] or {}

def get_ollama_cached():
    """Letzter guter Ollama-Stand (Background-Refresher); erst beim Start synchron nachladen."""
    if _oll_cache["data"] is None:
        try:
            _oll_cache["data"] = get_ollama()
            _oll_cache["ts"] = time.time()
        except Exception:
            pass
    return _oll_cache["data"] or {}


def build_metrics():
    # GPUs: frisch aus dem DCGM-Live-Cache (1 s-Scrape, ~17 ms) → Last kommt ~live rein;
    # Fallback auf Prometheus (5–15 s Latenz), falls der Exporter kurz nicht erreichbar ist.
    gpus = get_gpus_live() or get_gpus()
    ollama = get_ollama_cached()
    openai = get_openai_cached()
    # Der /metrics-Scrape sieht dieselben physischen Karten wie der DCGM-Exporter
    # (gpu="0"/"1"). Werte werden pro Karte ergänzt, nicht ersetzt: DCGM bleibt die
    # Primärquelle für Load/Temp/Power, der Scrape liefert, was DCGM nicht kennt.
    if isinstance(gpus, dict) and "error" not in gpus:
        for inst in openai.values():
            for gpu, cell in (inst.get("gpus") or {}).items():
                tgt = gpus.setdefault(str(gpu), {})
                for k, v in cell.items():
                    if tgt.get(k) is None:
                        tgt[k] = v
    ai_online = any(g.get("load") is not None for g in gpus.values()) if isinstance(gpus, dict) and "error" not in gpus else False
    return {
        "ts": round(time.time(), 2),
        "ai_server_online": ai_online or bool(openai) or any(i.get("online") for i in ollama.values()),
        "gpus": gpus,
        "ollama": ollama,
        "openai": openai,
        "instances": {**ollama, **openai},
        "targets": sorted(TARGETS.keys()),
        "target_info": {n: {"type": target_type(n), "url": t["url"]} for n, t in TARGETS.items()},
        # Loki-Labels nur für ollama-Typen (openai-Typen liefern ihre t/s numerisch, ohne Loki)
        "loki_labels": sorted({n for n in TARGETS if target_type(n) == "ollama"}),
        "nap": get_nap() if NAP_URL else None,
        "power_limit_w": POWER_LIMIT_W,
        # Universell: GPU-Modellname + Hostname aus dem DCGM-Exporter (keine hartkodierten Strings);
        # LM_SERVER_NAME überschreibt die Bezeichnung (DCGM-Hostname ist oft nur eine Container-ID)
        "gpu_name": _dcg_cache.get("gpu_name"),
        "hostname": _dcg_cache.get("hostname"),
        "server_name": SERVER_NAME or None,
        "system": _sys_cache["data"] if _sys_cache["data"] and time.time() - _sys_cache["ts"] < 15 else None,
    }


# ── Loki (Ollama-Docker-Logs) ─────────────────────────────────────────────

# ── Synthetische Aktivitätszeilen ───────────────────────────────────────────
# Targets ohne Log-Stream (kein Loki) erzeugen Zeilen aus dem Scrape: nur wenn sich
# etwas ändert, damit der Log nicht zuspamt. Gleiches Format wie get_logs → das
# Frontend kann beide Quellen im selben Panel zeigen.
_ACTIVITY = {}
_PREV = {}
_ACT_LOCK = threading.Lock()
ACTIVITY_MAX = 100


def _activity(name, info):
    with _ACT_LOCK:
        deq = _ACTIVITY.setdefault(name, deque(maxlen=ACTIVITY_MAX))
        prev = _PREV.get(name)
        snap = {k: info.get(k) for k in ("state", "running", "waiting", "online", "error")}
        snap["draft"] = (info.get("mtp") or {}).get("draft_tokens")
        if prev is None:
            _PREV[name] = snap
            return
        now = time.time()

        def emit(line, kind="info"):
            deq.append({"ts": now, "line": line, "kind": kind})

        if snap["error"] and snap["error"] != prev.get("error"):
            emit(f"ERROR: {snap['error']}", "error")
        elif snap["online"] and not prev.get("online"):
            emit(f"{name}: online ({info.get('model') or '?'})", "ok")
        elif prev.get("online") and not snap["online"]:
            emit(f"{name}: offline", "warn")
        if snap["state"] != prev.get("state"):
            emit(f"state: {prev.get('state') or '?'} → {snap['state'] or '?'} "
                 f"(running {snap['running']}, waiting {snap['waiting']})",
                 "ok" if snap["state"] == "generating" else "info")
        elif snap["running"] != prev.get("running") or snap["waiting"] != prev.get("waiting"):
            emit(f"queue: running {snap['running']}, waiting {snap['waiting']}", "info")
        m = info.get("mtp") or {}
        d = m.get("draft_tokens") or 0
        if prev.get("draft") is not None and d > prev.get("draft"):
            emit(f"req fertig: decode {info.get('last_decode_tok_s')} tk/s · "
                 f"MTP acc {m.get('acc_rate')} · cache {info.get('cache_hit_pct')} %", "ok")
        _PREV[name] = snap

def get_activity(name, limit=80):
    with _ACT_LOCK:
        entries = list(_ACTIVITY.get(name, []))[-limit:]
    return {"gpu": name, "entries": entries}


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


# ── Nap (napd Control-API) ──────────────────────────────────────────────────
# Optional: LM_NAP_URL + LM_NAP_TOKEN. Ohne diese beiden Variablen bleibt die Sektion
# im Frontend ausgeblendet → nichts ist an einen bestimmten Host oder Daemon gebunden.
NAP_URL = os.environ.get("LM_NAP_URL", "")
NAP_TOKEN = os.environ.get("LM_NAP_TOKEN", "")


NAP_TTL = 5
_NAP_CACHE = {"ts": 0, "services": None, "host": {}}
_NAP_SEEN = {}   # name -> {first_seen, running_since, idle_since} (ISO)


def nap_get(path):
    if not NAP_URL:
        return None
    req = urllib.request.Request(NAP_URL.rstrip("/") + path, headers=auth_headers(NAP_TOKEN, "application/json"))
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)


def get_nap():
    """Services + Host-Ressourcen der Nap-Control-API (read-only)."""
    try:
        # Cache: Nap-API wird nur alle 5 s abgeholt, damit /api/metrics nicht jeden Poll blockiert
        now = time.time()
        cached = _NAP_CACHE
        if now - cached.get("ts", 0) < NAP_TTL and cached.get("services") is not None:
            return cached
        services = nap_get("/v1/services") or []
        host = nap_get("/v1/host") or {}
        # Idle-Timer: napd liefert keinen Event-Endpunkt → wir führen die Seen-Zeit pro Service
        # selbst nach. running_since ist der erste Poll, seit dem der Service läuft; idle_since der
        # erste Poll, seit dem er nicht mehr lief. Beides nur in diesem Prozess (kein Nap-Persist).
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for s in services:
            name = s.get("name")
            seen = _NAP_SEEN.setdefault(name, {"first_seen": now_iso, "running_since": None, "idle_since": None})
            if s.get("running"):
                if seen["running_since"] is None:
                    seen["running_since"] = now_iso
                seen["idle_since"] = None
            else:
                if seen["idle_since"] is None and seen["first_seen"] != now_iso:
                    seen["idle_since"] = now_iso
                seen["running_since"] = None
            s["first_seen"] = seen["first_seen"]
            s["running_since"] = seen["running_since"]
            s["idle_since"] = seen["idle_since"]
        _NAP_CACHE.update({"ts": now, "services": services, "host": host, "url": NAP_URL})
        return _NAP_CACHE
    except Exception as e:
        # Fehler nicht cachen → nächster Poll versucht erneut, aber nur, wenn noch nichts bekannt ist
        if _NAP_CACHE.get("services") is not None:
            _NAP_CACHE["error"] = str(e)
            return _NAP_CACHE
        return {"error": str(e), "services": [], "host": {}}


# ── Live t/s aus den Ollama-Logzeilen (llama.cpp print_timing) ─────────────
_TG_RE = re.compile(r"n_gen\s*=\s*(\d+),\s*tg\s*=\s*([\d.]+)\s*t/s,\s*tg_3s\s*=\s*([\d.]+)\s*t/s")
_PROMPT_RE = re.compile(r"prompt eval time\s*=\s*[\d.\s]+ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second")
# Aggregierte Eval-Zeile (kleine Batches, z. B. 1b-Modelle auf gpu0/gpu1 — dort fehlt die n_gen-Zeile):
# "eval time =     246.19 ms /    67 tokens" → tps = tokens / ms
_EVAL_RE = re.compile(r"(?<!prompt )(?<!total )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")
# MTP/speculative Decoding: "#mean acc len = 1.87, #acc rate/pos = (0.867)" (wird am Ende einer Generation geloggt)
_MTP_RE = re.compile(r"#mean acc len = ([\d.]+), #acc rate/pos = \(([\d.]+)\)")


def get_live_tps(gpu, window=900):
    """Kontinuierliche t/s-Punkte aus den Live-Logs (letzte `window` Sekunden).

    Punkte:
      kind=gen    → tg (Ø t/s der laufenden Generation), tg3s (Rolling 3s)
                   Fallback: aggregierte "eval time"-Zeile → tps = tokens/ms
      kind=prompt → tps des Prompt-Evals (eigener Punkt, wird nicht gemischt)
    """
    if gpu in TARGETS and target_type(gpu) == "openai":
        # Sparkline aus den Live-Gauges des Scrapes (kein Log-Parsing nötig) — Punkte aus dem
        # Puffer des Refreshers, gefiltert auf den angefragten Window.
        inst = get_openai_cached().get(gpu) or {}
        pts = [p for p in _openai_tps.get(gpu, []) if p["t"] >= time.time() - window]
        return {"gpu": gpu, "points": pts, "count": len(pts), "active": bool(inst.get("running")),
                "mtp": inst.get("mtp"),
                # Fallback für die "Jetzt"-Anzeige: live ist nur während eines laufenden Requests
                # gesetzt, sonst steht die Rate des letzten Laufs da.
                "last_decode_tok_s": inst.get("last_decode_tok_s"),
                "last_prefill_tok_s": inst.get("live_prefill_tok_s")}
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
                    continue
                ev = _EVAL_RE.search(line)
                if ev:
                    ms, n_tok = float(ev.group(1)), int(ev.group(2))
                    if ms > 0 and n_tok > 0:
                        pts.append({"t": int(ts_ns) // 1_000_000_000, "kind": "gen",
                                    "tg": round(n_tok / (ms / 1e3), 1), "tg3s": round(n_tok / (ms / 1e3), 1)})
                    continue
                mt = _MTP_RE.search(line)
                if mt:
                    pts.append({"t": int(ts_ns) // 1_000_000_000, "kind": "mtp",
                                "mean_acc_len": float(mt.group(1)), "acc_rate": float(mt.group(2))})
        pts.sort(key=lambda x: x["t"])
        now = int(time.time())
        last_gen = next((p for p in reversed(pts) if p["kind"] == "gen"), None)
        last_prompt = next((p for p in reversed(pts) if p["kind"] == "prompt"), None)
        last_mtp = next((p for p in reversed(pts) if p["kind"] == "mtp"), None)
        active = bool(last_gen and now - last_gen["t"] < 90)
        if not active and last_prompt and now - last_prompt["t"] < 90:
            active = True   # nur Prompt-Phase (Cold-Start) zählt auch als aktiv
        return {
            "gpu": gpu,
            "window_s": window,
            "count": len(pts),
            "points": pts,
            "active": active,
            "current_tps": round(last_gen["tg"], 1) if last_gen else None,
            "mtp": {"acc_rate": last_mtp["acc_rate"], "mean_acc_len": last_mtp["mean_acc_len"], "t": last_mtp["t"]} if last_mtp else None,
        }
    except Exception as e:
        return {"gpu": gpu, "error": str(e), "points": [], "count": 0}


# ── Ollama `ollama show` per HTTP-Äquivalent (/api/show) ──────────────────
def _mi_tail(model_info, tail):
    """Ersten Wert eines model_info-Keys mit Suffix (arch-Prefix variiert)."""
    for k, v in (model_info or {}).items():
        if k.endswith("." + tail):
            return v
    return None


def ollama_show(base, name, key=""):
    req = urllib.request.Request(base + "/api/show", data=json.dumps({"name": name}).encode(),
                                 headers={**auth_headers(key), "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)


def get_model_info(gpu):
    """Modell-Details eines Containers (dual/gpu0/gpu1) — das JSON-Äquivalent
    von `docker exec ollama-{gpu} ollama show <model>`."""
    if gpu in TARGETS and target_type(gpu) == "openai":
        inst = get_openai_cached().get(gpu) or {}
        vram_gb = round(sum(c.get("vram_used_mib", 0) for c in (inst.get("gpus") or {}).values()) / 1024, 1)
        return {"gpu": gpu, "online": inst.get("online", False), "type": "openai",
                "models": [{"name": inst.get("model"), "effective_ctx": inst.get("max_context"),
                            "size_vram_gb": vram_gb,
                            "spec": {"native_ctx": inst.get("max_context")},
                            "cache_hit_pct": inst.get("cache_hit_pct"),
                            "decode_tok_s": inst.get("last_decode_tok_s")}]}
    key = TARGETS.get(gpu, {}).get("key", "")
    base = TARGETS.get(gpu, {}).get("url") or OLLAMA_EP.get(gpu)
    out = {"gpu": gpu, "online": False, "models": []}
    if not base:
        return out
    try:
        with urllib.request.urlopen(base + "/api/version", timeout=TIMEOUT) as r:
            json.load(r)
        out["online"] = True
        _online_cache[gpu] = time.time()
    except Exception as e:
        if time.time() - _online_cache.get(gpu, 0) < ONLINE_TTL:
            out["online"] = True   # letzter guter Stand (unter Last teils Timeout)
        else:
            out["error"] = str(e)
            return out
    models_raw, _sysinfo, _stale = _ps_models_fresh(base, gpu, key)
    for m in models_raw:  # letztes Array-Element = zuletzt geladen
        name = m.get("name")
        rec = {
            "name": name,
            "effective_ctx": m.get("context_length"),
            "size_vram_gb": round(m.get("size_vram", 0) / 1e9, 1),
            "processor": m.get("processor"),
            # Offload-Rate: Anteil im VRAM (100 = voller GPU-Offload, kleiner = CPU-Spill-over)
            "offload_pct": round(100 * m.get("size_vram", 0) / m["size"], 1) if m.get("size") else None,
        }
        try:
            sh = ollama_show(base, name, key)
        except Exception as e:
            rec["spec_error"] = str(e)
            out["models"].append(rec)
            continue
        mi = sh.get("model_info") or {}
        pi = sh.get("projector_info") or {}
        det = sh.get("details") or {}
        par_raw = sh.get("parameters") or {}
        params = {}                      # `ollama show` liefert parameters als alignierten Text → parsen
        if isinstance(par_raw, dict):
            params = par_raw
        elif isinstance(par_raw, str):
            for ln in par_raw.splitlines():
                m = re.match(r"^(\S+)\s{2,}(\S.*?)\s*$", ln)
                if m:
                    try: params[m.group(1)] = float(m.group(2))
                    except ValueError: params[m.group(1)] = int(m.group(2))
        caps = sh.get("capabilities") or []
        arch = _mi_tail(mi, "general.architecture") or det.get("family")
        spec = {
            "architecture": arch,
            "param_count": _mi_tail(mi, "general.parameter_count"),
            "param_label": det.get("parameter_size"),
            "quantization": det.get("quantization_level"),
            "native_ctx": _mi_tail(mi, "context_length"),
            "blocks": _mi_tail(mi, "block_count"),
            "embedding": _mi_tail(mi, "embedding_length"),
            "attn_heads": _mi_tail(mi, "attention.head_count"),
            "kv_heads": _mi_tail(mi, "attention.head_count_kv"),
            "head_size": _mi_tail(mi, "attention.key_length"),
            "full_attn_every": _mi_tail(mi, "full_attention_interval"),
            "rope_dim": _mi_tail(mi, "rope.dimension_count"),
            "rope_base": _mi_tail(mi, "rope.freq_base"),
            "ssm_state": _mi_tail(mi, "ssm.state_size"),
            "ssm_groups": _mi_tail(mi, "ssm.group_count"),
            "mtp_layers": _mi_tail(mi, "nextn_predict_layers"),
        }
        vision = None
        if _mi_tail(pi, "vision.has_vision_encoder") or any(k.startswith("clip.") for k in pi):
            vision = {
                "projector": next((v for k, v in pi.items() if k.endswith("projector_type")), None),
                "image_size": next((v for k, v in pi.items() if k.endswith("vision.image_size")), None),
                "blocks": next((v for k, v in pi.items() if k.endswith("vision.block_count")), None),
            }
        sample = {k: params[k] for k in ("temperature", "top_k", "top_p", "min_p", "repeat_penalty") if k in params}
        rec["spec"] = spec
        rec["vision"] = vision
        rec["capabilities"] = caps
        rec["sampling"] = sample
        rec["modelfile_ctx"] = params.get("num_ctx")
        rec["draft_num_predict"] = params.get("draft_num_predict")
        out["models"].append(rec)
    return out


# ── Inference Benchmark (echte tk/s) ──────────────────────────────────────
def run_benchmark(target, prompt=None, max_tokens=None):
    if target in TARGETS and target_type(target) == "openai":
        t = TARGETS[target]
        inst = get_openai_cached().get(target) or {}
        model = inst.get("model") or next((x.get("name") for x in (inst.get("models") or [])), None)
        try:
            before = scrape_metrics(t["url"], t["key"])
            # Prompt/Output: Body > Env > Default. Längerer Lauf → mehrere t/s-Punkte im Dashboard.
            bprompt = prompt or os.environ.get("LM_BENCH_PROMPT", "Count from 1 to 20.")
            bmax = int(max_tokens or os.environ.get("LM_BENCH_MAX_TOKENS", "24"))
            payload = json.dumps({"model": model or "model", "max_tokens": bmax, "stream": False,
                                  "messages": [{"role": "user", "content": bprompt}]}).encode()
            req = urllib.request.Request(t["url"] + "/v1/chat/completions", data=payload,
                                         headers={**auth_headers(t["key"]), "Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.load(r)
            wall = round(time.time() - t0, 2)
            after = scrape_metrics(t["url"], t["key"])
            # Der Echo der API ist oft nur der übergebene Alias → für Labels die Modellname
            # aus dem Scrape nehmen (der ist die Wahrheit des Servers).
            mname = next((l.get("model_name") for l, _ in after.get("vllm:num_requests_running", [])), None) or d.get("model") or model
            def g(m, metric): return pick(m.get(metric, []), "model_name", mname)
            pt, gt = (d.get("usage") or {}).get("prompt_tokens", 0), (d.get("usage") or {}).get("completion_tokens", 0)
            # Prefill/Decode-Zerlegung: die API liefert keine Zeiten → Delta der Cumulative-Counter.
            dp = g(after, "strata:totals_prompt_seconds_total")
            dd = g(after, "strata:totals_decode_seconds_total")
            if dp is not None and g(before, "strata:totals_prompt_seconds_total") is not None:
                dp -= g(before, "strata:totals_prompt_seconds_total")
            if dd is not None and g(before, "strata:totals_decode_seconds_total") is not None:
                dd -= g(before, "strata:totals_decode_seconds_total")
            # vLLM/vLLM-kompatible Server zählen Prompt- und Decode-Zeit getrennt;
            # ein generischer OpenAI-Server nicht → dort bleibt nur wall time.
            prompt_tps = round(pt / dp, 1) if dp and dp > 0.01 else (round(pt / max(1e-6, wall), 1) if pt else None)
            gen_tps = round(gt / dd, 1) if dd and dd > 0.01 else (round(gt / max(1e-6, wall), 1) if gt else None)
            _push_history(gen_tps or 0, prompt_tps or 0)
            return {"target": target, "model": mname, "wall_s": wall, "prompt_tokens": pt,
                    "prompt_tps": prompt_tps, "gen_tokens": gt, "gen_tps": gen_tps}
        except Exception as e:
            return {"error": str(e)}
    key = TARGETS.get(target, {}).get("key", "")
    base = TARGETS.get(target, {}).get("url") or OLLAMA_EP.get(target)
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
                                  headers={**auth_headers(key), "Content-Type": "application/json"})
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
            gpu = resolve_target((q.get("gpu", [""])[0] or ""))
            limit_s = q.get("limit", ["80"])[0]
            try:
                limit = max(1, min(int(limit_s), 300))
            except ValueError:
                limit = 80
            if target_type(gpu) == "ollama":
                return self._send(200, get_logs(gpu, limit))
            return self._send(200, get_activity(gpu, limit))
        if path == "/api/live_tps":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            gpu = resolve_target((q.get("gpu", [""])[0] or ""))
            win_s = q.get("window", ["900"])[0]
            try:
                window = max(60, min(int(win_s), 3600))
            except ValueError:
                window = 900
            return self._send(200, get_live_tps(gpu, window))
        if path == "/api/model_info":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            gpu = resolve_target((q.get("gpu", [""])[0] or ""))
            return self._send(200, get_model_info(gpu))
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
        target = resolve_target(body.get("target", ""))
        if target not in TARGETS:
            return self._send(400, {"error": f"unknown target {target}"})
        result = run_benchmark(target, body.get("prompt"), body.get("max_tokens"))
        code = 200 if "error" not in result else 502
        return self._send(code, result)

    def log_message(self, fmt, *args):
        print("[llmetrix]", fmt % args, flush=True)


_load_history()
if __name__ == "__main__":
    port = int(os.environ.get("LM_PORT", "8790"))
    print(f"[llmetrix] starting on :{port} (prometheus={PROM_URL})", flush=True)
    threading.Thread(target=_ollama_refresher, daemon=True).start()
    threading.Thread(target=_dcgm_refresher, daemon=True).start()
    threading.Thread(target=_node_refresher, daemon=True).start()
    if any(target_type(n) == "openai" for n in TARGETS):
        threading.Thread(target=_openai_refresher, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
