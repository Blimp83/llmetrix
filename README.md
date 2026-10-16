# LLMetrix // Neural Matrix

Cyberpunk-Dashboard für Ollama-Cluster mit NVIDIA-GPUs: Live-Telemetrie (GPU-Load, VRAM, Temp, Power), Modell-Status, GPU-Offload-Rate, Token-Raten (t/s), MTP-Acceptance, System-Metriken (CPU/RAM) und echte Benchmarks — in einer Single-File-App ohne Dependencies.

```
┌─────────────────────────────────────────────────────────────┐
│  NEURAL ORB // INFERENCE CORE        (Token-Streams + Nadeln │
│                                          ∝ GPU-Aktivität)    │
├──────────────┬──────────────────────────────┬───────────────┤
│  GPU 0       │  CORE // ACTIVE MODEL        │   GPU 1       │
│  LOAD/VRAM   │  Modell · t/s-Sparklines     │   LOAD/VRAM   │
│  TEMP/POWER  │  MODEL SPEC                  │   TEMP/POWER  │
│  (Icons)     │  VRAM FOOTPRINT + OFFLOAD    │   (Icons)     │
│              │  AI-SERVER SYSTEM (CPU/RAM)  │               │
└──────────────┴──────────────────────────────┴───────────────┘
```

## Features

- **Neural Orb**: rotierende Aktivitäts-Nadeln + Token-Bits (0/1) aus einer Binär-Matrix, die in den Kern fliegen — links GPU0 (cyan), rechts GPU1 (emerald), DUAL = teal. Aktivität skaliert mit t/s/GPU-Load.
- **GPU-Kacheln** (je GPU): LOAD (Tacho), VRAM (3D-Quader mit Füllstand), TEMP (Thermometer), POWER (Energiezelle) — Werte live, Warn-Farbe bei >92% bzw. ≥85°C.
- **Core-Panel**: aktives Modell, LIVE t/s-Sparklines (Generation + Prompt-Eval), MODEL SPEC (Arch/DNA via `ollama show`), VRAM-FOOTPRINT, GPU-OFFLOAD-Rate (Donut, z. B. `66/66 LAYERS · VOLL GPU`), AI-SERVER SYSTEM (CPU/RAM).
- **Benchmarks**: echte `/api/generate`-Läufe gegen jede Ollama-Instanz (`GPU 0`, `GPU 1`, `DUAL 0+1`) mit Prompt- + Generation-t/s.
- **Live-Logs** der Ollama-Container (via Loki), MTP-Acceptance-Rate aus den Logzeilen.
- Mobile-first, Drag&Drop-Layout (pro Modus persistiert), kein Build-Schritt.

## Architektur

```
┌────────────┐  scrape   ┌──────────────────────────┐
│ DCGM-      ├──────────►│                          │
│ Exporter   │           │  LLMetrix (python:3.13)  │──► :8790 (Dashboard)
│ :9400      │           │                          │
├────────────┤  scrape   │  server.py (Stdlib nur) │
│ node-      ├──────────►│  web/index.html (Single) │
│ exporter   │           └───────────┬──────────────┘
│ :9100      │                       │ HTTP (urllib)
├────────────┤                       ▼
│ Loki       │            ┌──────────────────────┐
│ :3100      │◄── Promtail│  Ollama-Instanzen    │
└────────────┘             │  gpu0 / gpu1 / dual │
                           └──────────────────────┘
```

Das Backend ist **reine Python-Stdlib** (kein pip, kein venv). Das Frontend ist eine **einzige HTML-Datei** (Tailwind via CDN).

## Voraussetzungen

| Komponente | Pflicht? | Zweck |
|---|---|---|
| Docker | ja | läuft als Container (`python:3.13-slim`) |
| NVIDIA GPU(s) + Ollama | ja | die Instanzen, deren Status angezeigt wird |
| [DCGM-Exporter](https://github.com/NVIDIA/dcgm-exporter) | ja (empfohlen) | GPU-Load/VRAM/Temp/Power **live** (~1 s) + GPU-Modellname |
| [node-exporter](https://github.com/prometheus/node_exporter) | optional | CPU/RAM des AI-Servers |
| Loki + Promtail | optional | Live-t/s, MTP, Log-Panel, "ERZEUGUNG AKTIV" |
| Prometheus | optional | Fallback für GPU-Metriken (5–15 s Latenz), nur wenn DCGM-Exporter kurz weg ist |

**Ohne Loki** fehlen: Log-Panel, t/s-Sparklines, MTP-Badge, "LIVE"-Badge. Der Orb animiert dann nur über den GPU-Load (DCGM). Alle Status-/Metriken-Kacheln laufen trotzdem.

## Setup

```bash
git clone <repo-url> llmetrix && cd llmetrix

# 1. URLs in docker-compose.yml setzen (oder docker-compose.override.yml anlegen)
nano docker-compose.yml

# 2. Starten
docker compose up -d

# Dashboard: http://<host>:8790
```

### Env-Vars

| Var | Default | Beschreibung |
|---|---|---|
| `LM_OLLAMA_GPU0` | `http://127.0.0.1:11434` | Ollama-Instanz "GPU 0" (URL) |
| `LM_OLLAMA_GPU1` | `http://127.0.0.1:11435` | Ollama-Instanz "GPU 1" (URL) |
| `LM_OLLAMA_DUAL` | `http://127.0.0.1:11436` | Ollama-Instanz "DUAL" (Tensor-Split, URL) |
| `LM_DCGM_URL` | `http://127.0.0.1:9400` | DCGM-Exporter (GPU-Metriken, primär) |
| `LMPROM_URL` | `http://127.0.0.1:9090` | Prometheus (Fallback-Queries) |
| `LM_NODE_URL` | `http://127.0.0.1:9100` | node-exporter (CPU/RAM) |
| `LM_LOKI_URL` | `http://127.0.0.1:3100` | Loki (Logs + Live-t/s) |
| `LM_POWER_LIMIT_W` | `180` | Max-Power der GPU in W (Power-Anzeiger) |
| `LM_SERVER_NAME` | — | Freundliche Server-Bezeichnung im Header |
| `LM_TIMEOUT` | `8` | HTTP-Timeout (s) für Ollama-Calls |
| `LM_PORT` | `8790` | Dashboard-Port |
| `LM_HIST_FILE` | `/data/history.json` | Persistenz der Benchmark-History |

### Loki-Labeling (falls genutzt)

Promtail muss die Ollama-Container-Logs mit `job="ollama"` und `gpu="gpu0"|"gpu1"|"dual"` labeln, z. B.:

```yaml
scrape_configs:
  - job_name: ollama
    static_configs:
      - targets: [localhost]
        labels:
          job: ollama
          gpu: dual
          __path__: /var/lib/docker/containers/*ollama-dual*json
```

## API

| Endpoint | Beschreibung |
|---|---|
| `GET /api/metrics` | GPU-Metriken + Ollama-Status + System (1 s Poll) |
| `GET /api/live_tps?gpu=dual` | Live t/s-Punkte aus den Logs (Window 60–3600 s) |
| `GET /api/logs?gpu=dual` | Neueste Logzeilen |
| `GET /api/model_info?gpu=dual` | Modell-Spec (`ollama show`-Äquivalent) |
| `GET /api/history` | Benchmark-History (Sparkline) |
| `POST /api/inference` | `{target: gpu0\|gpu1\|dual}` → echtes Benchmark |

## Hinweise

- **GPU-Mapping**: Das Dashboard zeigt GPUs physisch (DCGM `gpu="0"/"1"`), Ollama-Instanzen logisch (`gpu0`/`gpu1`/`dual`). Bei dynamischer GPU-Zuweisung (z. B. Nap) können sich die beiden voneinander unterscheiden — bewusste Design-Entscheidung, da die Service-Ebene (Modelle, t/s, Logs) korrekt zugeordnet ist.
- **Caching**: Ollama-Status wird alle 3 s im Hintergrund aktualisiert (unter Inferenzlast antworten Ollama-Calls langsam), DCGM 1 s, node-exporter 2 s. Letzter guter Stand wird vorgehalten, damit das Dashboard nicht flackert.
- **Anzahl GPUs**: Die UI ist auf die Topologie `GPU0 + GPU1 + DUAL` ausgelegt (3 Ollama-Instanzen). Andere Topologien laufen mit angepasster Env-Konfiguration, die UI-Anpassung wäre ein Feature.

## Lizenz

MIT — siehe [LICENSE](LICENSE).
