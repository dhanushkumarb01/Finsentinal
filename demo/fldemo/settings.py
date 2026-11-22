"""Ports, file locations and run parameters shared by every demo component."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from fldemo.paths import CONFIG_DIR, DEMO_DIR, RECORDINGS_DIR, STATE_DIR, TOOLS_DIR

FS_PORT = 8000            # FinSentinel (its own UI proxies to :8000)
COORD_PORT = 8200         # coordinator
CADDY_TLS_PORT = 8443     # TLS front door (virtual devices via Toxiproxy)
CADDY_HTTP_PORT = 8480    # plain-HTTP origin for the public tunnel
TOXI_API_PORT = 8474
TOXI_BASE_PORT = 9100     # virtual device i -> 9100 + i
MC_PORT = 5180            # Mission Control static site
FS_UI_PORT = 5173
FORWARDER_PORT = 9999     # FinSentinel's SIEM forwarder (only with --wazuh)

KEYS_DIR = STATE_DIR / "keys"
TLS_DIR = STATE_DIR / "tls"
LOG_DIR = STATE_DIR / "logs"
FS_DB = STATE_DIR / "finsentinel_demo.db"
SECRETS_FILE = STATE_DIR / "secrets.json"     # operator/admin/webhook tokens
REGISTRY_FILE = STATE_DIR / "registry.json"   # device public keys for FinSentinel
BEACON_FILE = STATE_DIR / "beacon.json"
TUNNEL_URL_FILE = STATE_DIR / "tunnel_url.txt"
RUNTIME_FILE = STATE_DIR / "runtime.json"     # what run_demo started (urls, run id)

DEVICES_FILE = CONFIG_DIR / "devices.json"
THRESHOLDS_FILE = CONFIG_DIR / "thresholds.json"
MC_DIST = DEMO_DIR / "mission-control" / "dist"

CA_CERT = TLS_DIR / "ca.crt"
SERVER_CERT = TLS_DIR / "server.crt"
SERVER_KEY = TLS_DIR / "server.key"

EXE = {
    "caddy": TOOLS_DIR / "caddy.exe",
    "toxiproxy": TOOLS_DIR / "toxiproxy-server.exe",
    "cloudflared": TOOLS_DIR / "cloudflared.exe",
}

for _d in (STATE_DIR, KEYS_DIR, TLS_DIR, LOG_DIR, RECORDINGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainConfig:
    epochs: int = 1
    lr: float = 0.05
    batch_size: int = 32


@dataclass
class RunConfig:
    rounds: int = 30
    round_seconds: float = 25.0          # how long a round stays open for submissions
    min_round_seconds: float = 6.0       # close early once every online device reported, but not before this
    grace_seconds: float = 3.0           # after close: wait for approvals before aggregating
    pause_seconds: float = 2.0           # pause between rounds (lets the audience follow)
    wokwi: bool = False                  # boards expected -> longer rounds
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 0


def load_devices() -> list[dict]:
    return json.loads(DEVICES_FILE.read_text())["devices"]


def load_secrets() -> dict:
    return json.loads(SECRETS_FILE.read_text())
