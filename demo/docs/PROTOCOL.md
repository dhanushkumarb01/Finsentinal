# Device ↔ Server protocol (contract for coordinator, virtual devices and ESP32 firmware)

All numbers here are normative. Reference implementation for devices: `demo/fldemo/wire.py`
(+ `demo/devices/agent.py`). Test vectors: `demo/tests/vectors/finsentinel_vectors.json`.

## 1. Topology and ports (laptop)

| Component | Listen | Notes |
|---|---|---|
| FinSentinel (unmodified) | `127.0.0.1:8000` | own SQLite at `demo/.state/finsentinel_demo.db`; its UI (Vite, :5173) proxies to :8000 |
| Coordinator (FastAPI) | `127.0.0.1:8200` | public device API + local-only admin/event API |
| Caddy TLS front door | `https://127.0.0.1:8443` (cert for `localhost`, `127.0.0.1` from the demo CA) | routes below |
| Caddy plain-HTTP tunnel origin | `http://127.0.0.1:8480` | same routes; the public tunnel terminates TLS and forwards here |
| Toxiproxy | API `127.0.0.1:8474`; device links `127.0.0.1:9101…9106` → `127.0.0.1:8443` | virtual device *i* uses port `9100+i`; TLS passes through |
| Mission Control (static) | `http://127.0.0.1:5180` | served by `run_demo.py`; also serves `/config.json`, `/recordings/*`, `/femnist.json` |

Caddy exposes ONLY: `POST /api/v1/updates`, `GET /api/v1/health` → FinSentinel; and
`GET /fl/round`, `GET /fl/model/*`, `GET /fl/pubkey`, `POST /fl/status`, `POST /fl/report` →
coordinator. Everything else → 404 (operator/admin APIs are never reachable from outside).

## 2. Keys

* Each device: Ed25519 keypair. `client_id` = device id (`phone-01`…`phone-06`, `esp32-01`…`esp32-03`),
  `signing_key_id` = `<client_id>-key`. Public keys registered in FinSentinel's trusted-key table
  with `max_sample_count = 2000` at server start. Private keys in `demo/.state/keys/` (gitignored);
  for boards also emitted into `demo/wokwi/boardN/board_secrets.h` (gitignored).
* Coordinator: Ed25519 keypair. Public key is **pinned** in every device (32 raw bytes, hex in
  config). Used to sign (a) every global model, (b) the tunnel-address beacon.

## 3. Round flow (every device, every round)

1. **Poll** `GET /fl/round` (every 2 s virtual / 5 s board; reuse the TLS connection):
   ```json
   {"round": 3, "state": "open", "model_version": "har-mlp-v1",
    "model_sha256": "<hex of the 9112 model bytes>", "model_sig": "<base64 Ed25519>",
    "deadline": 1795356400.0, "server_time": 1795356340.2,
    "epochs": 1, "lr": 0.05, "batch_size": 32}
   ```
   `state` ∈ `waiting | open | aggregating | finished`. Devices act only when `state == "open"` and
   `round` is new for them. `model_sig` = Ed25519(coordinator key) over the UTF-8 string
   `fl-model|har-mlp-v1|<round>|<model_sha256>`.
2. **Download** `GET /fl/model/<round>` → `application/octet-stream`, exactly 9112 bytes:
   float32 little-endian, tensors in order `fc1.weight (32×64, row-major)`, `fc1.bias (32)`,
   `fc2.weight (6×32)`, `fc2.bias (6)`. Device computes SHA-256 of the bytes, compares with
   `model_sha256`, verifies `model_sig` with the pinned coordinator key. **Any mismatch → refuse
   the model, report `state="model_rejected"`, skip the round.**
3. **Decide to participate** (dropout causes, all reported honestly via `/fl/status`):
   offline switch/button → `offline`; battery < 15 % → `skipped/low_battery`; model download
   failed or exceeded the time budget → `skipped/weak_signal`; drift-causal rule (research
   mechanism) → `skipped/drift`.
4. **Train** `epochs` passes of mini-batch SGD (no momentum, lr, batch size from step 1), cross-entropy,
   over the device's own subject data (features = the 64-feature subset in
   `demo/config/har_features.json` order). Board shuffles with its own PRNG (order need not match).
5. **Payload**: NPZ with STORED entries `tensor_000000.npy … tensor_000003.npy`, `__manifest__.npy`
   exactly as `fldemo.wire.build_npz_stored` (byte-identical for identical weights).
   Hacked device (slide switch / attack): weights = trained weights × 8 (poisoning).
6. **Envelope** (sign as `fldemo.wire.make_envelope`): `update_id = <client_id>-r<round:04d>-<8 hex random>`,
   `created_at = YYYY-MM-DDTHH:MM:SSZ` (UTC from NTP), `sample_count` = number of local samples.
   Canonical signed bytes = `json.dumps(envelope_without_signature, sort_keys=True,
   separators=(",", ":"))`. Key order: `client_id, created_at, model_version, payload_sha256,
   payload_size_bytes, round_number, sample_count, signing_key_id, update_id`.
7. **Submit** `POST /api/v1/updates` body `{"envelope": {...}, "payload_base64": "..."}` → FinSentinel.
8. **Report** `POST /fl/report` (always, also on FinSentinel rejection):
   ```json
   {"client_id": "esp32-01", "round": 3,
    "submission": {"envelope": {...}, "payload_base64": "..."},
    "finsentinel": {"http_status": 200, "body": {...FinSentinel JSON or {"text": "..."}...}},
    "train": {"epochs": 1, "losses": [0.91], "samples": 341, "class_counts": [58,59,49,52,61,62],
              "seconds": 4.8},
    "battery": 72, "signal_dbm": -61, "hacked": false}
   ```
   The coordinator never trusts this for security: it re-checks the payload hash against the
   envelope, reads the real decision from FinSentinel's operator API, and aggregation is
   approval-gated by FinSentinel on the exact payload bytes.

## 4. Telemetry `POST /fl/status` (UI only; never used for security decisions)

```json
{"client_id": "phone-02", "state": "training", "round": 3, "battery": 64, "signal_dbm": -70,
 "network": "4g", "online": true, "hacked": false, "reason": null, "detail": "epoch 1/1"}
```
`state` ∈ `online | idle | downloading | model_rejected | training | signing | sending | done |
skipped | offline | error`. `reason` (when skipped) ∈ `low_battery | weak_signal | drift | offline
| model_rejected | tls_error | other`.

## 5. Tunnel-address beacon (how Wokwi boards find the laptop)

The free tunnel address changes often. Whenever it changes, the coordinator publishes to the ntfy
topic named in `demo/.state/beacon.json` (random, secret; also in `board_secrets.h`):
`POST https://ntfy.sh/<topic>` with plain-text body

```
FLB1|<https url without trailing slash>|<unix seconds>|<base64 Ed25519 signature>
```
signature = coordinator key over the UTF-8 string `FLB1|<url>|<unix seconds>`.
Board: `GET https://ntfy.sh/<topic>/raw?poll=1&since=2h` → one message per line → take the **last**
line that verifies with the pinned coordinator key and whose timestamp is newest. Re-read the
beacon whenever a request to the current URL fails twice. A forged beacon cannot redirect a
board (signature), and model downloads are separately signed.

## 6. TLS

* Virtual devices: connect to `https://localhost:<9100+i>`; trust ONLY the demo CA
  (`demo/.state/tls/ca.crt`). Any other certificate → handshake refused (`tls_error`).
* Boards: `WiFiClientSecure::setCACert()` with the public-root bundle in
  `demo/wokwi/common/ca_roots.h` (generated from certifi: ISRG Root X1/X2, GTS R1–R4,
  DigiCert Global Root G2, USERTrust RSA/ECC, Baltimore). No `setInsecure()` in the real firmware.
