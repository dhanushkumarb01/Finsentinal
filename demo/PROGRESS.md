# PROGRESS — Live FL Demo ("Mission Control")

*Single source of truth for the demo build. Read this first; update it every session.*

**Branch:** `demo/fl-live` · **Remote:** `finsentinal` (https://github.com/dhanushkumarb01/Finsentinal.git)
**Build instructions:** `demo/BUILD_PROMPT.md` · **How to run:** `demo/RUNBOOK.md`

| # | Phase | Status | Evidence |
|---|---|---|---|
| 0 | Verify & de-risk | 🟡 mostly done — Wokwi route + ESP32 speed pending (tunnel blocked on this network, see below) | `demo/tests` 14/14 pass; measurements below |
| 1 | Core engine | ⏳ | — |
| 2 | Device realism | ⏳ | — |
| 3 | Wokwi boards | ⏳ | — |
| 4 | Mission Control | ⏳ | — |
| 5 | Attack scenes | ⏳ | — |
| 6 | Rehearsal & hand-over | ⏳ | — |

## Decisions log
- 2026-11-22: Plan locked — all software; 2–3 Wokwi ESP32 boards + ~6 software devices;
  global server (FinSentinel + new coordinator) on the laptop; UCI HAR with a ~2.3K-param MLP;
  Mission Control is a standalone app in `demo/` (FinSentinel stays untouched).

- 2026-11-22: User approved downloading cloudflared 2026.11.1, Caddy 2.11.9, Toxiproxy 2.13.0
  into `demo/tools/` (gitignored). User is on the **free Wokwi plan** → Private Gateway
  (`host.wokwi.internal`) is paid-only and is not used; boards must reach the laptop via a
  public tunnel.
- 2026-11-22: FinSentinel is used via `PYTHONPATH` (no editable install) so nothing is written
  into `finsentinel/`. `demo/.venv` = Python 3.12, torch 2.15.0+cpu, FinSentinel's
  `requirements.lock` minus psycopg.
- 2026-11-22: User asked to free resources: Chrome, ChatGPT, Edge, OneDrive were closed.
  The research sweep was **not** touched (user chose not to stop it).

## Phase 0 findings (2026-11-22)

**Verified in code/tests**
- HAR: `demo/.data/download/har.zip` sha256 `c00b8030…1031` (nested zip, CC BY 4.0).
  Split fixed in `demo/config/har_split.json`; 64-feature subset in
  `demo/config/har_features.json` (ANOVA-F + |r|≤0.90 pruning, selection subjects only).
- Model `har-mlp-v1`: MLP 64→32→6, **2,278 params**; payload (STORED NPZ) **10,687 bytes**.
- Signature test vectors generated from FinSentinel's own code →
  `demo/tests/vectors/finsentinel_vectors.json` and `demo/wokwi/probe/vector.h`.
  Server normalises `created_at` to UTC and keeps sub-seconds; devices send
  `YYYY-MM-DDTHH:MM:SSZ`, which round-trips byte-identically.
- STORED-zip NPZ written by hand (`fldemo/wire.py`) is accepted by FinSentinel's
  deserializer and by the **real `/api/v1/updates` endpoint** (in-process TestClient, own
  SQLite): accepted → `pending_approval`; replay → 409; tampered byte → 422; unknown key → 401.
- FinSentinel requires `max_sample_count` on every trusted key (else 403
  `sample_count_limit_missing`) — the launcher must register it.

**Contradictions / risks found (reported to user)**
1. Wokwi simulated ESP32 is capped at ~8 MHz by default (not 240 MHz) → on-board training and
   TLS are slow; must be measured (probe ready in `demo/wokwi/probe/`).
2. **`api.trycloudflare.com` is blocked on the current network** (TLS reset; cloudflare.com,
   github.com, wokwi.com are fine). localhost.run (SSH, no download) reached a URL in a quick
   test but needs user approval as a different third-party service. Demo-day network may
   differ — re-test there.
3. FinSentinel's detector uses only absolute weight statistics; `max_update_l2_norm` is
   hard-disabled in `core/config.py`. An honest drifting device is therefore unlikely to be
   flagged → scene 7 must be measured honestly in Phase 2.
4. Quarantined updates cannot be approved later (by design); the Mission Control approve
   button applies to MEDIUM (pending) updates only. Approvals have no actor field → the
   coordinator writes `[actor=demo-coordinator]` into the reason text.
5. Wokwi CLI free tier: 50 CI min / 30 days, 5-min session cap → smoke tests only.
   Quick tunnels have no SSE → devices use plain request/response.

## Measurements
| What | Value | How |
|---|---|---|
| Central MLP test accuracy (9 UCI test subjects) | **0.883** (untrained 0.219) | `scripts/phase0_har.py`, SGD lr 0.05, bs 32, 30 epochs, 21 train subjects |
| Central model on each device subject | 0.894 – 0.997 | same run |
| ESP32 training speed in Wokwi | _pending (probe)_ | `demo/wokwi/probe` + `scripts/phase0_probe_server.py` |
