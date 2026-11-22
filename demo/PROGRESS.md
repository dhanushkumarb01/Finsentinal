# PROGRESS — Live FL Demo ("Mission Control")

*Single source of truth for the demo build. Read this first; update it every session.*

**Branch:** `demo/fl-live` · **Remote:** `finsentinal` (https://github.com/dhanushkumarb01/Finsentinal.git)
**Build instructions:** `demo/BUILD_PROMPT.md` · **How to run:** `demo/RUNBOOK.md`

| # | Phase | Status | Evidence |
|---|---|---|---|
| 0 | Verify & de-risk | ✅ done 2026-11-22 (reported to user) | `demo/tests` 14/14 pass; Wokwi probe ran end-to-end via localhost.run; measurements below |
| 1 | Core engine | ✅ built + verified 2026-11-22 | `tests/test_integration_round.py` (full in-process rounds); live runs: FinSentinel+coordinator+Caddy+Toxiproxy+6 phones, accuracy 0.22→0.55 in 4 rounds, assisted approvals, signed downlink |
| 2 | Device realism | ✅ built + verified | live rehearsal: low-battery skip, Toxiproxy weak-signal skip, drift-causal dropout (D_indep) |
| 3 | Wokwi boards | 🟡 built; host tests 24/24; **needs the Wokwi run by the user** | `demo/wokwi/board1-3`, `tests/test_firmware_host.py` |
| 4 | Mission Control | ✅ built; Vitest 41/41; typecheck + build clean; live stream + CORS POST verified; replay with no backend verified | `demo/mission-control` |
| 5 | Attack scenes | ✅ every scene verified live (replay 409, tamper 422, forge 401, poison quarantined, MITM no plaintext + impostor refused); Wazuh path built, not run (needs sweep paused) | `tests/test_mitm.py`, integration test, `scripts/rehearse.py` |
| 6 | Rehearsal & hand-over | 🟡 two consecutive cold-start runs of the 7-scene script: **13/13 checks each** (virtual fleet, `--wokwi off`); golden recording saved; runbook + presenter script written. Still open: a run with the Wokwi boards + user walkthrough | `scripts/rehearse.py`, `recordings/golden.jsonl` |

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
2. **`api.trycloudflare.com` is blocked on the user's network (a mobile hotspot)** (TLS reset;
   cloudflare.com, github.com, wokwi.com are fine). User approved **localhost.run** (SSH, no
   download, no account) as the tunnel; the Wokwi probe reached the laptop through it.
   Free localhost.run **rotates the domain every few minutes and drops connections** →
   `scripts/tunnel_lhr.sh` auto-restarts and writes the current URL to
   `.state/tunnel_url.txt`. For stage use the boards must learn the URL at runtime
   (or the user creates a free localhost.run account + SSH key for a stable domain — open
   decision). Wokwi's simulated WiFi took > 90 s to associate on the first attempts (status
   stuck at 0), then worked — firmware must wait/retry generously.
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
| ESP32 training speed in Wokwi (browser, free plan, default CPU cap) | **2.1 s simulated / ≤ 5.2 s real per epoch** (340 samples, 64→32→6, bs 32); 3 epochs = 6.4 s sim, 15.5 s wall incl. one HTTPS request; sim speed ≈ 41 % | `demo/wokwi/probe` + `scripts/phase0_probe_server.py`, 2026-11-22 |
| ESP32 Ed25519 sign (rweather Crypto) | **matches FinSentinel vector byte-for-byte**; 1.6 s simulated | probe `/ed25519?match=1` |
| ESP32 HTTPS request via tunnel (TLS, `setInsecure` in probe) | ~ 6–9 s real per request (incl. TLS handshake) | gap between probe requests |
| Wokwi browser compiler | arduino-esp32 **3.3.12**; static DRAM limit hit with 87 KB float array → store samples int8 (or const in flash) | build log |

## Phases 1–6 build (2026-11-22)

**Architecture as built** (contracts: `docs/PROTOCOL.md`, `docs/EVENTS.md`)
- `run_demo.py` supervises: FinSentinel `fldemo/fs_server.py` (:8000, own SQLite, schema + 9 keys
  registered, calibrated thresholds via env, audit webhook → coordinator) · coordinator
  `fldemo/coordinator.py` (:8200) · Caddy (:8443 TLS with demo CA, :8480 tunnel origin; only
  device routes exposed) · Toxiproxy (:8474, links :9101–9106) · Mission Control static (:5180) ·
  6 phones `fldemo/device.py` · tunnel (localhost.run via ssh, or cloudflared) + signed ntfy beacon.
- **FinSentinel's real routes** differ from the build prompt: approvals are
  `/api/v1/updates/{client}/{update}/approve|reject|status`, aggregation `/api/v1/aggregation/run`.
- **FinSentinel's real check order:** integrity (SHA-256) → signature → schema → policy → replay →
  anomaly. The 5 gate lights follow this order.
- Virtual phones train with a NumPy port of the model (`fldemo/npmodel.py`, tested equal to
  PyTorch) to save ~100 MB RAM per process.

**Calibration (`config/thresholds.json`, method in `scripts/calibrate_thresholds.py`)**
- 30 simulated clean rounds, 3 epochs, lr 0.1, all 9 device subjects; thresholds = max clean ×1.25:
  L2 norm 11.61, |mean| 0.155, std 0.243, zero-fraction 0.5. Clean false positives **0/270**;
  poisoned ×8 updates trigger **3/3 rules** (CRITICAL → quarantined) in 54/54 cases.
- Drift device (label shift to stair activities): FinSentinel flags **0/21** — honest updates look
  normal. A coordinator cosine-to-cohort monitor also cannot separate it (flag rate 0). Device-side
  D_indep (research estimator #2): clean 0.0, drifted ≈0.23 → drives drift-causal dropout
  (p = sigmoid(−3 + 20·D) ≈ 0.83).
- **Deviation from the locked design (reported to user):** scene 7 cannot honestly show the
  drifting device being *flagged*. It shows: FinSentinel lets the honest update through, the device
  drops out *because of* its drift, the server only sees the absence → research slide.

**Rehearsals (live, `--wokwi off`, 14 rounds × 15 s):** two consecutive cold starts, 13/13 checks
each (accuracy 0.22→0.54/0.55 by round 2–3; low battery; weak signal; poison quarantined
(L2 49.6 vs 11.6); replay 409; tamper 422; forge 401; MITM 3.6 KB captured, no plaintext,
impostor refused; drift dropout 1–2 rounds; drift not quarantined). Golden = run-20261122-200935.

**FEMNIST slide (`config/femnist.json`, `scripts/make_femnist_summary.py`):** Day-5 pilot +
12 finished MVP runs. Honest note: in these runs MNAR (drift-causal) accuracy is not visibly
different from MCAR yet; the slide states it is a separate, partial study.

**Open items**
1. Run the 3 Wokwi boards against a live `run_demo.py` (firmware compiled only on host so far):
   check pin names, core-3.3.12 compile, TLS to `*.lhr.life` (GTS) and ntfy.sh (LE YE1 via X1/X2
   cross-sign), heap during POST, round timing (90 s rounds).
2. `--wazuh` path: built (FinSentinel's own forwarder → `.state/siem/`), not run (needs sweep paused).
3. FinSentinel UI (:5173) starts only if `finsentinel/frontend/node_modules` exists (`npm ci` there).
