# Build Prompt — "Mission Control" Live Federated-Learning Demo

> **How to use:** open a new Claude Code chat at the repo root (`drift-causal-fl/`, the folder
> that contains `demo/`, `drift-causal-fl/` and `finsentinel/`), make sure branch
> `demo/fl-live` is checked out, and send:
> **"Read demo/BUILD_PROMPT.md and execute it, phase by phase."**

---

## 0. Your role and standard

You are the lead engineer building a live, presentation-grade simulation of a secured
federated-learning (FL) system. Work like a careful senior engineer: verify before you
build, test every phase, keep the system honest (every visual event must come from a real
system event — never fake an outcome), and keep the user informed in plain language.
The user is not a deep systems engineer; explain decisions simply.

The result must be **impressive on a projector** and **reliable on stage**.

---

## 1. Context you must know

**Repository:** `https://github.com/dhanushkumarb01/Finsentinal.git` (local git remote name
`finsentinal`). Push **only** to `finsentinal`. Never push to `origin` (old repo). Work on
branch `demo/fl-live`; never force-push; never commit to `master` directly.

**Layout:**
- `drift-causal-fl/` — our research: drift-causal client dropout in FL, benchmarked on
  FEMNIST (the "theory"). Read `drift-causal-fl/PROGRESS.md` for the research story.
- `finsentinel/` — Jaivardhan Vajpai's security layer, **finished**. Treat it as a
  **read-only dependency**. Read `finsentinel/docs/architecture.md`,
  `finsentinel/IMPLEMENTATION_STATUS.md`, `finsentinel/docs/threat_model.md`.
- `demo/` — everything you build goes here.

**Presentation story:** "We proved the effect on FEMNIST (handwriting) → we deployed it on a
lighter real device dataset (UCI HAR, smartphone activity sensors) in a secured FL pipeline
→ live demo." The demo *shows* the research idea; it does not *prove* that FEMNIST results
transfer — never word anything that way.

**Verified facts about FinSentinel (re-verify in code before relying on them):**
- Ingestion: `POST /api/v1/updates` with `{"envelope": {...}, "payload_base64": "..."}`.
  Pipeline: Ed25519 signature → replay check → policy (allowed model versions/rounds,
  freshness) → schema → SHA-256 payload integrity → rule-based anomaly detection →
  approval registration → audit outbox.
- Signature = Ed25519 over `json.dumps(envelope.model_dump(mode="json", exclude={"signature"}),
  sort_keys=True, separators=(",", ":"), ensure_ascii=False)` (`finsentinel/core/crypto.py`).
  `created_at` must be timezone-aware; its exact JSON string form matters — generate test
  vectors from the real Python code and make every client (Python and C++) match them.
- Payload = NPZ archive with a `__manifest__` entry (see `ingestion/serialization.py` and
  `scripts/demo_submit_updates.py::_make_payload`). Uncompressed (STORED) zip entries are
  accepted by `np.load` — confirm with a test, since firmware will write STORED zips.
- **No auto-approval, by design.** Approvals go through
  `POST /api/v1/approvals/{client_id}/{update_id}/approve|reject|quarantine` with the operator
  bearer token. High-risk updates are quarantined (`FINSENTINEL_HIGH_RISK_ACTION`).
- Aggregation: `POST /api/v1/aggregate/run` (operator auth) takes the approved payloads plus
  the trusted reference model (`reference_payload_base64`) and returns the FedAvg'd model as
  NPZ. There is **no** coordinator, no round management, no global-model distribution and no
  signed downlink — you build those.
- Size caps: aggregation 10 MB, nginx body 10 MB. The FEMNIST CNN (26 MB) does not fit;
  the HAR model does.
- Dev mode runs on SQLite via the pattern in `finsentinel/scripts/run_dev.py` (fake
  `core.supabase.connection` module injected before imports). Replicate this pattern in a
  demo launcher with **its own DB file under `demo/.state/`** so Jaivardhan's dev DB is never
  touched. All FinSentinel configuration goes through env vars (see `docs/install.md`) — set
  them in the launcher, don't edit his code.
- Behind a local proxy/tunnel every request arrives from 127.0.0.1, so the per-IP rate limit
  is shared by all devices: raise `FINSENTINEL_INGESTION_RATE_LIMIT_RPM` via env.
- Detection thresholds are uncalibrated (his open item S5-10). Calibrate them for the HAR
  model from clean runs (e.g. a high percentile of clean-update features × margin), and
  record the method and values in `demo/PROGRESS.md`.

**Machine constraints (critical):**
- Windows 11, 16 GB RAM, 12 logical CPUs, Python 3.11/3.12 available, Node 20, Docker
  Desktop stopped, WSL Ubuntu present.
- **A multi-day research sweep is running** (`drift-causal-fl/scripts/run_sweep.py`, several
  `python` processes, ~1 GB RAM free). **Never kill, restart, or slow it deliberately. Never
  run `taskkill`/`Stop-Process` on python by name. Never modify anything in
  `drift-causal-fl/`** (especially `results/`), and never install packages into
  `drift-causal-fl`'s environment.
- Use a separate venv at `demo/.venv` (install `finsentinel` editable plus demo deps). Keep
  the dev footprint small: no Docker, no Wazuh during development. Wazuh is optional and only
  for demo day (user pauses the sweep then).

---

## 2. Locked design (do not re-litigate; flag real problems to the user instead)

**All software. No physical hardware.**

| Part | What | Where |
|---|---|---|
| Global server ("the cloud") | FinSentinel (unmodified) + new **Coordinator** | User's laptop |
| Star devices | **2–3 Wokwi ESP32 boards** with real firmware | Wokwi (browser or VS Code) |
| Rest of fleet | **~6 software devices** (one OS process each) | Laptop |
| Big screen | **Mission Control** website (standalone, in `demo/`) | Laptop browser / projector |
| SIEM | Wazuh, demo day only | Laptop (WSL) |

**Dataset/model:** UCI HAR (30 subjects, 561 features, 6 activities). Fixed, documented
feature subset (~64 features, chosen on non-device subjects only, saved to a JSON file) →
MLP 64→32→6 (~2.3K params). Identical model on every device (PyTorch on software devices,
hand-written C++ on ESP32). Model version string: `har-mlp-v1`. One subject per device;
remaining subjects for test set and drift source.

**Coordinator (new, Python/FastAPI):** round management; holds the global model; serves the
global model **signed with the coordinator's Ed25519 key** (devices pin the public key and
refuse unsigned/altered models); tracks which updates were accepted; approval policy
"assisted": clean LOW-risk updates are approved through FinSentinel's operator API with an
audit trail naming the demo coordinator as actor, anything flagged waits for the human
(FinSentinel UI or a Mission Control approve button that calls the same operator API);
calls `/api/v1/aggregate/run`; publishes a live event stream (WebSocket) and records every
event to `demo/recordings/<run-id>.jsonl` for replay.

**Single public origin:** a TLS reverse proxy (single binary, no admin rights; e.g. Caddy)
routes `/api/v1/*` → FinSentinel and `/fl/*` → Coordinator. Software devices connect through
**Toxiproxy** (per-device TCP link with latency / packet loss / bandwidth from the device's
profile; TLS passes through untouched). Wokwi boards reach the laptop through a **Cloudflare
quick tunnel** (free) pointed at the proxy; fall back to Wokwi's Private Gateway
(`host.wokwi.internal`) only if the user has it. Verify both options in Phase 0.

**Software device behaviour:** each has a profile (device name, battery, network type,
drift schedule). Battery drains when training; low battery or bad signal → skips the round;
scripted drift (label-distribution shift, as in the research) changes its data. Dropout
therefore has visible, physical causes, and drift-caused dropout appears among them.

**Wokwi board:** ESP32 + SSD1306 OLED (round, status, loss) + green/red LED (accepted /
blocked) + potentiometer = battery level (turn down → skips round) + push-button = go
offline + slide switch = "hacked" (sends poisoned weights) + MPU6050 for show. Firmware:
WiFi (`Wokwi-GUEST`) → NTP time → download + verify signed global model → train on its
subject's data stored in flash → build STORED-zip NPZ → sign envelope (Ed25519) → HTTPS POST
to FinSentinel → show result. Private keys go in a **gitignored** generated header; public
keys are registered in FinSentinel's trusted-key table by the launcher.

**Attacks (real requests against the real server):** poisoning (boosted weights), replay
(resend captured body → 409), tamper (alter payload → 422), forged/unknown key (→ 401),
eavesdrop/MITM attempt (interceptor with an untrusted certificate → device refuses
handshake), drift (honest device becomes anomalous). Triggerable from the Mission Control
attack console and from the CLI. If an honest drifting device does **not** genuinely get
flagged after calibration, report that to the user honestly — do not force it.

**Mission Control (standalone Vite + React + TypeScript app, `demo/mission-control/`):**
projector-first (1920×1080, large type, dark theme, smooth animations), no runtime CDN
dependencies (works offline). Areas: header (round, global accuracy, Live/Replay toggle);
**device wall** (each device drawn as a phone or ESP32 board with battery, signal, status,
mini loss curve; Wokwi boards visibly marked); **secure tunnel** (animated locked
envelopes; attacks bounce off); **security gate** (5 lights: sender verified, not a replay,
correct shape, untampered, looks normal — each lit by the real FinSentinel result);
**global model** chart; **live alert feed**; **attack console** buttons; click a device →
"inside the device" panel (data sample, loss, weight-change heatmap, sign/send steps).
Label honestly: "Virtual device", "Wokwi-simulated ESP32". Replay mode plays a recording
with no backend running.

**One-command run:** `python demo/run_demo.py` starts and supervises everything (proxy,
Toxiproxy, FinSentinel, coordinator, software devices, Mission Control), prints the URLs, and
shuts everything down cleanly on Ctrl+C. Options at least: `--replay <file>`,
`--wokwi tunnel|gateway|off`, `--rounds N`, `--wazuh`. `python demo/attack.py <scene>` runs
attacks from the terminal. `python demo/setup.py` does the one-time setup (creates
`demo/.venv`, installs deps, downloads HAR, builds Mission Control, fetches helper tools
after asking). `demo/RUNBOOK.md` already describes the target user experience — make it
true, then mark each step verified.

---

## 3. Phases (complete each fully, with tests, before the next)

Commit at the end of every phase (conventional message ending with the attribution line your
harness specifies), push `demo/fl-live` to `finsentinal`, and update `demo/PROGRESS.md`.

| # | Phase | Exit criteria (must be demonstrated, not assumed) |
|---|---|---|
| 0 | **Verify & de-risk** | HAR downloaded to `demo/.data/` (gitignored) and checksummed; feature subset + MLP trained centrally with reported accuracy; signature test vectors generated from FinSentinel's code; STORED-zip NPZ accepted by FinSentinel's deserializer; Wokwi connectivity route confirmed (tunnel and/or gateway) with a minimal board that fetches a URL from the laptop; ESP32 training speed in Wokwi measured. **Stop and report to the user**, including anything that contradicts this plan. |
| 1 | **Core engine** | Launcher starts FinSentinel (own SQLite DB) + coordinator + 6 software devices; 10 real rounds over HTTPS; every update visible in FinSentinel's UI/API; global accuracy rises; signed downlink verified by devices; thresholds calibrated and recorded. Integration test runs a full round in-process. |
| 2 | **Device realism** | Profiles, battery, signal, Toxiproxy links, drift; devices drop out for visible, logged reasons; drift device behaviour measured and reported. |
| 3 | **Wokwi boards** | Firmware + `diagram.json` + `wokwi.toml` in `demo/wokwi/`; 2 boards complete the same rounds as software devices; OLED/LEDs/knob/button/switch all work; host-side tests for the C++ math against the PyTorch model (same input → same output within tolerance); Wokwi CLI smoke test if the user supplies `WOKWI_CLI_TOKEN`. |
| 4 | **Mission Control** | All areas live from real events; device zoom panel; replay of a recorded run works with every backend process stopped; Vitest tests for the event reducer and replay; production build clean. |
| 5 | **Attack scenes** | Every attack triggers a real FinSentinel outcome (status code / quarantine) and appears in the alert feed; with `--wazuh` the alerts reach Wazuh. Tests per attack. |
| 6 | **Rehearsal & hand-over** | Two consecutive cold-start runs of the full 7-scene script succeed; a golden recording saved in `demo/recordings/`; `demo/RUNBOOK.md` verified step by step on this machine; `demo/PRESENTER_SCRIPT.md` with what to click and what to say per scene, in plain language. |

**The 7 demo scenes** (design everything to serve these): (1) fleet comes online,
(2) learning together — envelopes, green gate, accuracy rising, (3) real-life dropout —
battery knob / weak signal, (4) hacked device — switch flipped, quarantined, red LED,
(5) replay + tamper blocked, (6) eavesdropper fails against TLS, (7) research twist —
honest drifting device flagged → absent → cut to FEMNIST results.

---

## 4. Engineering rules

- **Verify, don't trust:** anything in this prompt marked as a fact about external tools
  (Wokwi gateway licensing, HAR download URL, Caddy/Toxiproxy/cloudflared flags) must be
  checked against current docs/code before you depend on it.
- **Ask the user before:** downloading any executable (list them once, with sources;
  store under `demo/tools/`, gitignored), anything paid, anything needing admin rights,
  changing anything in `finsentinel/` (only if truly unavoidable — minimal, documented,
  his tests must still pass), or any change to the locked design.
- **Never commit:** private keys, tokens, `.env`, `demo/.state/`, `demo/.data/`,
  `demo/.venv/`, `demo/tools/`, `node_modules/`, generated firmware secrets. Add a
  `demo/.gitignore` in Phase 0 and check `git status` before every commit.
- **Tests:** pytest for all Python (unit + integration), Vitest for the UI, host-compiled
  tests for firmware math. Run them before every commit; report real output, never claim
  a pass you did not see.
- **Honesty in the UI:** simulated things are labelled; numbers on screen come from real
  runs; no hard-coded "success" animations.
- **Robustness on stage:** every process restartable; launcher health-checks each
  component and shows a clear red message instead of hanging; replay mode always available.
- **Keep `demo/PROGRESS.md` current** (status table per phase, decisions, deviations,
  measurements) so a new chat can resume from it — same discipline as
  `drift-causal-fl/PROGRESS.md`.
- Talk to the user in short, plain-language updates at each phase boundary.

---

## 5. Final deliverables

`demo/setup.py`, `demo/run_demo.py`, `demo/attack.py`, coordinator, software device agent, Wokwi project,
Mission Control app, tests, a golden recording, `demo/RUNBOOK.md` (verified),
`demo/PRESENTER_SCRIPT.md`, `demo/PROGRESS.md`, all pushed to `demo/fl-live` on
`finsentinal`. When everything is done, ask the user whether to open a pull request into
`master`.
