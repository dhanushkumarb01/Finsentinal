# RUNBOOK — How to run the live demo yourself

> Every step below is marked ✅ once it has been run on this laptop. Steps without a mark are
> built and unit-tested but still need the end-to-end check (see `demo/PROGRESS.md`).

---

## What you will have running

| Window | What it is | Address |
|---|---|---|
| Mission Control | The big screen for the audience | http://localhost:5180 |
| FinSentinel UI (optional) | The security guard's own control panel | http://localhost:5173 |
| Wokwi tabs (2–3) | The ESP32 boards | wokwi.com |
| Terminal | Runs everything; **Ctrl+C stops everything** | — |
| Wazuh (optional) | Professional security log room | https://localhost |

---

## A. One-time setup

1. **Get the latest code**
   ```powershell
   cd C:\Users\Dhanush\drift-causal-fl
   git checkout demo/fl-live
   git pull finsentinal demo/fl-live
   ```
2. **Install the demo** (its own Python environment; never touches the research sweep):
   ```powershell
   python demo\setup.py
   ```
   It installs packages, downloads the activity dataset (checksummed), asks before downloading
   the helper tools (Caddy, Toxiproxy, cloudflared — official GitHub releases, checksummed),
   builds Mission Control, creates the keys/certificates and generates the board files.
3. *(Optional)* FinSentinel's own UI: `cd finsentinel\frontend; npm ci` — the launcher starts it
   automatically when it is installed.

---

## B. Every time you run the demo

1. **Start everything** ✅ *(verified 2026-11-22 with `--wokwi off`: two cold starts, ALL SYSTEMS READY)*
   ```powershell
   cd C:\Users\Dhanush\drift-causal-fl
   demo\.venv\Scripts\python demo\run_demo.py
   ```
   Wait for the green **ALL SYSTEMS READY**. The terminal prints the addresses and, a few
   seconds later, the public tunnel address (`↳ Wokwi tunnel: https://….lhr.life`). You never
   need to copy that address anywhere: the launcher signs it and publishes it to the boards'
   beacon, and the boards find it by themselves.

   Options:
   | Option | Meaning |
   |---|---|
   | `--wokwi off` | no boards, no internet needed (virtual phones carry the demo) |
   | `--wokwi cloudflare` | use Cloudflare's tunnel instead of localhost.run (blocked on the current hotspot) |
   | `--rounds 20` | number of rounds (default 30) |
   | `--round-seconds 40` | round length (default 25 s; 90 s when boards are on) |
   | `--wazuh` | also send FinSentinel's audit events to Wazuh (section D) |

2. **Open the screens** ✅ *(Mission Control live stream + buttons verified)*
   - Mission Control → http://localhost:5180 (projector, full screen with **F11**)
   - FinSentinel UI → http://localhost:5173 — operator token: `demo\.state\secrets.json` → `operator_token`

3. **Start the Wokwi boards** ⏳ *(firmware built + host-tested; first live board run pending)* (2–3 browser tabs, Chrome is fastest)
   - For each `demo\wokwi\board1`, `board2`, `board3`: wokwi.com → **New Project → ESP32** →
     replace `sketch.ino` and `diagram.json`, then **▾ → Upload file(s)…** and add every other
     file in that folder (including `board_secrets.h`) → **▶ Play**.
     Do this once; afterwards the Wokwi project is saved in your account. Re-upload only
     `board_secrets.h` if you ever delete `demo\.state\keys`.
   - Wi-Fi in Wokwi can take 1–2 minutes. The screen then shows **Laptop found** and
     `R<n> …`. On Mission Control the board's card becomes live.

4. **Run the show** ✅ *(all 7 scenes verified by `demo/scripts/rehearse.py`, 13/13 checks, twice)* — follow `demo/PRESENTER_SCRIPT.md`.
   Attacks can also be run from a second terminal:
   ```powershell
   demo\.venv\Scripts\python demo\attack.py poison      # replay | tamper | forge | mitm | drift | weak_signal | low_battery | offline | restore
   ```

5. **Stop:** ✅ Ctrl+C in the terminal. Every recording is kept in `demo\recordings\run-*.jsonl`.

---

## C. Safety net (if anything breaks on stage) ✅ *(verified with no backend running)*

```powershell
demo\.venv\Scripts\python demo\run_demo.py --replay demo\recordings\golden.jsonl
```
Mission Control replays the recorded rehearsal. No server, no internet, no Wokwi needed.
Any `run-*.jsonl` recording works the same way.

---

## D. Demo day with Wazuh (optional, heavy)

1. Pause the research sweep first (it needs the memory).
2. In `finsentinel\deploy\wazuh\.env` set the SIEM log directory to
   `C:\Users\Dhanush\drift-causal-fl\demo\.state\siem` (as seen from WSL:
   `/mnt/c/Users/Dhanush/drift-causal-fl/demo/.state/siem`), then start Wazuh as described in
   `finsentinel\deploy\wazuh\README.md`.
3. `demo\.venv\Scripts\python demo\run_demo.py --wazuh` — FinSentinel's own SIEM forwarder
   writes `finsentinel-siem.jsonl` there; Wazuh picks it up.
4. Wazuh dashboard → https://localhost.

---

## E. Quick checks

| Problem | Fix |
|---|---|
| A red `✗` line at start | Read the log file it names in `demo\.state\logs\`; Ctrl+C and start again |
| Board stuck on `WiFi…` | Wokwi Wi-Fi is slow — wait 2 min; check the laptop has internet |
| Board says `No beacon` / `Laptop?` | Tunnel not up yet — watch the terminal for `↳ Wokwi tunnel`; it reconnects by itself |
| Board says `TLS refused` | The tunnel's certificate is not in the board's root list — use `--wokwi tunnel` (Google Trust Services) |
| Mission Control shows "connection lost" | The coordinator stopped — check the terminal; use the replay safety net |
| Update "waiting for review" | It was flagged MEDIUM: approve/reject it in Mission Control or the FinSentinel UI |
| Ports busy (8000, 8200, 8443, 5180…) | Another copy is running, or FinSentinel's own dev server uses 8000 — stop it first |
