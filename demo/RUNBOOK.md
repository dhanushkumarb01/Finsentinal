# RUNBOOK — How to run the live demo yourself

> Every step below is marked ✅ once it has been run on this laptop. Steps without a mark are
> built and unit-tested but still need the end-to-end check (see `demo/PROGRESS.md`).

---

## What you will have running

| Window | What it is | Address |
|---|---|---|
| Mission Control | The ONE screen: **Live** (audience), **Security Console** (FinSentinel + Wazuh decisions; replaces FinSentinel's UI), **Research** | http://localhost:5180 |
| Wokwi tabs (2–3) | The ESP32 boards | wokwi.com |
| Terminal | Runs everything; **Ctrl+C stops everything** | — |
| Wazuh dashboard | The SIEM that makes every security decision (its own detailed view) | https://localhost |

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
   | `--wazuh` | **Wazuh decides** every update and attack in real time (section D). Without it the coordinator decides and the screen says so |

2. **Open the screens** ✅ *(Mission Control live stream + buttons verified)*
   - Mission Control → http://localhost:5180 (projector, full screen with **F11**)
   - Security Console → the **Security Console** tab in Mission Control (FinSentinel results, Wazuh decisions, manual override)

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

## D. Wazuh makes the decisions

**How it works:** FinSentinel checks every update and reports. Its audit events flow to Wazuh.
Wazuh's rules (`demo/wazuh/fl_decisions.xml`, layered on FinSentinel's own rules) decide in real time:
approve, quarantine, reject, or suspend the device. Wazuh's active response writes the decision, and
the coordinator only carries it out through FinSentinel's operator API. Every decision names its
Wazuh rule (e.g. "wazuh rule 100211 (level 14)").

**Before using Wazuh: pause the research sweep** (Wazuh needs ~4–6 GB RAM). The sweep
checkpoints every 10 rounds and resumes exactly:
1. In the terminal where the sweep runs, press **Ctrl+C** (or close that window).
2. Afterwards, resume it with the same command:
   ```powershell
   cd C:\Users\Dhanush\drift-causal-fl\drift-causal-fl
   .venv\Scripts\python scripts\run_sweep.py --config configs/mvp.yaml --workers 4 --threads 1 --max-concurrent flfdms=1
   ```

**One-time setup** (installs Docker Engine in WSL Ubuntu and asks for your Ubuntu password; the
first image download takes 10–30 min):
```powershell
wsl -d Ubuntu bash /mnt/c/Users/Dhanush/drift-causal-fl/demo/wazuh/wazuh.sh setup
```
It runs FinSentinel's own `setup-wsl.sh` (unmodified) with the SIEM log pointed at
`demo\.state\siem`, then overlays the demo's decision rules and active response.

**Every time:** `demo\.venv\Scripts\python demo\run_demo.py --wazuh`. It starts Wazuh for you
(`wazuh.sh up`) and waits for it. Check anytime:
```powershell
wsl -d Ubuntu bash /mnt/c/Users/Dhanush/drift-causal-fl/demo/wazuh/wazuh.sh status      # containers + API
wsl -d Ubuntu bash /mnt/c/Users/Dhanush/drift-causal-fl/demo/wazuh/wazuh.sh logtest     # which rule fires for sample events
wsl -d Ubuntu bash /mnt/c/Users/Dhanush/drift-causal-fl/demo/wazuh/wazuh.sh decisions   # last decisions Wazuh made
```
Wazuh dashboard → https://localhost (user `admin`, local default password from
`finsentinel\deploy\wazuh\.env.example`). Stop Wazuh: `wazuh.sh down`.

If Wazuh does not come up, the launcher prints a red line and continues with the coordinator as
decision engine; Mission Control shows "Wazuh offline — coordinator decides".

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
