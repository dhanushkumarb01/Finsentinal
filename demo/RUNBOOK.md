# RUNBOOK — How to run the live demo yourself

> **Status: TARGET.** This is the contract the build must satisfy. Each step gets marked
> ✅ verified once it has been tested on this machine (see `demo/PROGRESS.md`). Until then,
> commands may change slightly; the final version will be exact.

---

## What you will have running

| Window | What it is | Address |
|---|---|---|
| Mission Control | The big screen for the audience | http://localhost:5180 |
| FinSentinel UI | The security guard's control panel (approve / reject) | http://localhost:5173 |
| Wokwi tabs (2–3) | The circuit-board devices | wokwi.com or VS Code |
| Terminal | Runs everything; Ctrl+C stops everything | — |
| Wazuh (optional) | Professional security log / CCTV room | https://localhost |

---

## A. One-time setup (≈20 minutes, done once)

1. **Get the latest code**
   ```powershell
   cd C:\Users\Dhanush\drift-causal-fl
   git checkout demo/fl-live
   git pull finsentinal demo/fl-live
   ```
2. **Install the demo** (creates its own private Python environment — does not touch the
   research sweep)
   ```powershell
   python demo/setup.py
   ```
   This creates `demo/.venv`, installs packages, downloads the activity dataset, builds the
   Mission Control website, and downloads the small helper tools (after asking you).
3. **Wokwi account:** sign up free at https://wokwi.com. If you want automatic board
   testing, create a CI token in your Wokwi dashboard and set it once:
   ```powershell
   setx WOKWI_CLI_TOKEN "paste-token-here"
   ```

---

## B. Every time you run the demo

1. **Start everything with one command**
   ```powershell
   cd C:\Users\Dhanush\drift-causal-fl
   demo\.venv\Scripts\python demo\run_demo.py
   ```
   Wait for the green line `ALL SYSTEMS READY`. It prints the addresses and, if Wokwi is on,
   the public tunnel address the boards use.

2. **Open the screens**
   - Mission Control → http://localhost:5180 (put this on the projector, full screen: F11)
   - FinSentinel UI → http://localhost:5173 (log in with the token the terminal prints)

3. **Start the Wokwi boards**
   - Open `demo/wokwi/board1` (and `board2`, `board3`) in Wokwi → press ▶ Play.
   - Each board's screen shows `Connected` and then `Round N · Learning…`.
   - On Mission Control the board's card turns live.

4. **Run the show** — follow `demo/PRESENTER_SCRIPT.md`. In short:

   | Scene | You do | Audience sees |
   |---|---|---|
   | 1 Fleet online | Press Play on boards | Devices appear |
   | 2 Learning | Nothing | Envelopes fly, gate goes green, accuracy climbs |
   | 3 Dropout | Turn a board's battery knob down | That device skips the round |
   | 4 Hacked device | Flip a board's "hacked" switch | Quarantined, red light, alert |
   | 5 Replay / tamper | Click **Replay**, then **Tamper** | Envelopes bounce off the gate |
   | 6 Eavesdropper | Click **Eavesdrop** | Lock holds, interception fails |
   | 7 Research twist | Click **Make device drift** | Honest device flagged → absent → show FEMNIST slide |

   Attacks can also be run from a second terminal:
   ```powershell
   demo\.venv\Scripts\python demo\attack.py poison    # or: replay, tamper, forge, mitm, drift
   ```

5. **Stop everything:** press **Ctrl+C** in the terminal. Everything shuts down cleanly.

---

## C. Safety net (if anything breaks on stage)

```powershell
demo\.venv\Scripts\python demo\run_demo.py --replay demo\recordings\golden.jsonl
```
Mission Control plays the recorded rehearsal run. No server, no internet, no Wokwi needed.

**No internet at the venue?** Run with `--wokwi off`: the software devices carry the whole
demo; or use replay.

---

## D. Demo day with Wazuh (optional, heavy)

1. Pause the research sweep first (it needs the memory) — ask before doing this if unsure how.
2. Start Wazuh from WSL as described in `finsentinel/deploy/wazuh/README.md`.
3. Run `demo\.venv\Scripts\python demo\run_demo.py --wazuh`.
4. Wazuh dashboard → https://localhost (accept the local certificate warning).
5. Afterwards, resume the sweep with the same command it was started with
   (`drift-causal-fl/PROGRESS.md` → "Exact next step").

---

## E. Quick checks

| Problem | Fix |
|---|---|
| Board says `WiFi failed` | Internet down or tunnel not started → restart with `run_demo.py`; or use `--wokwi off` |
| Board says `Time sync failed` | Same as above (boards need internet time to sign updates) |
| Mission Control blank | Check the terminal for a red component message; refresh the page |
| Updates stuck "pending" | They were flagged — approve/reject them in the FinSentinel UI |
| Anything else | Ctrl+C, run again; if still broken, use the replay safety net |
