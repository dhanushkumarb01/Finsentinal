# PROGRESS — Live FL Demo ("Mission Control")

*Single source of truth for the demo build. Read this first; update it every session.*

**Branch:** `demo/fl-live` · **Remote:** `finsentinal` (https://github.com/dhanushkumarb01/Finsentinal.git)
**Build instructions:** `demo/BUILD_PROMPT.md` · **How to run:** `demo/RUNBOOK.md`

| # | Phase | Status | Evidence |
|---|---|---|---|
| 0 | Verify & de-risk | ⏳ not started | — |
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

## Measurements
_(Phase 0 fills this: HAR accuracy, ESP32 training speed in Wokwi, round time, thresholds.)_
