"""Wazuh decides, the coordinator enforces.

Flow (see demo/wazuh/*):
  FinSentinel audit event → coordinator webhook → FinSentinel's SIEM forwarder → siem JSONL
  → Wazuh analysisd (FinSentinel rules + demo/wazuh/fl_decisions.xml) → active response
  `fl-decision` → demo/.state/wazuh/decisions.jsonl → THIS module → FinSentinel operator API.

Nothing here decides anything: every action is the execution of a Wazuh decision record, and every
resulting event names the Wazuh rule (id + level) that made it. Also writes FL-level events
(drift dropout, MITM refused, TLS errors) to the SIEM log for Wazuh, and watches Wazuh's health.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import httpx

from fldemo.paths import STATE_DIR

DECISIONS_FILE = STATE_DIR / "wazuh" / "decisions.jsonl"
FL_SIEM_FILE = STATE_DIR / "siem" / "fl-coordinator.jsonl"
API = os.environ.get("FLDEMO_WAZUH_API", "https://127.0.0.1:55000")
API_USER = os.environ.get("FLDEMO_WAZUH_API_USER", "wazuh-wui")
API_PASS = os.environ.get("FLDEMO_WAZUH_API_PASSWORD", "MyS3cr37P450r.*-")  # wazuh-docker 4.8 local default
DASHBOARD = os.environ.get("FLDEMO_WAZUH_DASHBOARD", "https://localhost")
FINAL = ("approved", "rejected", "quarantined")


def write_siem_event(event_type: str, *, severity: str, client_id: str = "", description: str = "",
                     **extra: Any) -> None:
    """Append one FL-level event for Wazuh (source = fl-coordinator, see fl_decisions.xml 100250+)."""
    FL_SIEM_FILE.parent.mkdir(parents=True, exist_ok=True)
    rec = {"source": "fl-coordinator", "event_type": event_type, "severity": severity, "client_id": client_id,
           "description": description, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
    with FL_SIEM_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")


class WazuhEngine:
    def __init__(self, coord, *, decisions_file=DECISIONS_FILE, poll_s: float = 0.3) -> None:
        self.coord = coord
        self.file = decisions_file
        self.poll_s = poll_s
        self.offset = 0
        self.up: bool | None = None
        self.decisions = 0
        self.last_decision_t: float | None = None
        self.early: dict[str, list[dict]] = {}   # update_id → decisions that arrived before the report
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.touch(exist_ok=True)
        self.offset = self.file.stat().st_size  # only decisions made during this run
        self._tasks = [asyncio.create_task(self._tail()), asyncio.create_task(self._health())]

    def stop(self) -> None:
        for t in self._tasks:
            t.cancel()

    def status(self) -> dict:
        return {"up": bool(self.up), "decisions": self.decisions, "last_decision_t": self.last_decision_t,
                "dashboard_url": DASHBOARD}

    # ------------------------------------------------------------------ health
    async def check_api(self) -> bool:
        try:
            async with httpx.AsyncClient(verify=False, timeout=5) as c:
                r = await c.post(f"{API}/security/user/authenticate", auth=(API_USER, API_PASS))
            return r.status_code == 200 and "token" in r.text
        except httpx.HTTPError:
            return False

    async def _health(self) -> None:
        while True:
            ok = await self.check_api()
            if ok != self.up:
                self.up = ok
                self.coord.bus.emit("component", name="wazuh", status="up" if ok else "down",
                                    detail="Wazuh manager API reachable" if ok else
                                    "Wazuh manager API not reachable — no decisions can be made")
                if not ok:
                    self.coord.bus.emit("alert", severity="critical", title="Wazuh is down",
                                        detail="Updates will wait (and be excluded) until Wazuh decides.",
                                        source="coordinator")
            await asyncio.sleep(10)

    # ------------------------------------------------------------------ decisions
    async def _tail(self) -> None:
        buf = ""
        while True:
            try:
                size = self.file.stat().st_size
                if size < self.offset:  # file truncated/rotated
                    self.offset = 0
                if size > self.offset:
                    with self.file.open("r", encoding="utf-8", errors="replace") as f:
                        f.seek(self.offset)
                        chunk = f.read()
                        self.offset = f.tell()
                    buf += chunk
                    *lines, buf = buf.split("\n")
                    for line in lines:
                        if line.strip():
                            try:
                                await self.apply(json.loads(line))
                            except (ValueError, KeyError) as exc:
                                self.coord.bus.emit("alert", severity="medium", title="Unreadable Wazuh decision",
                                                    detail=repr(exc)[:200], source="coordinator")
            except OSError:
                pass
            await asyncio.sleep(self.poll_s)

    def on_gate(self, rec: dict) -> list[dict]:
        """Decisions that arrived before the device's report for this update (race), to apply now."""
        return self.early.pop(rec["update_id"], [])

    async def apply(self, d: dict) -> None:
        """Enforce one Wazuh decision record (written by demo/wazuh/fl-decision)."""
        c = self.coord
        self.decisions += 1
        self.last_decision_t = time.time()
        decision, cid, uid = d.get("decision", "alert"), d.get("client_id", ""), d.get("update_id", "")
        rule = f"wazuh rule {d.get('rule_id')} (level {d.get('rule_level')})"
        reason = f"[actor={rule}] {d.get('rule_description', '')}"[:1900]
        rec = c.updates.get(uid) if uid else None
        enforced, detail = False, ""

        if decision in ("approve", "quarantine", "suspend_device") and uid and rec is None:
            # FinSentinel+Wazuh were faster than the device's report: apply when the report arrives.
            self.early.setdefault(uid, []).append(d)
            return

        if decision == "approve":
            if rec.get("approval") in FINAL:
                detail = f"already {rec['approval']}"
            elif cid in c.suspended:
                r = await c.decide(cid, uid, "reject", actor=f"wazuh suspension ({c.suspended[cid]['rule_id']})",
                                   reason=f"[actor=wazuh] device suspended by rule {c.suspended[cid]['rule_id']}")
                enforced, detail = r["ok"], "device is suspended → rejected instead"
            elif rec.get("late"):
                detail = "late update — kept out of averaging"
            else:
                r = await c.decide(cid, uid, "approve", actor=rule, reason=reason)
                enforced, detail = r["ok"], "approved in FinSentinel" if r["ok"] else f"FinSentinel refused ({r.get('status')})"
        elif decision in ("quarantine", "suspend_device"):
            if rec.get("approval") not in FINAL:
                r = await c.decide(cid, uid, "quarantine", actor=rule, reason=reason)
                enforced, detail = r["ok"], "quarantined in FinSentinel" if r["ok"] else f"FinSentinel refused ({r.get('status')})"
            else:
                detail = f"already {rec['approval']}"
            if decision == "suspend_device" and cid in c.devices:
                c.suspend(cid, rule_id=str(d.get("rule_id")))
                enforced = True
                detail += "; device suspended"
        elif decision == "reject":
            if rec is not None and rec.get("approval") not in FINAL:
                r = await c.decide(cid, uid, "reject", actor=rule, reason=reason)
                enforced, detail = r["ok"], "rejected in FinSentinel"
            else:
                enforced, detail = True, "blocked at the FinSentinel gate — nothing was stored"
            uid = c.attack_ids.get(uid, uid)
            rec = None  # a replay reuses the honest update's id: don't attribute latency/round to it
        else:  # alert
            enforced, detail = True, "alert raised (no enforcement)"
            c.bus.emit("alert", severity="high" if int(d.get("rule_level", 0)) >= 10 else "medium",
                       title=f"Wazuh: {d.get('rule_description', '')[:90]}", detail=reason[:300],
                       device=cid or None, source="wazuh", event_type=d.get("event_type"))

        latency = round((time.time() - rec["t"]) * 1000) if rec else None
        c.bus.emit("wazuh_decision", device=cid, update_id=uid or None, round=rec["round"] if rec else None,
                   decision=decision, rule_id=str(d.get("rule_id")), rule_level=int(d.get("rule_level", 0)),
                   rule_description=d.get("rule_description", ""), event_type=d.get("event_type", ""),
                   enforced=enforced, enforcement_detail=detail, latency_ms=latency)
