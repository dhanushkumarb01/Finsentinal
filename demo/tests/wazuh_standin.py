"""TEST-ONLY stand-in for Wazuh, used when the real Wazuh stack is not running.

It applies the demo's REAL rule/response configuration — parsed from demo/wazuh/fl_decisions.xml,
demo/wazuh/manager_overlay.conf and FinSentinel's finsentinel_rules.xml — to FinSentinel's real
audit events and writes decision records in exactly the format of demo/wazuh/fl-decision. This
lets the tests exercise the coordinator's enforcement and catch inconsistent rule/response config.
It is NOT used by the demo itself; on stage the decisions come from Wazuh.
"""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent
FS_RULES = DEMO.parent / "finsentinel" / "deploy" / "wazuh" / "rules" / "finsentinel_rules.xml"
DEMO_RULES = DEMO / "wazuh" / "fl_decisions.xml"
OVERLAY = DEMO / "wazuh" / "manager_overlay.conf"


def _xml(path: Path) -> ET.Element:
    text = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.S)
    text = re.sub(r"<\?xml[^>]*\?>", "", text)
    return ET.fromstring(f"<root>{text}</root>")


def load_table() -> list[dict]:
    """[{rule_id, level, description, event_types, severity_regex, decision}] most specific first."""
    parent_event = {}
    for r in _xml(FS_RULES).iter("rule"):
        f = r.find("field[@name='event_type']")
        if f is not None:
            parent_event[r.get("id")] = f.text.strip().strip("^$").replace("\\.", ".")
    overlay = _xml(OVERLAY)
    cmd_decision = {c.findtext("name"): c.findtext("extra_args") for c in overlay.iter("command")}
    rule_cmd = {}
    for ar in overlay.iter("active-response"):
        for rid in ar.findtext("rules_id").split(","):
            rule_cmd[rid.strip()] = ar.findtext("command")
    table = []
    for r in _xml(DEMO_RULES).iter("rule"):
        rid = r.get("id")
        if rid not in rule_cmd:
            continue
        sev = r.find("field[@name='severity']")
        evf = r.find("field[@name='event_type']")
        parents = [p.strip() for p in (r.findtext("if_sid") or "").split(",") if p.strip()]
        events = [evf.text.strip("^$").replace("\\.", ".")] if evf is not None else [parent_event[p] for p in parents if p in parent_event]
        table.append({"rule_id": rid, "level": int(r.get("level")), "description": r.findtext("description"),
                      "event_types": events, "severity_regex": sev.text if sev is not None else None,
                      "decision": cmd_decision[rule_cmd[rid]], "source": "fl-coordinator" if evf is not None else "finsentinel"})
    # Wazuh fires the most specific (deepest) matching child; field-constrained rules first.
    table.sort(key=lambda t: t["severity_regex"] is None)
    return table


def decide(table: list[dict], event: dict) -> dict | None:
    for t in table:
        if event.get("event_type") in t["event_types"] and \
                (t["severity_regex"] is None or re.search(t["severity_regex"], event.get("severity", ""))):
            desc = t["description"].replace("$(client_id)", event.get("client_id") or "") \
                                   .replace("$(update_id)", event.get("update_id") or "")
            return {"t": time.time(), "command": "add", "decision": t["decision"], "rule_id": t["rule_id"],
                    "rule_level": t["level"], "rule_description": desc, "event_type": event["event_type"],
                    "client_id": event.get("client_id") or "", "update_id": event.get("update_id") or "",
                    "event_id": "", "alert_id": "standin", "alert_timestamp": ""}
    return None


def run_on_finsentinel_db(engine, out_file: Path, stop) -> None:
    """Poll FinSentinel's audit_events table (via its SQLAlchemy engine) and write decisions."""
    from sqlalchemy import text
    table = load_table()
    last = 0
    while not stop.is_set():
        with engine.connect() as c:
            rows = c.execute(text("SELECT id, event_type, severity, client_id, update_id FROM audit_events "
                                  "WHERE id > :last ORDER BY id"), {"last": last}).fetchall()
        for rid, et, sev, cid, uid in rows:
            last = rid
            d = decide(table, {"event_type": et, "severity": sev or "", "client_id": cid, "update_id": uid})
            if d:
                with out_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(d) + "\n")
        time.sleep(0.3)
