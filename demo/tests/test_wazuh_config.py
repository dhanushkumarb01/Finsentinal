"""Wazuh decision layer: rule/response config consistency and the active-response script."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.wazuh_standin import decide, load_table

DEMO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def table():
    return load_table()


def test_every_decision_rule_has_an_active_response(table):
    ids = {t["rule_id"] for t in table}
    assert {"100201", "100202", "100211", "100203", "100204", "100205", "100206", "100212",
            "100251", "100252", "100253"} <= ids


@pytest.mark.parametrize("event,decision,rule", [
    ({"event_type": "security.anomaly_clean", "severity": "info"}, "approve", "100201"),
    ({"event_type": "security.anomaly_detected", "severity": "high"}, "quarantine", "100202"),
    ({"event_type": "security.anomaly_detected", "severity": "medium"}, "quarantine", "100202"),
    ({"event_type": "security.anomaly_detected", "severity": "critical"}, "suspend_device", "100211"),
    ({"event_type": "security.replay_detected", "severity": "high"}, "reject", "100203"),
    ({"event_type": "security.signature_verification_failed", "severity": "high"}, "reject", "100204"),
    ({"event_type": "security.unknown_signing_key", "severity": "high"}, "reject", "100204"),
    ({"event_type": "security.payload_integrity_failed", "severity": "high"}, "reject", "100205"),
    ({"event_type": "fl.drift_dropout", "severity": "low"}, "alert", "100251"),
    ({"event_type": "fl.mitm_refused", "severity": "high"}, "alert", "100252"),
])
def test_rule_table_decisions(table, event, decision, rule):
    d = decide(table, {**event, "client_id": "phone-03", "update_id": "phone-03-r0001-ab"})
    assert d["decision"] == decision and d["rule_id"] == rule
    if rule != "100252":  # the MITM rule names no device
        assert "phone-03" in d["rule_description"]


def test_routine_events_get_no_decision(table):
    for et in ("security.update_authenticated", "security.update_pending_approval", "key.registered"):
        assert decide(table, {"event_type": et, "severity": "info"}) is None


def test_active_response_script_records_wazuh_message(tmp_path):
    msg = {"version": 1, "origin": {"name": "node01", "module": "wazuh-execd"}, "command": "add",
           "parameters": {"extra_args": ["quarantine"], "alert": {
               "timestamp": "2026-11-22T15:00:00.000+0000", "id": "1795.123",
               "rule": {"level": 12, "description": "FL decision: QUARANTINE update u1 from phone-03", "id": "100202"},
               "data": {"source": "finsentinel", "event_type": "security.anomaly_detected", "severity": "high",
                        "client_id": "phone-03", "update_id": "phone-03-r0004-aa", "idempotency_key": "k1"}},
               "program": "active-response/bin/fl-decision"}}
    out = tmp_path / "decisions.jsonl"
    p = subprocess.run([sys.executable, str(DEMO / "wazuh" / "fl-decision")], input=json.dumps(msg) + "\n",
                       text=True, capture_output=True, env={**os.environ, "FL_DECISIONS_FILE": str(out)})
    assert p.returncode == 0, p.stderr
    rec = json.loads(out.read_text().strip())
    assert rec["decision"] == "quarantine" and rec["rule_id"] == "100202" and rec["rule_level"] == 12
    assert rec["client_id"] == "phone-03" and rec["update_id"] == "phone-03-r0004-aa"


def test_wazuh_files_use_lf_line_endings():
    for f in ("wazuh.sh", "fl-decision"):
        assert b"\r\n" not in (DEMO / "wazuh" / f).read_bytes(), f
