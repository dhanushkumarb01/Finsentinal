"""Full rounds in-process: FinSentinel + coordinator + 6 device agents + attacks (subprocess-isolated)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def summary(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("it")
    env = {**os.environ, "FLDEMO_STATE_DIR": str(tmp / "state"), "FLDEMO_RECORDINGS_DIR": str(tmp / "rec"),
           "IT_ROUNDS": "3", "OMP_NUM_THREADS": "1"}
    p = subprocess.run([sys.executable, str(HERE / "integration_round.py")], env=env, capture_output=True,
                       text=True, timeout=600)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("SUMMARY ")), None)
    assert line, f"no summary\nSTDOUT:{p.stdout[-3000:]}\nSTDERR:{p.stderr[-5000:]}"
    return json.loads(line[8:])


def test_all_rounds_complete_and_aggregate(summary):
    assert summary["state"] == "finished"
    assert summary["rounds_completed"] == 3
    assert summary["aggregated_rounds"] >= 2


def test_global_accuracy_rises(summary):
    acc = summary["accuracy"]
    assert acc[-1] > acc[0] + 0.2, acc


def test_clean_updates_are_approved_by_assisted_policy(summary):
    assert summary["approvals"] >= 10


def test_attacks_get_real_finsentinel_outcomes(summary):
    assert summary["attacks"] == {"replay": 409, "tamper": 422, "forge": 401}


def test_poisoned_update_is_quarantined(summary):
    assert "phone-03" in summary["quarantined"]


def test_recording_contains_every_event(summary):
    assert summary["recording_lines"] == summary["events"]
    for t in ("run_started", "round_started", "local_training", "update_submitted", "gate_result",
              "approval", "round_completed", "global_eval", "attack", "device_status", "run_finished"):
        assert t in summary["event_types"], t
