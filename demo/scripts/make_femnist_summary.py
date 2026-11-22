"""Build demo/config/femnist.json for Mission Control's research slide (scene 7). READ-ONLY on the
research results — nothing in drift-causal-fl/ is modified.

Source: the completed Day-5 FEMNIST pilot (25 clients, 25 rounds, seed 0, model-independent drift
estimator) plus any finished runs of the MVP sweep (100 clients, 100 rounds), each labelled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent
RES = DEMO.parent / "drift-causal-fl" / "results"
OUT = DEMO / "config" / "femnist.json"
NAMES = {"mcar": "Random dropout (MCAR)", "mar": "Dropout explained by observables (MAR)",
         "mnar": "Drift-causal dropout (MNAR)", "mnar_threshold": "Drift-causal, threshold (MNAR-thr.)"}


def main() -> int:
    pilot = RES / "day5_v3" / "day5_v3_model_independent_summary.json"
    series = []
    if pilot.exists():
        s = json.loads(pilot.read_text())["summaries"]
        for mech in ("mcar", "mar", "mnar", "mnar_threshold"):
            row = s.get(f"{mech}_fedavg")
            if row:
                series.append({"label": f"FedAvg · {NAMES[mech]}", "final_accuracy": round(row["final_acc_global"], 4),
                               "drift_cohort_accuracy": round(row["final_acc_drift_cohort"], 4)})
    mvp = []
    for d in sorted((RES / "mvp").glob("s*__*")):
        f = d / "result.json"
        if not f.exists():
            continue
        try:
            r = json.loads(f.read_text())
            last = r["rounds"][-1]
            if last["round"] < r["config"]["rounds"]:
                continue
            mvp.append({"label": f"MVP {r['config']['method']} · {NAMES.get(r['config']['mechanism'], r['config']['mechanism'])}",
                        "final_accuracy": round(last["acc_global"], 4),
                        "drift_cohort_accuracy": round(last["acc_drift_cohort"], 4)})
        except (KeyError, ValueError, IndexError):
            continue
    if not series and not mvp:
        print("no research results found")
        return 1
    OUT.write_text(json.dumps({
        "title": "FEMNIST study: does drift-caused absence change what FL learns?",
        "note": "Separate simulation study (FEMNIST handwriting, 62 classes), not produced by this demo. "
                "Pilot = 25 clients x 25 rounds, seed 0; MVP sweep rows appear only for finished runs "
                "(sweep in progress). Same dropout rate in every row — only the cause differs.",
        "series": series + mvp,
        "source": "drift-causal-fl/results/day5_v3 (pilot) and results/mvp (finished runs only)",
    }, indent=1))
    print(f"wrote {OUT} ({len(series)} pilot rows, {len(mvp)} MVP rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
