"""Run an attack scene against the running demo from a terminal.

    demo\\.venv\\Scripts\\python demo\\attack.py poison [--target phone-03]
    scenes: poison replay tamper forge mitm drift weak_signal low_battery offline restore

The same code path as Mission Control's attack console: the coordinator fires real requests at
the real FinSentinel and the outcome appears in the live alert feed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402

from fldemo.attacks import SCENES  # noqa: E402
from fldemo.settings import COORD_PORT, load_secrets  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", choices=sorted(SCENES))
    ap.add_argument("--target", help="device id, e.g. phone-03")
    args = ap.parse_args()
    try:
        r = httpx.post(f"http://127.0.0.1:{COORD_PORT}/fl/admin/attack/{args.scene}", timeout=60,
                       headers={"Authorization": f"Bearer {load_secrets()['admin_token']}"},
                       json={"target": args.target})
    except httpx.HTTPError as exc:
        print(f"Coordinator not reachable ({exc}). Is run_demo.py running?")
        return 1
    print(json.dumps(r.json(), indent=1))
    return 0 if r.status_code == 200 and r.json().get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
