import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent
for p in (DEMO, DEMO.parent / "finsentinel"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
