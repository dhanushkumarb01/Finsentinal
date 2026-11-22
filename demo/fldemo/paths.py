"""Filesystem locations used by the demo. Everything lives under demo/."""

from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = DEMO_DIR.parent
FINSENTINEL_DIR = REPO_ROOT / "finsentinel"

DATA_DIR = DEMO_DIR / ".data"            # gitignored: downloaded dataset
STATE_DIR = DEMO_DIR / ".state"          # gitignored: SQLite DB, keys, tokens, logs
CONFIG_DIR = DEMO_DIR / "config"         # committed: split + feature subset
RECORDINGS_DIR = DEMO_DIR / "recordings"
TOOLS_DIR = DEMO_DIR / "tools"           # gitignored: helper binaries

HAR_ZIP = DATA_DIR / "download" / "har.zip"
HAR_ROOT = DATA_DIR / "extract" / "UCI HAR Dataset"
HAR_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
# SHA-256 of the outer archive as downloaded on 2026-11-22.
HAR_ZIP_SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"

SPLIT_FILE = CONFIG_DIR / "har_split.json"
FEATURES_FILE = CONFIG_DIR / "har_features.json"
