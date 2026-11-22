"""One-time setup for the live FL demo.

    python demo/setup.py            (any Python 3.11+; creates and uses demo/.venv)
    python demo/setup.py --yes      (do not ask before downloading helper tools)

Steps: venv → Python packages (FinSentinel's pinned deps, CPU torch) → UCI HAR (checksummed) →
helper tools into demo/tools (asks first; checksummed) → Mission Control build → keys, TLS,
firmware headers → threshold calibration (if missing) → FEMNIST summary for the research slide.
Never touches drift-causal-fl/ or finsentinel/ (FinSentinel is used read-only via PYTHONPATH).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

DEMO = Path(__file__).resolve().parent
VENV = DEMO / ".venv"
VPY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
TOOLS = DEMO / "tools"

TOOL_DOWNLOADS = [  # (file, url, sha256 of the downloaded file)
    ("cloudflared.exe", "https://github.com/cloudflare/cloudflared/releases/download/2026.11.1/cloudflared-windows-amd64.exe",
     "341deda3db5ec498a0482254d6c38c0b7cf01a30c37bb5dd95d2216d08955e35"),
    ("toxiproxy-server.exe", "https://github.com/Shopify/toxiproxy/releases/download/v2.13.0/toxiproxy-server-windows-amd64.exe",
     "04b710eefd7690c89f7cc67289e3e14618f1ff3c26e4a5f752a34a879b895bab"),
    ("toxiproxy-cli.exe", "https://github.com/Shopify/toxiproxy/releases/download/v2.13.0/toxiproxy-cli-windows-amd64.exe",
     "6585d19999abfc1d0c3b3276343ddfe81fbb4b50883ce810337678243a90499b"),
    ("caddy.exe", "https://github.com/caddyserver/caddy/releases/download/v2.11.9/caddy_2.11.9_windows_amd64.zip",
     "2622e239c79df700b0eb92cbd950d1edee365e155a7859f358f0e55f002df591"),  # sha256 of caddy.exe inside the zip
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, **kw) -> None:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def step(title: str) -> None:
    print(f"\n== {title}", flush=True)


def ensure_venv() -> None:
    if Path(sys.executable).resolve() == VPY.resolve():
        return
    if not VPY.exists():
        step("Creating demo/.venv")
        run([sys.executable, "-m", "venv", VENV])
    os.execv(str(VPY), [str(VPY), __file__, *sys.argv[1:]])


def install_packages() -> None:
    step("Installing Python packages (CPU-only torch + FinSentinel's pinned dependencies)")
    lock = (DEMO.parent / "finsentinel" / "requirements.lock").read_text().splitlines()
    torch_pin = next((l for l in lock if l.lower().startswith("torch==")), "torch")
    reqs = [l for l in lock if l.strip() and not l.startswith("#")
            and not l.lower().startswith(("psycopg", "torch=="))]
    req_file = DEMO / ".state" / "requirements.generated.txt"
    req_file.parent.mkdir(parents=True, exist_ok=True)
    req_file.write_text("\n".join(reqs + (DEMO / "requirements.txt").read_text().splitlines()) + "\n")
    run([VPY, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run([VPY, "-m", "pip", "install", "-q", torch_pin, "--index-url", "https://download.pytorch.org/whl/cpu"])
    run([VPY, "-m", "pip", "install", "-q", "-r", req_file])


def download_har() -> None:
    step("UCI HAR dataset")
    sys.path.insert(0, str(DEMO))
    from fldemo.paths import HAR_ROOT, HAR_URL, HAR_ZIP, HAR_ZIP_SHA256
    if not HAR_ZIP.exists() or sha256(HAR_ZIP) != HAR_ZIP_SHA256:
        HAR_ZIP.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {HAR_URL} (~58 MB; UCI can be slow)")
        urllib.request.urlretrieve(HAR_URL, HAR_ZIP)
    got = sha256(HAR_ZIP)
    if got != HAR_ZIP_SHA256:
        raise SystemExit(f"  ✗ HAR checksum mismatch: {got}")
    print("  ✓ checksum ok")
    if not HAR_ROOT.exists():
        extract = HAR_ROOT.parent
        extract.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(HAR_ZIP) as outer:
            outer.extract("UCI HAR Dataset.zip", extract)
        with zipfile.ZipFile(extract / "UCI HAR Dataset.zip") as inner:
            for m in inner.namelist():
                if not m.startswith("__MACOSX"):
                    inner.extract(m, extract)
        (extract / "UCI HAR Dataset.zip").unlink()
    from fldemo.har import subject_arrays
    subject_arrays([1])  # builds the fast cache
    print("  ✓ extracted + cached")


def fetch_tools(yes: bool) -> None:
    step("Helper tools (demo/tools, gitignored)")
    missing = [t for t in TOOL_DOWNLOADS if not (TOOLS / t[0]).exists() or sha256(TOOLS / t[0]) != t[2]]
    if not missing:
        print("  ✓ all present and verified")
        return
    print("  These official free binaries will be downloaded from GitHub releases:")
    for name, url, _ in missing:
        print(f"    - {name:22s} {url}")
    if not yes and input("  Download them now? [y/N] ").strip().lower() != "y":
        print("  skipped — run_demo.py needs caddy.exe and toxiproxy-server.exe")
        return
    TOOLS.mkdir(exist_ok=True)
    for name, url, digest in missing:
        dst = TOOLS / name
        if url.endswith(".zip"):
            tmp = TOOLS / "download.zip"
            urllib.request.urlretrieve(url, tmp)
            with zipfile.ZipFile(tmp) as z:
                z.extract(name, TOOLS)
            tmp.unlink()
        else:
            urllib.request.urlretrieve(url, dst)
        ok = sha256(dst) == digest
        print(f"  {'✓' if ok else '✗ CHECKSUM MISMATCH'} {name}")
        if not ok:
            dst.unlink()
            raise SystemExit(1)


def build_mission_control() -> None:
    step("Mission Control (npm ci + build)")
    mc = DEMO / "mission-control"
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    if not npm:
        print("  ✗ npm not found — install Node.js 20+ and re-run")
        return
    run([npm, "ci", "--no-audit", "--no-fund"], cwd=mc)
    run([npm, "run", "build"], cwd=mc)


def generate_material() -> None:
    step("Keys, TLS certificates, firmware headers")
    run([VPY, "-c", "import sys; sys.path.insert(0, r'%s'); from fldemo import keys as K; "
                    "from fldemo.settings import load_devices; K.ensure_secrets(); "
                    "K.ensure_device_keys([d['id'] for d in load_devices()]); K.ensure_tls(); "
                    "K.ensure_beacon_topic(); print('  ok')" % DEMO])
    if (DEMO / "scripts" / "gen_firmware.py").exists():
        run([VPY, DEMO / "scripts" / "gen_firmware.py"])
    if not (DEMO / "config" / "thresholds.json").exists():
        step("Calibrating detection thresholds (~10 min)")
        run([VPY, DEMO / "scripts" / "calibrate_thresholds.py", "--rounds", "30", "--drift-round", "10"])
    if (DEMO / "scripts" / "make_femnist_summary.py").exists():
        run([VPY, DEMO / "scripts" / "make_femnist_summary.py"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="download helper tools without asking")
    ap.add_argument("--skip-npm", action="store_true")
    args = ap.parse_args()
    ensure_venv()
    install_packages()
    download_har()
    fetch_tools(args.yes)
    if not args.skip_npm:
        build_mission_control()
    generate_material()
    print("\n✓ Setup complete. Next: demo\\.venv\\Scripts\\python demo\\run_demo.py")


if __name__ == "__main__":
    main()
