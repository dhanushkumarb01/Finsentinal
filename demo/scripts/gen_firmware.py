"""Generate the Wokwi ESP32 board projects demo/wokwi/board1..3/.

Run:  demo/.venv/Scripts/python demo/scripts/gen_firmware.py

KEY-FILE LAYOUT (shared contract with the demo launcher, which reads exactly these):

    demo/.state/keys/<id>.seed.hex   Ed25519 private seed, 32 bytes as 64 lowercase hex chars
    demo/.state/keys/<id>.pub.hex    Ed25519 raw public key, 32 bytes as 64 lowercase hex chars
        <id> in: phone-01 .. phone-06, esp32-01 .. esp32-03, coordinator
    demo/.state/beacon.json          {"topic": "fl-demo-<24 hex>"}   (ntfy.sh beacon topic)

Missing files are created here (fresh keys from the `cryptography` package; a missing
.pub.hex is derived from an existing .seed.hex; existing files are never overwritten), so the
launcher and the firmware always agree. Signing key id in FinSentinel = "<id>-key".

Outputs per board N (esp32-0N):
    sketch.ino, fl_*.h, ca_roots.h    copied from demo/wokwi/common/ (browser Wokwi needs every file)
    board_config.h                    committed: BOARD_ID, label, subject, SIGNING_KEY_ID (no secrets)
    board_secrets.h                   GITIGNORED: DEVICE_SEED, COORD_PUBKEY, BEACON_TOPIC
    board_data.h                      committed: the board's HAR subject as int8 (x*127) in flash
    diagram.json, libraries.txt, wokwi.toml, platformio.ini, README.md
Also regenerates demo/wokwi/common/ca_roots.h from certifi.
"""

from __future__ import annotations

import json
import secrets
import shutil
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO))

import numpy as np  # noqa: E402

from fldemo.har import ACTIVITY_NAMES, subject_arrays  # noqa: E402

WOKWI = DEMO / "wokwi"
COMMON = WOKWI / "common"
STATE = DEMO / ".state"
KEYS = STATE / "keys"
BEACON = STATE / "beacon.json"
SPLIT = DEMO / "config" / "har_split.json"

DEVICE_IDS = [f"phone-{i:02d}" for i in range(1, 7)] + [f"esp32-{i:02d}" for i in range(1, 4)] + ["coordinator"]
COMMON_FILES = ["sketch.ino", "fl_app.h", "fl_model.h", "fl_npz.h", "fl_json.h", "fl_b64.h", "fl_sha256.h", "ca_roots.h"]
CA_NAMES = [
    "ISRG Root X1", "ISRG Root X2",
    "GTS Root R1", "GTS Root R2", "GTS Root R3", "GTS Root R4",
    "DigiCert Global Root G2", "DigiCert Global Root CA",
    "USERTrust RSA Certification Authority", "USERTrust ECC Certification Authority",
    "Baltimore CyberTrust Root",
]
LIBRARIES = ["Crypto", "Adafruit SSD1306", "Adafruit GFX Library", "Adafruit MPU6050",
             "Adafruit Unified Sensor", "Adafruit BusIO"]


def warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- keys
def ensure_keys() -> dict[str, tuple[bytes, bytes]]:
    from cryptography.hazmat.primitives import serialization as ser
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    KEYS.mkdir(parents=True, exist_ok=True)
    out = {}
    for dev in DEVICE_IDS:
        seed_f, pub_f = KEYS / f"{dev}.seed.hex", KEYS / f"{dev}.pub.hex"
        if seed_f.exists():
            seed = bytes.fromhex(seed_f.read_text().strip())
            assert len(seed) == 32, f"{seed_f}: expected 32 bytes"
        else:
            seed = Ed25519PrivateKey.generate().private_bytes(ser.Encoding.Raw, ser.PrivateFormat.Raw, ser.NoEncryption())
            seed_f.write_text(seed.hex() + "\n")
            print(f"generated key {seed_f.relative_to(DEMO)}")
        pub = Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes(ser.Encoding.Raw, ser.PublicFormat.Raw)
        if pub_f.exists():
            if bytes.fromhex(pub_f.read_text().strip()) != pub:
                raise SystemExit(f"{pub_f} does not match {seed_f}; fix or delete one of them.")
        else:
            pub_f.write_text(pub.hex() + "\n")
        out[dev] = (seed, pub)
    return out


def ensure_beacon() -> str:
    data = {}
    if BEACON.exists():
        data = json.loads(BEACON.read_text() or "{}")
    if not data.get("topic"):
        data["topic"] = "fl-demo-" + secrets.token_hex(12)
        BEACON.parent.mkdir(parents=True, exist_ok=True)
        BEACON.write_text(json.dumps(data, indent=1) + "\n")
        print(f"generated beacon topic in {BEACON.relative_to(DEMO)}")
    return data["topic"]


# --------------------------------------------------------------------------- CA roots
def gen_ca_roots() -> None:
    import certifi
    from cryptography import x509
    from cryptography.hazmat.primitives.serialization import Encoding
    from cryptography.x509.oid import NameOID

    text = Path(certifi.where()).read_text(encoding="ascii")
    pems = []
    for block in text.split("-----END CERTIFICATE-----")[:-1]:
        pem = block[block.index("-----BEGIN CERTIFICATE-----"):] + "-----END CERTIFICATE-----\n"
        pems.append(pem)
    by_cn = {}
    for pem in pems:
        cert = x509.load_pem_x509_certificate(pem.encode())
        cns = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
        if cns:
            by_cn.setdefault(cns[0].value, cert)
    lines = [
        "// GENERATED by demo/scripts/gen_firmware.py from certifi " + certifi.__version__ + " - do not edit.",
        "// Public root CAs for WiFiClientSecure::setCACert (laptop tunnel + ntfy.sh).",
        "#pragma once",
        "static const char CA_ROOTS[] =",
    ]
    found = 0
    for name in CA_NAMES:
        cert = by_cn.get(name)
        if cert is None:
            warn(f"CA root '{name}' not found in certifi; skipped")
            continue
        found += 1
        lines.append(f"    // {name} (expires {cert.not_valid_after_utc.date()})")
        for ln in cert.public_bytes(Encoding.PEM).decode().strip().splitlines():
            lines.append(f'    "{ln}\\n"')
    lines[-1] += ";"
    (COMMON / "ca_roots.h").write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    print(f"ca_roots.h: {found}/{len(CA_NAMES)} roots")


# --------------------------------------------------------------------------- per-board files
def c_bytes(b: bytes) -> str:
    return ", ".join(f"0x{x:02x}" for x in b)


def board_data_h(dev: str, subject: int) -> str:
    x, y = subject_arrays([subject])
    q = np.clip(np.round(x * 127.0), -127, 127).astype(np.int8)
    rows = ",\n".join("  {" + ",".join(str(int(v)) for v in row) + "}" for row in q)
    counts = np.bincount(y, minlength=6).tolist()
    names = ", ".join(f'"{n}"' for n in ACTIVITY_NAMES)
    return f"""// GENERATED by demo/scripts/gen_firmware.py - do not edit.
// UCI HAR subject {subject} for {dev}: 64 features (demo/config/har_features.json order),
// stored as int8 = round(x * 127) (features are in [-1, 1]); kept in flash (PROGMEM).
// Class counts: {counts}
#pragma once
#include <stdint.h>
#if defined(ARDUINO)
#include <pgmspace.h>
#else
#define PROGMEM
#endif

#define FL_N {len(y)}
static const char* const FL_ACTIVITY[6] = {{{names}}};
static const uint8_t FL_Y[FL_N] PROGMEM = {{{",".join(str(int(v)) for v in y)}}};
static const int8_t FL_X[FL_N][64] PROGMEM = {{
{rows}
}};
"""


def board_config_h(n: int, dev: str, subject: int) -> str:
    return f"""// GENERATED by demo/scripts/gen_firmware.py - committed, contains NO secrets.
#pragma once
#define BOARD_ID "{dev}"
#define BOARD_LABEL "ESP32 board {n}"
#define BOARD_NUM {n}
#define BOARD_SUBJECT {subject}
#define SIGNING_KEY_ID "{dev}-key"
"""


def board_secrets_h(dev: str, seed: bytes, coord_pub: bytes, topic: str) -> str:
    return f"""// GENERATED by demo/scripts/gen_firmware.py from demo/.state/ - GITIGNORED, never commit.
// Regenerate after any key / beacon change. Device private key: anyone holding it can sign as {dev}.
#pragma once
#include <stdint.h>
static const uint8_t DEVICE_SEED[32] = {{{c_bytes(seed)}}};
// Pinned coordinator public key (verifies the beacon and every global model).
static const uint8_t COORD_PUBKEY[32] = {{{c_bytes(coord_pub)}}};
#define BEACON_TOPIC "{topic}"
"""


def diagram_json(n: int) -> str:
    parts = [
        {"type": "board-esp32-devkit-c-v4", "id": "esp", "top": 0, "left": 0, "attrs": {}},
        {"type": "board-ssd1306", "id": "oled1", "top": -150, "left": 160, "attrs": {"i2cAddress": "0x3c"}},
        {"type": "wokwi-mpu6050", "id": "imu1", "top": -150, "left": 320, "attrs": {}},
        {"type": "wokwi-led", "id": "ledG", "top": 40, "left": 250, "attrs": {"color": "green", "label": "accepted"}},
        {"type": "wokwi-resistor", "id": "rG", "top": 110, "left": 200, "rotate": 90, "attrs": {"value": "220"}},
        {"type": "wokwi-led", "id": "ledR", "top": 40, "left": 310, "attrs": {"color": "red", "label": "rejected"}},
        {"type": "wokwi-resistor", "id": "rR", "top": 110, "left": 260, "rotate": 90, "attrs": {"value": "220"}},
        {"type": "wokwi-potentiometer", "id": "bat", "top": 190, "left": -160, "attrs": {"value": "800", "label": "battery"}},
        {"type": "wokwi-pushbutton", "id": "btnOff", "top": 200, "left": 200, "attrs": {"color": "blue", "label": "offline", "bounce": "0"}},
        {"type": "wokwi-slide-switch", "id": "swHack", "top": 280, "left": 200, "attrs": {"value": "", "label": "HACKED"}},
    ]
    conns = [
        ["esp:TX", "$serialMonitor:RX", "", []],
        ["esp:RX", "$serialMonitor:TX", "", []],
        # I2C bus: OLED + MPU6050
        ["oled1:SDA", "esp:21", "blue", []],
        ["oled1:SCL", "esp:22", "orange", []],
        ["oled1:VCC", "esp:3V3", "red", []],
        ["oled1:GND", "esp:GND.2", "black", []],
        ["imu1:SDA", "esp:21", "blue", []],
        ["imu1:SCL", "esp:22", "orange", []],
        ["imu1:VCC", "esp:3V3", "red", []],
        ["imu1:GND", "esp:GND.2", "black", []],
        # LEDs: GPIO -> resistor -> anode, cathode -> GND
        ["esp:25", "rG:1", "green", []],
        ["rG:2", "ledG:A", "green", []],
        ["ledG:C", "esp:GND.1", "black", []],
        ["esp:26", "rR:1", "red", []],
        ["rR:2", "ledR:A", "red", []],
        ["ledR:C", "esp:GND.1", "black", []],
        # Battery potentiometer on GPIO 34 (ADC1)
        ["bat:VCC", "esp:3V3", "red", []],
        ["bat:GND", "esp:GND.1", "black", []],
        ["bat:SIG", "esp:34", "violet", []],
        # Offline push-button: GPIO 27 (INPUT_PULLUP) to GND
        ["btnOff:1.l", "esp:27", "blue", []],
        ["btnOff:2.l", "esp:GND.1", "black", []],
        # HACKED slide switch: common (2) -> GPIO 14 (INPUT_PULLUP); left (1) -> GND = normal,
        # right (3) open -> pulled HIGH = HACKED
        ["swHack:2", "esp:14", "magenta", []],
        ["swHack:1", "esp:GND.1", "black", []],
    ]
    return json.dumps({"version": 1, "author": "fl-demo", "editor": "wokwi", "parts": parts,
                       "connections": conns, "dependencies": {}}, indent=2) + "\n"


WOKWI_TOML = """[wokwi]
version = 1
firmware = '.pio/build/esp32dev/firmware.bin'
elf = '.pio/build/esp32dev/firmware.elf'
"""

PLATFORMIO_INI = """; Local build (VS Code + PlatformIO + Wokwi extension). Browser Wokwi does not need this.
; arduino-esp32 3.x comes from the pioarduino platform (the official one is still on 2.x).
[platformio]
src_dir = .

[env:esp32dev]
platform = https://github.com/pioarduino/platform-espressif32/releases/download/stable/platform-espressif32.zip
board = esp32dev
framework = arduino
monitor_speed = 115200
build_flags = -Os
lib_deps =
    rweather/Crypto
    adafruit/Adafruit SSD1306
    adafruit/Adafruit GFX Library
    adafruit/Adafruit MPU6050
    adafruit/Adafruit Unified Sensor
    adafruit/Adafruit BusIO
"""


def readme(n: int, dev: str, subject: int, files: list[str]) -> str:
    file_list = "\n".join(f"   - `{f}`" for f in files)
    return f"""# ESP32 board {n} (`{dev}`, HAR subject {subject}) - Wokwi

Generated by `demo/scripts/gen_firmware.py`. Do not edit by hand: change `demo/wokwi/common/`
and run the generator again.

## Run it in browser Wokwi

1. Run `demo/.venv/Scripts/python demo/scripts/gen_firmware.py` on the laptop. It also writes
   `board_secrets.h`, which holds this board's private key and is never committed.
2. Open https://wokwi.com, choose **New Project → ESP32**.
3. Replace `sketch.ino` and `diagram.json` with the files from this folder. Then use the
   file menu (▾ next to the tabs) → **Upload file(s)…** to add every other file:
{file_list}
4. Press **Play**. The board joins `Wokwi-GUEST` (this can take 1–2 minutes), syncs the time,
   reads the signed beacon from ntfy.sh, and shows **Laptop found**. It then polls the
   laptop every 5 s.

## Controls

| Part | Meaning |
|---|---|
| Potentiometer (GPIO 34) | Battery level 0–100 %. Below 15 %, the board skips the round (`low_battery`). |
| Blue button (GPIO 27) | Toggle offline. The OLED shows OFFLINE and the board stops taking part. |
| Slide switch (GPIO 14) | Right = **HACKED**. The board multiplies its weights by 8 (poisoned update) and the OLED badge blinks. |
| Green LED (GPIO 25) | FinSentinel accepted the last update (pending approval or approved). |
| Red LED (GPIO 26) | Rejected, quarantined, or TLS refused. Both LEDs are off while the board trains. |

OLED lines: `R<round> <state>`, `loss`, `bat % / rssi`, last verdict, accelerometer magnitude.

## Local build (optional)

With VS Code, PlatformIO and the Wokwi extension: open this folder, run `pio run`, then
start the simulator. `wokwi.toml` points at the build output.
"""


def main() -> None:
    keys = ensure_keys()
    topic = ensure_beacon()
    gen_ca_roots()
    split = json.loads(SPLIT.read_text())["wokwi_boards"]
    coord_pub = keys["coordinator"][1]
    for n in (1, 2, 3):
        dev = f"esp32-{n:02d}"
        subject = int(split[dev])
        out = WOKWI / f"board{n}"
        out.mkdir(parents=True, exist_ok=True)
        for f in COMMON_FILES:
            shutil.copyfile(COMMON / f, out / f)
        gen = {
            "board_config.h": board_config_h(n, dev, subject),
            "board_secrets.h": board_secrets_h(dev, keys[dev][0], coord_pub, topic),
            "board_data.h": board_data_h(dev, subject),
            "diagram.json": diagram_json(n),
            "libraries.txt": "# Wokwi library manager\n" + "\n".join(LIBRARIES) + "\n",
            "wokwi.toml": WOKWI_TOML,
            "platformio.ini": PLATFORMIO_INI,
        }
        files = [f for f in COMMON_FILES if f != "sketch.ino"] + [
            "board_config.h", "board_secrets.h", "board_data.h", "libraries.txt"]
        gen["README.md"] = readme(n, dev, subject, files)
        for name, text in gen.items():
            (out / name).write_text(text, encoding="utf-8", newline="\n")
        print(f"{out.relative_to(DEMO)}: {dev}, subject {subject}")
    print(f"beacon topic: {topic}  |  coordinator pub: {coord_pub.hex()}")


if __name__ == "__main__":
    main()
