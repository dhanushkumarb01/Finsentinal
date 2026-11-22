"""Key and certificate material for the demo (all under demo/.state, gitignored).

Layout (shared with demo/scripts/gen_firmware.py):
  .state/keys/<id>.seed.hex   Ed25519 private seed (32 bytes, hex) for every device and 'coordinator'
  .state/keys/<id>.pub.hex    matching raw public key (32 bytes, hex)
  .state/beacon.json          {"topic": "fl-demo-<24 hex>"}  ntfy topic for the tunnel beacon
  .state/secrets.json         operator / admin / webhook tokens
  .state/tls/                 demo CA + server cert for localhost (Caddy)
"""

from __future__ import annotations

import base64
import datetime as dt
import ipaddress
import json
import secrets

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.x509.oid import NameOID

from fldemo.settings import (
    BEACON_FILE, CA_CERT, KEYS_DIR, REGISTRY_FILE, SECRETS_FILE, SERVER_CERT, SERVER_KEY,
)

COORDINATOR = "coordinator"


def _raw_pub(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def load_or_create_key(key_id: str) -> Ed25519PrivateKey:
    seed_file = KEYS_DIR / f"{key_id}.seed.hex"
    if seed_file.exists():
        key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_file.read_text().strip()))
    else:
        key = Ed25519PrivateKey.generate()
        seed = key.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()
        )
        seed_file.write_text(seed.hex())
    pub_file = KEYS_DIR / f"{key_id}.pub.hex"
    pub_hex = _raw_pub(key).hex()
    if not pub_file.exists() or pub_file.read_text().strip() != pub_hex:
        pub_file.write_text(pub_hex)
    return key


def public_key(key_id: str) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(bytes.fromhex((KEYS_DIR / f"{key_id}.pub.hex").read_text().strip()))


def coordinator_pub_hex() -> str:
    load_or_create_key(COORDINATOR)
    return (KEYS_DIR / f"{COORDINATOR}.pub.hex").read_text().strip()


def sign_b64(key: Ed25519PrivateKey, message: str) -> str:
    return base64.b64encode(key.sign(message.encode("utf-8"))).decode("ascii")


def verify_b64(pub: Ed25519PublicKey, message: str, sig_b64: str) -> bool:
    try:
        pub.verify(base64.b64decode(sig_b64, validate=True), message.encode("utf-8"))
        return True
    except Exception:
        return False


def model_sig_message(model_version: str, round_number: int, sha256_hex: str) -> str:
    return f"fl-model|{model_version}|{round_number}|{sha256_hex}"


def beacon_message(url: str, ts: int) -> str:
    return f"FLB1|{url}|{ts}"


def ensure_beacon_topic() -> str:
    if BEACON_FILE.exists():
        return json.loads(BEACON_FILE.read_text())["topic"]
    topic = f"fl-demo-{secrets.token_hex(12)}"
    BEACON_FILE.write_text(json.dumps({"topic": topic}))
    return topic


def ensure_secrets() -> dict:
    data = json.loads(SECRETS_FILE.read_text()) if SECRETS_FILE.exists() else {}
    changed = False
    for name in ("operator_token", "admin_token", "webhook_token", "forwarder_token"):
        if len(data.get(name, "")) < 32:
            data[name] = secrets.token_hex(32)
            changed = True
    if changed:
        SECRETS_FILE.write_text(json.dumps(data, indent=1))
    return data


def ensure_device_keys(device_ids: list[str]) -> dict[str, str]:
    """Create missing keys; write the public-key registry FinSentinel loads at start."""
    registry = {}
    for did in device_ids:
        load_or_create_key(did)
        registry[did] = (KEYS_DIR / f"{did}.pub.hex").read_text().strip()
    load_or_create_key(COORDINATOR)
    REGISTRY_FILE.write_text(json.dumps(registry, indent=1))
    return registry


def ensure_tls() -> None:
    """Demo CA + server cert for localhost/127.0.0.1 (ECDSA P-256, 1-year validity)."""
    if CA_CERT.exists() and SERVER_CERT.exists() and SERVER_KEY.exists():
        cert = x509.load_pem_x509_certificate(SERVER_CERT.read_bytes())
        if cert.not_valid_after_utc > dt.datetime.now(dt.UTC) + dt.timedelta(days=7):
            return
    now = dt.datetime.now(dt.UTC) - dt.timedelta(minutes=5)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "FL Demo Local CA")])
    ca = (
        x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name)
        .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now).not_valid_after(now + dt.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, False, False), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]))
        .issuer_name(ca_name).public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now).not_valid_after(now + dt.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName([
            x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
        .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    CA_CERT.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    SERVER_CERT.write_bytes(cert.public_bytes(serialization.Encoding.PEM)
                            + ca.public_bytes(serialization.Encoding.PEM))
    SERVER_KEY.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
