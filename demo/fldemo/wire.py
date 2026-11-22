"""Device-side wire format, implemented independently of FinSentinel.

This is the *reference implementation* the ESP32 firmware mirrors:

* ``build_npz_stored`` writes an NPZ (a ZIP of .npy files) with STORED
  (uncompressed) entries by hand — no ``zipfile``/``numpy.savez`` — so the C++
  port can reproduce the exact bytes.
* ``canonical_envelope`` reproduces FinSentinel's signed-data format:
  ``json.dumps(envelope_without_signature, sort_keys=True,
  separators=(",", ":"), ensure_ascii=False)`` after Pydantic normalised
  ``created_at`` to UTC. Devices send ``created_at`` as
  ``YYYY-MM-DDTHH:MM:SSZ`` (UTC, whole seconds), which Pydantic re-serialises
  to the identical string. Test vectors generated from FinSentinel's own code
  (``tests/vectors``) pin this down.
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import zlib
from collections.abc import Mapping
from datetime import UTC, datetime

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

MANIFEST_KEY = "__manifest__"
# Fixed DOS timestamp (1980-01-01 00:00) so payload bytes are deterministic.
_DOS_TIME, _DOS_DATE = 0, (0 << 9) | (1 << 5) | 1


# --------------------------------------------------------------------------- NPY
def npy_bytes(array: np.ndarray) -> bytes:
    """Serialize a C-contiguous array as .npy format version 1.0."""
    shape = tuple(int(d) for d in array.shape)
    shape_txt = "(" + ", ".join(str(d) for d in shape) + ("," if len(shape) == 1 else "") + ")"
    header = "{'descr': '%s', 'fortran_order': False, 'shape': %s, }" % (array.dtype.str, shape_txt)
    # magic(6) + version(2) + len(2) + header + '\n' padded so the total is a multiple of 64.
    pad = 64 - ((10 + len(header) + 1) % 64)
    if pad == 64:
        pad = 0
    header_bytes = (header + " " * pad + "\n").encode("latin1")
    return (
        b"\x93NUMPY\x01\x00"
        + struct.pack("<H", len(header_bytes))
        + header_bytes
        + np.ascontiguousarray(array).tobytes()
    )


# --------------------------------------------------------------------------- ZIP
def zip_stored(entries: list[tuple[str, bytes]]) -> bytes:
    """Minimal ZIP writer: STORED entries, no extra fields, no data descriptors."""
    out = bytearray()
    central = bytearray()
    for name, data in entries:
        fname = name.encode("utf-8")
        crc = zlib.crc32(data) & 0xFFFFFFFF
        offset = len(out)
        out += struct.pack(
            "<IHHHHHIIIHH", 0x04034B50, 20, 0, 0, _DOS_TIME, _DOS_DATE,
            crc, len(data), len(data), len(fname), 0,
        )
        out += fname + data
        central += struct.pack(
            "<IHHHHHHIIIHHHHHII", 0x02014B50, 20, 20, 0, 0, _DOS_TIME, _DOS_DATE,
            crc, len(data), len(data), len(fname), 0, 0, 0, 0, 0, offset,
        )
        central += fname
    cd_offset = len(out)
    out += central
    out += struct.pack(
        "<IHHHHIIH", 0x06054B50, 0, 0, len(entries), len(entries), len(central), cd_offset, 0,
    )
    return bytes(out)


def manifest_bytes(state: Mapping[str, np.ndarray]) -> bytes:
    tensors = [
        {
            "archive_key": f"tensor_{i:06d}",
            "dtype": np.asarray(arr).dtype.str,
            "name": name,
            "shape": list(np.asarray(arr).shape),
        }
        for i, (name, arr) in enumerate(state.items())
    ]
    manifest = {"format_version": 1, "tensor_count": len(tensors), "tensors": tensors}
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def build_npz_stored(state: Mapping[str, np.ndarray]) -> bytes:
    """NPZ accepted by FinSentinel's deserializer, written with STORED entries."""
    entries = [
        (f"tensor_{i:06d}.npy", npy_bytes(np.asarray(arr, dtype="<f4")))
        for i, arr in enumerate(state.values())
    ]
    m = np.frombuffer(manifest_bytes(state), dtype=np.uint8)
    entries.append((MANIFEST_KEY + ".npy", npy_bytes(m)))
    return zip_stored(entries)


# --------------------------------------------------------------------------- ENVELOPE
def utc_timestamp(dt: datetime | None = None) -> str:
    dt = (dt or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_envelope(envelope: Mapping[str, object]) -> bytes:
    unsigned = {k: v for k, v in envelope.items() if k != "signature"}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def make_envelope(
    *,
    update_id: str,
    client_id: str,
    model_version: str,
    round_number: int,
    sample_count: int,
    payload: bytes,
    signing_key_id: str,
    private_key: Ed25519PrivateKey,
    created_at: str | None = None,
) -> dict:
    env = {
        "update_id": update_id,
        "client_id": client_id,
        "model_version": model_version,
        "round_number": int(round_number),
        "sample_count": int(sample_count),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_size_bytes": len(payload),
        "created_at": created_at or utc_timestamp(),
        "signing_key_id": signing_key_id,
    }
    env["signature"] = base64.b64encode(private_key.sign(canonical_envelope(env))).decode("ascii")
    return env


def submission_body(envelope: dict, payload: bytes) -> dict:
    return {"envelope": envelope, "payload_base64": base64.b64encode(payload).decode("ascii")}
