"""Toxiproxy helpers: one TCP link per virtual device with its network profile as real toxics."""

from __future__ import annotations

import json

import httpx

from fldemo.settings import CADDY_TLS_PORT, DEVICES_FILE, TOXI_API_PORT, TOXI_BASE_PORT

API = f"http://127.0.0.1:{TOXI_API_PORT}"


def link_name(device: str) -> str:
    return f"link-{device}"


def ensure_link(device: str) -> None:
    idx = int(device.split("-")[1])
    with httpx.Client(base_url=API, timeout=5) as c:
        c.delete(f"/proxies/{link_name(device)}")
        r = c.post("/proxies", json={"name": link_name(device), "listen": f"127.0.0.1:{TOXI_BASE_PORT + idx}",
                                     "upstream": f"127.0.0.1:{CADDY_TLS_PORT}", "enabled": True})
        r.raise_for_status()


def apply_profile(device: str, profile: str) -> None:
    prof = json.loads(DEVICES_FILE.read_text())["network_profiles"][profile]
    name = link_name(device)
    with httpx.Client(base_url=API, timeout=5) as c:
        for t in c.get(f"/proxies/{name}/toxics").json():
            c.delete(f"/proxies/{name}/toxics/{t['name']}")
        for stream in ("downstream", "upstream"):
            c.post(f"/proxies/{name}/toxics", json={
                "name": f"lat-{stream}", "type": "latency", "stream": stream, "toxicity": 1.0,
                "attributes": {"latency": prof["latency"] // 2, "jitter": prof["jitter"] // 2}}).raise_for_status()
            if prof["bandwidth"]:
                c.post(f"/proxies/{name}/toxics", json={
                    "name": f"bw-{stream}", "type": "bandwidth", "stream": stream, "toxicity": 1.0,
                    "attributes": {"rate": prof["bandwidth"]}}).raise_for_status()
