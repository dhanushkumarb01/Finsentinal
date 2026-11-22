"""Eavesdropper scene: a passive tap sees only ciphertext; an impostor certificate is refused."""

import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from fldemo import keys as K
from fldemo.attacks import _mitm_sync
from fldemo.settings import SERVER_CERT, SERVER_KEY


class _H(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(422)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *a):
        pass


@pytest.fixture()
def tls_server():
    K.ensure_tls()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(SERVER_CERT, SERVER_KEY)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


def test_tap_sees_no_plaintext_and_impostor_is_refused(tls_server):
    out = _mitm_sync(upstream_port=tls_server)
    assert out["tap_request_status"] == 422          # the real request went through the tap
    assert out["tap_bytes_captured"] > 500           # the tap did capture the traffic …
    assert out["tap_plaintext_found"] is False       # … but it is all TLS ciphertext
    assert out["device_refused"] is True             # impostor certificate rejected
