"""Photoshop bridge tests.

Photoshop itself is not available in CI, so these drive the real client
against a socket server that speaks the same wire protocol: PBKDF2-derived
3DES, the same framing, the same reply shape. That covers the parts most
likely to break — key derivation, padding, framing, message routing between
commands and subscribed events.
"""
from __future__ import annotations

import socket
import struct
import threading

import pytest

from compositor.io.photoshop_bridge import (
    PHOTOSHOP_BLEND_MODES,
    PhotoshopConnection,
    PhotoshopError,
    PhotoshopMessage,
    _Cipher,
    derive_key,
    parse_message,
    script_place_layers,
    script_save_active_document,
)

PASSWORD = "Swordfish"


# ---------------------------------------------------------------------------
# A stand-in for Photoshop's ExtendScript listener
# ---------------------------------------------------------------------------


class FakePhotoshop(threading.Thread):
    """Answers every script with a canned reply, and can push events."""

    def __init__(self, password: str = PASSWORD):
        super().__init__(daemon=True)
        self.password = password
        self.received = []
        self.event_payloads = []
        self._server = socket.socket()
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        self.port = self._server.getsockname()[1]
        self._cipher = _Cipher(derive_key(password))
        self._conn = None
        self._stop = threading.Event()

    def run(self):
        self._server.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            self._conn = conn
            conn.settimeout(0.5)
            self._serve(conn)
            return

    def _serve(self, conn):
        while not self._stop.is_set():
            try:
                header = _recv_exactly(conn, 8)
            except socket.timeout:
                continue
            except OSError:
                return
            if header is None:
                continue
            length, status = struct.unpack(">ii", header)
            payload = _recv_exactly(conn, length - 4)
            if payload is None:
                return
            try:
                message = parse_request(self._cipher.decrypt(payload))
            except PhotoshopError:
                # A client using the wrong password: the frame decrypts to
                # garbage, which is exactly how a real mismatch looks.
                return
            self.received.append(message.content)
            self._send(conn, message.id, self._reply_for(message))

    def _reply_for(self, message: PhotoshopMessage) -> str:
        script = message.content
        if "networkEventSubscribe" in script:
            # Photoshop hands back an id for the event stream.
            return "subscribed"
        return "24.7"

    def _send(self, conn, msg_id: int, content: str) -> None:
        body = _reply_body(msg_id, "Done", content)
        encrypted = self._cipher.encrypt(body)
        conn.sendall(struct.pack(">ii", 4 + len(encrypted), 0) + encrypted)

    def push_event(self, msg_id: int, content: str) -> None:
        """Send an unsolicited event on the open connection."""
        if self._conn is None:
            raise RuntimeError("no client connected")
        encrypted = self._cipher.encrypt(_reply_body(msg_id, "Tool", content))
        self._conn.sendall(struct.pack(">ii", 4 + len(encrypted), 0) + encrypted)

    def close(self):
        self._stop.set()
        try:
            self._server.close()
        except OSError:
            pass
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass


def _recv_exactly(sock, count):
    chunks = []
    received = 0
    while received < count:
        chunk = sock.recv(min(count - received, 4096))
        if not chunk:
            return None
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def _reply_body(msg_id: int, command: str, content: str) -> bytes:
    """The reply frame body: header, then ``command\\0\\rcontent\\r``."""
    return (
        struct.pack(">iii", 1, msg_id, 1)
        + command.encode("ascii")
        + b"\x00\r"
        + content.encode("utf-8")
        + b"\r"
    )


def parse_request(payload: bytes) -> PhotoshopMessage:
    """Decode a *request* frame.

    Requests are asymmetric with replies: the body is the raw script with no
    ``command\\rcontent\\r`` envelope.
    """
    version, msg_id, msg_type = struct.unpack(">iii", payload[:12])
    return PhotoshopMessage(
        version=version,
        id=msg_id,
        type=msg_type,
        command="",
        content=payload[12:].decode("utf-8"),
    )


@pytest.fixture()
def photoshop():
    server = FakePhotoshop()
    server.start()
    yield server
    server.close()


# ---------------------------------------------------------------------------
# Crypto and framing
# ---------------------------------------------------------------------------


def test_key_derivation_is_24_bytes():
    assert len(derive_key("")) == 24
    assert derive_key("a") != derive_key("b")
    assert derive_key("same") == derive_key("same")


def test_cipher_round_trips_unicode_scripts():
    cipher = _Cipher(derive_key(PASSWORD))
    script = 'app.open(new File("C:/文档/未命名.psd"));'
    payload = struct.pack(">iii", 1, 1, 2) + script.encode("utf-8")
    encrypted = cipher.encrypt(payload)
    assert len(encrypted) % 8 == 0  # 3DES block alignment
    assert cipher.decrypt(encrypted) == payload


def test_parse_message_splits_command_and_content():
    body = _reply_body(42, "Done", "hello world")
    message = parse_message(body)
    assert message.id == 42
    assert message.command == "Done"
    assert message.content == "hello world"
    assert not message.is_error


def test_parse_message_rejects_truncated_payload():
    with pytest.raises(PhotoshopError):
        parse_message(b"\x00\x00")


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def test_send_script_round_trip(photoshop):
    with PhotoshopConnection(password=PASSWORD, host="127.0.0.1", port=photoshop.port) as conn:
        reply = conn.send_script("app.version;")
        assert reply.content == "24.7"
        assert photoshop.received == ["app.version;"]


def test_evaluate_returns_content(photoshop):
    with PhotoshopConnection(password=PASSWORD, host="127.0.0.1", port=photoshop.port) as conn:
        assert conn.evaluate("app.version;") == "24.7"


def test_wrong_password_is_reported(photoshop):
    # The fake server cannot decrypt a frame made with the wrong key, so it
    # never answers; the client must fail on the timeout rather than hang.
    conn = PhotoshopConnection(password="wrong", host="127.0.0.1", port=photoshop.port)
    try:
        with pytest.raises(PhotoshopError):
            conn.connect()
            conn.send_script("app.version;", timeout=2.0)
    finally:
        conn.close()


def test_connection_refused_is_reported():
    # Port 1 is not listening; connecting must raise, not block.
    conn = PhotoshopConnection(password="", host="127.0.0.1", port=1)
    with pytest.raises(PhotoshopError) as excinfo:
        conn.connect(timeout=1.0)
    assert "Could not reach Photoshop" in str(excinfo.value)
    conn.close()


def test_send_without_connect_raises():
    conn = PhotoshopConnection(password=PASSWORD, host="127.0.0.1", port=1)
    with pytest.raises(PhotoshopError):
        conn.send_script("app.version;")


def test_subscription_id_routes_events_to_the_callback(photoshop):
    got = []
    with PhotoshopConnection(password=PASSWORD, host="127.0.0.1", port=photoshop.port) as conn:
        conn.subscribe("toolChanged", got.append)
        # The reply's id is the stream id Photoshop will reuse for events.
        msg_id = list(conn._subscribed.values())[0]
        photoshop.push_event(msg_id, "brush")

        deadline = threading.Event()
        for _ in range(100):
            if got:
                break
            deadline.wait(0.05)

    assert got, "callback never fired"
    assert got[0].command == "Tool"
    assert got[0].content == "brush"


def test_close_is_idempotent(photoshop):
    conn = PhotoshopConnection(password=PASSWORD, host="127.0.0.1", port=photoshop.port)
    conn.connect()
    conn.close()
    conn.close()
    assert not conn.is_connected


# ---------------------------------------------------------------------------
# ExtendScript builders
# ---------------------------------------------------------------------------


def test_place_layers_emits_every_row():
    script = script_place_layers(
        ["/tmp/a.png", "/tmp/b.png"],
        ["Bottom", "Top"],
        opacities=[1.0, 0.5],
        blend_modes=["Multiply", "Linear Dodge (Add)"],
        offsets=[(10, 20), (0, 0)],
    )
    assert "/tmp/a.png" in script
    assert '"Bottom"' in script
    assert "100" in script and "50" in script
    # Compositor's spelling is translated to Photoshop's enum member.
    assert PHOTOSHOP_BLEND_MODES["Linear Dodge (Add)"] == "LINEAR_DODGE"
    assert '"LINEAR_DODGE"' in script
    # The offset rides along in the spec row and is applied by ``apply``.
    assert '"/tmp/a.png", "Bottom", 100, "MULTIPLY", 10, 20' in script
    assert "layer.translate(row[4], row[5])" in script


def test_place_layers_rejects_mismatched_names():
    with pytest.raises(PhotoshopError):
        script_place_layers(["/tmp/a.png"], ["one", "two"])


def test_place_layers_rejects_empty():
    with pytest.raises(PhotoshopError):
        script_place_layers([], [])


def test_save_active_document_escapes_paths():
    script = script_save_active_document(r'C:\Users\me\out"weird.psd')
    assert r"C:\\Users\\me\\out\"weird.psd" in script


def test_unix_paths_are_not_escaped():
    script = script_save_active_document("/tmp/out.psd")
    assert '"/tmp/out.psd"' in script
