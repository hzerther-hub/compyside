"""Photoshop bridge — talk to a running Photoshop over its ExtendScript listener.

Photoshop exposes a TCP server on port 49494 that accepts JavaScript and
replies with the script's result. The wire format is fixed:

* the payload is ``>iii`` (protocol version, message id, message type)
  followed by the UTF-8 script;
* it is encrypted with 3DES-CBC, key derived by PBKDF2-HMAC-SHA1 from the
  listener password and the salt ``"Adobe Photoshop"`` (1000 iterations,
  24-byte key), IV of eight zero bytes, PKCS#7 padding;
* the frame on the wire is ``>ii`` (4 + len(ciphertext), status) followed by
  the ciphertext.

That protocol was reverse-engineered by the ``pyps`` project
(https://github.com/theiviaxx/photoshopConnection); this module reimplements
it with a background reader thread and pluggable crypto backends.

Photoshop must have *Enable scripting* turned on in Preferences, and the
password passed to :class:`PhotoshopConnection` must match the one its
listener is configured with — a mismatch comes back as a reply mentioning
"decrypting", which is turned into a clear error here.
"""
from __future__ import annotations

import hashlib
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

DEFAULT_PORT = 49494
PROTOCOL_VERSION = 1
# Message type for "run a script and send me the result".
MESSAGE_TYPE_COMMAND = 2

_SALT = b"Adobe Photoshop"
_ITERATIONS = 1000
_KEY_LENGTH = 24
_BLOCK_SIZE = 8

# How long the reader waits for the *start* of a message before looping again.
_POLL_TIMEOUT = 0.5
# Once a message has started arriving, how long to keep waiting for the rest.
_MESSAGE_TIMEOUT = 30.0

SUBSCRIBE_TEMPLATE = """
var idNS = stringIDToTypeID( 'networkEventSubscribe' );
var desc1 = new ActionDescriptor();
desc1.putClass( stringIDToTypeID( 'eventIDAttr' ), stringIDToTypeID( '%s' ) );
executeAction( idNS, desc1, DialogModes.NO );
"""

# Compositor blend mode spelling -> Photoshop's BlendMode enum member. The
# names differ ("Linear Dodge (Add)" vs "LINEAR_DODGE"), so the mapping is
# explicit rather than derived.
PHOTOSHOP_BLEND_MODES = {
    "Normal": "NORMAL",
    "Darken": "DARKEN",
    "Multiply": "MULTIPLY",
    "Color Burn": "COLOR_BURN",
    "Linear Burn": "LINEAR_BURN",
    "Lighten": "LIGHTEN",
    "Screen": "SCREEN",
    "Color Dodge": "COLOR_DODGE",
    "Linear Dodge (Add)": "LINEAR_DODGE",
    "Overlay": "OVERLAY",
    "Soft Light": "SOFT_LIGHT",
    "Hard Light": "HARD_LIGHT",
    "Vivid Light": "VIVID_LIGHT",
    "Linear Light": "LINEAR_LIGHT",
    "Pin Light": "PIN_LIGHT",
    "Hard Mix": "HARD_MIX",
    "Difference": "DIFFERENCE",
    "Exclusion": "EXCLUSION",
    "Subtract": "SUBTRACT",
    "Divide": "DIVIDE",
    "Hue": "HUE",
    "Saturation": "SATURATION",
    "Color": "COLOR",
    "Luminosity": "LUMINOSITY",
}


class PhotoshopError(RuntimeError):
    """Anything that goes wrong talking to Photoshop."""


# ---------------------------------------------------------------------------
# Crypto
# ---------------------------------------------------------------------------


def derive_key(password: str) -> bytes:
    """PBKDF2-HMAC-SHA1 exactly as Photoshop's listener does it."""
    return hashlib.pbkdf2_hmac(
        "sha1", password.encode("utf-8"), _SALT, _ITERATIONS, dklen=_KEY_LENGTH
    )


def _pad(data: bytes) -> bytes:
    padding = _BLOCK_SIZE - (len(data) % _BLOCK_SIZE)
    return data + bytes([padding] * padding)


def _unpad(data: bytes) -> bytes:
    if not data:
        return data
    padding = data[-1]
    if padding < 1 or padding > _BLOCK_SIZE or padding > len(data):
        raise PhotoshopError("malformed reply from Photoshop (bad padding)")
    return data[:-padding]


class _Cipher:
    """3DES-CBC over a 24-byte key, using whichever crypto lib is installed."""

    def __init__(self, key: bytes):
        self._backend, self._enc, self._dec = self._load(key)

    @staticmethod
    def _load(key: bytes):
        try:
            from Crypto.Cipher import DES3  # type: ignore

            # Two objects: a pycryptodome CBC context is single-direction.
            return (
                "pycryptodome",
                DES3.new(key, DES3.MODE_CBC, b"\0" * _BLOCK_SIZE),
                DES3.new(key, DES3.MODE_CBC, b"\0" * _BLOCK_SIZE),
            )
        except ImportError:
            pass
        try:
            from cryptography.hazmat.primitives.ciphers import (  # type: ignore
                Cipher,
                algorithms,
                modes,
            )
        except ImportError:
            raise PhotoshopError(
                "No 3DES backend available. Install one with: "
                'pip install "compositor-pyside[ps]"'
            ) from None
        try:
            factory = Cipher(algorithms.TripleDES(key), modes.CBC(b"\0" * _BLOCK_SIZE))
        except Exception as exc:  # noqa: BLE001 - some builds ship TripleDES but disable it
            raise PhotoshopError(
                f"3DES is unavailable in this cryptography build ({exc}); "
                "install pycryptodome instead"
            ) from exc
        return "cryptography", factory.encryptor(), factory.decryptor()

    def encrypt(self, data: bytes) -> bytes:
        if self._backend == "pycryptodome":
            return self._enc.encrypt(_pad(data))
        return self._enc.update(_pad(data)) + self._enc.finalize()

    def decrypt(self, data: bytes) -> bytes:
        if self._backend == "pycryptodome":
            return _unpad(self._dec.decrypt(data))
        return _unpad(self._dec.update(data) + self._dec.finalize())


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass
class PhotoshopMessage:
    """One reply (or event) from Photoshop."""

    version: int
    id: int
    type: int
    command: str
    content: str

    @property
    def is_error(self) -> bool:
        return self.command.upper() == "ERROR"

    def __str__(self) -> str:
        return f"<{self.command}: {self.content}>"


def parse_message(payload: bytes) -> PhotoshopMessage:
    """Decode one decrypted reply frame body."""
    if len(payload) < 12:
        raise PhotoshopError("short reply from Photoshop")
    version, msg_id, msg_type = struct.unpack(">iii", payload[:12])
    command, _, rest = payload[12:].partition(b"\r")
    content, _, _ = rest.partition(b"\r")
    return PhotoshopMessage(
        version=version,
        id=msg_id,
        type=msg_type,
        # Photoshop terminates the command word with a NUL before the
        # separator; it is an internal marker, not part of the name.
        command=command.rstrip(b"\x00").decode("utf-8", "replace"),
        content=content.decode("utf-8", "replace"),
    )


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

MessageCallback = Callable[[PhotoshopMessage], None]


class _Idle(Exception):
    """No data arrived within the poll window — not an error."""


class PhotoshopConnection:
    """A connection to one running Photoshop.

    A single background thread owns the read side of the socket so commands
    and subscribed events can arrive without either blocking the other.
    Use it as a context manager, or call :meth:`close`.
    """

    def __init__(
        self,
        password: str = "",
        host: Optional[str] = None,
        port: int = DEFAULT_PORT,
    ):
        self.password = password
        self.host = host
        self.port = port
        self._socket: Optional[socket.socket] = None
        self._cipher: Optional[_Cipher] = None
        self._next_id = 0
        self._send_lock = threading.Lock()
        self._replies: Dict[int, "queue.Queue[PhotoshopMessage]"] = {}
        self._event_callbacks: Dict[int, MessageCallback] = {}
        self._subscribed: Dict[str, int] = {}
        self._unsolicited: "queue.Queue[PhotoshopMessage]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._running = False
        self._closed_reason: Optional[str] = None

    # ---- lifecycle ------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._running and self._socket is not None

    def __enter__(self) -> "PhotoshopConnection":
        self.connect()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def connect(self, timeout: float = 5.0) -> None:
        """Open the socket and start the reader thread."""
        if self.is_connected:
            return
        candidates = [self.host] if self.host else ["127.0.0.1", _local_address()]
        last_error: Optional[Exception] = None
        for host in [h for h in candidates if h]:
            try:
                self._socket = socket.create_connection((host, self.port), timeout=timeout)
                break
            except OSError as exc:
                last_error = exc
        if self._socket is None:
            raise PhotoshopError(
                f"Could not reach Photoshop on port {self.port}. Check that Photoshop "
                f"is running and that Enable scripting is on ({last_error})"
            )

        # Build the cipher before starting the reader, so a missing 3DES
        # backend fails the connect rather than the first command.
        self._cipher = _Cipher(derive_key(self.password))
        self._socket.settimeout(_POLL_TIMEOUT)
        self._running = True
        self._reader = threading.Thread(
            target=self._read_loop, name="photoshop-bridge", daemon=True
        )
        self._reader.start()

    def close(self) -> None:
        """Stop the reader thread and drop the socket."""
        self._running = False
        reader, self._reader = self._reader, None
        if reader is not None:
            reader.join(timeout=2.0)
        sock, self._socket = self._socket, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        for waiting in list(self._replies.values()):
            waiting.put(PhotoshopMessage(0, -1, 0, "CLOSED", self._closed_reason or "closed"))

    # ---- sending --------------------------------------------------------

    def send_script(self, script: str, timeout: float = _MESSAGE_TIMEOUT) -> PhotoshopMessage:
        """Run ``script`` in Photoshop and return its reply.

        Blocks until Photoshop answers or ``timeout`` elapses.
        """
        if not self.is_connected or self._socket is None or self._cipher is None:
            raise PhotoshopError("not connected to Photoshop")

        with self._send_lock:
            msg_id = self._next_id
            self._next_id += 1
        waiting: "queue.Queue[PhotoshopMessage]" = queue.Queue(maxsize=1)
        self._replies[msg_id] = waiting

        payload = (
            struct.pack(">iii", PROTOCOL_VERSION, msg_id, MESSAGE_TYPE_COMMAND)
            + script.encode("utf-8")
        )
        encrypted = self._cipher.encrypt(payload)
        frame = struct.pack(">ii", 4 + len(encrypted), 0) + encrypted

        try:
            self._socket.sendall(frame)
        except OSError as exc:
            self._replies.pop(msg_id, None)
            raise PhotoshopError(f"lost the connection to Photoshop: {exc}") from exc

        try:
            message = waiting.get(timeout=timeout)
        except queue.Empty:
            self._replies.pop(msg_id, None)
            raise PhotoshopError(f"Photoshop did not answer within {timeout:g}s") from None

        self._replies.pop(msg_id, None)
        if message.command == "CLOSED":
            raise PhotoshopError(self._closed_reason or "connection closed while waiting for Photoshop")
        if "decrypting" in message.content.lower():
            raise PhotoshopError("Photoshop rejected the password — check the listener password")
        if message.is_error:
            raise PhotoshopError(message.content or "Photoshop reported an error")
        return message

    def evaluate(self, script: str, timeout: float = _MESSAGE_TIMEOUT) -> str:
        """Run ``script`` and return just its result string."""
        return self.send_script(script, timeout=timeout).content

    # ---- events ---------------------------------------------------------

    def subscribe(self, event: str, callback: Optional[MessageCallback] = None) -> None:
        """Ask Photoshop to stream ``event`` (e.g. ``toolChanged``)."""
        reply = self.send_script(SUBSCRIBE_TEMPLATE % event)
        self._subscribed[event] = reply.id
        if callback is not None:
            self._event_callbacks[reply.id] = callback

    def unsubscribe(self, event: str) -> None:
        msg_id = self._subscribed.pop(event, None)
        if msg_id is not None:
            self._event_callbacks.pop(msg_id, None)

    def next_message(self, timeout: float = 0.0) -> Optional[PhotoshopMessage]:
        """Pop an event Photoshop pushed that nobody subscribed to."""
        try:
            return self._unsolicited.get(timeout=timeout)
        except queue.Empty:
            return None

    # ---- reader ---------------------------------------------------------

    def _read_loop(self) -> None:
        while self._running:
            message = self._read_message()
            if message is None:
                continue
            callback = self._event_callbacks.get(message.id)
            if callback is not None:
                try:
                    callback(message)
                except Exception:  # noqa: BLE001 - a bad callback must not kill the reader
                    pass
                continue
            waiting = self._replies.get(message.id)
            if waiting is not None and not waiting.full():
                waiting.put(message)
            else:
                self._unsolicited.put(message)

    def _read_message(self) -> Optional[PhotoshopMessage]:
        if self._socket is None:
            self._running = False
            return None
        try:
            header = self._recv_exactly(self._socket, 8)
        except _Idle:
            return None
        except (OSError, PhotoshopError) as exc:
            return self._fail(f"connection to Photoshop failed: {exc}")

        length, status = struct.unpack(">ii", header)
        payload_length = length - 4
        if payload_length < 0:
            return self._fail("Photoshop sent a malformed frame")

        try:
            payload = self._recv_exactly(self._socket, payload_length)
        except _Idle:
            return None  # pragma: no cover - a message never starts idle
        except (OSError, PhotoshopError) as exc:
            return self._fail(f"connection to Photoshop failed: {exc}")

        if status != 0:
            # Photoshop reports a protocol-level failure (bad password, …).
            return PhotoshopMessage(
                0, 0, 0, "ERROR", payload.decode("utf-8", "replace")
            )

        assert self._cipher is not None
        try:
            plain = self._cipher.decrypt(payload)
            return parse_message(plain)
        except PhotoshopError as exc:
            return self._fail(str(exc))

    def _fail(self, reason: str) -> None:
        self._running = False
        self._closed_reason = reason
        for waiting in list(self._replies.values()):
            if not waiting.full():
                waiting.put(PhotoshopMessage(0, -1, 0, "CLOSED", reason))
        return None

    @staticmethod
    def _recv_exactly(sock: socket.socket, count: int) -> bytes:
        """Read exactly ``count`` bytes, or raise ``_Idle`` if none arrived.

        Once the first byte is in, keep going: the socket timeout is a poll
        window, not a deadline for a partially received message.
        """
        if count == 0:
            return b""
        deadline = time.monotonic() + _MESSAGE_TIMEOUT
        chunks: List[bytes] = []
        received = 0
        while received < count:
            try:
                chunk = sock.recv(min(count - received, 4096))
            except socket.timeout:
                if received == 0:
                    raise _Idle()
                if time.monotonic() > deadline:
                    raise PhotoshopError("Photoshop stopped responding mid-message")
                continue
            if not chunk:
                raise OSError("Photoshop closed the connection")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)


def _local_address() -> Optional[str]:
    """The address Photoshop's listener usually binds to on this machine."""
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return None


# ---------------------------------------------------------------------------
# ExtendScript builders
# ---------------------------------------------------------------------------


def _js_string(value: Union[str, Path]) -> str:
    """Quote a path as an ExtendScript string literal."""
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def script_version() -> str:
    return "#target photoshop\napp.version;"


def script_open_document(path: Union[str, Path]) -> str:
    """Open ``path`` as a new Photoshop document and return its name."""
    return (
        "#target photoshop\n"
        "(function () {\n"
        "  var f = new File(" + _js_string(path) + ");\n"
        "  if (!f.exists) { throw new Error('File not found: ' + f.fsName); }\n"
        "  var d = app.open(f);\n"
        "  d.activeLayer.name = d.name;\n"
        "  return d.name;\n"
        "})();\n"
    )


def script_place_layers(
    paths: Sequence[Union[str, Path]],
    names: Sequence[str],
    *,
    opacities: Sequence[float] = (),
    blend_modes: Sequence[str] = (),
    offsets: Sequence[Tuple[int, int]] = (),
) -> str:
    """Build one Photoshop document out of layer PNGs, bottom layer first.

    ``paths``/``names``/``opacities``/``blend_modes``/``offsets`` run
    bottom-to-top, the same order Compositor keeps its layer list in. Each
    ``offsets`` entry is the document-space position of that layer's cropped
    image, which is applied once the layer has been moved into the document.
    """
    if not paths:
        raise PhotoshopError("no layers to send")
    if len(names) != len(paths):
        raise PhotoshopError("layer names and paths must line up")

    entries = []
    for index, path in enumerate(paths):
        opacity = opacities[index] if index < len(opacities) else 1.0
        mode = PHOTOSHOP_BLEND_MODES.get(
            blend_modes[index] if index < len(blend_modes) else "Normal", "NORMAL"
        )
        offset = offsets[index] if index < len(offsets) else (0, 0)
        entries.append(
            "[%s, %s, %d, %s, %d, %d]"
            % (
                _js_string(path),
                _js_string(names[index]),
                int(round(max(0.0, min(1.0, float(opacity))) * 100)),
                _js_string(mode),
                int(offset[0]),
                int(offset[1]),
            )
        )

    spec = ",\n    ".join(entries)
    return (
        "#target photoshop\n"
        "(function () {\n"
        "  var spec = [\n    " + spec + "\n  ];\n"
        "  var apply = function (layer, row) {\n"
        "    layer.name = row[1];\n"
        "    layer.opacity = row[2];\n"
        "    try { layer.blendMode = BlendMode[row[3]]; } catch (e) {}\n"
        "    if (row[4] !== 0 || row[5] !== 0) { layer.translate(row[4], row[5]); }\n"
        "  };\n"
        "  var doc = app.open(new File(spec[0][0]));\n"
        "  apply(doc.activeLayer, spec[0]);\n"
        "  for (var i = 1; i < spec.length; i++) {\n"
        "    var src = app.open(new File(spec[i][0]));\n"
        "    var moved = src.activeLayer;\n"
        "    doc.activeLayer = doc.layers[0];\n"
        "    moved.move(doc, ElementPlacement.PLACEBEFORE);\n"
        "    apply(moved, spec[i]);\n"
        "    src.close(SaveOptions.DONOTSAVECHANGES);\n"
        "  }\n"
        "  doc.activeLayer = doc.layers[0];\n"
        "  return doc.name;\n"
        "})();\n"
    )


def script_save_active_document(path: Union[str, Path], *, as_psd: bool = True) -> str:
    """Save the active Photoshop document to ``path``."""
    kind = "PhotoshopSaveOptions" if as_psd else "PNGSaveOptions"
    return (
        "#target photoshop\n"
        "(function () {\n"
        "  if (app.documents.length === 0) { throw new Error('No open document'); }\n"
        "  var d = app.activeDocument;\n"
        "  var f = new File(" + _js_string(path) + ");\n"
        "  d.saveAs(f, new " + kind + "(), true, Extension.LOWERCASE);\n"
        "  return f.fsName;\n"
        "})();\n"
    )


def script_document_info() -> str:
    """Return a JSON blob describing Photoshop and the active document."""
    return (
        "#target photoshop\n"
        "(function () {\n"
        "  var out = { photoshop: app.version, documents: app.documents.length };\n"
        "  if (app.documents.length) {\n"
        "    var d = app.activeDocument;\n"
        "    out.name = d.name;\n"
        "    out.width = Math.round(d.width * 100) / 100;\n"
        "    out.height = Math.round(d.height * 100) / 100;\n"
        "    out.layers = [];\n"
        "    for (var i = 0; i < d.layers.length; i++) {\n"
        "      var l = d.layers[i];\n"
        '      out.layers.push({ name: l.name, kind: l.typename, visible: l.visible });\n'
        "    }\n"
        "  }\n"
        "  return JSON.stringify(out);\n"
        "})();\n"
    )
