"""
Encryption at rest for the operator console's data folder: saved detection frames, the detection log, the Discord
outbox and the user accounts file are written encrypted with Fernet (AES-128-CBC with an HMAC), and decrypted in
memory when the console shows or sends them.

The key is made on first use and kept in the operating system's credential store through keyring (on Windows,
Credential Manager, protected by DPAPI for the signed-in Windows user), never in the data folder itself.

Encryption is off until enable() is called, so raven.pipeline and the tests read and write plain files. Files
written before encryption was turned on are still read: anything that isn't a Fernet token is returned as is.
"""

import threading
from pathlib import Path

import cv2
import numpy as np

KEYRING_SERVICE = "RAVEN operator console"
KEYRING_NAME = "data-folder-key"

# every Fernet token starts with these bytes (version byte 0x80, base64 encoded)
TOKEN_PREFIX = b"gAAAAA"

_fernet = None
_lock = threading.Lock()


class VaultError(RuntimeError):
    pass


def enable(key: bytes | None = None):
    """
    Turn encryption on with key, or with the key from the credential store (made there on first use).
    Raises VaultError if there's no credential store to keep the key in.
    """
    global _fernet
    from cryptography.fernet import Fernet

    if key is None:
        key = _stored_key()
    with _lock:
        _fernet = Fernet(key)


def disable():
    global _fernet
    with _lock:
        _fernet = None


def enabled() -> bool:
    return _fernet is not None


def _stored_key() -> bytes:
    from cryptography.fernet import Fernet

    try:
        import keyring

        key = keyring.get_password(KEYRING_SERVICE, KEYRING_NAME)
        if not key:
            key = Fernet.generate_key().decode()
            keyring.set_password(KEYRING_SERVICE, KEYRING_NAME, key)
            # read it back, so a store that silently drops it fails here rather than locking the data away
            if keyring.get_password(KEYRING_SERVICE, KEYRING_NAME) != key:
                raise VaultError("the credential store didn't keep the data key")
        return key.encode()
    except VaultError:
        raise
    except Exception as error:  # no backend, access denied, ...
        raise VaultError(f"can't reach the system credential store for the data key: {error}") from error


# ---------------------------------------------------------------------------------------------------- bytes

def encrypt(data: bytes) -> bytes:
    fernet = _fernet
    return fernet.encrypt(data) if fernet is not None else data


def decrypt(data: bytes) -> bytes:
    """
    Decrypt a Fernet token; data that isn't one (a file from before encryption) is returned unchanged.
    Raises VaultError for a token made with another key, or a damaged one.
    """
    if not data.startswith(TOKEN_PREFIX):
        return data
    fernet = _fernet
    if fernet is None:
        raise VaultError("this file is encrypted; open it from the RAVEN operator console")
    from cryptography.fernet import InvalidToken

    try:
        return fernet.decrypt(data)
    except InvalidToken:
        raise VaultError("can't decrypt: the file was encrypted with another key, or it's damaged") from None


def write_bytes(path, data: bytes):
    """
    Write data to path, encrypted when encryption is on. Written to a temporary file first, then moved into
    place, so a crash never leaves half a file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encrypt(data))
    temporary.replace(path)


def read_bytes(path) -> bytes:
    return decrypt(Path(path).read_bytes())


# ---------------------------------------------------------------------------------------------------- frames

def write_image(path, frame) -> bool:
    """
    Save an OpenCV frame as a JPEG at path (encrypted when encryption is on). Returns False if it can't be saved,
    like cv2.imwrite.
    """
    ok, encoded = cv2.imencode(Path(path).suffix or ".jpg", frame)
    if not ok:
        return False
    try:
        write_bytes(path, encoded.tobytes())
    except OSError:
        return False
    return True


def read_image(path):
    """
    Load a saved frame as an OpenCV image (BGR), or None if it can't be read or decoded.
    """
    try:
        data = read_bytes(path)
    except (OSError, VaultError):
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------------------------------- text lines

def encode_line(text: str) -> str:
    """
    One line of a JSON Lines file: the text itself, or a Fernet token (which has no newlines) when encryption is on.
    """
    return encrypt(text.encode("utf-8")).decode("ascii") if enabled() else text


def decode_line(line: str) -> str:
    return decrypt(line.strip().encode("utf-8")).decode("utf-8")
