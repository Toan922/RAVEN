"""backend/vault.py: encryption at rest for frames, the detection log and the outbox (NFR-02)."""
import json

import numpy as np
import pytest

from raven.backend import vault
from raven.backend.detection_log import DetectionLog, read_log
from raven.backend.outbox import Outbox


def test_vault_encrypts_frames_logs_and_outbox(tmp_path, make_event):
    from cryptography.fernet import Fernet

    vault.enable(Fernet.generate_key())
    frame = np.full((48, 64, 3), 120, np.uint8)
    path = tmp_path / "frame.jpg"
    assert vault.write_image(path, frame)
    assert path.read_bytes().startswith(vault.TOKEN_PREFIX)
    assert vault.read_image(path).shape == (48, 64, 3)

    log = DetectionLog(tmp_path / "detections.jsonl")
    log.on_human_detected(make_event(frame=str(path)))
    assert (tmp_path / "detections.jsonl").read_text().startswith("gAAAAA")
    records, skipped = read_log(tmp_path / "detections.jsonl")
    assert len(records) == 1 and skipped == 0

    outbox = Outbox(tmp_path / "outbox.jsonl")
    outbox.add({"id": "a"})
    assert "gAAAAA" in (tmp_path / "outbox.jsonl").read_text()
    assert [r["id"] for r in Outbox(tmp_path / "outbox.jsonl").pending()] == ["a"]


def test_vault_still_reads_plain_files_and_refuses_another_key(tmp_path):
    from cryptography.fernet import Fernet

    plain = tmp_path / "old.jsonl"
    plain.write_text(json.dumps({"id": "1", "frame": "x.jpg"}) + "\n")
    vault.enable(Fernet.generate_key())
    assert len(read_log(plain)[0]) == 1            # written before encryption was turned on
    secret = tmp_path / "secret.bin"
    vault.write_bytes(secret, b"hello")
    vault.enable(Fernet.generate_key())
    with pytest.raises(vault.VaultError):
        vault.read_bytes(secret)
