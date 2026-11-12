"""Backend pieces the operator console relies on: config checks, sampling rate, geotagging, outbox, vault, accounts."""
import json
import math
import time

import numpy as np
import pytest

from raven.backend import vault
from raven.backend.accounts import AuthError, UserStore, check_password
from raven.backend.config import (
    FeedConfig, RavenConfig, parse_udp_address, port_clashes, stride_for,
)
from raven.backend.detection_log import DetectionLog, read_log
from raven.backend.events import HumanDetected
from raven.backend.geotag import APPROXIMATE, ESTIMATED, UNAVAILABLE, geotag
from raven.backend.latency import LatencyTracker
from raven.backend.outbox import Outbox


@pytest.fixture(autouse=True)
def no_encryption():
    # every test starts and ends with encryption off, like raven.pipeline
    vault.disable()
    yield
    vault.disable()


def _event(**extra):
    fields = dict(frame="x.jpg", frame_index=1, time=time.time(), telemetry=None,
                  bounding_boxes=[{"box_xyxy": [0, 0, 1, 1], "confidence": 0.9}])
    fields.update(extra)
    return HumanDetected(**fields)


# ---------------------------------------------------------------------------------------------------- config

def test_stride_from_analysis_fps():
    assert stride_for(30, 1) == 30
    assert stride_for(30, 2) == 15
    assert stride_for(10, 2) == 5
    assert stride_for(1, 2) == 1      # never below one frame
    assert stride_for(0, 1) == 1      # streams that don't report a frame rate


def test_analysis_fps_must_be_one_to_two():
    assert any("Analysis rate" in p for p in RavenConfig(analysis_fps=3).problems())
    assert any("Analysis rate" in p for p in RavenConfig(analysis_fps=0.5).problems())
    assert not any("Analysis rate" in p for p in RavenConfig(analysis_fps=1.5).problems())


def test_old_presets_fall_back_to_one_fps_and_keep_session():
    config = RavenConfig.from_dict({"frame_stride": 5, "session": {"name": "Cedar", "area": "B2", "created": 5}})
    assert config.analysis_fps == 1.0
    assert (config.session.name, config.session.area, config.session.created) == ("Cedar", "B2", 5.0)
    assert RavenConfig().min_confidence == 0.7


def test_udp_addresses_and_port_clashes():
    assert parse_udp_address("udp://127.0.0.1:5001") == ("127.0.0.1", 5001)
    for bad in ("udp://host", "udp://:5000", "udp://h:99999", "rtsp://h:5000"):
        with pytest.raises(ValueError):
            parse_udp_address(bad)
    feeds = [FeedConfig(name="A", source="udp://127.0.0.1:5000", telemetry_path="udp://127.0.0.1:5001"),
             FeedConfig(name="B", source="udp://127.0.0.1:5002", telemetry_path="udp://127.0.0.1:5001")]
    assert len(port_clashes(feeds)) == 1
    feeds[1].telemetry_path = "udp://127.0.0.1:5003"
    assert port_clashes(feeds) == []


def test_udp_telemetry_needs_a_stream():
    feed = FeedConfig(name="A", source=__file__, telemetry_path="udp://127.0.0.1:5001")
    assert any("needs a stream link" in p for p in feed.problems())


# ---------------------------------------------------------------------------------------------------- geotag

TELEMETRY = {"latitude": 47.411, "longitude": 10.282, "altitude_agl_m": 85.0, "heading_deg": 0.0,
             "camera_depression_deg": 50.0}


def test_geotag_estimates_ground_position_ahead_of_the_drone():
    box = [{"box_xyxy": [310, 246, 330, 266], "confidence": 0.9}]   # the middle of a 640x512 frame
    location, status = geotag(TELEMETRY, box, 640, 512)
    assert status == ESTIMATED
    north = (location["latitude"] - TELEMETRY["latitude"]) * 111_320
    assert math.isclose(north, 85 / math.tan(math.radians(50)), rel_tol=0.02)
    assert abs(location["longitude"] - TELEMETRY["longitude"]) < 1e-6


def test_geotag_falls_back_to_drone_position_then_unavailable():
    box = [{"box_xyxy": [0, 0, 10, 10], "confidence": 0.9}]
    position_only = {"latitude": 1.0, "longitude": 2.0}
    assert geotag(position_only, box, 640, 512) == ({"latitude": 1.0, "longitude": 2.0}, APPROXIMATE)
    level_camera = dict(TELEMETRY, camera_depression_deg=0.0)     # the ray never reaches the ground nearby
    assert geotag(level_camera, [{"box_xyxy": [310, 0, 330, 20], "confidence": 0.9}], 640, 512)[1] == APPROXIMATE
    assert geotag(None, box, 640, 512) == (None, UNAVAILABLE)


def test_discord_labels_approximate_and_unavailable_locations():
    from raven.discord_bot.subscriber import create_discord_alert

    estimated = create_discord_alert(_event(telemetry={"latitude": 1.0, "longitude": 2.0},
                                            location={"latitude": 1.5, "longitude": 2.5}, location_status="estimated"))
    assert estimated.fields[0].name == "Location" and estimated.fields[0].value.startswith("1.50000, 2.50000")
    approximate = create_discord_alert(_event(telemetry={"latitude": 1.0, "longitude": 2.0}))
    assert "Approximate (drone position)" in approximate.fields[0].name
    missing = create_discord_alert(_event(location_status="unavailable"))
    assert missing.fields[0].value == "Location not available"


# ---------------------------------------------------------------------------------------------------- outbox

def test_outbox_keeps_unsent_alerts_across_a_restart(tmp_path):
    path = tmp_path / "discord_outbox.jsonl"
    outbox = Outbox(path)
    outbox.add({"id": "a", "frame": "1.jpg"})
    outbox.add({"id": "b", "frame": "2.jpg"})
    outbox.add({"id": "a", "frame": "1.jpg"})      # queued twice: kept once
    outbox.mark_sent("a")
    with path.open("a", encoding="utf-8") as f:
        f.write('{"queued": {"id": "c"')            # a line cut short by a crash
    reopened = Outbox(path)
    assert [record["id"] for record in reopened.pending()] == ["b"]
    assert len(path.read_text().splitlines()) == 1  # opening compacts the file


# ---------------------------------------------------------------------------------------------------- vault

def test_vault_encrypts_frames_logs_and_outbox(tmp_path):
    from cryptography.fernet import Fernet

    vault.enable(Fernet.generate_key())
    frame = np.full((48, 64, 3), 120, np.uint8)
    path = tmp_path / "frame.jpg"
    assert vault.write_image(path, frame)
    assert path.read_bytes().startswith(vault.TOKEN_PREFIX)
    assert vault.read_image(path).shape == (48, 64, 3)

    log = DetectionLog(tmp_path / "detections.jsonl")
    log.on_human_detected(_event(frame=str(path)))
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


# ---------------------------------------------------------------------------------------------------- accounts

def test_accounts_roles_and_lockout(tmp_path):
    store = UserStore(tmp_path / "accounts.dat")
    assert store.is_empty()
    admin = store.create("admin", "Ada", "admin", "correct-horse-1")
    viewer = store.create("val", "Val", "viewer", "viewer-pass-2")
    assert admin.can("manage_users") and admin.can("run_feeds")
    assert not viewer.can("run_feeds") and not viewer.can("review_alerts")
    assert "correct-horse-1" not in (tmp_path / "accounts.dat").read_text()
    assert UserStore(tmp_path / "accounts.dat").authenticate("ADMIN", "correct-horse-1").name == "Ada"

    with pytest.raises(AuthError):
        store.create("val", "Again", "viewer", "viewer-pass-2")
    with pytest.raises(AuthError, match="only administrator"):
        store.remove("admin")
    for _ in range(4):
        with pytest.raises(AuthError, match="Wrong"):
            store.authenticate("val", "nope")
    with pytest.raises(AuthError, match="locked"):
        store.authenticate("val", "nope")
    with pytest.raises(AuthError, match="Try again"):
        store.authenticate("val", "viewer-pass-2")


def test_password_rules():
    assert check_password("short1") is not None
    assert check_password("lettersonly") is not None
    assert check_password("letters-and-1") is None


def test_latency_p90():
    tracker = LatencyTracker()
    assert tracker.p90_ms() == 0
    for ms in range(1, 51):            # the window keeps the newest 50
        tracker.record(ms / 1000)
    assert 45 <= tracker.p90_ms() <= 47


def test_weights_in_a_folder_with_an_apostrophe_load_from_a_copy(tmp_path):
    # ultralytics drops apostrophes from weight paths, so "Fall '26" would become "Fall 26" and not be found
    from raven.backend.detection import loadable_path

    folder = tmp_path / "Fall '26"
    folder.mkdir()
    weights = folder / "best.pt"
    weights.write_bytes(b"weights")
    path = loadable_path(weights)
    assert "'" not in path and path.endswith("best.pt")
    assert open(path, "rb").read() == b"weights"
    assert loadable_path(weights) == path                  # copied once, then reused

    plain = tmp_path / "plain.pt"
    plain.write_bytes(b"x")
    assert loadable_path(plain) == str(plain)              # paths without one are used as they are
