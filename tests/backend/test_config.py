"""backend/config.py: console settings, sampling rate and feed checks (FR-02, FR-05)."""
import pytest

from raven.backend.config import FeedConfig, RavenConfig, parse_udp_address, port_clashes, stride_for


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
