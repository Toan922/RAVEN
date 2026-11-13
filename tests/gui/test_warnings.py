"""Operator console warnings: feed problems and falling behind show, then clear (FR-13, NFR-05)."""
import pytest

pytest.importorskip("PySide6.QtWebEngineWidgets")   # skip (not fail) the console tests without PySide6



def test_warnings_show_and_clear(window):
    window._on_pipeline_status("Drone 1", "feed_interrupted", "lost it")
    assert "Video interrupted" in window.warning_notice.text() and not window.warning_notice.isHidden()
    window._on_progress("Drone 1", {"consecutive_failures": 0, "inference_p90_ms": 800, "lag_s": 0})
    assert "Falling behind" in window.warning_text("Drone 1")
    window._on_pipeline_status("Drone 1", "feed_resumed", "")
    window._on_progress("Drone 1", {"consecutive_failures": 0, "inference_p90_ms": 50, "lag_s": 0})
    assert window.warning_text("Drone 1") == "" and window.warning_notice.isHidden()
