"""backend/latency.py: classification latency tracking (NFR-04)."""
from raven.backend.latency import LatencyTracker


def test_latency_p90():
    tracker = LatencyTracker()
    assert tracker.p90_ms() == 0
    for ms in range(1, 51):            # the window keeps the newest 50
        tracker.record(ms / 1000)
    assert 45 <= tracker.p90_ms() <= 47
