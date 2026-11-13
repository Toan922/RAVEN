"""Fixtures shared by every test folder."""
import time

import pytest

from raven.backend.events import HumanDetected


@pytest.fixture
def make_event():
    """Build a HumanDetected with one person box; keyword arguments replace any field."""
    def make(**extra):
        fields = dict(frame="x.jpg", frame_index=1, time=time.time(), telemetry=None,
                      bounding_boxes=[{"box_xyxy": [0, 0, 1, 1], "confidence": 0.9}])
        fields.update(extra)
        return HumanDetected(**fields)
    return make



