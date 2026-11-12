"""backend/geotag.py: ground position of detected people (FR-08, FR-09)."""
import math

from raven.backend.geotag import APPROXIMATE, ESTIMATED, UNAVAILABLE, geotag
from raven.backend.scene_similarity import distance_m

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


def test_geotag_locates_every_person_and_holds_still_while_the_drone_moves():
    # FR-12 tells people apart by these per-box positions, so a still person must keep about the same one
    # while the drone flies toward them (here 5 m north, one frame at 5 m/s).
    vfov = 2 * math.atan(math.tan(math.radians(22.5)) * 512 / 640)

    def row_for(forward_m):   # the frame row where a person forward_m ahead of the drone appears
        down = math.atan(85 / forward_m) - math.radians(50)
        return (math.tan(down) / math.tan(vfov / 2) + 1) * 256

    seen = []
    for drone_north in (0.0, 5.0):
        telemetry = dict(TELEMETRY, latitude=TELEMETRY["latitude"] + drone_north / 111_320)
        y = row_for(80.0 - drone_north)
        boxes = [{"box_xyxy": [310, y - 8, 330, y + 8], "confidence": 0.9},
                 {"box_xyxy": [100, 200, 120, 216], "confidence": 0.5}]
        geotag(telemetry, boxes, 640, 512)
        assert all("location" in box for box in boxes)
        seen.append(boxes[0]["location"])
    assert distance_m(*seen) < 1.0
