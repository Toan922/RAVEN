"""
Geotag Human Detection (FR-08, FR-09): the estimated ground position of a detected person, from the drone's
position data and where the person is in the frame.

The ground is taken as flat at the drone's take-off height (altitude_agl_m is above ground level). A pixel's
viewing ray leaves the camera at the camera's depression angle below the horizon, plus or minus where the pixel
sits in the camera's field of view, and meets the ground at altitude / tan(angle) metres in front of the drone.

- "estimated": the drone's telemetry has latitude, longitude, altitude_agl_m, heading_deg and
  camera_depression_deg, and the person's ray meets the ground nearby
- "approximate": the drone's own position is used, because the rest is missing or the ray is close to the horizon
- "unavailable": the frame has no position data at all
"""

import math

# the thermal camera's horizontal field of view; the vertical one follows from the frame's aspect ratio.
# 45 degrees matches a 640 px radiometric camera with a 13 mm lens (e.g. the Zenmuse XT2 the HIT-UAV set used).
CAMERA_HFOV_DEG = 45.0

# a ray this close to the horizon would land too far away to trust, so the drone's position is used instead
MIN_RAY_DEPRESSION_DEG = 5.0

# estimates further than this from the drone are treated as unreliable
MAX_GROUND_DISTANCE_M = 1500.0

# metres per degree of latitude
METRES_PER_DEGREE = 111_320.0

ESTIMATED = "estimated"
APPROXIMATE = "approximate"
UNAVAILABLE = "unavailable"

_NEEDED = ("altitude_agl_m", "heading_deg", "camera_depression_deg")


def _number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def ground_position(telemetry, box_xyxy, frame_width, frame_height, hfov_deg=CAMERA_HFOV_DEG):
    """
    The (latitude, longitude) on flat ground under the centre of box_xyxy, or None if it can't be estimated.
    """
    if not telemetry or not all(_number(telemetry.get(key)) for key in ("latitude", "longitude", *_NEEDED)):
        return None
    if frame_width <= 0 or frame_height <= 0:
        return None

    altitude = telemetry["altitude_agl_m"]
    if altitude <= 0:
        return None

    hfov = math.radians(hfov_deg)
    vfov = 2 * math.atan(math.tan(hfov / 2) * frame_height / frame_width)

    # where the box centre sits in the frame, from -1 (left / top) to 1 (right / bottom)
    x1, y1, x2, y2 = box_xyxy
    nx = ((x1 + x2) / 2 - frame_width / 2) / (frame_width / 2)
    ny = ((y1 + y2) / 2 - frame_height / 2) / (frame_height / 2)

    # angles of the pixel's ray: below the horizon, and to the right of the camera's heading
    down = math.radians(telemetry["camera_depression_deg"]) + math.atan(ny * math.tan(vfov / 2))
    right = math.atan(nx * math.tan(hfov / 2))
    if down < math.radians(MIN_RAY_DEPRESSION_DEG):
        return None

    # metres in front of and to the right of the drone
    forward = altitude / math.tan(down)
    side = math.hypot(forward, altitude) * math.tan(right)
    if math.hypot(forward, side) > MAX_GROUND_DISTANCE_M:
        return None

    heading = math.radians(telemetry["heading_deg"])
    north = forward * math.cos(heading) - side * math.sin(heading)
    east = forward * math.sin(heading) + side * math.cos(heading)

    latitude = telemetry["latitude"] + north / METRES_PER_DEGREE
    longitude = telemetry["longitude"] + east / (METRES_PER_DEGREE * math.cos(math.radians(telemetry["latitude"])))
    return latitude, longitude


def geotag(telemetry, boxes, frame_width, frame_height):
    """
    Where the detection is and how it was found: ({"latitude", "longitude"} or None, status), using the
    most confident person in the frame.
    """
    if not telemetry or not (_number(telemetry.get("latitude")) and _number(telemetry.get("longitude"))):
        return None, UNAVAILABLE
    if boxes:
        best = max(boxes, key=lambda box: box.get("confidence", 0))
        position = ground_position(telemetry, best["box_xyxy"], frame_width, frame_height)
        if position is not None:
            return {"latitude": position[0], "longitude": position[1]}, ESTIMATED
    return {"latitude": telemetry["latitude"], "longitude": telemetry["longitude"]}, APPROXIMATE
