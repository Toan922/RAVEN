"""
Tell whether a detection shows people already alerted (FR-12), without inspecting or loading frame images.

- RecentPeople: by each person's estimated ground position (from geotag). Works while the drone moves.
- scenes_are_similar: by person boxes in the image. The fallback when a frame has no ground positions
  (no telemetry), since boxes shift with the camera and only match while the drone hovers.
"""

import math

from raven.backend.geotag import METRES_PER_DEGREE

# Each person's box must overlap a previous box by at least this intersection-over-union.
# Lower values tolerate more movement; higher values require closer matches.
DEFAULT_IOU_THRESHOLD = 0.5

# Two ground positions this close are taken as the same person. Covers geotag error and telemetry up to
# 2 s old (FR-06); people this close together are still told apart by count, since each match is one-to-one.
SAME_PERSON_RADIUS_M = 15.0


def _one_to_one(candidates):
    """
    candidates[i] lists the previous items that current item i may be. Return {previous index: current index}
    pairing as many current items as possible, each previous item at most once.
    """
    matched = {}

    def find_match(index, visited):
        for previous_index in candidates[index]:
            if previous_index in visited:
                continue
            visited.add(previous_index)
            # Reassign an earlier match if needed, so crowded items are order-independent.
            if previous_index not in matched or find_match(matched[previous_index], visited):
                matched[previous_index] = index
                return True
        return False

    for index in range(len(candidates)):
        find_match(index, set())
    return matched


def box_iou(box, previous_box):
    """Return intersection area / union area for two [x1, y1, x2, y2] boxes."""
    x1, y1, x2, y2 = box
    px1, py1, px2, py2 = previous_box
    intersection = max(0, min(x2, px2) - max(x1, px1)) * max(
        0, min(y2, py2) - max(y1, py1)
    )
    area = max(0, x2 - x1) * max(0, y2 - y1)
    previous_area = max(0, px2 - px1) * max(0, py2 - py1)
    union = area + previous_area - intersection
    return intersection / union if union > 0 else 0.0


def scenes_are_similar(boxes, previous_boxes, iou_threshold=DEFAULT_IOU_THRESHOLD):
    """
    Require the same person count and a distinct overlapping match for every box.
    Boxes use the HumanDetected.bounding_boxes format. Order and confidence are ignored.
    Empty scenes match each other; a zero-area box never matches.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be greater than 0 and at most 1")
    if len(boxes) != len(previous_boxes):
        return False

    candidates = [
        [
            j for j, previous in enumerate(previous_boxes)
            if box_iou(box["box_xyxy"], previous["box_xyxy"]) >= iou_threshold
        ]
        for box in boxes
    ]
    return len(_one_to_one(candidates)) == len(boxes)


def distance_m(location, other):
    """Metres between two {"latitude", "longitude"} points (flat-earth approximation, fine within a few km)."""
    north = (location["latitude"] - other["latitude"]) * METRES_PER_DEGREE
    east = (location["longitude"] - other["longitude"]) * METRES_PER_DEGREE * math.cos(math.radians(location["latitude"]))
    return math.hypot(north, east)


class RecentPeople:
    """
    People already alerted for one drone, by ground position. A person is forgotten once they go unseen for
    window_s; every sighting restarts their window, so someone in view the whole time is never alerted twice.
    """

    def __init__(self, window_s, radius_m=SAME_PERSON_RADIUS_M):
        self.window_s = window_s
        self.radius_m = radius_m
        self._people = []   # {"location": {"latitude", "longitude"}, "seen": time last seen}

    def new_people(self, locations, now):
        """
        Return the locations that aren't a known person. Known people found in locations are marked seen at now.
        """
        self._people = [person for person in self._people if now - person["seen"] <= self.window_s]
        candidates = [
            [j for j, person in enumerate(self._people) if distance_m(location, person["location"]) <= self.radius_m]
            for location in locations
        ]
        matched = _one_to_one(candidates)
        for j in matched:
            self._people[j]["seen"] = now
        known = set(matched.values())
        return [location for i, location in enumerate(locations) if i not in known]

    def remember(self, locations, now):
        """Add people who were just alerted."""
        self._people += [{"location": dict(location), "seen": now} for location in locations]
