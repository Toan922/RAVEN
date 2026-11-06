"""Compare scenes using person boxes, without inspecting or loading frame images."""

# Each person's box must overlap a previous box by at least this intersection-over-union.
# Lower values tolerate more movement; higher values require closer matches.
DEFAULT_IOU_THRESHOLD = 0.5


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
    matched = {}

    def find_match(index, visited):
        for previous_index in candidates[index]:
            if previous_index in visited:
                continue
            visited.add(previous_index)
            # Reassign an earlier match if needed, so crowded boxes are order-independent.
            if previous_index not in matched or find_match(matched[previous_index], visited):
                matched[previous_index] = index
                return True
        return False

    return all(find_match(index, set()) for index in range(len(boxes)))
