"""Box matching tests; run with pytest or python tests/backend/test_scene_similarity.py."""

from raven.backend.scene_similarity import scenes_are_similar


def box(x, y=0, width=20, height=40, confidence=0.9):
    return {"box_xyxy": [x, y, x + width, y + height], "confidence": confidence}


def test_five_people_with_jitter_and_reordered_detections_are_similar():
    previous = [box(x, y) for x, y in [(0, 0), (40, 0), (80, 0), (0, 80), (40, 80)]]
    current = [box(x + 1, y - 1, confidence=0.6)
               for x, y in [(40, 80), (0, 0), (0, 80), (80, 0), (40, 0)]]

    assert scenes_are_similar(current, previous)
    assert scenes_are_similar(previous, current)


def test_one_moved_person_or_changed_count_is_a_new_scene():
    previous = [box(0), box(40), box(80), box(120), box(160)]

    assert not scenes_are_similar(previous[:-1] + [box(240)], previous)
    assert not scenes_are_similar(previous[:-1], previous)
    assert not scenes_are_similar(previous + [box(240)], previous)


def test_each_person_requires_a_separate_match():
    assert not scenes_are_similar([box(0), box(0)], [box(0), box(60)])


def test_crowded_boxes_find_a_full_match_instead_of_greedy_pairing():
    # The first current box can match either previous box. The second can only
    # match the first previous box, so a greedy first-choice match would fail.
    previous = [box(0, width=10, height=10), box(4, width=10, height=10)]
    current = [box(2, width=10, height=10), box(0, width=10, height=10)]

    assert scenes_are_similar(current, previous)
    assert scenes_are_similar(previous, current)


def test_empty_scenes_and_iou_threshold_boundary():
    assert scenes_are_similar([], [])
    assert not scenes_are_similar([], [box(0)])
    assert not scenes_are_similar([box(0)], [])

    previous = [box(0, width=30)]
    current = [box(10, width=30)]  # Intersection / union is exactly 0.5.
    assert scenes_are_similar(current, previous, iou_threshold=0.5)
    assert not scenes_are_similar(current, previous, iou_threshold=0.51)
    assert scenes_are_similar(previous, previous, iou_threshold=1.0)


def test_invalid_iou_thresholds_are_rejected():
    for threshold in (0, -0.1, 1.01):
        try:
            scenes_are_similar([box(0)], [box(0)], iou_threshold=threshold)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid IoU threshold {threshold}")


if __name__ == "__main__":
    test_five_people_with_jitter_and_reordered_detections_are_similar()
    test_one_moved_person_or_changed_count_is_a_new_scene()
    test_each_person_requires_a_separate_match()
    test_crowded_boxes_find_a_full_match_instead_of_greedy_pairing()
    test_empty_scenes_and_iou_threshold_boundary()
    test_invalid_iou_thresholds_are_rejected()
    print("scene similarity tests passed")
