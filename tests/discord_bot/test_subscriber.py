"""discord_bot/subscriber.py: alert queue, duplicate suppression (FR-12) and alert contents (FR-10.2).
Run with pytest, or python tests/discord_bot/test_subscriber.py (skips tests that need fixtures)."""

import asyncio
from types import SimpleNamespace

from raven.discord_bot.subscriber import (
    COOLDOWN_SECONDS,
    DUPLICATE_WINDOW_SECONDS,
    DiscordSubscriber,
    alert_people,
    create_discord_alert,
)

# detections without "location" on their boxes (no telemetry) use the image-box fallback, where FR-12's
# window is the cooldown
WINDOW = DUPLICATE_WINDOW_SECONDS


def detection(timestamp, positions=(0,), drone=""):
    return SimpleNamespace(
        time=timestamp,
        drone=drone,
        bounding_boxes=[{"box_xyxy": [x, 0, x + 30, 40], "confidence": 0.9}
                        for x in positions],
    )


def queued_times(subscriber):
    return [subscriber.alerts.get_nowait().time
            for _ in range(subscriber.alerts.qsize())]


def test_continuously_observed_scene_stays_suppressed_after_cooldown():
    async def scenario():
        subscriber = DiscordSubscriber(object())
        for timestamp in range(3 * WINDOW + 1):
            await subscriber.on_human_detected(detection(timestamp, (0, 60, 120, 180, 240)))

        assert subscriber.last_alert_times[""] == 0
        assert subscriber.last_detection_times[""] == 3 * WINDOW
        assert queued_times(subscriber) == [0]

    asyncio.run(scenario())


def test_scene_seen_during_cooldown_can_alert_once_cooldown_expires():
    async def scenario():
        subscriber = DiscordSubscriber(object())
        await subscriber.on_human_detected(detection(0))
        await subscriber.on_human_detected(detection(1, (100,)))
        assert subscriber.alerts.qsize() == 1

        await subscriber.on_human_detected(detection(WINDOW, (100,)))
        assert queued_times(subscriber) == [0, WINDOW]

    asyncio.run(scenario())


def test_gradual_movement_is_compared_with_last_queued_alert():
    async def scenario():
        subscriber = DiscordSubscriber(object())
        # Adjacent frames overlap strongly, but movement away from the alert's
        # original position eventually exceeds the 0.5 IoU threshold.
        for timestamp in range(0, 15, 2):
            await subscriber.on_human_detected(detection(timestamp, (timestamp,)))

        assert queued_times(subscriber) == [0, 12]

    asyncio.run(scenario())


def test_changed_person_count_alerts_after_cooldown():
    async def scenario():
        subscriber = DiscordSubscriber(object())
        await subscriber.on_human_detected(detection(0))
        await subscriber.on_human_detected(detection(WINDOW, (0, 60)))

        assert queued_times(subscriber) == [0, WINDOW]

    asyncio.run(scenario())


def test_quiet_gap_rearms_scene_but_exact_reset_boundary_does_not():
    async def scenario():
        subscriber = DiscordSubscriber(object())
        await subscriber.on_human_detected(detection(0))
        await subscriber.on_human_detected(detection(WINDOW))
        await subscriber.on_human_detected(detection(2 * WINDOW))
        assert subscriber.alerts.qsize() == 1

        returning_time = 3 * WINDOW + 1
        await subscriber.on_human_detected(detection(returning_time))
        assert queued_times(subscriber) == [0, returning_time]

    asyncio.run(scenario())


def test_pending_delivery_does_not_accumulate_duplicate_alerts():
    async def scenario():
        # Leave delivery stopped to simulate a backlog during a Discord outage.
        # The queued alert must already anchor similarity before upload succeeds.
        subscriber = DiscordSubscriber(object())
        for timestamp in range(61):
            await subscriber.on_human_detected(detection(timestamp))

        assert subscriber.alerts.qsize() == 1
        await subscriber.on_human_detected(detection(61, (100,)))
        assert queued_times(subscriber) == [0, 61]

    asyncio.run(scenario())


def test_drones_keep_separate_scenes():
    async def scenario():
        # Two drones see the same layout: drone B must not be compared with drone A's alert,
        # and drone A detecting nonstop must not keep drone B's scene from re-arming.
        subscriber = DiscordSubscriber(object())
        await subscriber.on_human_detected(detection(0, drone="A"))
        await subscriber.on_human_detected(detection(0, drone="B"))
        assert subscriber.alerts.qsize() == 2

        for timestamp in range(1, 3 * WINDOW + 1):
            await subscriber.on_human_detected(detection(timestamp, drone="A"))
        returning_time = 3 * WINDOW + 1
        await subscriber.on_human_detected(detection(returning_time, drone="B"))
        alerts = [subscriber.alerts.get_nowait() for _ in range(subscriber.alerts.qsize())]
        assert [(alert.drone, alert.time) for alert in alerts] == [("A", 0), ("B", 0), ("B", returning_time)]

    asyncio.run(scenario())


def test_queue_alert_sees_only_alerts_that_pass_every_check():
    # The console's outbox subclass records alerts in queue_alert, so suppressed ones must never reach it.
    recorded = []

    class RecordingSubscriber(DiscordSubscriber):
        def queue_alert(self, event):
            recorded.append(event.time)
            super().queue_alert(event)

    async def scenario():
        subscriber = RecordingSubscriber(object())
        for timestamp in range(3 * WINDOW + 1):
            await subscriber.on_human_detected(detection(timestamp))
        assert recorded == [0]
        assert queued_times(subscriber) == [0]

    asyncio.run(scenario())


def located(timestamp, people, drone=""):
    """A detection whose boxes carry ground positions; people is a list of (pixel x, metres north of a start point)."""
    return SimpleNamespace(
        time=timestamp,
        drone=drone,
        bounding_boxes=[{"box_xyxy": [x, 0, x + 13, 17], "confidence": 0.9,
                         "location": {"latitude": 32.0 + north / 111_320, "longitude": -96.0}}
                        for x, north in people],
    )


def test_same_person_alerts_once_while_drone_flies_over():
    async def scenario():
        # The box moves ~45 px a frame (drone at 5 m/s), but the ground position only wobbles by a few metres.
        subscriber = DiscordSubscriber(object())
        for timestamp in range(14):
            await subscriber.on_human_detected(located(timestamp, [(20 + 45 * timestamp, (-1) ** timestamp * 3)]))
        assert queued_times(subscriber) == [0]

    asyncio.run(scenario())


def test_new_person_alerts_once_the_short_cooldown_passes():
    async def scenario():
        # Person B shows up 1 s after A's alert: held back by the 2 s cooldown, then alerted on the next frame.
        subscriber = DiscordSubscriber(object())
        for timestamp in range(6):
            people = [(300, 0)] + ([(100, 200)] if timestamp >= 1 else [])
            await subscriber.on_human_detected(located(timestamp, people))
        assert queued_times(subscriber) == [0, COOLDOWN_SECONDS]

    asyncio.run(scenario())


def test_people_standing_together_are_counted_separately():
    async def scenario():
        subscriber = DiscordSubscriber(object())
        await subscriber.on_human_detected(located(0, [(300, 0)]))
        await subscriber.on_human_detected(located(COOLDOWN_SECONDS, [(300, 0), (320, 3)]))
        assert queued_times(subscriber) == [0, COOLDOWN_SECONDS]

    asyncio.run(scenario())


def test_person_is_forgotten_only_after_the_window_unseen():
    async def scenario():
        subscriber = DiscordSubscriber(object())
        await subscriber.on_human_detected(located(0, [(300, 0)]))
        await subscriber.on_human_detected(located(WINDOW, [(300, 0)]))           # seen again: window restarts
        await subscriber.on_human_detected(located(2 * WINDOW, [(300, 0)]))       # still within it
        returning_time = 3 * WINDOW + 1                                           # unseen for more than the window
        await subscriber.on_human_detected(located(returning_time, [(300, 0)]))
        assert queued_times(subscriber) == [0, returning_time]

    asyncio.run(scenario())


def test_frame_without_ground_positions_falls_back_to_the_window():
    async def scenario():
        subscriber = DiscordSubscriber(object())
        await subscriber.on_human_detected(located(0, [(300, 0)]))
        await subscriber.on_human_detected(detection(COOLDOWN_SECONDS, (100,)))   # no telemetry: no "location"
        await subscriber.on_human_detected(detection(WINDOW, (100,)))
        assert queued_times(subscriber) == [0, WINDOW]

    asyncio.run(scenario())


def test_failed_outbox_write_does_not_mark_people_as_alerted():
    # If queueing raises (the console's outbox write failed), the person must stay new so the next frame retries.
    class FailingOnce(DiscordSubscriber):
        failures = 1

        def queue_alert(self, event):
            if self.failures:
                self.failures -= 1
                raise OSError("disk full")
            super().queue_alert(event)

    async def scenario():
        subscriber = FailingOnce(object())
        try:
            await subscriber.on_human_detected(located(0, [(300, 0)]))
        except OSError:
            pass
        for timestamp in (1, 2):
            await subscriber.on_human_detected(located(timestamp, [(300, 0)]))
        assert queued_times(subscriber) == [1]

    asyncio.run(scenario())


def test_alert_has_one_line_per_person_with_a_labelled_location(make_event):
    # FR-10.2, FR-09, NFR-06: one field per person, numbered by confidence, with the classification, their
    # confidence and their own location in decimal degrees; approximate and unavailable locations are labelled
    event = make_event(telemetry={"latitude": 1.0, "longitude": 2.0}, bounding_boxes=[
        {"box_xyxy": [0, 0, 1, 1], "confidence": 0.84},                                                  # not placed
        {"box_xyxy": [5, 5, 6, 6], "confidence": 0.91, "location": {"latitude": 1.5, "longitude": 2.5}},
    ])
    fields = create_discord_alert(event).fields
    assert [field.name for field in fields] == ["Person 1 · Human · 0.91", "Person 2 · Human · 0.84"]
    assert fields[0].value == "1.50000, 2.50000 · [Open in Google Maps](https://www.google.com/maps/search/?api=1&query=1.5,2.5)"
    assert fields[1].value.startswith("Approximate (drone position): 1.00000, 2.00000 · [Open in Google Maps](")

    missing = create_discord_alert(make_event(telemetry=None)).fields
    assert [(field.name, field.value) for field in missing] == [("Person 1 · Human · 0.90", "Location unavailable")]


def test_crowded_alert_lists_the_most_confident_people_then_a_count(make_event):
    # Discord rejects embeds over 25 fields, and a rejected alert would retry forever (FR-11.2)
    boxes = [{"box_xyxy": [i, 0, i + 1, 1], "confidence": 0.5 + i / 100} for i in range(30)]
    fields = create_discord_alert(make_event(bounding_boxes=boxes)).fields
    assert len(fields) == 11
    assert fields[0].name == "Person 1 · Human · 0.79"
    assert (fields[-1].name, fields[-1].value) == ("More people", "+20 more in the image")


def test_people_are_numbered_by_confidence_with_their_own_location(make_event):
    # FR-08, FR-09, FR-10.2: every person gets a number (highest confidence first), their own ground location,
    # and a status; a person geotag couldn't place falls back to the drone's position, labelled approximate
    drone = {"latitude": 32.0, "longitude": -96.0}
    telemetry = dict(drone, altitude_agl_m=85.0, heading_deg=0.0, camera_depression_deg=50.0, t=100.0)
    a = {"latitude": 32.0001, "longitude": -96.0001}
    c = {"latitude": 32.0002, "longitude": -96.0002}
    event = make_event(telemetry=telemetry, bounding_boxes=[
        {"box_xyxy": [10, 10, 23, 27], "confidence": 0.72, "location": a},    # A
        {"box_xyxy": [100, 50, 113, 67], "confidence": 0.91},                 # B: no ground position
        {"box_xyxy": [200, 90, 213, 107], "confidence": 0.84, "location": c},  # C
    ])

    assert alert_people(event) == [
        {"number": 1, "box": [100, 50, 113, 67], "confidence": 0.91, "location": drone, "status": "approximate"},
        {"number": 2, "box": [200, 90, 213, 107], "confidence": 0.84, "location": c, "status": "estimated"},
        {"number": 3, "box": [10, 10, 23, 27], "confidence": 0.72, "location": a, "status": "estimated"},
    ]


def test_people_without_position_data_are_unavailable(make_event):
    # FR-09: no telemetry, or a row without both coordinates, means no location at all, and the alert still sends
    box = {"box_xyxy": [10, 10, 23, 27], "confidence": 0.9}
    unavailable = [{"number": 1, "box": [10, 10, 23, 27], "confidence": 0.9, "location": None, "status": "unavailable"}]
    assert alert_people(make_event(telemetry=None, bounding_boxes=[box])) == unavailable
    assert alert_people(make_event(telemetry={"latitude": 32.0, "altitude_agl_m": 85.0}, bounding_boxes=[box])) == unavailable

if __name__ == "__main__":
    test_continuously_observed_scene_stays_suppressed_after_cooldown()
    test_scene_seen_during_cooldown_can_alert_once_cooldown_expires()
    test_gradual_movement_is_compared_with_last_queued_alert()
    test_changed_person_count_alerts_after_cooldown()
    test_quiet_gap_rearms_scene_but_exact_reset_boundary_does_not()
    test_pending_delivery_does_not_accumulate_duplicate_alerts()
    test_drones_keep_separate_scenes()
    test_queue_alert_sees_only_alerts_that_pass_every_check()
    test_same_person_alerts_once_while_drone_flies_over()
    test_new_person_alerts_once_the_short_cooldown_passes()
    test_people_standing_together_are_counted_separately()
    test_person_is_forgotten_only_after_the_window_unseen()
    test_frame_without_ground_positions_falls_back_to_the_window()
    test_failed_outbox_write_does_not_mark_people_as_alerted()
    print("Discord subscriber tests passed")
