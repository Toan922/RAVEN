"""discord_bot/subscriber.py: alert queue, duplicate suppression (FR-12) and alert contents (FR-10.2).
Run with pytest, or python tests/discord_bot/test_subscriber.py (skips tests that need fixtures)."""

import asyncio
from types import SimpleNamespace

from raven.discord_bot.subscriber import (
    COOLDOWN_SECONDS,
    DUPLICATE_WINDOW_SECONDS,
    DiscordSubscriber,
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


def test_discord_labels_approximate_and_unavailable_locations(make_event):
    estimated = create_discord_alert(make_event(telemetry={"latitude": 1.0, "longitude": 2.0},
                                            location={"latitude": 1.5, "longitude": 2.5}, location_status="estimated"))
    assert estimated.fields[0].name == "Location" and estimated.fields[0].value.startswith("1.50000, 2.50000")
    approximate = create_discord_alert(make_event(telemetry={"latitude": 1.0, "longitude": 2.0}))
    assert "Approximate (drone position)" in approximate.fields[0].name
    missing = create_discord_alert(make_event(location_status="unavailable"))
    assert missing.fields[0].value == "Location not available"


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
    print("Discord subscriber tests passed")
