"""Alert queue tests; run with pytest or python tests/test_discord_subscriber.py."""

import asyncio
from types import SimpleNamespace

from raven.discord_bot.subscriber import (
    COOLDOWN_SECONDS,
    SCENE_RESET_SECONDS,
    DiscordSubscriber,
)


def detection(timestamp, positions=(0,)):
    return SimpleNamespace(
        time=timestamp,
        bounding_boxes=[{"box_xyxy": [x, 0, x + 30, 40], "confidence": 0.9}
                        for x in positions],
    )


def queued_times(subscriber):
    return [subscriber.alerts.get_nowait().time
            for _ in range(subscriber.alerts.qsize())]


def test_continuously_observed_scene_stays_suppressed_after_cooldown():
    async def scenario():
        subscriber = DiscordSubscriber(object())
        for timestamp in range(3 * COOLDOWN_SECONDS + 1):
            await subscriber.on_human_detected(detection(timestamp, (0, 60, 120, 180, 240)))

        assert subscriber.last_alert_time == 0
        assert subscriber.last_detection_time == 3 * COOLDOWN_SECONDS
        assert queued_times(subscriber) == [0]

    asyncio.run(scenario())


def test_scene_seen_during_cooldown_can_alert_once_cooldown_expires():
    async def scenario():
        subscriber = DiscordSubscriber(object())
        await subscriber.on_human_detected(detection(0))
        await subscriber.on_human_detected(detection(1, (100,)))
        assert subscriber.alerts.qsize() == 1

        await subscriber.on_human_detected(detection(COOLDOWN_SECONDS, (100,)))
        assert queued_times(subscriber) == [0, COOLDOWN_SECONDS]

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
        await subscriber.on_human_detected(detection(COOLDOWN_SECONDS, (0, 60)))

        assert queued_times(subscriber) == [0, COOLDOWN_SECONDS]

    asyncio.run(scenario())


def test_quiet_gap_rearms_scene_but_exact_reset_boundary_does_not():
    async def scenario():
        subscriber = DiscordSubscriber(object())
        await subscriber.on_human_detected(detection(0))
        await subscriber.on_human_detected(detection(SCENE_RESET_SECONDS))
        await subscriber.on_human_detected(detection(2 * SCENE_RESET_SECONDS))
        assert subscriber.alerts.qsize() == 1

        returning_time = 3 * SCENE_RESET_SECONDS + 1
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


if __name__ == "__main__":
    test_continuously_observed_scene_stays_suppressed_after_cooldown()
    test_scene_seen_during_cooldown_can_alert_once_cooldown_expires()
    test_gradual_movement_is_compared_with_last_queued_alert()
    test_changed_person_count_alerts_after_cooldown()
    test_quiet_gap_rearms_scene_but_exact_reset_boundary_does_not()
    test_pending_delivery_does_not_accumulate_duplicate_alerts()
    print("Discord subscriber tests passed")
