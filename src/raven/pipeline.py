"""
RAVEN's entry point. Wires the pipe-and-filter stages (sample_frame -> detect_humans) to the event bus:
the stages run in a worker thread and publish HumanDetected and PipelineStatus events, which subscribers
receive on the bus's event loop thread. Run with:

    python -m raven.pipeline
"""

import asyncio
import threading
import time
import os
from pathlib import Path

import cv2
import discord
from bubus import EventBus
from dotenv import load_dotenv

from raven.backend.bus import ThreadSafeBus
from raven.backend.detection import detect_humans, load_model
from raven.backend.events import HumanDetected, PipelineStatus
from raven.backend.frame_sampler import sample_frame
from raven.backend.telemetry import TelemetryReceiver
from raven.discord_bot.subscriber import DiscordSubscriber

# temp settings until frontend is config'd after, will read from config file
STREAM_URL = "udp://127.0.0.1:5000"
TELEMETRY_ADDRESS = ("127.0.0.1", 5001)
FRAMES_PER_SECOND_RATE = 1        # FR-05, allowed 1-2
MIN_CONFIDENCE = 0.5              # FR-10.1
SHUTDOWN_ALERT_TIMEOUT_SECONDS = 5.0  # how long shutdown waits for queued alerts to send

# get the root directory of the project
ROOT_DIR = Path(__file__).resolve().parents[2]

# frames with detections are saved here, HumanDetected.frame points at them
FRAMES_DIR = ROOT_DIR / "runs" / "ingest"

def run_detections(worker_bus, telemetry_receiver, model, stop):
    """
    The worker thread's main loop: pulls frames from the stream through sample_frame and detect_humans,
    saves each frame with people in it to FRAMES_DIR, then publishes a HumanDetected event pointing at it.
    The frame is saved before the event is published, so subscribers can always open HumanDetected.frame;
    if saving fails, that detection is skipped and nothing is published for it.
    Runs until stop (a threading.Event) is set: sample_frame checks it on every frame, so the loop ends within
    about one frame while streaming, 3 s if the feed is stalled, and 10 s while reconnecting.
    worker_bus must be the ThreadSafeBus, since this runs outside the bus's event loop thread.
    """

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    samples = sample_frame(STREAM_URL, telemetry_receiver, FRAMES_PER_SECOND_RATE, worker_bus, stop)
    detections = detect_humans(samples, model, MIN_CONFIDENCE, worker_bus)

    for frame, frame_index, frame_time, telemetry_row, boxes in detections:
        # save the frame with detections
        frame_path = FRAMES_DIR / f"frame_{frame_time:.3f}.jpg"
        if not cv2.imwrite(str(frame_path), frame):
            print(f"Failed to save frame {frame_time:.3f} to {frame_path}, detection not published")
            continue

        # dispatch the HumanDetected event
        worker_bus.dispatch(HumanDetected(
            frame=str(frame_path),
            frame_index=frame_index,
            time=frame_time,
            telemetry=telemetry_row,
            bounding_boxes=boxes
        ))

# shortcut: prints status to the terminal, replace with the Detection Log and Operator Console subscribers once they're on the bus
def print_status(event):
    """Print a PipelineStatus event (feed and detection problems) to the terminal."""
    print(f"{time.strftime('%H:%M:%S')}  STATUS    {event.status} ({event.source}): {event.message}", flush=True)

async def main():
    """
    Start RAVEN: create the event bus and the ThreadSafeBus wrapper for worker threads, register the subscribers,
    start the Discord alert delivery task (FR-11.1, FR-11.2), load and warm up the model, start the telemetry
    receiver thread, then run run_detections in its own worker thread and wait.
    On exit, an error, or Ctrl+C: set stop, wait for the worker to finish so its last events still reach the bus,
    hand any remaining events to the subscribers, give queued alerts a limited time to send (NFR-09), report any that
    didn't, then cancel the delivery task and stop the bus.
    """

    # get the discord webhook and create the human detected subscriber
    load_dotenv()
    webhook = discord.SyncWebhook.from_url(os.environ["DISCORD_WEBHOOK_URL"])
    human_detected_subscriber = DiscordSubscriber(webhook)

    # event and worker threads
    bus = EventBus(name = "raven_event_bus", parallel_handlers=True)
    worker_bus = ThreadSafeBus(bus, asyncio.get_running_loop())

    # subscribers, registered before anything can publish so no early event (e.g. model_unavailable) is missed
    bus.on(PipelineStatus, print_status)
    bus.on(HumanDetected, human_detected_subscriber.on_human_detected)

    # send queued alerts in the background, so Discord never holds up the bus or the pipeline (NFR-05)
    delivery_task = asyncio.create_task(human_detected_subscriber.deliver_alerts())

    # tell the worker thread to stop
    stop = threading.Event()
    worker = None

    try:
        # load the model
        model = load_model(bus)

        # receive telemetry from the stream
        telemetry_receiver = TelemetryReceiver(TELEMETRY_ADDRESS)
        threading.Thread(target=telemetry_receiver.run, daemon=True).start()

        # run main loop in its own thread and wait till it stops
        worker = threading.Thread(target=run_detections, args=(worker_bus, telemetry_receiver, model, stop))
        worker.start()
        await asyncio.to_thread(worker.join)

    finally:
        # ensure the worker thread is stopped
        stop.set()

        # tell worker to stop and wait till it has
        if worker is not None:
            await asyncio.to_thread(worker.join)

        # hand any remaining events to the subscribers; alerts are only queued here, not necessarily sent yet
        await bus.wait_until_idle()

        # wait for queued alerts to send, with a limit so stopping during a Discord outage can't hang forever
        try:
            await asyncio.wait_for(human_detected_subscriber.alerts.join(), timeout=SHUTDOWN_ALERT_TIMEOUT_SECONDS)
        except TimeoutError:
            # list each alert that won't be sent, since the queue is lost on exit; the one being retried isn't listed, see docs/TODO.md
            alerts = human_detected_subscriber.alerts

            while not alerts.empty():
                event = alerts.get_nowait()
                print(f"Alert not sent: {event.frame}, detected at {time.strftime('%H:%M:%S', time.localtime(event.time))}", flush=True)

        delivery_task.cancel()

        # stop the event bus
        await bus.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # stop the program gracefully on Ctrl+C
        pass