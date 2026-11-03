"""
RAVEN's entry point. Wires the pipe-and-filter stages (sample_frame -> detect_humans) to the event bus:
the stages run in a worker thread and publish HumanDetected and PipelineStatus events, which subscribers
receive on the bus's event loop thread. Run with:

    python -m raven.pipeline
"""

import asyncio
import threading
import time
from pathlib import Path

import cv2
from bubus import EventBus

from raven.backend.detection import detect_humans, load_model
from raven.backend.events import HumanDetected, PipelineStatus
from raven.backend.frame_sampler import sample_frame
from raven.backend.telemetry import TelemetryReceiver

# temp settings until frontend is config'd after, will read from config file
STREAM_URL = "udp://127.0.0.1:5000"
TELEMETRY_ADDRESS = ("127.0.0.1", 5001)
FRAMES_PER_SECOND_RATE = 1        # FR-05, allowed 1-2
MIN_CONFIDENCE = 0.5              # FR-10.1

# get the root directory of the project
ROOT_DIR = Path(__file__).resolve().parents[2]

# frames with detections are saved here, HumanDetected.frame points at them
FRAMES_DIR = ROOT_DIR / "runs" / "ingest"

class ThreadSafeBus:
    """
    Wraps the event bus so code running in worker threads can publish to it.
    bubus only accepts dispatch() from the thread running its event loop, so dispatch() here hands each
    event to that thread instead of calling the bus directly. It returns right away and never blocks the worker.
    """
    
    def __init__(self, event_bus, loop):
        # the real bus, and the event loop (bus thread) it runs on
        self.event_bus = event_bus
        self.loop = loop

    def dispatch(self, event):
        # schedule the event dispatch to happen in the event loop thread
        self.loop.call_soon_threadsafe(self.event_bus.dispatch, event)

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

# shortcut: print subscribers so runs are visible in the terminal; replace with the Discord, Detection Log
# and Operator Console subscribers once they're on the bus
def print_status(event):
    """Print a PipelineStatus event (feed and detection problems) to the terminal."""
    print(f"{time.strftime('%H:%M:%S')}  STATUS    {event.status} ({event.source}): {event.message}", flush=True)

def print_detection(event):
    """Print a HumanDetected event to the terminal: how many people, best confidence, location and saved frame."""
    best = max(box["confidence"] for box in event.bounding_boxes)
    if event.telemetry is None:
        location = "location unavailable"
    else:
        location = f"drone at {event.telemetry['latitude']:.5f}, {event.telemetry['longitude']:.5f}"
    print(f"{time.strftime('%H:%M:%S', time.localtime(event.time))}  DETECTED  "
          f"{len(event.bounding_boxes)} person(s), best {best:.0%}, {location}, frame {event.frame}", flush=True)

async def main():
    """
    Start RAVEN: create the event bus and the ThreadSafeBus wrapper for worker threads, load and warm up the
    model, start the telemetry receiver thread, then run run_detections in its own worker thread and wait.
    On exit, an error, or Ctrl+C: set stop, wait for the worker to finish so its last events still reach the bus,
    deliver any queued events, then stop the bus.
    """

    # event and worker threads
    bus = EventBus(name = "raven_event_bus", parallel_handlers=True)
    worker_bus = ThreadSafeBus(bus, asyncio.get_running_loop())

    # subscribers, registered before anything can publish so no early event (e.g. model_unavailable) is missed
    bus.on(PipelineStatus, print_status)
    bus.on(HumanDetected, print_detection)

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

        # deliver any queued events
        await bus.wait_until_idle()

        # stop the event bus
        await bus.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # stop the program gracefully on Ctrl+C
        pass