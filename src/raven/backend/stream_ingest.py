"""
Live ingest for the operator console: a video stream from the ground station with live UDP telemetry.
Uses the frame_sampler filter (sample_frame with a TelemetryReceiver), which reports feed_missing,
feed_interrupted and feed_resumed, and adds what the console needs:
several drones at once, progress for the feeds bar, and every frame for the Live view.

The stream is read on its own thread, which keeps only the newest sampled frame: if the model is slower than the
analysis rate, older samples are dropped instead of queueing up, so alerts stay close to real time.
"""

import threading
import time
from pathlib import Path


from raven.backend.config import drone_folder
from raven.backend.detection import MAX_CONSECUTIVE_FAILED_FRAMES, run_model
from raven.backend.events import HumanDetected, PipelineStatus
from raven.backend.frame_sampler import sample_frame
from raven.backend.geotag import geotag
from raven.backend.latency import LatencyTracker
from raven.backend.vault import write_image
from raven.backend.telemetry import TelemetryReceiver


def ingest_stream(stream_url, telemetry_address, model, bus, min_confidence, analysis_fps, output_dir,
                  stop_event, on_progress=None, drone="", on_frame=None):
    """
    Run the person detector on about analysis_fps frames per second of the stream at stream_url, geotagging
    each sampled frame with the newest telemetry row from telemetry_address (host, port), at most 2 s old.
    Each sampled frame with people is saved, then published on bus as HumanDetected. Runs until stop_event is set.

    - on_progress: called with a stats dict after every sampled frame
    - on_frame: called with (frame_index, frame, boxes, checked, telemetry_row, seconds) for every frame read,
      where seconds counts from the start of the run and telemetry_row may be None (telemetry lost)

    The telemetry port is opened first, so a port already in use fails right away with OSError.
    """
    receiver = TelemetryReceiver(telemetry_address)
    listener = threading.Thread(target=receiver.run, name=f"telemetry {drone}", daemon=True)
    listener.start()

    started = time.time()
    stats = {
        "frames_read": 0,
        "frames_checked": 0,
        "detections": 0,
        "events_published": 0,
        "total_frames": 0,   # a live stream has no end
        "fps": 0.0,
        "time": 0.0,
        # checked frames in a row the model failed on, 0 once one succeeds again
        "consecutive_failures": 0,
        # True while the newest telemetry row is more than 2 s old (or none has arrived)
        "telemetry_lost": False,
        # 90th percentile of the recent model times; seconds from reading a frame to its result; samples skipped
        # because the model was still busy
        "inference_p90_ms": 0.0,
        "lag_s": 0.0,
        "samples_dropped": 0,
    }

    frames_dir = Path(output_dir) / drone_folder(drone or "drone") / time.strftime("live_%Y%m%d_%H%M%S")
    frames_dir.mkdir(parents=True, exist_ok=True)

    last_results = []
    consecutive_failures = 0
    latency = LatencyTracker()

    # the newest sample the reader thread has handed over, and whether the stream has ended
    newest = {"sample": None, "done": False, "error": None}
    handoff = threading.Condition()

    # the reader stops when the operator stops the feed, or when the analysis loop ends on an error
    halt = threading.Event()

    class Stop:
        @staticmethod
        def is_set():
            return stop_event.is_set() or halt.is_set()

    def frame_read(frame_index, frame, kept):
        # frames between samples go straight to the Live view, with the latest detections drawn on them
        # frame_index restarts after a reconnect, so count reads here
        stats["frames_read"] += 1
        stats["time"] = time.time() - started
        if on_frame is not None and not kept:
            on_frame(frame_index, frame, last_results, False, receiver.at_time(time.time()), stats["time"])

    def read_stream():
        try:
            for sample in sample_frame(stream_url, receiver, analysis_fps, bus, Stop, on_frame=frame_read,
                                       drone=drone):
                with handoff:
                    if newest["sample"] is not None:
                        stats["samples_dropped"] += 1
                    newest["sample"] = sample
                    handoff.notify()
        except Exception as error:  # hand it to the analysis loop, which reports it
            newest["error"] = error
        finally:
            with handoff:
                newest["done"] = True
                handoff.notify()

    def next_sample():
        with handoff:
            while newest["sample"] is None and not newest["done"]:
                handoff.wait(0.5)
            sample, newest["sample"] = newest["sample"], None
            return sample

    reader = threading.Thread(target=read_stream, name=f"stream {drone}", daemon=True)
    reader.start()

    try:
        while True:
            sample = next_sample()
            if sample is None:
                if newest["error"] is not None:
                    raise newest["error"]
                break
            frame, frame_index, frame_time, telemetry_row = sample
            results = []
            inference_started = time.monotonic()
            try:
                results = run_model(model, frame, min_confidence)
            except Exception as e:
                consecutive_failures += 1
                bus.dispatch(PipelineStatus(
                    source="human_detection", status="frame_skipped", drone=drone,
                    message=f"Failed to process frame {frame_index}: {type(e).__name__}: {e}"))
                if consecutive_failures == MAX_CONSECUTIVE_FAILED_FRAMES:
                    bus.dispatch(PipelineStatus(
                        source="human_detection", status="persistent_frames_skipped", drone=drone,
                        message=f"{MAX_CONSECUTIVE_FAILED_FRAMES} frames in a row failed analysis, last error: "
                                f"{type(e).__name__}: {e}. Detection is still running"))
            else:
                consecutive_failures = 0
            last_results = results
            latency.record(time.monotonic() - inference_started)
            stats["inference_p90_ms"] = latency.p90_ms()
            stats["lag_s"] = max(0.0, time.time() - frame_time)
            stats["consecutive_failures"] = consecutive_failures
            stats["telemetry_lost"] = telemetry_row is None
            stats["frames_checked"] += 1
            stats["detections"] += len(results)
            elapsed = frame_time - started
            if elapsed > 0:
                stats["fps"] = stats["frames_read"] / elapsed

            if on_frame is not None:
                on_frame(frame_index, frame, results, True, telemetry_row, elapsed)
            if on_progress is not None:
                on_progress(dict(stats))
            if not results:
                continue

            frame_path = frames_dir / f"frame_{frame_index:06d}.jpg"
            # encrypted when the console turned encryption on (raven.backend.vault)
            if not write_image(frame_path, frame):
                print(f"Failed to save frame {frame_index} to {frame_path}, detection not published", flush=True)
                continue

            location, location_status = geotag(telemetry_row, results, frame.shape[1], frame.shape[0])
            bus.dispatch(HumanDetected(
                frame=str(frame_path),
                frame_index=frame_index,
                time=frame_time,
                telemetry=telemetry_row,
                bounding_boxes=results,
                drone=drone or "",
                source=str(stream_url),
                location=location,
                location_status=location_status,
            ))
            stats["events_published"] += 1
    finally:
        halt.set()
        reader.join(timeout=15)
        receiver.close()

    if on_progress is not None:
        on_progress(dict(stats))
    return stats
