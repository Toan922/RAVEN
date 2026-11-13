"""
Ingest for the operator console: reads a recorded video file (or a stream link) with a telemetry CSV,
runs the person detector on about analysis_fps frames per second of video, and publishes a HumanDetected event on the
Detection Event Bus for each checked frame with people in it.
stream_ingest is the live counterpart: a UDP stream from the ground station with live UDP telemetry.
"""

import csv
import time
from pathlib import Path

import cv2

from raven.backend.config import drone_folder, stride_for
from raven.backend.detection import MAX_CONSECUTIVE_FAILED_FRAMES, run_model
from raven.backend.events import HumanDetected, PipelineStatus
from raven.backend.geotag import geotag
from raven.backend.latency import LatencyTracker
from raven.backend.vault import write_image


def probe_source(video_path):
    """
    Open a video file or stream link and return its basic info without reading it through.
    Returns {"width", "height", "fps", "frame_count"}; frame_count is 0 for live streams.
    Raises RuntimeError if the source can't be opened.
    """

    # open the source; a stream link gets frame_sampler's time limits so a dead link fails instead of hanging
    if "://" in str(video_path):
        from raven.backend.frame_sampler import STREAM_TIMEOUT_MSEC, open_stream

        capture = open_stream(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"No stream at {video_path} within {STREAM_TIMEOUT_MSEC // 1000} s")
    else:
        capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # read the properties, live streams report 0 (or negative) frames
    info = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": capture.get(cv2.CAP_PROP_FPS) or 0.0,
        "frame_count": max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
    }
    capture.release()
    return info


def check_decodes(video_path):
    """
    Make sure a video file opens and its first frame decodes (FR-04.2), so an unsupported file is caught before
    a run starts. Raises RuntimeError naming the problem.
    """
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"can't open {Path(str(video_path)).name}: not a video format RAVEN can read")
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError(f"can't decode {Path(str(video_path)).name}: the video codec isn't supported")
    finally:
        capture.release()


def ingest_video(video_path, telemetry_data, model, bus, min_confidence, analysis_fps, output_dir,
                 stop_event=None, on_progress=None, drone="", on_frame=None, realtime=False):
    """
    Read the video frame by frame and run the model on about analysis_fps frames per second of video (FR-05).
    Each checked frame with people is saved to disk, matched to its telemetry row, and published on bus
    as a HumanDetected event. The frame is saved before the event is published, so subscribers can always
    open HumanDetected.frame. bus must be safe to call from this thread (a ThreadSafeBus).

    A frame the model fails on is skipped and reported as frame_skipped; the MAX_CONSECUTIVE_FAILED_FRAMES-th
    failure in a row is also reported as persistent_frames_skipped (FR-14.1, FR-14.2), as raven.backend.detection does.

    Optional hooks for the GUI:
    - stop_event: a threading.Event; the loop ends cleanly once it is set
    - on_progress: called with a stats dict after every checked frame
    - on_frame: called with (frame_index, frame, boxes, checked) for every frame read, for a live view;
      boxes are the latest detections (from this frame when checked is True, else from the last checked frame)
    - realtime: read video files at their own frame rate, like a live drone, instead of as fast as possible
      (streams already arrive in real time)

    drone names the drone this feed comes from: it tags every event and keeps that drone's frames in their own
    folder, so several feeds can run at once.
    """

    # open the video
    capture = cv2.VideoCapture(str(video_path))

    # stop with a clear error if the video can't be opened (wrong path, bad stream URL, ...)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # frame counter, starts at -1 so the first frame read is index 0
    frame_index = -1

    # get the frames per second of the video, some streams report 0 so fall back to 30 to avoid dividing by zero
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    # step this many frames between checks, so about analysis_fps frames are checked each second
    frame_stride = stride_for(fps, analysis_fps)

    # running totals reported to on_progress, total_frames is 0 for live streams
    stats = {
        "frames_read": 0,
        "frames_checked": 0,
        "detections": 0,
        "events_published": 0,
        "total_frames": max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        "fps": fps,
        "frame_stride": frame_stride,
        "time": 0.0,
        # checked frames in a row the model failed on, 0 once one succeeds again
        "consecutive_failures": 0,
        # 90th percentile of the recent model times, and how far behind real time a real-speed run is
        "inference_p90_ms": 0.0,
        "lag_s": 0.0,
    }

    # this video's frames go in <output_dir>/<drone>/<video name>, so drones and videos don't overwrite each other
    frames_dir = Path(output_dir) / Path(str(video_path)).stem
    if drone:
        frames_dir = Path(output_dir) / drone_folder(drone) / Path(str(video_path)).stem
    frames_dir.mkdir(parents=True, exist_ok=True)

    # the latest detections, shown on the frames between checks in a live view
    last_results = []

    # failed frames in a row, reset by any successful frame
    consecutive_failures = 0
    latency = LatencyTracker()

    # when to show each frame in real time: frame i is due i / fps seconds after the start
    started = time.monotonic()
    pace = realtime and stats["total_frames"] > 0

    try:
        while True:
            # stop if the operator asked to
            if stop_event is not None and stop_event.is_set():
                break

            # read a frame from the video, and stop at the end of it
            ok, frame = capture.read()
            if not ok:
                break

            # count every frame read, including the ones we skip, so the time stays correct
            frame_index += 1
            stats["frames_read"] = frame_index + 1
            stats["time"] = frame_index / fps

            # wait until this frame is due when playing a file at real speed
            if pace:
                delay = started + frame_index / fps - time.monotonic()
                if delay > 0:
                    if stop_event is not None:
                        stop_event.wait(delay)
                    else:
                        time.sleep(delay)

            # only run the model on every frame_stride-th frame
            checked = frame_index % frame_stride == 0
            results = []
            if checked:
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
                if pace:
                    stats["lag_s"] = max(0.0, time.monotonic() - (started + frame_index / fps))
                stats["consecutive_failures"] = consecutive_failures
                stats["frames_checked"] += 1
                stats["detections"] += len(results)

            # hand every frame to the live view, with the latest detections
            if on_frame is not None:
                on_frame(frame_index, frame, last_results, checked)

            if not checked:
                continue

            # report progress after every checked frame
            if on_progress is not None:
                on_progress(dict(stats))

            # threshold check (FR-10.1): run_model only returns people at or above min_confidence
            if not results:
                continue

            # get the time of this frame in the video and the telemetry row for that moment
            video_time = frame_index / fps
            telemetry_row = telemetry_at_time(telemetry_data, video_time)

            # save the frame; if that fails, nothing is published for it
            frame_path = frames_dir / f"frame_{frame_index:06d}.jpg"
            # encrypted when the console turned encryption on (raven.backend.vault)
            if not write_image(frame_path, frame):
                print(f"Failed to save frame {frame_index} to {frame_path}, detection not published", flush=True)
                continue

            # estimated ground position of the most confident person (FR-08, FR-09)
            location, location_status = geotag(telemetry_row, results, frame.shape[1], frame.shape[0])

            bus.dispatch(HumanDetected(
                frame=str(frame_path),
                frame_index=frame_index,
                time=time.time(),
                telemetry=telemetry_row,
                bounding_boxes=results,
                drone=drone or "",
                source=str(video_path),
                video_time=video_time,
                location=location,
                location_status=location_status,
            ))
            stats["events_published"] += 1
    finally:
        capture.release()

    # final report so the totals include the last frames read after the last checked frame
    if on_progress is not None:
        on_progress(dict(stats))

    return stats


def load_telemetry(telemetry_path):
    """
    Load the telemetry rows from a CSV file, with every value as a float, sorted by time (t, seconds into the video).
    """

    rows = []
    with open(telemetry_path, newline="") as f:
        reader = csv.DictReader(f)

        # stop early if the columns needed to geotag a frame are missing
        missing = {"t", "latitude", "longitude"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{telemetry_path} is missing columns: {sorted(missing)}")

        for row in reader:
            # convert the values in the row to floats, a bad value raises an error naming it
            for key in row:
                row[key] = float(row[key])
            rows.append(row)

    # telemetry_at_time needs at least one row
    if not rows:
        raise ValueError(f"{telemetry_path} has no telemetry rows")

    return sorted(rows, key=lambda row: row["t"])


def telemetry_at_time(telemetry_data, time):
    """
    Return the latest telemetry row at or before the given video time, never one from the future.
    If the time is before the first row, return the first row.
    """

    latest_row = telemetry_data[0]

    # walk forward through the sorted rows until the next one is in the future
    for row in telemetry_data:
        if row["t"] <= time:
            latest_row = row
        else:
            break

    return latest_row
