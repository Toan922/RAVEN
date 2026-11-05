import argparse
import cv2
import csv
import sys
import redis
import json
import time
from pathlib import Path

# get the root directory of the project, and make it importable when this file is run as a script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_DIR, DEFAULT_STREAM_NAME, drone_folder

# get the path to the videos directory
VIDEOS_DIR = ROOT_DIR / "videos"

# save the frames with detections here, one subfolder per video
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

# the trained person detector
MODEL_PATH = DEFAULT_MODEL_PATH

# name the redis stream
REDIS_STREAM_NAME = DEFAULT_STREAM_NAME

def probe_source(video_path):
    """
    Open a video file or stream link and return its basic info without reading it through.
    Returns {"width", "height", "fps", "frame_count"}; frame_count is 0 for live streams.
    Raises RuntimeError if the source can't be opened.
    """

    # open the source
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

def ingest_video(video_path, telemetry_data, model, redis_client, min_confidence, frame_stride,
                 stop_event=None, on_progress=None, output_dir=OUTPUT_DIR, stream_name=REDIS_STREAM_NAME,
                 drone=None, on_frame=None, realtime=False):
    """
    Read the video frame by frame and run the model on every frame_stride-th frame.
    Each sampled frame with people is saved to disk, matched to its telemetry row,
    and published to the Redis stream as one event.

    Optional hooks for the GUI:
    - stop_event: a threading.Event; the loop ends cleanly once it is set
    - on_progress: called with a stats dict after every sampled frame

    - on_frame: called with (frame_index, frame, boxes, checked) for every frame read, for a live view;
      boxes are the latest detections (from this frame when checked is True, else from the last checked frame)
    - realtime: read video files at their own frame rate, like a live drone, instead of as fast as possible
      (streams already arrive in real time)

    drone names the drone this feed comes from: it tags every event and keeps that drone's frames in their own
    folder, so several feeds can publish to the same stream at once.
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

    # running totals reported to on_progress, total_frames is 0 for live streams
    stats = {
        "frames_read": 0,
        "frames_checked": 0,
        "detections": 0,
        "events_published": 0,
        "total_frames": max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        "fps": fps,
        "time": 0.0,
    }

    # create the output directory for this video's frames, named after the video without extension,
    # so frames from different videos don't overwrite each other
    frames_dir = Path(output_dir) / Path(str(video_path)).stem
    if drone:
        frames_dir = Path(output_dir) / drone_folder(drone) / Path(str(video_path)).stem
    frames_dir.mkdir(parents=True, exist_ok=True)

    # the latest detections, shown on the frames between checks in a live view
    last_results = []

    # when to show each frame in real time: frame i is due i / fps seconds after the start
    started = time.monotonic()
    pace = realtime and stats["total_frames"] > 0

    while True:
        # stop if the operator asked to
        if stop_event is not None and stop_event.is_set():
            break

        # Read a frame from the video
        ok, frame = capture.read()

        # If the frame was not read successfully, break the loop
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
        if checked:
            # detect people in the frame using the YOLO model
            results = detect_objects(model, frame, min_confidence)
            last_results = results
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

        # skip if no people are detected in the frame
        if not results:
            continue

        # get the time of this frame in seconds and the telemetry row for that moment
        current_time = frame_index / fps
        telemetry_row = telemetry_at_time(telemetry_data, current_time)

        # save the frame
        frame_path = frames_dir / f"frame_{frame_index:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)

        # create the event with the frame, its telemetry and the detected people, in the format subscribers expect
        event = {
            "frame": str(frame_path),
            # the video or stream the frame came from, so footage can be reviewed later
            "source": str(video_path),
            # which drone sent it, so alerts from several feeds can be told apart
            "drone": drone or "",
            "frame_index": frame_index,
            "time": current_time,
            "telemetry": telemetry_row,
            "bounding_boxes": results,
        }

        # convert the event to a JSON string and push it to the Redis stream
        redis_client.xadd(stream_name, {"event": json.dumps(event)}, maxlen = 10000, approximate = True)
        stats["events_published"] += 1

    capture.release()

    # final report so the totals include the last frames read after the last checked frame
    if on_progress is not None:
        on_progress(dict(stats))

    return stats

def load_telemetry(telemetry_path):
    """
    Load the telemetry rows from a CSV file, with every value as a float, sorted by time (t).
    """

    # initialize an empty list to store the telemetry rows
    rows = []

    # read the telemetry data from the CSV file
    with open(telemetry_path, newline="") as f:
        # read the csv file line by line
        reader = csv.DictReader(f)

        # stop early if the columns needed to geotag a frame are missing
        missing = {"t", "latitude", "longitude"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{telemetry_path} is missing columns: {sorted(missing)}")

        for row in reader:
            # convert the values in the row to floats, a bad value raises an error naming it
            for key in row:
                row[key] = float(row[key])

            # append the row to the rows list
            rows.append(row)

    # telemetry_at_time needs at least one row
    if not rows:
        raise ValueError(f"{telemetry_path} has no telemetry rows")

    # sort the rows by the time column and return them
    return sorted(rows, key=lambda row: row["t"])

def telemetry_at_time(telemetry_data, time):
    """
    Return the latest telemetry row at or before the given time, never one from the future.
    If the time is before the first row, return the first row.
    """

    # start from the first row
    latest_row = telemetry_data[0]

    # walk forward through the sorted rows until the next one is in the future
    for row in telemetry_data:
        if row["t"] <= time:
            latest_row = row
        else:
            break
    
    # return the telemetry data at the latest time
    return latest_row

def detect_objects(model, frame, min_confidence):
    """
    Detect people in a frame using the YOLO model.
    Return a list of {"box_xyxy": [x1, y1, x2, y2], "confidence": float}, the same format as discord_bot/mock_alert.py
    """

    # run the model on the frame and get the results
    model_results = model(frame, conf=min_confidence, verbose = False)[0]

    # list of detections in our own plain format
    results = []

    # pair each box with its confidence score
    for box, confidence in zip(model_results.boxes.xyxy, model_results.boxes.conf):
        # convert the box tensor to a list of floats
        box = [float(x) for x in box]
        
        # append the box and confidence to the results list
        results.append({"box_xyxy": box, "confidence": float(confidence)})
    
    # return the results
    return results

if __name__ == "__main__":
    # pick the feed on the command line, so several drones can run at once (one process each)
    parser = argparse.ArgumentParser(description="Detect people in one drone feed and publish alerts to Redis")
    parser.add_argument("--video", default=str(VIDEOS_DIR / "hit-uav.mp4"), help="video file or stream link")
    parser.add_argument("--telemetry", default=str(VIDEOS_DIR / "hit-uav_telemetry.csv"), help="telemetry CSV")
    parser.add_argument("--drone", default="Drone 1", help="drone name that tags every alert")
    parser.add_argument("--confidence", type=float, default=0.5, help="minimum detection confidence")
    parser.add_argument("--stride", type=int, default=5, help="check every Nth frame")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args()

    # imported here so the GUI (and --help) work without loading the model library
    from ultralytics import YOLO

    # load the telemetry
    telemetry_data = load_telemetry(args.telemetry)

    # connect to redis
    redis_client = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)

    # load the model
    model = YOLO(str(MODEL_PATH))

    print(f"Model is loaded and ready to detect objects on {args.drone}.")

    ingest_video(args.video, telemetry_data, model, redis_client, min_confidence=args.confidence,
                 frame_stride=args.stride, drone=args.drone)
