import cv2
import csv
import redis
import json
from pathlib import Path
from ultralytics import YOLO

# get the root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent

# get the path to the videos directory
VIDEOS_DIR = ROOT_DIR / "videos"

# save the frames with detections here, one subfolder per video
OUTPUT_DIR = ROOT_DIR / "runs" / "ingest"

# the trained person detector
MODEL_PATH = ROOT_DIR / "runs/detect/runs/train/yolov8n_raven/weights/best.pt"

# name the redis stream
REDIS_STREAM_NAME = "raven:detections"

def ingest_video(video_path, telemetry_data, model, redis_client, min_confidence, frame_stride):
    """
    Read the video frame by frame and run the model on every frame_stride-th frame.
    Each sampled frame with people is saved to disk, matched to its telemetry row,
    and published to the Redis stream as one event.
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

    # create the output directory for this video's frames, named after the video without extension,
    # so frames from different videos don't overwrite each other
    frames_dir = OUTPUT_DIR / Path(video_path).stem
    frames_dir.mkdir(parents=True, exist_ok=True)

    while True:
        # Read a frame from the video
        ok, frame = capture.read()
            
        # If the frame was not read successfully, break the loop
        if not ok:
            break

        # count every frame read, including the ones we skip, so the time stays correct
        frame_index += 1

        # only run the model on every frame_stride-th frame
        if frame_index % frame_stride != 0:
            continue

        # detect people in the frame using the YOLO model
        results = detect_objects(model, frame, min_confidence)

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
            "frame_index": frame_index,
            "time": current_time,
            "telemetry": telemetry_row,
            "bounding_boxes": results,
        }

        # convert the event to a JSON string and push it to the Redis stream
        redis_client.xadd(REDIS_STREAM_NAME, {"event": json.dumps(event)}, maxlen = 10000, approximate = True)

    capture.release()

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
    # get the path to the video to ingest and the telemetry data path
    video_path = VIDEOS_DIR / "hit-uav.mp4"
    telemetry_path = VIDEOS_DIR / "hit-uav_telemetry.csv"
    
    # load the telemetry
    telemetry_data = load_telemetry(telemetry_path)
    
    # connect to redis
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

    # load the model
    model = YOLO(str(MODEL_PATH))

    print("Model is loaded and ready to detect objects.")

    # ingest the video, keeping detections at 50%+ confidence and checking every 5th frame
    ingest_video(video_path, telemetry_data, model, redis_client, min_confidence=0.5, frame_stride=5)