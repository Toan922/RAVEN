"""
Human Detection and Threshold Check filters (FR-07, FR-10.1, FR-14).
Runs the trained YOLO person detector on each sampled frame and passes on only frames with people
at or above the configured confidence threshold.
Frames that fail analysis are skipped and reported on the bus, and the operator is notified if
failures persist (FR-14.1, FR-14.2).
"""

import numpy as np
from pathlib import Path
from ultralytics import YOLO
from raven.backend.events import PipelineStatus

# get the root directory of the project
ROOT_DIR = Path(__file__).resolve().parents[3]

# the trained person detector
MODEL_PATH = ROOT_DIR / "runs/detect/runs/train/yolov8n_raven/weights/best.pt"

# how many frames in a row can fail analysis before the operator is notified (FR-14.2)
MAX_CONSECUTIVE_FAILED_FRAMES = 3

def load_model(bus):
    """
    Load the trained YOLO person detector from the specified MODEL_PATH.
    If loading succeeds, return the loaded model.
    If loading fails, publish a model_unavailable PipelineStatus event and re-raise the error.
    """
    
    try:
        model = YOLO(str(MODEL_PATH))
        # warm up the model with a dummy input to ensure it is ready for inference
        model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        return model
    except Exception as e:
        # build the error message
        error_message = f"Failed to load model from {MODEL_PATH}: {type(e).__name__}: {e}"
        bus.dispatch(PipelineStatus(source="human_detection", status="model_unavailable", message=error_message))
        raise

def run_model(model, frame, min_confidence):
    """
    Detect people in a frame using the YOLO model.
    Return a list of {"box_xyxy": [x1, y1, x2, y2], "confidence": float}, the format HumanDetected.bounding_boxes expects
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

def detect_humans(samples, model, min_confidence, bus):
    """
    Call run_model on each (frame, frame_index, frame_time, telemetry_row) from samples, e.g. sample_frame.
    Yields (frame, frame_index, frame_time, telemetry_row, boxes) only for frames with at least one person
    at or above min_confidence (FR-10.1), where boxes is the list returned by run_model.
    If a frame fails analysis, publishes frame_skipped and moves on to the next frame (FR-14.1).
    On the MAX_CONSECUTIVE_FAILED_FRAMES-th failure in a row, also publishes one persistent_frames_skipped
    and keeps going (FR-14.2). A successful frame resets the count, so a later streak is reported again.
    """

    # failed frames in a row, reset by any successful frame
    consecutive_failures = 0

    for frame, frame_index, frame_time, telemetry_row in samples:
        try:
            boxes = run_model(model, frame, min_confidence)
        except Exception as e:
            # count as a failure
            consecutive_failures += 1
            
            # publish the frame skipped PipelineStatus event (FR-14.1)
            error_message = f"Failed to process frame {frame_index}: {type(e).__name__}: {e}"
            bus.dispatch(PipelineStatus(source="human_detection", status="frame_skipped", message=error_message))

            # notify once per streak, on exactly the MAX_CONSECUTIVE_FAILED_FRAMES-th failure, not on every failure after it
            if consecutive_failures == MAX_CONSECUTIVE_FAILED_FRAMES:
                # publish the persistent frames skipped PipelineStatus event (FR-14.2)
                bus.dispatch(PipelineStatus(source="human_detection", status="persistent_frames_skipped", message=f"{MAX_CONSECUTIVE_FAILED_FRAMES} frames in a row failed analysis, last error: {type(e).__name__}: {e}. Detection is still running"))

            # continue to the next frame
            continue

        # reset the consecutive failures counter on success
        consecutive_failures = 0

        # threshold check (FR-10.1): run_model only returns boxes at or above min_confidence,
        # so frames with no boxes have nobody in them and are dropped here
        if boxes:
            yield frame, frame_index, frame_time, telemetry_row, boxes