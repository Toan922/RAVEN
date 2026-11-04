"""
Events published on RAVEN's Detection Event Bus.
The pipeline (video ingestion -> frame sampling -> human detection -> threshold check) publishes these events,
and subscribers such as Alert Notification and Detection Log react to them without blocking the pipeline.
Each event is a bubus BaseEvent, so subscribers register for an event by its class instead of parsing JSON.
"""

from bubus import BaseEvent
from typing import Literal

class HumanDetected(BaseEvent):
    """
    The event with the frame, its telemetry and the detected people, in the format subscribers expect.
    Published once per sampled frame that has detections at or above the confidence threshold.
    """

    # path to the saved frame on disk
    frame: str

    # index of the frame in the video, counting every frame read, including the ones skipped
    frame_index: int

    # wall-clock time the frame was read (time.time(), seconds since the epoch), the same clock telemetry is stamped with
    time: float

    # the newest live telemetry row when the frame was read, or None if none arrived within 2 s, see TelemetryReceiver.at_time
    telemetry: dict[str, float] | None

    # the detected people, a list of {"box_xyxy": [x1, y1, x2, y2], "confidence": float}
    bounding_boxes: list[dict]

class PipelineStatus(BaseEvent):
    """
    The event representing the status of the detection pipeline.
    Published when a feed or analysis problem happens or clears.
    """

    # the source of the status update, either "video_ingestion" or "human_detection"
    source: Literal[
        "video_ingestion", 
        "human_detection"
    ]

    # the current status of the pipeline, based on requirements doc
    status: Literal[
        "feed_unsupported",             # maps to FR-04.2 w/ video not being decoded
        "feed_missing",                 # maps to FR-13.1 w/ no feed available w/ in 10 seconds of starting
        "feed_interrupted",             # maps to FR-13.2 w/ feed being dropped mid-ingestion
        "feed_resumed",                 # maps to FR-13.2 w/ feed being resumed after interruption
        "frame_skipped",                # maps to FR-14.1 w/ single frame analysis failed -> log only
        "persistent_frames_skipped",    # maps to FR-14.2 w/ 3 consecutive frames being skipped
        "model_unavailable"             # maps to FR-14.2 w/ the detection model being unavailable
    ]

    # details regarding the status change
    message: str