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

    # the detected people, a list of {"box_xyxy": [x1, y1, x2, y2], "confidence": float}, plus
    # "location": {"latitude", "longitude"} on each box whose ground position geotag could estimate
    bounding_boxes: list[dict]

    # optional, set by the operator console, which can run several drone feeds at once:
    # which drone sent the frame, so alerts from several feeds can be told apart ("" for a single feed)
    drone: str = ""

    # the video file or stream link the frame came from, so its footage can be reviewed later
    source: str = ""

    # seconds into the video when the frame was read (frame_index / fps), or None when not tracked
    video_time: float | None = None

    # where the person is, see raven.backend.geotag (FR-08, FR-09): {"latitude", "longitude"} or None
    location: dict[str, float] | None = None

    # how location was found: "estimated" (ground position), "approximate" (the drone's own position) or
    # "unavailable" (no position data); "" when not geotagged, then telemetry is the drone's position
    location_status: str = ""

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

    # optional: which drone feed the status is about ("" for a single feed)
    drone: str = ""