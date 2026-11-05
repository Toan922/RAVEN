"""
Detection Log subsystem: subscribes to HumanDetected on the Detection Event Bus and appends each event to a
JSON Lines file (one JSON object per line), so the alerts of earlier runs can be shown again after a restart.
"""

import json
import threading
from pathlib import Path

# the HumanDetected fields kept in the log, besides the event's id
LOGGED_FIELDS = ("frame", "frame_index", "time", "telemetry", "bounding_boxes", "drone", "source", "video_time")

# how many of the newest records read_log returns, so a long-lived log doesn't flood the alert list
MAX_REPLAYED = 10000


def event_record(event) -> dict:
    """
    A HumanDetected event as a plain dict: its bubus event_id as "id", plus LOGGED_FIELDS.
    """
    record = {"id": str(event.event_id)}
    for name in LOGGED_FIELDS:
        record[name] = getattr(event, name, None)
    return record


class DetectionLog:
    """
    Appends every HumanDetected event to the log file at path. The path can be changed between runs.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def on_human_detected(self, event):
        """
        Bus handler for HumanDetected. The write is a single short line, so it runs right on the bus thread.
        """
        line = json.dumps(event_record(event))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


def read_log(path, limit=MAX_REPLAYED) -> tuple[list[dict], int]:
    """
    Read the log at path, oldest first. Returns (records, skipped), where records holds at most the newest limit
    records and skipped counts lines that weren't valid records (e.g. a line cut short by a crash).
    A missing file is an empty log.
    """
    path = Path(path)
    if not path.is_file():
        return [], 0

    records, skipped = [], 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            if isinstance(record, dict) and record.get("id") and record.get("frame"):
                records.append(record)
            else:
                skipped += 1
    return records[-limit:], skipped
