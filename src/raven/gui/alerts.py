"""
Alert data for the GUI: one Alert per detection event, a list model for the sidebar,
and helpers that draw the detected boxes on saved frames.
"""
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QAbstractListModel, QModelIndex, QRectF, QSortFilterProxyModel, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap

ALERT_ROLE = Qt.UserRole + 1

STATUSES = ("new", "reviewed", "dismissed")
STATUS_COLORS = {"new": "#e5484d", "reviewed": "#30a46c", "dismissed": "#8b8d98"}
BOX_COLOR = QColor("#e5484d")

# one color per drone on cards and map pins, kept apart from the status colors above
DRONE_PALETTE = ["#0090ff", "#f76b15", "#8e4ec6", "#12a594", "#d6409f", "#ad7f58"]
UNNAMED_DRONE = "Unnamed drone"
# the first colors belong to the feeds in Settings (at most 3), by position
MAX_FEED_COLORS = 3


class DroneColors:
    """
    A color per drone. Feeds in the Settings list own the first colors by position (feed 1 blue, feed 2 orange,
    feed 3 purple), so renaming a feed keeps its color. Drones that only appear in alerts (older runs, other
    consoles) share the remaining colors, in order of first appearance.
    """

    def __init__(self):
        self._feeds: dict[str, str] = {}
        self._others: dict[str, str] = {}

    def set_feeds(self, names) -> bool:
        """
        Give the configured feeds their colors. Returns True if any drone's color changed.
        """
        before = dict(self._feeds)
        self._feeds = {}
        for i, name in enumerate(names):
            if name and name not in self._feeds:
                self._feeds[name] = DRONE_PALETTE[i % MAX_FEED_COLORS]
        return before != self._feeds

    def color(self, name: str) -> str:
        name = name or UNNAMED_DRONE
        if name in self._feeds:
            return self._feeds[name]
        if name not in self._others:
            spare = DRONE_PALETTE[MAX_FEED_COLORS:]
            self._others[name] = spare[len(self._others) % len(spare)]
        return self._others[name]

    def colors(self, names) -> dict:
        return {name: self.color(name) for name in names}


drone_colors = DroneColors()


def _natural_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


@dataclass
class Alert:
    id: str       # the HumanDetected event's bubus event_id
    event: dict   # the event as a plain dict, see raven.backend.detection_log.event_record
    status: str = "new"

    @property
    def boxes(self) -> list:
        return self.event.get("bounding_boxes", [])

    @property
    def people(self) -> int:
        return len(self.boxes)

    @property
    def confidence(self) -> float:
        return max((box.get("confidence", 0) for box in self.boxes), default=0.0)

    @property
    def location_status(self) -> str:
        """
        "estimated" (the person's ground position), "approximate" (the drone's position) or "unavailable".
        Alerts from before geotagging only have the drone's position, which is approximate.
        """
        status = self.event.get("location_status") or ""
        if status in ("estimated", "approximate", "unavailable"):
            return status
        telemetry = self.event.get("telemetry") or {}
        return "approximate" if "latitude" in telemetry and "longitude" in telemetry else "unavailable"

    @property
    def is_approximate(self) -> bool:
        return self.location_status == "approximate"

    def _position(self) -> dict:
        if self.location_status == "estimated" and self.event.get("location"):
            return self.event["location"]
        if self.location_status == "unavailable":
            return {}
        return self.event.get("telemetry") or {}

    @property
    def lat(self):
        return self._position().get("latitude")

    @property
    def lon(self):
        return self._position().get("longitude")

    @property
    def where(self) -> str:
        """
        The coordinates for cards and the log, marked when they're only the drone's position.
        """
        if not self.has_location:
            return "no location"
        text = f"{self.lat:.5f}, {self.lon:.5f}"
        return f"≈ {text} (approx.)" if self.is_approximate else text

    @property
    def has_location(self) -> bool:
        return isinstance(self.lat, (int, float)) and isinstance(self.lon, (int, float))

    @property
    def frame_path(self) -> str:
        return self.event.get("frame", "")

    @property
    def video_name(self) -> str:
        return Path(self.frame_path).parent.name if self.frame_path else ""

    @property
    def drone(self) -> str:
        # alerts from before feeds were named don't say which drone sent them
        return self.event.get("drone") or UNNAMED_DRONE

    @property
    def drone_color(self) -> str:
        return drone_colors.color(self.drone)

    @property
    def source(self) -> str:
        # the video/stream ingest read; events from before this field existed don't have it
        return self.event.get("source", "")

    @property
    def frame_index(self) -> int | None:
        index = self.event.get("frame_index")
        return int(index) if isinstance(index, (int, float)) else None

    @property
    def seconds(self) -> float:
        # seconds into the video; events from raven.pipeline's live stream don't track it
        return float(self.event.get("video_time") or 0.0)

    @property
    def video_time(self) -> str:
        return _clock(self.seconds)

    @property
    def is_live(self) -> bool:
        # live streams don't have a time into the video
        return self.event.get("video_time") is None

    @property
    def when(self) -> str:
        """
        "video 0:12" for a recorded video, "live 18:20:53" for a live stream.
        """
        return f"live {self.received:%H:%M:%S}" if self.is_live else f"video {self.video_time}"

    @property
    def received(self) -> datetime:
        # the wall-clock time the frame was read
        return datetime.fromtimestamp(float(self.event.get("time") or 0.0))

    @property
    def title(self) -> str:
        people = "1 person" if self.people == 1 else f"{self.people} people"
        return f"{people} · {self.confidence:.0%}"

    def map_payload(self) -> dict:
        return {"id": self.id, "lat": self.lat, "lon": self.lon, "status": self.status, "approx": self.is_approximate,
                "drone": self.drone, "color": self.drone_color,
                "label": f"{self.drone}: {self.title} · {self.when}"}


class AlertModel(QAbstractListModel):
    """
    Every alert received, newest first.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._alerts: list[Alert] = []
        self._by_id: dict[str, Alert] = {}

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._alerts)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        alert = self._alerts[index.row()]
        if role == ALERT_ROLE:
            return alert
        if role == Qt.DisplayRole:
            return alert.title
        if role == Qt.ToolTipRole:
            return f"{alert.drone} · {alert.frame_path}\nDouble-click to review the footage"
        return None

    def add_events(self, batch, statuses: dict) -> list[Alert]:
        """
        Add (event id, event) pairs, oldest first, skipping ids already in the list.
        statuses holds remembered reviewed/dismissed states. Returns the alerts that were added.
        """
        new = [Alert(event_id, event, statuses.get(event_id, "new"))
               for event_id, event in batch if event_id not in self._by_id]
        if not new:
            return []
        self.beginInsertRows(QModelIndex(), 0, len(new) - 1)
        self._alerts[0:0] = reversed(new)
        for alert in new:
            self._by_id[alert.id] = alert
        self.endInsertRows()
        return new

    def alerts(self) -> list[Alert]:
        return list(self._alerts)

    def get(self, alert_id: str) -> Alert | None:
        return self._by_id.get(alert_id)

    def row_of(self, alert_id: str) -> int:
        alert = self._by_id.get(alert_id)
        return self._alerts.index(alert) if alert else -1

    def set_status(self, alert_id: str, status: str):
        row = self.row_of(alert_id)
        if row < 0 or self._alerts[row].status == status:
            return
        self._alerts[row].status = status
        index = self.index(row)
        self.dataChanged.emit(index, index)

    def drones(self) -> list[str]:
        """
        Every drone with an alert in the list, sorted by name (Drone 2 before Drone 10).
        """
        return sorted({alert.drone for alert in self._alerts}, key=_natural_key)

    def count(self, status: str) -> int:
        return sum(1 for alert in self._alerts if alert.status == status)

    def clear(self):
        self.beginResetModel()
        self._alerts.clear()
        self._by_id.clear()
        self.endResetModel()


class AlertFilter(QSortFilterProxyModel):
    """
    Show all alerts, only one status, or everything except dismissed ("active"), optionally for one drone only.
    """
    MODES = {"Active": None, "All": None, "New": "new", "Reviewed": "reviewed", "Dismissed": "dismissed"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "Active"
        self._drone: str | None = None

    def set_drone(self, drone: str | None):
        self._drone = drone
        self.invalidateFilter()

    def drone(self) -> str | None:
        return self._drone

    def set_mode(self, mode: str):
        self._mode = mode
        self.invalidateFilter()

    def mode(self) -> str:
        return self._mode

    def filterAcceptsRow(self, row, parent):
        alert = self.sourceModel().index(row, 0, parent).data(ALERT_ROLE)
        if self._drone is not None and alert.drone != self._drone:
            return False
        if self._mode == "All":
            return True
        if self._mode == "Active":
            return alert.status != "dismissed"
        return alert.status == self.MODES[self._mode]


def load_frame(path: str) -> QImage:
    """
    A saved frame as a QImage, decrypted in memory (a null image if it's missing or can't be read).
    """
    from raven.backend.vault import VaultError, read_bytes

    try:
        return QImage.fromData(read_bytes(path)) if path else QImage()
    except (OSError, VaultError):
        return QImage()


def frame_with_boxes(alert: Alert, max_size=None) -> QPixmap | None:
    """
    Load the alert's saved frame and draw its bounding boxes, optionally scaled down to fit max_size (w, h).
    Boxes are drawn after scaling so they stay visible on small thumbnails. Returns None if the frame is missing.
    """
    image = load_frame(alert.frame_path)
    if image.isNull():
        return None

    scale = 1.0
    if max_size is not None:
        scale = min(max_size[0] / image.width(), max_size[1] / image.height(), 1.0)
        if scale < 1.0:
            image = image.scaled(round(image.width() * scale), round(image.height() * scale),
                                 Qt.KeepAspectRatio, Qt.SmoothTransformation)

    image = image.convertToFormat(QImage.Format_RGB32)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(BOX_COLOR)
    # same line width rule as the Discord alert image, with a floor for thumbnails
    pen.setWidthF(2 if max_size is not None else max(3, image.width() // 400))
    painter.setPen(pen)
    for box in alert.boxes:
        x1, y1, x2, y2 = (v * scale for v in box["box_xyxy"])
        painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
    painter.end()
    return QPixmap.fromImage(image)


class ThumbnailCache:
    """
    Thumbnails for the alert list, loaded the first time a card is drawn and kept for the most recent cards.
    """

    def __init__(self, size=(104, 78), capacity=600):
        self.size = size
        self._capacity = capacity
        self._items: OrderedDict[str, QPixmap | None] = OrderedDict()

    def get(self, alert: Alert) -> QPixmap | None:
        if alert.id in self._items:
            self._items.move_to_end(alert.id)
            return self._items[alert.id]
        pixmap = frame_with_boxes(alert, self.size)
        self._items[alert.id] = pixmap
        if len(self._items) > self._capacity:
            self._items.popitem(last=False)
        return pixmap

    def clear(self):
        self._items.clear()
