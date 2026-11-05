"""
Live view (Feed tab > Live): the frames a drone feed is being read from right now, with the latest detections,
for one drone at a time picked from a dropdown.
"""
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from raven.gui.alerts import AlertModel, drone_colors
from raven.gui.alerts_panel import drone_icon
from raven.gui.elided_label import ElidedLabel
from raven.gui.video_canvas import VideoCanvas


def _clock(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


# what the banner pill says for each feed state, and its color
STATES = {
    "idle": ("OFFLINE", "#8b8d98"),
    "starting": ("STARTING", "#f5a524"),
    "running": ("LIVE", "#e5484d"),
    "finished": ("ENDED", "#8b8d98"),
    "stopped": ("STOPPED", "#8b8d98"),
    "failed": ("FAILED", "#e5484d"),
}


class LiveView(QWidget):
    """
    The video area for the Live subtab. Its banner part (drone dropdown, status pill, details) is
    self.banner, which the Feed tab places in its top banner.
    """
    review_requested = Signal(str)  # alert id

    def __init__(self, model: AlertModel, parent=None):
        super().__init__(parent)
        self.model = model
        self._names: list[str] = []
        self._frames: dict[str, tuple[QImage, dict]] = {}
        self._states: dict[str, str] = {}
        self._messages: dict[str, str] = {}

        # ---- banner: drone dropdown, status pill, what the drone is doing
        self.drone_combo = QComboBox()
        self.drone_combo.setMinimumWidth(140)
        self.drone_combo.setToolTip("Which drone to watch")
        self.drone_combo.currentIndexChanged.connect(lambda _: self._render())
        self.state_pill = QLabel()
        self.details = ElidedLabel()  # changes every frame: cut short rather than widen the Feed tab
        self.details.setObjectName("hint")

        self.banner = QWidget()
        banner_layout = QHBoxLayout(self.banner)
        banner_layout.setContentsMargins(0, 0, 0, 0)
        banner_layout.addWidget(self.drone_combo)
        banner_layout.addWidget(self.state_pill, 0, Qt.AlignVCenter)
        banner_layout.addWidget(self.details, 1)

        # ---- video, and the drone's latest alert underneath
        self.canvas = VideoCanvas()
        self.latest_label = ElidedLabel()
        self.latest_label.setObjectName("hint")
        self.review_button = QPushButton("Review latest alert")
        self.review_button.setToolTip("Open this drone's newest alert in Review")
        self.review_button.clicked.connect(self._review_latest)

        bottom = QHBoxLayout()
        bottom.addWidget(self.latest_label, 1)
        bottom.addWidget(self.review_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(bottom)

        for signal in (model.rowsInserted, model.modelReset):
            signal.connect(lambda *_: self._update_latest())
        self._render()

    # ------------------------------------------------------------------ feeds

    def drone_names(self) -> list[str]:
        return list(self._names)

    def status(self, name: str) -> tuple[str, dict | None]:
        """
        A drone's feed state and the details of its latest frame (None before its first frame).
        """
        frame = self._frames.get(name)
        return self._states.get(name, "idle"), frame[1] if frame else None

    def current(self) -> str | None:
        return self.drone_combo.currentData()

    def select(self, name: str):
        index = self.drone_combo.findData(name)
        if index >= 0:
            self.drone_combo.setCurrentIndex(index)

    def set_feeds(self, feeds):
        """
        Offer the feeds that are turned on in Settings, plus any drone still showing a frame from this session.
        """
        names = [feed.name.strip() for feed in feeds if feed.enabled and feed.name.strip()]
        for name in self._frames:
            if name not in names and self._states.get(name) == "running":
                names.append(name)
        if names == self._names:
            self._refresh_icons()
            return
        current = self.current()
        self._names = names
        self.drone_combo.blockSignals(True)
        self.drone_combo.clear()
        for name in names:
            self.drone_combo.addItem(drone_icon(drone_colors.color(name)), name, name)
        index = self.drone_combo.findData(current)
        self.drone_combo.setCurrentIndex(index if index >= 0 else 0)
        self.drone_combo.blockSignals(False)
        self._render()

    def _refresh_icons(self):
        for i in range(self.drone_combo.count()):
            self.drone_combo.setItemIcon(i, drone_icon(drone_colors.color(self.drone_combo.itemData(i))))

    def set_state(self, name: str, state: str, message: str = ""):
        """
        A feed started, is running, or ended (state is a key of STATES).
        """
        self._states[name] = state
        self._messages[name] = message
        if state == "starting":
            self._frames.pop(name, None)  # a new run: don't show the last run's final frame
        if name == self.current():
            self._render()

    def on_frame(self, name: str, image: QImage, info: dict):
        self._frames[name] = (image, info)
        if self._states.get(name) != "running":
            self._states[name] = "running"
        if name == self.current():
            # a new frame doesn't change the latest alert (new alerts update it on their own)
            self._render(update_latest=False)

    # ------------------------------------------------------------------ drawing

    def _render(self, update_latest=True):
        name = self.current()
        state = self._states.get(name, "idle") if name else "idle"
        label, color = STATES[state]
        self.state_pill.setText(f" {label} ")
        self.state_pill.setStyleSheet(f"background:{color}; color:white; border-radius:8px; font-weight:bold;"
                                      "padding:1px 6px;")
        self.state_pill.setToolTip(self._messages.get(name, "") if name else "")

        if not name:
            self.details.setText("No drone feeds are turned on. Add one in Settings.")
            self.canvas.show_frame(None, message="No drone feeds are turned on. Add one in Settings > Drone feeds.")
            self._update_latest()
            return

        frame = self._frames.get(name)
        if frame is None:
            self.details.setText("")
            waiting = {
                "starting": f"Connecting to {name}…",
                "failed": f"{name} failed: {self._messages.get(name, '')}",
            }.get(state, f"{name} isn't running. Press Start under the map (or F5) to start the feeds.")
            self.canvas.show_frame(None, message=waiting)
            self._update_latest()
            return

        image, info = frame
        position = _clock(info["time"])
        if info["total_frames"] and info["fps"]:
            position += f" / {_clock(info['total_frames'] / info['fps'])}"
        people = len(info["boxes"])
        parts = [position, f"frame {info['frame_index'] + 1}",
                 "1 person in view" if people == 1 else f"{people} people in view"]
        if isinstance(info.get("lat"), (int, float)) and isinstance(info.get("lon"), (int, float)):
            parts.append(f"drone at {info['lat']:.5f}, {info['lon']:.5f}")
        source = info.get("source", "")
        parts.append(source if "://" in source else Path(source).name)
        self.details.setText(" · ".join(parts))

        overlay = "" if state == "running" else {
            "finished": "Feed ended. Showing its last frame.",
            "stopped": "Feed stopped. Showing its last frame.",
            "failed": f"Feed failed: {self._messages.get(name, '')}",
        }.get(state, "")
        # boxes are solid on a frame the detector checked, dashed while carried to the next check
        self.canvas.show_frame(image, info["boxes"], held=not info["checked"], message=overlay, centered=True)
        if update_latest:
            self._update_latest()

    def _latest_alert(self):
        name = self.current()
        # the model lists alerts newest first
        return next((alert for alert in self.model.alerts() if alert.drone == name), None) if name else None

    def _update_latest(self):
        alert = self._latest_alert()
        self.review_button.setEnabled(alert is not None)
        if alert is None:
            self.latest_label.setText("No alerts from this drone yet.")
        else:
            self.latest_label.setText(f"Latest alert: {alert.title} at video {alert.video_time} "
                                      f"(received {alert.received:%H:%M:%S})")

    def _review_latest(self):
        alert = self._latest_alert()
        if alert is not None:
            self.review_requested.emit(alert.id)
