"""
Feed tab: review the footage behind an alert, with playback, frame-by-frame stepping, jumps between alerts,
and its own alerts panel (sharing the alert list and statuses with the Map tab).

Footage is decoded with OpenCV rather than a media player so stepping lands on exact frames,
and the detection boxes are drawn on the frames the detector actually checked.
"""
from bisect import bisect_left, bisect_right
from pathlib import Path

import cv2
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QSplitter, QStyle,
    QStyleOptionSlider, QToolButton, QVBoxLayout, QWidget,
)

from gui.alerts import STATUS_COLORS, Alert, AlertModel
from gui.alerts_panel import STRIP_WIDTH, AlertsSidebar
from gui.shortcuts import registry
from gui.theme import media_icon

SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]
BOX_COLOR = QColor("#e5484d")


def _clock(seconds: float, tenths=True) -> str:
    minutes, rest = divmod(max(0.0, seconds), 60)
    return f"{int(minutes)}:{rest:04.1f}" if tenths else f"{int(minutes)}:{int(rest):02d}"


def resolve_footage(alert: Alert, feed_sources: list[str]) -> tuple[str | None, str]:
    """
    Find the video file an alert came from. Returns (path or None, reason when None).
    """
    source = alert.source
    if source:
        if "://" in source:
            return None, "This alert came from a live stream, which isn't recorded, so only its saved frame can be shown."
        if Path(source).is_file():
            return source, ""
        return None, f"The original video is no longer at {source}. Showing the saved frame."
    # alerts published before events recorded their source: match the video name against the feeds in Settings
    for candidate in feed_sources:
        if candidate and "://" not in candidate and Path(candidate).stem == alert.video_name \
                and Path(candidate).is_file():
            return candidate, ""
    return None, "This alert doesn't say which video it came from. Showing the saved frame."


class VideoCanvas(QWidget):
    """
    Draws the current frame scaled to fit, with detection boxes (solid on a checked frame, dashed while held),
    or a message when there is nothing to show.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 240)
        self._image: QImage | None = None
        self._boxes: list = []
        self._held = False
        self._message = "Double-click an alert, or pick one from the list on the right, to review its footage."

    def show_frame(self, image: QImage | None, boxes=(), held=False, message=""):
        self._image = image
        self._boxes = list(boxes)
        self._held = held
        self._message = message
        self.update()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))
        if self._image is None or self._image.isNull():
            painter.setPen(QColor("#c8ccd2"))
            painter.drawText(self.rect().adjusted(40, 40, -40, -40), Qt.AlignCenter | Qt.TextWordWrap, self._message)
            return

        # fit the frame in the widget, keeping its aspect ratio
        size = self._image.size().scaled(self.size(), Qt.KeepAspectRatio)
        x = (self.width() - size.width()) / 2
        y = (self.height() - size.height()) / 2
        target = QRectF(x, y, size.width(), size.height())
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawImage(target, self._image)

        scale = size.width() / self._image.width()
        pen = QPen(BOX_COLOR)
        pen.setWidthF(2.5)
        if self._held:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing)
        for box in self._boxes:
            x1, y1, x2, y2 = box["box_xyxy"]
            painter.drawRect(QRectF(QPointF(x + x1 * scale, y + y1 * scale), QPointF(x + x2 * scale, y + y2 * scale)))

        if self._message:
            painter.setPen(QColor("#ffffff"))
            band = QRectF(0, self.height() - 34, self.width(), 34)
            painter.fillRect(band, QColor(0, 0, 0, 160))
            painter.drawText(band, Qt.AlignCenter, self._message)


class TimelineSlider(QSlider):
    """
    A horizontal slider with a red tick for every frame that has an alert.
    """

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self._markers: list[int] = []

    def set_markers(self, frames: list[int]):
        self._markers = frames
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._markers or self.maximum() <= self.minimum():
            return
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        groove = self.style().subControlRect(QStyle.CC_Slider, option, QStyle.SC_SliderGroove, self)
        handle = self.style().subControlRect(QStyle.CC_Slider, option, QStyle.SC_SliderHandle, self)
        span = groove.width() - handle.width()
        painter = QPainter(self)
        painter.setPen(QPen(QColor(STATUS_COLORS["new"]), 2))
        for frame in self._markers:
            pos = QStyle.sliderPositionFromValue(self.minimum(), self.maximum(), frame, span)
            x = groove.left() + handle.width() / 2 + pos
            painter.drawLine(QPointF(x, groove.top() - 3), QPointF(x, groove.top() + 2))


class FootagePlayer(QWidget):
    """
    Plays a video file with OpenCV: play/pause, speed, ±1 frame, jump to previous/next alert, timeline scrubbing.
    """
    frame_shown = Signal(int)
    previous_alert_requested = Signal()
    next_alert_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._capture = None
        self.path = ""
        self.total = 0
        self.fps = 30.0
        self.index = -1
        self._next_read = 0
        self._overlays: dict[int, list] = {}
        self._overlay_frames: list[int] = []
        self._hold = 1
        self._theme = "light"

        self.canvas = VideoCanvas()

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._tick)

        def tool(_icon, tip, slot):
            # icons are set (and re-colored per theme) in set_theme
            button = QToolButton()
            button.setIconSize(QSize(22, 22))
            button.setToolTip(tip)
            button.setAutoRaise(True)
            button.clicked.connect(slot)
            return button

        self.prev_alert_button = tool(QStyle.SP_MediaSkipBackward, "", self.previous_alert_requested)
        self.step_back_button = tool(QStyle.SP_MediaSeekBackward, "", lambda: self.step(-1))
        self.play_button = tool(QStyle.SP_MediaPlay, "", self.toggle_play)
        self.step_fwd_button = tool(QStyle.SP_MediaSeekForward, "", lambda: self.step(1))
        self.next_alert_button = tool(QStyle.SP_MediaSkipForward, "", self.next_alert_requested)
        self._tooltips = {"previous_alert": (self.prev_alert_button, "Previous alert"),
                          "frame_back": (self.step_back_button, "Previous frame"),
                          "play_pause": (self.play_button, "Play / pause"),
                          "frame_forward": (self.step_fwd_button, "Next frame"),
                          "next_alert": (self.next_alert_button, "Next alert")}
        registry().changed.connect(self._update_tooltips)
        self._update_tooltips()

        self.speed_combo = QComboBox()
        for speed in SPEEDS:
            self.speed_combo.addItem(f"{speed:g}×", speed)
        self.speed_combo.setCurrentIndex(SPEEDS.index(1.0))
        self.speed_combo.setToolTip("Playback speed")
        self.speed_combo.currentIndexChanged.connect(self._update_timer)

        self.boxes_check = QCheckBox("Show detections")
        self.boxes_check.setChecked(True)
        self.boxes_check.setToolTip("Solid boxes: the frame the detector checked. Dashed: carried over until the next check.")
        self.boxes_check.toggled.connect(lambda _: self.show(self.index))

        self.timeline = TimelineSlider()
        self.timeline.setEnabled(False)
        self.timeline.valueChanged.connect(self._timeline_moved)

        self.position_label = QLabel()
        self.position_label.setObjectName("hint")
        self.position_label.setMinimumWidth(240)
        self.position_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # playback buttons centered under the video; options on the left, position on the right
        transport = QHBoxLayout()
        for widget in (self.prev_alert_button, self.step_back_button, self.play_button,
                       self.step_fwd_button, self.next_alert_button):
            transport.addWidget(widget)

        left = QWidget()
        left_layout = QHBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.speed_combo)
        left_layout.addSpacing(8)
        left_layout.addWidget(self.boxes_check)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QHBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addStretch(1)
        right_layout.addWidget(self.position_label)

        # both sides get the same width, so the buttons stay in the exact middle
        side = max(left.sizeHint().width(), right.sizeHint().width())
        left.setMinimumWidth(side)
        right.setMinimumWidth(side)

        controls = QHBoxLayout()
        controls.addWidget(left, 1)
        controls.addLayout(transport)
        controls.addWidget(right, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.timeline)
        layout.addLayout(controls)

        self._set_controls_enabled(False)
        self.set_theme(self._theme)

    def _update_tooltips(self, *_):
        for action_id, (button, text) in self._tooltips.items():
            key = registry().text(action_id)
            button.setToolTip(f"{text} ({key})" if key else text)

    def set_theme(self, name: str):
        self._theme = name
        for button, kind in ((self.prev_alert_button, "prev_alert"), (self.step_back_button, "step_back"),
                             (self.step_fwd_button, "step_forward"), (self.next_alert_button, "next_alert")):
            button.setIcon(media_icon(kind, name))
        self.play_button.setIcon(media_icon("pause" if self.is_playing() else "play", name))

    # ------------------------------------------------------------------ opening

    def open(self, path: str) -> bool:
        """
        Open a video file. Returns False if OpenCV can't read it.
        """
        if path == self.path and self._capture is not None:
            return True
        self.close_video()
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            return False
        self._capture = capture
        self.path = path
        self.total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        self._next_read = 0
        self.index = -1
        self.timeline.blockSignals(True)
        self.timeline.setRange(0, max(0, self.total - 1))
        self.timeline.setValue(0)
        self.timeline.blockSignals(False)
        self._set_controls_enabled(True)
        self._update_timer()
        return True

    def close_video(self):
        self.pause()
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self.path = ""
        self.total = 0
        self.index = -1
        self._set_controls_enabled(False)

    def show_still(self, image: QImage | None, boxes, message: str):
        """
        Show a single saved frame when the footage itself isn't available.
        """
        self.close_video()
        self.canvas.show_frame(image, boxes, message=message if image is not None else "")
        if image is None:
            self.canvas.show_frame(None, message=message + "\nThe saved frame is missing too.")
        self.position_label.setText("Footage unavailable")
        self.timeline.set_markers([])

    def set_overlays(self, overlays: dict[int, list]):
        """
        overlays: frame index -> detection boxes, for every alert in this video.
        Boxes stay up (dashed) until the next checked frame, estimated from the spacing between alert frames.
        """
        self._overlays = overlays
        self._overlay_frames = sorted(overlays)
        gaps = [b - a for a, b in zip(self._overlay_frames, self._overlay_frames[1:]) if b > a]
        self._hold = min(gaps) if gaps else 1
        self.timeline.set_markers(self._overlay_frames)
        if self.index >= 0:
            self.show(self.index)

    # ------------------------------------------------------------------ playback

    def _read(self, index: int):
        """
        Read one frame, seeking only when it isn't the next frame in order (seeking is slower).
        """
        if index != self._next_read:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._capture.read()
        self._next_read = index + 1
        return frame if ok else None

    def show(self, index: int):
        if self._capture is None or self.total <= 0:
            return
        index = max(0, min(self.total - 1, index))
        frame = self._read(index)
        if frame is None:
            self.pause()
            return
        self.index = index
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy()

        boxes, held = [], False
        if self.boxes_check.isChecked():
            if index in self._overlays:
                boxes = self._overlays[index]
            else:
                # carry the latest checked frame's boxes until the next check
                i = bisect_right(self._overlay_frames, index) - 1
                if i >= 0 and index - self._overlay_frames[i] < self._hold:
                    boxes, held = self._overlays[self._overlay_frames[i]], True
        self.canvas.show_frame(image, boxes, held)

        self.timeline.blockSignals(True)
        self.timeline.setValue(index)
        self.timeline.blockSignals(False)
        self.position_label.setText(
            f"{_clock(index / self.fps)} / {_clock(self.total / self.fps)}  ·  frame {index + 1:,} / {self.total:,}"
            f"  ·  {self.fps:g} fps")
        self.frame_shown.emit(index)

    def step(self, delta: int):
        self.pause()
        self.show(self.index + delta)

    def toggle_play(self):
        if self._timer.isActive():
            self.pause()
        else:
            self.play()

    def play(self):
        if self._capture is None:
            return
        if self.index >= self.total - 1:
            self.show(0)
        self._timer.start()
        self.play_button.setIcon(media_icon("pause", self._theme))

    def pause(self):
        self._timer.stop()
        self.play_button.setIcon(media_icon("play", self._theme))

    def is_playing(self) -> bool:
        return self._timer.isActive()

    def _tick(self):
        if self.index >= self.total - 1:
            self.pause()
            return
        self.show(self.index + 1)

    def _update_timer(self, *_):
        speed = self.speed_combo.currentData() or 1.0
        self._timer.setInterval(max(1, round(1000 / (self.fps * speed))))

    def _timeline_moved(self, value: int):
        self.show(value)

    def _set_controls_enabled(self, enabled: bool):
        for widget in (self.prev_alert_button, self.step_back_button, self.play_button, self.step_fwd_button,
                       self.next_alert_button, self.speed_combo, self.boxes_check, self.timeline):
            widget.setEnabled(enabled)

    # ------------------------------------------------------------------ alert navigation helpers

    def previous_alert_frame(self) -> int | None:
        i = bisect_left(self._overlay_frames, self.index) - 1
        return self._overlay_frames[i] if i >= 0 else None

    def next_alert_frame(self) -> int | None:
        i = bisect_right(self._overlay_frames, self.index)
        return self._overlay_frames[i] if i < len(self._overlay_frames) else None


class FeedTab(QWidget):
    """
    The Feed page: footage player and alert details on the left, alerts panel on the right.
    """
    status_changed = Signal(str, str)
    show_on_map_requested = Signal(str)
    log = Signal(str)

    def __init__(self, model: AlertModel, feed_sources, parent=None):
        """
        feed_sources: callable returning the feeds' sources in Settings, used for alerts that don't record their video.
        """
        super().__init__(parent)
        self.model = model
        self._feed_sources = feed_sources
        self._alert_id: str | None = None
        self._frame_to_alert: dict[int, str] = {}

        self.player = FootagePlayer()
        self.player.frame_shown.connect(self._on_frame_shown)
        self.player.previous_alert_requested.connect(lambda: self._jump_alert(-1))
        self.player.next_alert_requested.connect(lambda: self._jump_alert(1))

        # details of the selected alert, with the same status actions as the alerts panel
        self.alert_title = QLabel("No alert selected")
        title_font = self.alert_title.font()
        title_font.setBold(True)
        title_font.setPointSizeF(title_font.pointSizeF() * 1.1)
        self.alert_title.setFont(title_font)
        self.alert_details = QLabel()
        self.alert_details.setObjectName("hint")
        self.alert_details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status_pill = QLabel()
        self.reviewed_button = QPushButton("Reviewed")
        self.reviewed_button.clicked.connect(lambda: self._set_status("reviewed"))
        self.dismiss_button = QPushButton("Dismiss")
        self.dismiss_button.clicked.connect(lambda: self._set_status("dismissed"))
        self.map_button = QPushButton("Show on map")
        self.map_button.clicked.connect(lambda: self._alert_id and self.show_on_map_requested.emit(self._alert_id))

        text = QVBoxLayout()
        text.setSpacing(2)
        title_row = QHBoxLayout()
        title_row.addWidget(self.alert_title)
        title_row.addWidget(self.status_pill)
        title_row.addStretch(1)
        text.addLayout(title_row)
        text.addWidget(self.alert_details)

        info = QFrame()
        info.setObjectName("alertInfo")
        info_layout = QHBoxLayout(info)
        info_layout.setContentsMargins(10, 8, 10, 8)
        info_layout.addLayout(text, 1)
        for button in (self.reviewed_button, self.dismiss_button, self.map_button):
            info_layout.addWidget(button)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)
        left_layout.addWidget(info)
        left_layout.addWidget(self.player, 1)

        self.sidebar = AlertsSidebar(model)
        self.sidebar.alert_selected.connect(self.open_alert)
        self.sidebar.alert_activated.connect(self.open_alert)
        self.sidebar.status_changed.connect(self.status_changed)
        self.sidebar.collapsed_changed.connect(self._on_sidebar_collapsed)
        self._sidebar_width = 360

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(left)
        self.splitter.addWidget(self.sidebar)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)

        # keyboard (rebindable in Settings): play/pause, one frame back/forward, previous/next alert
        for action_id, slot in (("play_pause", self.player.toggle_play),
                                ("frame_back", lambda: self.player.step(-1)),
                                ("frame_forward", lambda: self.player.step(1)),
                                ("previous_alert", lambda: self._jump_alert(-1)),
                                ("next_alert", lambda: self._jump_alert(1))):
            registry().bind(action_id, self, slot, Qt.WidgetWithChildrenShortcut)

        # keep tick marks and boxes current while new alerts arrive for the open video
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(500)
        self._refresh_timer.timeout.connect(self._refresh_overlays)
        model.rowsInserted.connect(lambda *_: self._refresh_timer.start())
        model.dataChanged.connect(lambda *_: self._update_details())
        model.modelReset.connect(self._on_model_reset)

        self._update_details()

    def set_sidebar_width(self, width: int):
        self._sidebar_width = width
        self.splitter.setSizes([max(400, self.width() - width), width])

    def _on_sidebar_collapsed(self, collapsed: bool):
        total = sum(self.splitter.sizes()) or self.width()
        side = STRIP_WIDTH if collapsed else self._sidebar_width
        self.splitter.setSizes([max(400, total - side), side])

    # ------------------------------------------------------------------ opening alerts

    def open_alert(self, alert_id: str):
        alert = self.model.get(alert_id)
        if alert is None:
            return
        self._alert_id = alert_id
        self.sidebar.select_alert(alert_id)

        footage, reason = resolve_footage(alert, self._feed_sources())
        if footage is not None and self.player.open(footage):
            self._refresh_overlays()
            self.player.pause()
            frame = alert.frame_index if alert.frame_index is not None else round(alert.seconds * self.player.fps)
            self.player.show(frame)
        else:
            if footage is not None:
                reason = f"OpenCV couldn't open {footage}. Showing the saved frame."
            self._frame_to_alert = {}
            image = QImage(alert.frame_path)
            self.player.show_still(None if image.isNull() else image, alert.boxes, reason)
            self.log.emit(f"Feed: {reason}")
        self._update_details()

    def _refresh_overlays(self):
        """
        Collect every alert from the open video: boxes per frame for the player, and frame -> alert for navigation.
        """
        if not self.player.path:
            return
        feed_sources = self._feed_sources()
        drone = self.model.get(self._alert_id).drone if self._alert_id and self.model.get(self._alert_id) else None
        overlays, frame_to_alert = {}, {}
        for alert in self.model.alerts():
            # two drones may have flown the same video file (tests, replays): keep to the open alert's drone
            if drone is not None and alert.drone != drone:
                continue
            footage, _ = resolve_footage(alert, feed_sources)
            if footage == self.player.path and alert.frame_index is not None:
                overlays[alert.frame_index] = alert.boxes
                frame_to_alert[alert.frame_index] = alert.id
        self._frame_to_alert = frame_to_alert
        self.player.set_overlays(overlays)

    def _jump_alert(self, direction: int):
        frame = self.player.previous_alert_frame() if direction < 0 else self.player.next_alert_frame()
        if frame is not None and frame in self._frame_to_alert:
            self.open_alert(self._frame_to_alert[frame])

    def _on_frame_shown(self, index: int):
        # landing on another alert's frame (stepping or playing) selects that alert, without stopping playback
        alert_id = self._frame_to_alert.get(index)
        if alert_id is not None and alert_id != self._alert_id:
            self._alert_id = alert_id
            self.sidebar.select_alert(alert_id)
            self._update_details()

    def _on_model_reset(self):
        self._alert_id = None
        self._frame_to_alert = {}
        self.player.set_overlays({})
        self._update_details()

    # ------------------------------------------------------------------ details / status

    def _set_status(self, status: str):
        if self._alert_id is not None:
            self.status_changed.emit(self._alert_id, status)

    def _update_details(self):
        alert = self.model.get(self._alert_id) if self._alert_id else None
        for button in (self.reviewed_button, self.dismiss_button, self.map_button):
            button.setEnabled(alert is not None)
        if alert is None:
            self.alert_title.setText("No alert selected")
            self.alert_details.setText("Double-click an alert on the Map tab, or pick one from the list.")
            self.status_pill.hide()
            return
        self.alert_title.setText(f'<span style="color:{alert.drone_color}">●</span> {alert.drone} · {alert.title}')
        where = f"{alert.lat:.5f}, {alert.lon:.5f}" if alert.has_location else "no coordinates"
        source = Path(alert.source).name if alert.source and "://" not in alert.source else (alert.source or alert.video_name)
        self.alert_details.setText(
            f"Video {alert.video_time} · frame {alert.frame_index if alert.frame_index is not None else '?'} · "
            f"drone at {where} · received {alert.received:%H:%M:%S} · {source}")
        self.status_pill.setText(f" {alert.status.upper()} ")
        self.status_pill.setStyleSheet(f"background:{STATUS_COLORS[alert.status]}; color:white; border-radius:8px;"
                                       "font-weight:bold; padding:1px 6px;")
        self.status_pill.show()
        self.map_button.setEnabled(alert.has_location)

    def clear_thumbnails(self):
        self.sidebar.clear_thumbnails()

    def set_theme(self, name: str):
        self.player.set_theme(name)

    def shutdown(self):
        self.player.close_video()
