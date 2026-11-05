"""
Feeds bar under the map: one row per drone feed (source, progress, counts, status), with Start / Stop for all
feeds and the Redis / Discord status lights.
"""
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from gui.alerts import drone_colors

COLORS = {"green": "#30a46c", "red": "#e5484d", "amber": "#f5a524", "gray": "#8b8d98"}


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}" if seconds >= 3600 \
        else f"{seconds // 60}:{seconds % 60:02d}"


class StatusLight(QLabel):
    """
    A colored dot followed by a short label, e.g. "● Redis".
    """

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self._name = name
        self.set_state("gray", "")

    def set_state(self, color: str, detail: str, name: str | None = None):
        self.state = color
        if name is not None:
            self._name = name
        self.setText(f'<span style="color:{COLORS[color]}">●</span> {self._name}')
        self.setToolTip(detail)


class FeedRow:
    """
    The widgets for one drone feed, laid out as one row of the bar's grid.
    """

    def __init__(self, index: int, name: str, browse):
        self.index = index  # position in the Settings feeds list
        self.name = name
        # the last probed source details, kept while the source stays the same
        self.info: dict | None = None
        self.info_source = ""

        self.name_label = QLabel()
        self.name_label.setMinimumWidth(70)
        self.source_label = QLabel()
        self.source_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.source_label.setMinimumWidth(80)
        self.info_label = QLabel()
        self.info_label.setObjectName("hint")

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("Idle")
        self.progress.setMinimumWidth(140)

        self.stats_label = QLabel()
        self.state_light = StatusLight("Idle")
        self.browse_button = QPushButton("Browse…")
        self.browse_button.setToolTip(f"Choose the video for {name}")
        self.browse_button.clicked.connect(lambda: browse(self.index))

        self.set_name(name)
        self.set_stats(None)
        self.state_light.set_state("gray", "Not running", "Idle")

    def widgets(self):
        return [self.name_label, self.source_label, self.info_label, self.progress, self.stats_label,
                self.state_light, self.browse_button]

    def set_name(self, name: str):
        self.name = name
        color = drone_colors.color(name)
        self.name_label.setText(f'<span style="color:{color}; font-size:15px">●</span> <b>{name}</b>')
        self.name_label.setToolTip(f"{name}: pins and cards for this drone use this color")

    def set_source(self, source: str, info: dict | None):
        if not source:
            self.source_label.setText("<i>No video selected</i>")
            self.source_label.setToolTip("")
        else:
            self.source_label.setText(source if "://" in source else Path(source).name)
            self.source_label.setToolTip(source)

        if info:
            text = f"{info['width']}×{info['height']} · {info['fps']:.1f} fps"
            if info["frame_count"] and info["fps"]:
                text += f" · {_clock(info['frame_count'] / info['fps'])}"
            else:
                text += " · live"
            self.info_label.setText(text)
        else:
            self.info_label.setText("")

    def set_running(self, running: bool):
        self.browse_button.setEnabled(not running)
        if running:
            self.progress.setRange(0, 0)  # busy until the first progress report
            self.progress.setFormat("Starting…")
            self.state_light.set_state("amber", "Loading telemetry, source and model", "Starting")

    def set_stats(self, stats: dict | None):
        if not stats:
            self.stats_label.setText("Read 0 · checked 0 · people 0 · alerts 0")
            return

        self.state_light.set_state("green", "Running", "Running")
        self.stats_label.setText(
            f"Read {stats['frames_read']:,} · checked {stats['frames_checked']:,} · "
            f"people {stats['detections']:,} · alerts {stats['events_published']:,}")

        total = stats["total_frames"]
        elapsed = _clock(stats["time"])
        if total:
            self.progress.setRange(0, 1000)
            self.progress.setValue(round(1000 * min(1.0, stats["frames_read"] / total)))
            self.progress.setFormat(f"{elapsed} / {_clock(total / stats['fps'])}  (%p%)")
        else:
            # live stream: no end, show elapsed time on a busy bar
            self.progress.setRange(0, 0)
            self.progress.setFormat(f"Live · {elapsed}")

    def set_finished(self, message: str, ok: bool):
        self.progress.setRange(0, 1000)
        self.progress.setFormat(message)
        self.state_light.set_state("gray" if ok else "red", message, message)
        self.browse_button.setEnabled(True)


class VideoBar(QFrame):
    browse_requested = Signal(int)   # index of the feed in the Settings list
    start_requested = Signal()
    stop_requested = Signal()
    edit_feeds_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("videoBar")
        self.rows: dict[str, FeedRow] = {}
        self._running = False

        self.feeds_grid = QGridLayout()
        self.feeds_grid.setContentsMargins(0, 0, 0, 0)
        self.feeds_grid.setHorizontalSpacing(10)
        self.feeds_grid.setVerticalSpacing(4)
        self.feeds_grid.setColumnStretch(1, 1)
        self.feeds_grid.setColumnStretch(3, 1)
        self.no_feeds = QLabel("No drone feeds are turned on. Use Edit feeds to add one.")
        self.no_feeds.setObjectName("hint")

        feeds = QWidget()
        feeds_layout = QVBoxLayout(feeds)
        feeds_layout.setContentsMargins(0, 0, 0, 0)
        feeds_layout.addLayout(self.feeds_grid)
        feeds_layout.addWidget(self.no_feeds)

        self.start_button = QPushButton(" Start")
        self.start_button.setToolTip("Start every feed that's turned on")
        self.start_button.clicked.connect(self.start_requested)
        self.stop_button = QPushButton(" Stop")
        self.stop_button.setToolTip("Stop every running feed")
        self.stop_button.clicked.connect(self.stop_requested)
        self.stop_button.setEnabled(False)
        self.edit_button = QPushButton("Edit feeds…")
        self.edit_button.setToolTip("Add, remove or rename drone feeds in Settings")
        self.edit_button.clicked.connect(self.edit_feeds_requested)

        self.redis_light = StatusLight("Redis")
        self.discord_light = StatusLight("Discord")

        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        lights = QHBoxLayout()
        lights.setSpacing(12)
        lights.addWidget(self.redis_light)
        lights.addWidget(self.discord_light)
        lights.addStretch(1)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(4)
        controls_layout.addLayout(buttons)
        controls_layout.addLayout(lights)
        controls_layout.addWidget(self.edit_button)

        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Plain)
        separator.setObjectName("barSeparator")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(12)
        layout.addWidget(feeds, 1)
        layout.addWidget(separator)
        layout.addWidget(controls, 0, Qt.AlignTop)

        self.discord_light.set_state("gray", "Off")

    def set_theme(self, name: str):
        from gui.theme import media_icon

        self.start_button.setIcon(media_icon("play", name))
        self.stop_button.setIcon(media_icon("stop", name))

    # ------------------------------------------------------------------ feeds

    def set_feeds(self, feeds):
        """
        Show one row per turned-on feed (FeedConfig list from Settings). Rows of feeds that keep their name
        keep their progress, so renaming another feed doesn't wipe a finished run's numbers.
        """
        if self._running:
            return  # the feeds can't change during a run; Settings is locked
        wanted = [(i, feed) for i, feed in enumerate(feeds) if feed.enabled]
        old = self.rows
        self.rows = {}
        while self.feeds_grid.count():
            item = self.feeds_grid.takeAt(0)
            if item.widget() is not None:
                item.widget().setParent(None)

        for row_number, (index, feed) in enumerate(wanted):
            name = feed.name.strip() or f"Feed {index + 1}"
            row = old.pop(name, None) or FeedRow(index, name, self.browse_requested.emit)
            row.index = index
            row.set_name(name)
            row.set_source(feed.source, row.info if row.info_source == feed.source else None)
            for column, widget in enumerate(row.widgets()):
                self.feeds_grid.addWidget(widget, row_number, column)
            self.rows[name] = row
        for row in old.values():
            for widget in row.widgets():
                widget.deleteLater()
        self.no_feeds.setVisible(not self.rows)
        self.start_button.setEnabled(bool(self.rows) and not self._running)

    def set_source_info(self, name: str, source: str, info: dict):
        row = self.rows.get(name)
        if row is not None:
            row.info, row.info_source = info, source
            row.set_source(source, info)

    def set_running(self, running: bool, names=()):
        self._running = running
        self.start_button.setEnabled(not running and bool(self.rows))
        self.stop_button.setEnabled(running)
        self.edit_button.setEnabled(not running)
        for name in names:
            if name in self.rows:
                self.rows[name].set_running(True)
                self.rows[name].set_stats(None)
        if not running:
            for row in self.rows.values():
                row.browse_button.setEnabled(True)

    def set_stats(self, name: str, stats: dict | None):
        if name in self.rows:
            self.rows[name].set_stats(stats)

    def set_finished(self, name: str, message: str, ok: bool):
        if name in self.rows:
            self.rows[name].set_finished(message, ok)

    def set_stopping(self):
        self.stop_button.setEnabled(False)
        for row in self.rows.values():
            if row.state_light.state in ("green", "amber"):
                row.progress.setFormat("Stopping…")
