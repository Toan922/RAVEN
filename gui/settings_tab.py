"""
Settings tab: everything the operator inputs for a run (drone feeds, detection, alerts, backend),
keyboard shortcuts, and the button that opens the log panel.
"""
import os
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QScrollArea, QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

from backend.config import MAX_FEEDS, ROOT_DIR, FeedConfig, RavenConfig
from gui.alerts import drone_colors
from gui.shortcut_settings import ShortcutSettings
from gui.theme import ACCENTS, accent_color
from gui.workers import JobRunner, redis_client, send_test_alert

# assumed frame rate for the "checks per second" hint until a source has been tested
ASSUMED_FPS = 30.0

VIDEO_FILTER = "Videos (*.mp4 *.avi *.mov *.mkv *.m4v *.ts);;All files (*)"


def _row(*widgets, stretch_first=True) -> QWidget:
    """
    Lay out widgets side by side in one form row.
    """
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    for i, widget in enumerate(widgets):
        layout.addWidget(widget, 1 if (stretch_first and i == 0) else 0)
    return row


def _hint(text="") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    label.setObjectName("hint")  # muted text color comes from the theme stylesheet
    return label


def _collapsible(title: str, body: QWidget) -> QWidget:
    """
    A bold arrow button that shows or hides body, like the Advanced section.
    """
    wrapper = QWidget()
    layout = QVBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    toggle = QToolButton()
    toggle.setText(title)
    toggle.setCheckable(True)
    toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    toggle.setArrowType(Qt.RightArrow)
    toggle.setObjectName("advancedToggle")
    body.setVisible(False)
    toggle.toggled.connect(body.setVisible)
    toggle.toggled.connect(lambda on: toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow))
    layout.addWidget(toggle)
    layout.addWidget(body)
    wrapper.toggle = toggle
    return wrapper


def is_stream_link(source: str) -> bool:
    return "://" in source


class FeedEditor(QFrame):
    """
    One drone feed in Settings: on/off, drone name, video or stream link (with Test), and telemetry CSV.
    """
    changed = Signal()
    probed = Signal(dict)      # source details after a successful test
    remove_requested = Signal()
    log = Signal(str)

    def __init__(self, feed: FeedConfig, jobs: JobRunner, parent=None):
        super().__init__(parent)
        self.setObjectName("feedCard")
        self._jobs = jobs
        self.fps = None

        self.enabled_check = QCheckBox()
        self.enabled_check.setToolTip("Run this feed when Start is pressed")
        self.enabled_check.toggled.connect(self._emit_changed)
        self.color_dot = QLabel()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Drone name, e.g. Drone 1")
        self.name_edit.setMaxLength(40)
        self.name_edit.setToolTip("Tags every alert from this feed, and its Discord alerts")
        # typing counts as a change right away (Save changes lights up); checks run when the field is left
        self.name_edit.textChanged.connect(self._emit_changed)
        self.remove_button = QToolButton()
        self.remove_button.setText("Remove")
        self.remove_button.setToolTip("Remove this feed")
        self.remove_button.clicked.connect(self.remove_requested)

        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText(r"C:\videos\flight.mp4  or  rtsp://host:554/stream")
        self.source_edit.editingFinished.connect(self._source_edited)
        self.source_edit.textChanged.connect(self._emit_changed)
        self.browse_button = QPushButton("Browse…")
        self.browse_button.clicked.connect(self.browse_source)
        self.test_button = QPushButton("Test source")
        self.test_button.clicked.connect(self.test_source)
        self.source_status = _hint()

        self.telemetry_edit = QLineEdit()
        self.telemetry_edit.setPlaceholderText("CSV with t, latitude, longitude columns")
        self.telemetry_edit.editingFinished.connect(self._telemetry_edited)
        self.telemetry_edit.textChanged.connect(self._emit_changed)
        self.browse_telemetry_button = QPushButton("Browse…")
        self.browse_telemetry_button.clicked.connect(self._browse_telemetry)
        self.telemetry_status = _hint()

        grid = QGridLayout(self)
        grid.setContentsMargins(10, 8, 10, 8)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        top = _row(self.enabled_check, self.color_dot, self.name_edit, self.remove_button, stretch_first=False)
        top.layout().setStretch(2, 1)
        grid.addWidget(QLabel("Drone"), 0, 0)
        grid.addWidget(top, 0, 1)
        grid.addWidget(QLabel("Video / stream link"), 1, 0)
        grid.addWidget(_row(self.source_edit, self.browse_button, self.test_button), 1, 1)
        grid.addWidget(self.source_status, 2, 1)
        grid.addWidget(QLabel("Telemetry CSV"), 3, 0)
        grid.addWidget(_row(self.telemetry_edit, self.browse_telemetry_button), 3, 1)
        grid.addWidget(self.telemetry_status, 4, 1)

        self.set_feed(feed)

    # ------------------------------------------------------------------ data

    def feed(self) -> FeedConfig:
        return FeedConfig(name=self.name_edit.text().strip(), source=self.source_edit.text().strip(),
                          telemetry_path=self.telemetry_edit.text().strip(), enabled=self.enabled_check.isChecked())

    def set_feed(self, feed: FeedConfig):
        for widget in (self.enabled_check, self.name_edit, self.source_edit, self.telemetry_edit):
            widget.blockSignals(True)
        self.enabled_check.setChecked(feed.enabled)
        self.name_edit.setText(feed.name)
        self.source_edit.setText(feed.source)
        self.telemetry_edit.setText(feed.telemetry_path)
        for widget in (self.enabled_check, self.name_edit, self.source_edit, self.telemetry_edit):
            widget.blockSignals(False)
        self.fps = None
        self.source_status.setText(self._idle_source_text())
        self._check_telemetry()

    def set_color(self, color: str):
        self.color_dot.setText(f'<span style="color:{color}; font-size:18px">●</span>')
        self.color_dot.setToolTip("This drone's color on the map and alert cards")

    def set_locked(self, locked: bool, can_remove: bool):
        for widget in (self.enabled_check, self.name_edit, self.source_edit, self.browse_button, self.test_button,
                       self.telemetry_edit, self.browse_telemetry_button):
            widget.setEnabled(not locked)
        self.remove_button.setEnabled(not locked and can_remove)

    def _emit_changed(self, *_):
        self.changed.emit()

    def _label(self) -> str:
        return self.name_edit.text().strip() or "this feed"

    # ------------------------------------------------------------------ source

    def _idle_source_text(self) -> str:
        source = self.source_edit.text().strip()
        if not source:
            return "Pick a video file or paste a stream link, then press Test source."
        if is_stream_link(source):
            return "Stream link set. Press Test source to check the connection."
        return "" if Path(source).is_file() else "✗ File not found."

    def browse_source(self):
        current = self.source_edit.text()
        start = current if current and not is_stream_link(current) else str(ROOT_DIR / "videos")
        path, _ = QFileDialog.getOpenFileName(self, f"Choose the video for {self._label()}", start, VIDEO_FILTER)
        if path:
            self.set_source(str(Path(path)))

    def set_source(self, source: str):
        self.source_edit.setText(source)
        self._source_edited()

    def _source_edited(self):
        self.fps = None
        source = self.source_edit.text().strip()
        self.source_status.setText(self._idle_source_text())
        if source and not is_stream_link(source) and Path(source).is_file():
            # opening a stream can take a while, so only files are checked right away
            self.test_source()
        self.changed.emit()

    def test_source(self):
        source = self.source_edit.text().strip()
        if not source:
            self.source_status.setText("✗ No source set.")
            return
        from backend.ingest import probe_source

        self.test_button.setEnabled(False)
        self.source_status.setText("Opening source…")

        def done(info):
            self.test_button.setEnabled(True)
            if self.source_edit.text().strip() == source:
                self.apply_source_info(info)
                self.probed.emit(info)

        def fail(message):
            self.test_button.setEnabled(True)
            self.source_status.setText(f"✗ {message}")
            self.log.emit(f"Source test failed for {self._label()}: {message}")

        self._jobs.run(lambda: probe_source(source), done, fail)

    def apply_source_info(self, info: dict):
        self.fps = info["fps"] or None
        text = f"✓ {info['width']}×{info['height']}, {info['fps']:.1f} fps"
        if info["frame_count"] and info["fps"]:
            seconds = info["frame_count"] / info["fps"]
            text += f", {info['frame_count']} frames ({int(seconds // 60)}:{int(seconds % 60):02d})"
        else:
            text += ", live stream"
        self.source_status.setText(text)
        self.changed.emit()

    # ------------------------------------------------------------------ telemetry

    def _browse_telemetry(self):
        path, _ = QFileDialog.getOpenFileName(self, f"Choose telemetry for {self._label()}",
                                              self.telemetry_edit.text() or str(ROOT_DIR / "videos"),
                                              "CSV files (*.csv);;All files (*)")
        if path:
            self.telemetry_edit.setText(str(Path(path)))
            self._telemetry_edited()

    def _telemetry_edited(self):
        self._check_telemetry()
        self.changed.emit()

    def _check_telemetry(self):
        """
        Validate the telemetry CSV right away using the same loader ingest uses.
        """
        from backend.ingest import load_telemetry

        path = self.telemetry_edit.text().strip()
        if not path:
            self.telemetry_status.setText("Needed to geotag this drone's alerts on the map.")
            return
        try:
            rows = load_telemetry(path)
        except FileNotFoundError:
            self.telemetry_status.setText("✗ File not found.")
        except Exception as error:  # show the loader's own message (missing columns, bad value, ...)
            self.telemetry_status.setText(f"✗ {error}")
        else:
            self.telemetry_status.setText(f"✓ {len(rows)} rows covering {rows[0]['t']:g}–{rows[-1]['t']:g} s")


class SettingsTab(QWidget):
    config_changed = Signal(object)        # RavenConfig
    source_probed = Signal(str, str, dict)  # drone name, source, {"width", "height", "fps", "frame_count"}
    save_requested = Signal()
    log_toggle_requested = Signal()
    appearance_changed = Signal(str)        # "system", "light" or "dark"
    accent_changed = Signal(str)            # a key of gui.theme.ACCENTS
    log = Signal(str)

    def __init__(self, config: RavenConfig, parent=None):
        super().__init__(parent)
        self._jobs = JobRunner(self)
        self._loading = False
        self._running = False
        self.feed_editors: list[FeedEditor] = []

        content = QWidget()
        content.setMaximumWidth(860)
        column = QVBoxLayout(content)
        column.setSpacing(14)
        column.addWidget(self._build_feeds_group())
        column.addWidget(self._build_detection_group())
        column.addWidget(self._build_alerts_group())
        column.addWidget(self._build_appearance_group())
        column.addWidget(self._build_advanced_group())
        self.shortcut_settings = ShortcutSettings()
        self.shortcut_settings.log.connect(self.log)
        self.shortcuts_section = _collapsible("Keyboard shortcuts", self.shortcut_settings)
        column.addWidget(self.shortcuts_section)
        column.addWidget(self._build_preset_row())
        column.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        holder = QWidget()
        holder_layout = QHBoxLayout(holder)
        # let the form grow up to its maximum width, then keep it left-aligned
        holder_layout.addWidget(content, 10)
        holder_layout.addStretch(1)
        scroll.setWidget(holder)

        # top row, outside the scroll area so the log button is always in reach
        self.log_button = QPushButton("Show log")
        self.log_button.setCheckable(True)
        self.log_button.setObjectName("logButton")
        self.log_button.clicked.connect(lambda: self.log_toggle_requested.emit())
        top = QHBoxLayout()
        top.setContentsMargins(12, 8, 12, 0)
        top.addWidget(_hint("Settings apply to the next run. Save changes keeps them for the next launch."), 1)
        top.addWidget(self.log_button)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(top)
        outer.addWidget(scroll, 1)

        self.set_config(config)

    # ------------------------------------------------------------------ building

    def _build_feeds_group(self) -> QGroupBox:
        group = QGroupBox(f"Drone feeds (up to {MAX_FEEDS})")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        layout.addWidget(_hint("Each feed is one drone: its video or stream link and its telemetry. Start runs every "
                               "feed that's ticked, at the same time. Detection settings below apply to all of them."))
        self.feeds_layout = QVBoxLayout()
        self.feeds_layout.setSpacing(8)
        layout.addLayout(self.feeds_layout)
        self.add_feed_button = QPushButton("+ Add feed")
        self.add_feed_button.clicked.connect(self._add_feed)
        layout.addWidget(self.add_feed_button, 0, Qt.AlignLeft)
        return group

    def _build_detection_group(self) -> QGroupBox:
        group = QGroupBox("Detection (all feeds)")
        form = QFormLayout(group)

        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(5, 95)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.05, 0.95)
        self.confidence_spin.setSingleStep(0.01)
        self.confidence_spin.setDecimals(2)
        # keep the slider (whole percent) and the spin box in sync
        self.confidence_slider.valueChanged.connect(lambda v: self.confidence_spin.setValue(v / 100))
        self.confidence_spin.valueChanged.connect(lambda v: self.confidence_slider.setValue(round(v * 100)))
        self.confidence_spin.valueChanged.connect(self._changed)
        form.addRow("Confidence threshold", _row(self.confidence_slider, self.confidence_spin))
        form.addRow("", _hint("Detections below this confidence are ignored. Lower finds more people "
                              "but raises more false alarms."))

        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 300)
        self.stride_spin.setPrefix("every ")
        self.stride_spin.setSuffix(" frame(s)")
        self.stride_spin.valueChanged.connect(self._changed)
        self.rate_hint = _hint()
        form.addRow("Frame sample rate", _row(self.stride_spin, self.rate_hint, stretch_first=False))

        self.model_edit = QLineEdit()
        self.model_edit.textChanged.connect(self._changed)
        browse_model = QPushButton("Browse…")
        browse_model.clicked.connect(self._browse_model)
        form.addRow("Model weights", _row(self.model_edit, browse_model))

        self.realtime_check = QCheckBox("Play video files at real speed")
        self.realtime_check.setToolTip("Off: files are processed as fast as possible. Streams always run in real time.")
        self.realtime_check.toggled.connect(self._changed)
        form.addRow("", self.realtime_check)

        self._detection_widgets = [self.confidence_slider, self.confidence_spin, self.stride_spin,
                                   self.model_edit, browse_model, self.realtime_check]
        return group

    def _build_alerts_group(self) -> QGroupBox:
        group = QGroupBox("Alerts")
        form = QFormLayout(group)

        self.discord_check = QCheckBox("Send alerts to Discord while running")
        self.discord_check.toggled.connect(self._changed)
        form.addRow("", self.discord_check)

        self.webhook_edit = QLineEdit()
        self.webhook_edit.setEchoMode(QLineEdit.Password)
        self.webhook_edit.setPlaceholderText("https://discord.com/api/webhooks/…")
        self.webhook_edit.textChanged.connect(self._changed)
        show = QToolButton()
        show.setText("Show")
        show.setCheckable(True)
        show.toggled.connect(lambda on: self.webhook_edit.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password))
        self.test_alert_button = QPushButton("Send test alert")
        self.test_alert_button.clicked.connect(self._send_test_alert)
        form.addRow("Discord webhook", _row(self.webhook_edit, show, self.test_alert_button))
        self.webhook_status = _hint("Loaded from DISCORD_WEBHOOK_URL in .env. Edits here last until the console closes.")
        form.addRow("", self.webhook_status)

        self.cooldown_spin = QDoubleSpinBox()
        self.cooldown_spin.setRange(0, 600)
        self.cooldown_spin.setDecimals(0)
        self.cooldown_spin.setSuffix(" s")
        self.cooldown_spin.valueChanged.connect(self._changed)
        form.addRow("Discord cooldown", _row(self.cooldown_spin, _hint(
            "Minimum video time between two Discord alerts from the same drone (each drone has its own). "
            "The alert list in the console shows every detection."), stretch_first=False))

        self._alert_widgets = [self.discord_check, self.webhook_edit, self.cooldown_spin]
        return group

    def _build_appearance_group(self) -> QGroupBox:
        group = QGroupBox("Appearance")
        form = QFormLayout(group)
        self.appearance_combo = QComboBox()
        for label, value in (("System", "system"), ("Light", "light"), ("Dark", "dark")):
            self.appearance_combo.addItem(label, value)
        self.appearance_combo.setToolTip("System follows the Windows light/dark setting")
        self.appearance_combo.currentIndexChanged.connect(
            lambda i: self.appearance_changed.emit(self.appearance_combo.itemData(i)))
        form.addRow("Default appearance", _row(self.appearance_combo, _hint(
            "How the console looks at launch; changes apply and save right away. The sun / moon button "
            "switches for this session only."), stretch_first=False))

        # accent color: one round swatch per color, applied and saved right away
        self._accent_group = QButtonGroup(self)
        swatches = QHBoxLayout()
        swatches.setContentsMargins(0, 0, 0, 0)
        swatches.setSpacing(6)
        for name in ACCENTS:
            swatch = QToolButton()
            swatch.setObjectName("accentSwatch")
            swatch.setCheckable(True)
            swatch.setCursor(Qt.PointingHandCursor)
            swatch.setIconSize(QSize(20, 20))
            swatch.setToolTip(name.capitalize())
            swatch.setProperty("accent", name)
            self._accent_group.addButton(swatch)
            swatches.addWidget(swatch)
        swatches.addStretch(1)
        self._accent_group.buttonClicked.connect(lambda button: self.accent_changed.emit(button.property("accent")))
        swatch_row = QWidget()
        swatch_row.setLayout(swatches)
        form.addRow("Accent color", swatch_row)
        self.show_accent_shades("light")
        return group

    def set_accent(self, name: str):
        for button in self._accent_group.buttons():
            button.setChecked(button.property("accent") == name)

    def show_accent_shades(self, theme: str):
        """
        Draw each swatch in the shade the current theme uses.
        """
        for button in self._accent_group.buttons():
            pixmap = QPixmap(40, 40)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(accent_color(button.property("accent"), theme)))
            painter.drawEllipse(0, 0, 40, 40)
            painter.end()
            button.setIcon(QIcon(pixmap))

    def set_appearance(self, value: str):
        self.appearance_combo.blockSignals(True)
        self.appearance_combo.setCurrentIndex(max(0, self.appearance_combo.findData(value)))
        self.appearance_combo.blockSignals(False)

    def _build_advanced_group(self) -> QWidget:
        group = QGroupBox()
        form = QFormLayout(group)

        self.redis_host_edit = QLineEdit()
        self.redis_host_edit.textChanged.connect(self._changed)
        self.redis_port_spin = QSpinBox()
        self.redis_port_spin.setRange(1, 65535)
        self.redis_port_spin.valueChanged.connect(self._changed)
        self.test_redis_button = QPushButton("Test Redis")
        self.test_redis_button.clicked.connect(self._test_redis)
        form.addRow("Redis host / port", _row(self.redis_host_edit, self.redis_port_spin, self.test_redis_button))
        self.redis_status = _hint()
        form.addRow("", self.redis_status)

        self.stream_edit = QLineEdit()
        self.stream_edit.textChanged.connect(self._changed)
        form.addRow("Stream name", self.stream_edit)

        self.output_edit = QLineEdit()
        self.output_edit.textChanged.connect(self._changed)
        browse_output = QPushButton("Browse…")
        browse_output.clicked.connect(self._browse_output)
        form.addRow("Saved frames folder", _row(self.output_edit, browse_output))
        form.addRow("", _hint("Each drone's frames go in their own subfolder."))

        self._advanced_widgets = [self.redis_host_edit, self.redis_port_spin, self.stream_edit,
                                  self.output_edit, browse_output]
        return _collapsible("Advanced", group)

    def _build_preset_row(self) -> QWidget:
        load = QPushButton("Load preset…")
        load.clicked.connect(self._load_preset)
        save = QPushButton("Save preset…")
        save.clicked.connect(self._save_preset)
        self.save_changes_button = QPushButton("Save changes")
        self.save_changes_button.setEnabled(False)
        self.save_changes_button.clicked.connect(self.save_requested)
        defaults = QPushButton("Restore defaults")
        defaults.clicked.connect(self._restore_defaults)
        self._preset_widgets = [load, defaults]

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(load)
        layout.addWidget(save)
        layout.addStretch(1)
        layout.addWidget(self.save_changes_button)
        layout.addWidget(defaults)
        self.set_unsaved(False)
        return row

    def set_unsaved(self, unsaved: bool, saving_allowed: bool = True):
        """
        Enable Save changes only when the form differs from the saved settings.
        """
        from gui.shortcuts import registry

        key = registry().text("save_settings")
        key = f" ({key})" if key else ""
        self.save_changes_button.setEnabled(unsaved and saving_allowed)
        if not saving_allowed:
            self.save_changes_button.setToolTip("Settings can't be saved right now")
        elif unsaved:
            self.save_changes_button.setToolTip(f"You have unsaved changes. Save them for the next launch{key}")
        else:
            self.save_changes_button.setToolTip("All changes saved")

    def set_log_visible(self, visible: bool):
        self.log_button.setChecked(visible)
        self.log_button.setText("Hide log" if visible else "Show log")

    # ------------------------------------------------------------------ feeds list

    def _set_feeds(self, feeds: list[FeedConfig]):
        for editor in self.feed_editors:
            editor.setParent(None)
            editor.deleteLater()
        self.feed_editors = []
        for feed in feeds[:MAX_FEEDS]:
            self._append_editor(feed)
        self._update_feed_controls()

    def _append_editor(self, feed: FeedConfig) -> FeedEditor:
        editor = FeedEditor(feed, self._jobs)
        editor.changed.connect(self._changed)
        editor.log.connect(self.log)
        editor.probed.connect(lambda info, e=editor: self.source_probed.emit(e.feed().name, e.feed().source, info))
        editor.remove_requested.connect(lambda e=editor: self._remove_feed(e))
        self.feeds_layout.addWidget(editor)
        self.feed_editors.append(editor)
        return editor

    def _add_feed(self):
        if len(self.feed_editors) >= MAX_FEEDS:
            return
        taken = {editor.feed().name.lower() for editor in self.feed_editors}
        number = next(n for n in range(1, MAX_FEEDS + 2) if f"drone {n}" not in taken)
        editor = self._append_editor(FeedConfig(name=f"Drone {number}"))
        self._update_feed_controls()
        self._changed()
        editor.source_edit.setFocus()
        self.log.emit(f"Added feed Drone {number}")

    def _remove_feed(self, editor: FeedEditor):
        if len(self.feed_editors) <= 1:
            return
        name = editor.feed().name or "a feed"
        if editor.feed().source:
            answer = QMessageBox.question(self, "Remove feed", f"Remove {name} from the feeds list?")
            if answer != QMessageBox.Yes:
                return
        self.feed_editors.remove(editor)
        editor.setParent(None)
        editor.deleteLater()
        self._update_feed_controls()
        self._changed()
        self.log.emit(f"Removed feed {name}")

    def _update_feed_controls(self):
        full = len(self.feed_editors) >= MAX_FEEDS
        self.add_feed_button.setEnabled(not full and not self._running)
        self.add_feed_button.setToolTip(f"At most {MAX_FEEDS} feeds" if full else "Add another drone feed")
        for editor in self.feed_editors:
            editor.set_locked(self._running, can_remove=len(self.feed_editors) > 1)

    def browse_source(self, index: int = 0):
        """
        Pick the video for one feed (the Browse buttons under the map use this).
        """
        if 0 <= index < len(self.feed_editors):
            self.feed_editors[index].browse_source()

    def test_sources(self):
        """
        Check every feed's video file once at launch so the bar under the map shows their details.
        """
        for editor in self.feed_editors:
            source = editor.feed().source
            if source and not is_stream_link(source) and Path(source).is_file():
                editor.test_source()

    # ------------------------------------------------------------------ config in / out

    def config(self) -> RavenConfig:
        return RavenConfig(
            feeds=[editor.feed() for editor in self.feed_editors],
            min_confidence=round(self.confidence_spin.value(), 2),
            frame_stride=self.stride_spin.value(),
            model_path=self.model_edit.text().strip(),
            realtime=self.realtime_check.isChecked(),
            alert_cooldown_s=self.cooldown_spin.value(),
            discord_enabled=self.discord_check.isChecked(),
            webhook_url=self.webhook_edit.text().strip(),
            redis_host=self.redis_host_edit.text().strip() or "localhost",
            redis_port=self.redis_port_spin.value(),
            stream_name=self.stream_edit.text().strip() or RavenConfig.stream_name,
            output_dir=self.output_edit.text().strip() or RavenConfig.output_dir,
        )

    def set_config(self, config: RavenConfig):
        """
        Fill every field from a config, then announce the change once.
        """
        self._loading = True
        self._set_feeds(config.feeds or [FeedConfig()])
        self.confidence_spin.setValue(config.min_confidence)
        self.stride_spin.setValue(config.frame_stride)
        self.model_edit.setText(config.model_path)
        self.realtime_check.setChecked(config.realtime)
        self.cooldown_spin.setValue(config.alert_cooldown_s)
        self.discord_check.setChecked(config.discord_enabled)
        self.webhook_edit.setText(config.webhook_url)
        self.redis_host_edit.setText(config.redis_host)
        self.redis_port_spin.setValue(config.redis_port)
        self.stream_edit.setText(config.stream_name)
        self.output_edit.setText(config.output_dir)
        self._loading = False
        self._changed()

    def set_running(self, running: bool):
        """
        Lock the inputs that a running ingest depends on. Shortcuts and the log stay usable.
        """
        self._running = running
        for widget in self._detection_widgets + self._alert_widgets + self._advanced_widgets + self._preset_widgets:
            widget.setEnabled(not running)
        self._update_feed_controls()

    def _changed(self, *_):
        if self._loading:
            return
        # the sample-rate hint uses the first tested feed's real frame rate
        tested = next((editor for editor in self.feed_editors if editor.fps), None)
        fps = tested.fps if tested else ASSUMED_FPS
        note = f" ({tested.feed().name})" if tested and len(self.feed_editors) > 1 else \
            ("" if tested else " (assumed until a source is tested)")
        self.rate_hint.setText(f"≈ {fps / self.stride_spin.value():.1f} checks/sec at {fps:.0f} fps{note}")
        # the main window assigns drone colors from the new feeds list, then calls refresh_colors()
        self.config_changed.emit(self.config())

    def refresh_colors(self):
        for editor in self.feed_editors:
            name = editor.feed().name
            editor.set_color(drone_colors.color(name) if name else "#8b8d98")

    # ------------------------------------------------------------------ model / output

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose model weights", self.model_edit.text(),
                                              "PyTorch weights (*.pt);;All files (*)")
        if path:
            self.model_edit.setText(str(Path(path)))
            self._changed()

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Saved frames folder", self.output_edit.text())
        if path:
            self.output_edit.setText(str(Path(path)))
            self._changed()

    # ------------------------------------------------------------------ alerts / redis tests

    def _send_test_alert(self):
        url = self.webhook_edit.text().strip()
        if not url:
            self.webhook_status.setText("✗ No webhook URL set.")
            return
        self.test_alert_button.setEnabled(False)
        self.webhook_status.setText("Sending…")

        def done(message):
            self.test_alert_button.setEnabled(True)
            self.webhook_status.setText(f"✓ {message}. Check the Discord channel.")
            self.log.emit("Sent a Discord test alert")

        def fail(message):
            self.test_alert_button.setEnabled(True)
            self.webhook_status.setText(f"✗ {message}")
            self.log.emit(f"Discord test alert failed: {message}")

        self._jobs.run(lambda: send_test_alert(url), done, fail)

    def _test_redis(self):
        config = self.config()
        self.test_redis_button.setEnabled(False)
        self.redis_status.setText("Connecting…")

        def ping():
            client = redis_client(config)
            client.ping()
            return client.xlen(config.stream_name)

        def done(length):
            self.test_redis_button.setEnabled(True)
            self.redis_status.setText(f"✓ Connected. {length} event(s) on {config.stream_name}.")

        def fail(message):
            self.test_redis_button.setEnabled(True)
            self.redis_status.setText(f"✗ {message}")

        self._jobs.run(ping, done, fail)

    # ------------------------------------------------------------------ presets

    def _load_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load preset", str(ROOT_DIR), "RAVEN presets (*.json)")
        if not path:
            return
        try:
            loaded = RavenConfig.load(path)
        except Exception as error:  # bad JSON or wrong value types
            QMessageBox.warning(self, "Load preset", f"Could not load {path}:\n{error}")
            return
        # presets never hold the webhook, keep the current one
        loaded.webhook_url = self.webhook_edit.text().strip()
        self.set_config(loaded)
        self.test_sources()
        self.log.emit(f"Loaded preset {path}")

    def _save_preset(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save preset", str(ROOT_DIR / "preset.json"), "RAVEN presets (*.json)")
        if path:
            self.config().save(path)
            self.log.emit(f"Saved preset {path} (the webhook is not saved in presets)")

    def _restore_defaults(self):
        defaults = RavenConfig(webhook_url=os.environ.get("DISCORD_WEBHOOK_URL", ""))
        self.set_config(defaults)
        self.log.emit("Restored default settings (shortcuts have their own reset)")

    def wait_for_jobs(self):
        self._jobs.wait_all()
