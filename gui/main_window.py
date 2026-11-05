"""
Main window: Map tab (map + feeds bar + alerts sidebar), Feed tab, Settings tab (with the log panel),
and the wiring between them.
"""
import json
import os

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QMessageBox, QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from backend.config import RavenConfig
from gui.alerts import AlertModel, drone_colors
from gui.alerts_panel import STRIP_WIDTH, AlertsSidebar
from gui.feed_tab import FeedTab
from gui.header import HeaderBar
from gui.log_panel import LogPanel
from gui.map_view import LinkViewer, MapView
from gui.profile import ProfileButton
from gui.settings_tab import SettingsTab
from gui.shortcuts import registry
from gui.theme import ACCENTS, DEFAULT_ACCENT, THEMES, apply_theme, set_accent, system_theme
from gui.video_bar import VideoBar
from gui.workers import AlertFeed, DiscordWorker, IngestWorker

PAGE_NAMES = ["Map", "Feed", "Settings"]
PAGE_ACTIONS = ["page_map", "page_feed", "page_settings"]

DEFAULT_SIDEBAR_WIDTH = 360
DEFAULT_LOG_WIDTH = 520
DEFAULT_BAR_HEIGHT = 0   # 0: the feeds bar's natural height

# how long the Discord bot keeps running after ingest ends, so the last events still get sent
DISCORD_GRACE_MS = 5000


class MainWindow(QMainWindow):
    def __init__(self, demo=None):
        """
        demo: a gui.demo.DemoBackend when started with --demo (fake Redis + simulated detector), else None.
        """
        super().__init__()
        self._demo = demo
        self.setWindowTitle("RAVEN — Rescue Aerial Vision and Emergency Navigation" + (" DEMO" if demo else ""))
        self.store = QSettings()

        if demo is not None:
            # demo runs always start from an empty fake Redis; its settings are kept apart from the real ones
            config = self._load_demo_config(demo)
            self.store.remove("alert_statuses")
        else:
            config = self._load_config()
        self._config = config
        drone_colors.set_feeds([feed.name for feed in config.feeds])
        self._statuses: dict[str, str] = self._load_statuses()

        self._feed: AlertFeed | None = None
        self._feed_key = None
        self._ingests: dict[str, IngestWorker] = {}   # drone name -> running worker
        self._ingest_errors: dict[str, str] = {}
        self._discord: DiscordWorker | None = None
        self._sidebar_width = int(self.store.value("sidebar_width", DEFAULT_SIDEBAR_WIDTH))
        self._log_width = int(self.store.value("log_width", DEFAULT_LOG_WIDTH))
        self._feed_sidebar_width = int(self.store.value("feed_sidebar_width", self._sidebar_width))
        self._bar_height = int(self.store.value("feeds_bar_height", DEFAULT_BAR_HEIGHT))

        # ---- widgets
        self.model = AlertModel(self)
        self.map_view = MapView()
        self.link_viewer = LinkViewer()
        self.video_bar = VideoBar()
        self.sidebar = AlertsSidebar(self.model)
        self.settings_tab = SettingsTab(config)
        self.feed_tab = FeedTab(self.model, feed_sources=lambda: [feed.source for feed in self._config.feeds])
        self.log_panel = LogPanel()

        # links clicked on the map open in the link viewer, layered over the map so the map is never reloaded
        self.map_stack = QStackedWidget()
        self.map_stack.addWidget(self.map_view)
        self.map_stack.addWidget(self.link_viewer)
        # the map and the feeds bar under it, split by a handle that drags the bar taller or shorter
        left = QSplitter(Qt.Vertical)
        left.addWidget(self.map_stack)
        left.addWidget(self.video_bar)
        left.setStretchFactor(0, 1)
        left.setStretchFactor(1, 0)
        left.setCollapsible(0, False)
        left.setCollapsible(1, False)
        left.splitterMoved.connect(lambda *_: self._remember_bar_height())
        self.map_split = left

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(left)
        self.splitter.addWidget(self.sidebar)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.splitterMoved.connect(self._remember_sidebar_width)

        # Settings page: the form, with the log panel opening on its right
        self.settings_page = QSplitter(Qt.Horizontal)
        self.settings_page.addWidget(self.settings_tab)
        self.settings_page.addWidget(self.log_panel)
        self.settings_page.setStretchFactor(0, 1)
        self.settings_page.setCollapsible(0, False)
        self.settings_page.setCollapsible(1, False)
        self.settings_page.splitterMoved.connect(self._remember_log_width)
        self.log_panel.hide()

        # header row (title, page buttons, theme toggle) above the page area
        self.header = HeaderBar(PAGE_NAMES, PAGE_ACTIONS, demo=demo is not None)
        self.profile_button = ProfileButton(self.feed_tab.live)
        self.header.add_trailing(self.profile_button)
        self.profile_button.panel.duty_changed.connect(
            lambda key: self.log(f"Status: {self.profile_button.panel.duty_combo.currentText()}"))
        self.profile_button.panel.logout_requested.connect(
            lambda: self.log("Log out isn't available yet: the console has no sign-in."))
        self.pages = QStackedWidget()
        for page in (self.splitter, self.feed_tab, self.settings_page):
            self.pages.addWidget(page)
        self.header.page_selected.connect(self.show_page)
        self.header.theme_toggle_requested.connect(self.toggle_theme)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.header)
        central_layout.addWidget(self.pages, 1)
        self.setCentralWidget(central)

        # default appearance (Settings): system, light or dark; the header button overrides it for this session
        appearance = self.store.value("appearance", "")
        if appearance not in ("system", "light", "dark"):
            # before this setting existed, the header button's last choice was remembered
            old = self.store.value("theme", "")
            appearance = old if old in THEMES else "system"
        self.store.remove("theme")
        self._appearance = appearance
        self._theme_override = False
        accent = self.store.value("accent", DEFAULT_ACCENT)
        self._accent = accent if accent in ACCENTS else DEFAULT_ACCENT
        set_accent(self._accent)
        self.set_theme(self._default_theme())
        QGuiApplication.styleHints().colorSchemeChanged.connect(self._system_scheme_changed)

        # ---- window-wide shortcuts (keys come from Settings > Keyboard shortcuts)
        for i, action_id in enumerate(PAGE_ACTIONS):
            registry().bind(action_id, self, lambda i=i: self.show_page(i))
        registry().bind("show_log", self, self.toggle_log)
        registry().bind("toggle_theme", self, self.toggle_theme)
        registry().bind("save_settings", self, self.save_settings)
        registry().bind("start_ingest", self, self.start_ingest)
        registry().bind("stop_ingest", self, self.stop_ingest)
        registry().changed.connect(lambda *_: self._apply_config(self._config))

        # ---- wiring
        self.settings_tab.config_changed.connect(self._apply_config)
        self.settings_tab.source_probed.connect(self.video_bar.set_source_info)
        self.settings_tab.log.connect(self.log)
        self.settings_tab.save_requested.connect(self.save_settings)
        self.settings_tab.log_toggle_requested.connect(self.toggle_log)
        self.settings_tab.set_appearance(self._appearance)
        self.settings_tab.appearance_changed.connect(self.set_appearance)
        self.settings_tab.set_accent(self._accent)
        self.settings_tab.accent_changed.connect(self.set_accent)
        self.log_panel.close_requested.connect(lambda: self.set_log_visible(False))

        self.video_bar.browse_requested.connect(self.settings_tab.browse_source)
        self.video_bar.start_requested.connect(self.start_ingest)
        self.video_bar.stop_requested.connect(self.stop_ingest)
        self.video_bar.edit_feeds_requested.connect(lambda: self.show_page(PAGE_NAMES.index("Settings")))

        self.sidebar.alert_selected.connect(self._focus_alert)
        self.sidebar.alert_activated.connect(self.review_alert)
        self.sidebar.status_changed.connect(self._set_status)
        self.sidebar.clear_requested.connect(self._clear_alerts)
        self.sidebar.collapsed_changed.connect(self._on_sidebar_collapsed)

        self.map_view.marker_clicked.connect(self.sidebar.select_alert)
        self.map_view.link_requested.connect(self._open_link)
        self.link_viewer.back_requested.connect(self._back_to_map)
        self.map_view.log.connect(self.log)

        self.feed_tab.status_changed.connect(self._set_status)
        self.feed_tab.show_on_map_requested.connect(self._show_alert_on_map)
        self.feed_tab.log.connect(self.log)
        self.feed_tab.sidebar.clear_requested.connect(self._clear_alerts)

        # restart the alert feed shortly after Redis settings stop changing
        self._feed_timer = QTimer(self)
        self._feed_timer.setSingleShot(True)
        self._feed_timer.setInterval(600)
        self._feed_timer.timeout.connect(self._restart_feed)

        # ---- restore layout
        geometry = self.store.value("geometry")
        if geometry is None or not self.restoreGeometry(geometry):
            self.resize(1400, 860)
        # both alerts panels (Map and Feed) are open at launch; they can still be collapsed into the right edge
        self.splitter.setSizes([max(400, self.width() - self._sidebar_width), self._sidebar_width])
        self.feed_tab.set_sidebar_width(self._feed_sidebar_width)
        bar = self._bar_height or self.video_bar.sizeHint().height()
        self.map_split.setSizes([max(300, self.height() - bar), bar])

        self.log("RAVEN operator console started")
        if demo is not None:
            self.log(f"Demo mode: fake Redis on 127.0.0.1:{config.redis_port}, simulated detector, "
                     f"{len(config.feeds)} generated drone flights in {demo.folder}. Press Start to run them.")
        # what's on disk, as the form shows it; the Save changes button lights up when the form differs
        self._saved_settings = self.settings_tab.config().to_dict()
        self._apply_config(self.settings_tab.config())
        self.settings_tab.test_sources()

    # ------------------------------------------------------------------ pages / theme / log

    def show_page(self, index: int):
        self.pages.setCurrentIndex(index)
        self.header.set_page(index)

    def set_theme(self, name: str):
        self._theme = name
        apply_theme(QApplication.instance(), name)
        self.header.set_theme(name)
        self.profile_button.set_theme(name)
        self.map_view.set_theme(name)
        self.feed_tab.set_theme(name)
        self.video_bar.set_theme(name)
        self.settings_tab.show_accent_shades(name)

    def _default_theme(self) -> str:
        return system_theme() if self._appearance == "system" else self._appearance

    def toggle_theme(self):
        # the header button only lasts for this session; Settings > Appearance sets the default
        self._theme_override = True
        self.set_theme("light" if self._theme == "dark" else "dark")

    def set_appearance(self, appearance: str):
        self._appearance = appearance
        self._theme_override = False
        self.store.setValue("appearance", appearance)
        self.set_theme(self._default_theme())
        self.log(f"Default appearance: {appearance}")

    def set_accent(self, accent: str):
        self._accent = accent
        self.store.setValue("accent", accent)
        set_accent(accent)
        self.set_theme(self._theme)
        self.log(f"Accent color: {accent}")

    def _system_scheme_changed(self, *_):
        # "System" follows Windows switching between light and dark while the console is open
        if self._appearance == "system" and not self._theme_override:
            self.set_theme(system_theme())

    def toggle_log(self):
        """
        The log lives beside the Settings form. From another page, the shortcut opens Settings with the log shown.
        """
        settings = PAGE_NAMES.index("Settings")
        if self.pages.currentIndex() != settings:
            self.show_page(settings)
            self.set_log_visible(True)
        else:
            self.set_log_visible(not self.log_panel.isVisible())

    def set_log_visible(self, visible: bool):
        self.log_panel.setVisible(visible)
        self.settings_tab.set_log_visible(visible)
        if visible:
            total = sum(self.settings_page.sizes()) or self.width()
            width = min(self._log_width, max(300, total - 500))
            self.settings_page.setSizes([max(400, total - width), width])
            self.log_panel.text.verticalScrollBar().setValue(self.log_panel.text.verticalScrollBar().maximum())

    def _remember_log_width(self, *_):
        if self.log_panel.isVisible():
            self._log_width = self.settings_page.sizes()[1]

    # ------------------------------------------------------------------ settings

    def has_unsaved_settings(self) -> bool:
        # the webhook is not part of to_dict(): it always comes from .env
        return self.settings_tab.config().to_dict() != self._saved_settings

    def save_settings(self):
        settings = self.settings_tab.config().to_dict()
        self.store.setValue("config", json.dumps(settings))
        self.store.sync()
        self._saved_settings = settings
        self.settings_tab.set_unsaved(False)
        self.log("Settings saved")

    def _load_config(self) -> RavenConfig:
        raw = self.store.value("config")
        try:
            config = RavenConfig.from_dict(json.loads(raw)) if raw else RavenConfig()
        except (ValueError, TypeError):
            config = RavenConfig()
        # the webhook is a secret: it always comes from .env, never from saved settings
        config.webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
        return config

    def _load_demo_config(self, demo) -> RavenConfig:
        """
        The generated demo flights and fake Redis, with the demo's own saved choices (feeds, confidence, sample
        rate, Discord) on top. The fake Redis port, detector and frames folder always come from this launch.
        """
        config = demo.config()
        raw = self.store.value("config")
        if raw:
            try:
                saved = RavenConfig.from_dict(json.loads(raw))
            except (ValueError, TypeError):
                saved = None
            if saved is not None:
                for name in ("min_confidence", "frame_stride", "alert_cooldown_s", "discord_enabled", "stream_name",
                             "realtime"):
                    setattr(config, name, getattr(saved, name))
                # saved feeds are used while their files still exist (the demo folder can be regenerated)
                if all(os.path.isfile(f.source) and os.path.isfile(f.telemetry_path)
                       for f in saved.feeds if f.source or f.telemetry_path):
                    config.feeds = saved.feeds
        config.webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
        return config

    def _load_statuses(self) -> dict:
        try:
            return json.loads(self.store.value("alert_statuses", "{}"))
        except (ValueError, TypeError):
            return {}

    def _apply_config(self, config: RavenConfig):
        self._config = config
        if hasattr(self, "_saved_settings"):
            self.settings_tab.set_unsaved(self.has_unsaved_settings())

        # feed colors follow the feeds list; recolor what's already shown if a feed moved or was renamed
        if drone_colors.set_feeds([feed.name.strip() for feed in config.feeds]):
            names = set(self.model.drones()) | {feed.name.strip() for feed in config.feeds}
            self.map_view.set_drone_colors(drone_colors.colors(names))
            self.sidebar.list.viewport().update()
            self.feed_tab.sidebar.list.viewport().update()
            self.sidebar._update_drones()
            self.feed_tab.sidebar._update_drones()
        self.settings_tab.refresh_colors()
        self.video_bar.set_feeds(config.feeds)
        self.feed_tab.live.set_feeds(config.feeds)

        if (config.redis_host, config.redis_port, config.stream_name) != self._feed_key:
            self._feed_timer.start()

    # ------------------------------------------------------------------ alert feed

    def _restart_feed(self):
        config = self._config
        key = (config.redis_host, config.redis_port, config.stream_name)
        if self._feed is not None:
            self._feed.stop()
            self._feed.wait(3000)
            self._feed = None
        if self._feed_key is not None and self._feed_key[2] != config.stream_name:
            # a different stream holds different alerts
            self._clear_view()
        self._feed_key = key

        self.video_bar.redis_light.set_state("amber", f"Connecting to {config.redis_host}:{config.redis_port}")
        feed = AlertFeed(config)
        feed.events.connect(self._on_events)
        feed.connection_changed.connect(self._on_redis_connection)
        feed.log.connect(self.log)
        feed.start()
        self._feed = feed

    def _on_events(self, batch):
        new = self.model.add_events(batch, self._statuses)
        if not new:
            return
        self.map_view.add_alerts(new)
        if len(new) <= 3:
            # live alerts: one line each
            for alert in new:
                where = f"{alert.lat:.5f}, {alert.lon:.5f}" if alert.has_location else "no coordinates"
                self.log(f"Alert from {alert.drone}: {alert.title} at video {alert.video_time} ({where})")
        else:
            # a replay or burst: one summary line
            missing = sum(1 for alert in new if not alert.has_location)
            note = f", {missing} without coordinates (not on the map)" if missing else ""
            drones = sorted({alert.drone for alert in new})
            self.log(f"Received {len(new)} alerts from {', '.join(drones)}{note}")

    def _on_redis_connection(self, connected: bool, detail: str):
        if connected:
            self.video_bar.redis_light.set_state("green", f"Connected to {detail}")
            self.log(f"Connected to Redis at {detail}, reading {self._config.stream_name}")
        else:
            self.video_bar.redis_light.set_state("red", f"Not reachable: {detail}")
            self.log(f"Redis not reachable ({detail}). Retrying every 2 s.")

    # ------------------------------------------------------------------ map links

    def _open_link(self, url):
        self.link_viewer.open(url)
        self.map_stack.setCurrentWidget(self.link_viewer)
        self.log(f"Opened {url.toString()} (Go back to map or Esc returns)")

    def _back_to_map(self):
        self.map_stack.setCurrentWidget(self.map_view)
        self.link_viewer.close_page()

    def _focus_alert(self, alert_id: str):
        # selecting an alert while a link is open brings the map back
        if self.map_stack.currentWidget() is not self.map_view:
            self._back_to_map()
        self.map_view.focus_alert(alert_id)

    # ------------------------------------------------------------------ alerts

    def _set_status(self, alert_id: str, status: str):
        self.model.set_status(alert_id, status)
        self.map_view.set_status(alert_id, status)
        if status == "new":
            self._statuses.pop(alert_id, None)
        else:
            self._statuses[alert_id] = status
        self.store.setValue("alert_statuses", json.dumps(self._statuses))

    def review_alert(self, alert_id: str):
        """
        Double-clicked alert: open the Feed tab on that alert's footage.
        """
        self.show_page(PAGE_NAMES.index("Feed"))
        self.feed_tab.review(alert_id)

    def _show_alert_on_map(self, alert_id: str):
        self.show_page(PAGE_NAMES.index("Map"))
        self.sidebar.select_alert(alert_id)
        self._focus_alert(alert_id)

    def _clear_alerts(self):
        answer = QMessageBox.question(
            self, "Clear alerts",
            "Remove every alert from the list and the map?\n\n"
            "Nothing is deleted from Redis; restarting the console brings them back.")
        if answer == QMessageBox.Yes:
            self._clear_view()
            self.log("Cleared the alert list")

    def _clear_view(self):
        self.model.clear()
        self.map_view.clear_alerts()
        self.sidebar.clear_thumbnails()
        self.feed_tab.clear_thumbnails()

    # ------------------------------------------------------------------ sidebar layout

    def _remember_sidebar_width(self, *_):
        if not self.sidebar.is_collapsed():
            self._sidebar_width = self.splitter.sizes()[1]

    def _remember_bar_height(self):
        self._bar_height = self.map_split.sizes()[1]

    def _on_sidebar_collapsed(self, collapsed: bool):
        # hand the space to the map when collapsing (the splitter would otherwise leave it empty)
        total = sum(self.splitter.sizes()) or self.width()
        side = STRIP_WIDTH if collapsed else self._sidebar_width
        self.splitter.setSizes([max(400, total - side), side])

    # ------------------------------------------------------------------ ingest (one worker per drone feed)

    def is_running(self) -> bool:
        return bool(self._ingests)

    def start_ingest(self):
        if self._ingests:
            return
        config = self.settings_tab.config()
        problems = config.problems()
        if problems:
            QMessageBox.warning(self, "Can't start yet", "Fix these settings first:\n\n• " + "\n• ".join(problems))
            self.show_page(PAGE_NAMES.index("Settings"))
            return

        feeds = config.enabled_feeds()
        self._ingest_errors = {}
        for feed in feeds:
            # demo mode swaps the YOLO model for a simulated detector
            worker = IngestWorker(config, feed, model_factory=self._demo.make_detector if self._demo else None)
            worker.progress.connect(self.video_bar.set_stats)
            worker.frame_ready.connect(self.feed_tab.live.on_frame)
            worker.source_info.connect(self.video_bar.set_source_info)
            worker.log.connect(self.log)
            worker.failed.connect(self._on_ingest_failed)
            worker.finished.connect(lambda name=feed.name: self._on_ingest_finished(name))
            self._ingests[feed.name] = worker

        self.video_bar.set_running(True, [feed.name for feed in feeds])
        for feed in feeds:
            self.feed_tab.live.set_state(feed.name, "starting")
        self.settings_tab.set_running(True)
        self.log(f"Starting {len(feeds)} feed(s): " + ", ".join(f"{feed.name} ({feed.source})" for feed in feeds))
        for worker in self._ingests.values():
            worker.start()

        if config.discord_enabled:
            self._start_discord(config)

    def stop_ingest(self):
        if not self._ingests:
            return
        for worker in self._ingests.values():
            worker.stop()
        self.video_bar.set_stopping()
        self.log("Stopping all feeds…")

    def _on_ingest_failed(self, name: str, message: str):
        self._ingest_errors[name] = message
        self.log(f"[{name}] Ingest failed: {message}")
        # the other feeds keep running; one message box per failure is enough
        QMessageBox.warning(self, f"{name} failed", message)

    def _on_ingest_finished(self, name: str):
        worker = self._ingests.pop(name, None)
        if name in self._ingest_errors:
            self.video_bar.set_finished(name, "Failed", ok=False)
            self.feed_tab.live.set_state(name, "failed", self._ingest_errors[name])
        elif worker is not None and worker.stopped_by_operator():
            self.video_bar.set_finished(name, "Stopped", ok=True)
            self.feed_tab.live.set_state(name, "stopped")
        else:
            self.video_bar.set_finished(name, "Finished", ok=True)
            self.feed_tab.live.set_state(name, "finished")

        if self._ingests:
            return
        # every feed is done
        self.video_bar.set_running(False)
        self.settings_tab.set_running(False)
        failed = len(self._ingest_errors)
        self.log("All feeds finished" + (f" ({failed} failed)" if failed else ""))
        if self._discord is not None:
            QTimer.singleShot(DISCORD_GRACE_MS, self._stop_discord_if_idle)

    # ------------------------------------------------------------------ discord

    def _start_discord(self, config: RavenConfig):
        if self._discord is not None:
            return
        worker = DiscordWorker(config)
        worker.log.connect(self.log)
        worker.failed.connect(self._on_discord_failed)
        worker.finished.connect(self._on_discord_finished)
        self._discord = worker
        self.video_bar.discord_light.set_state("green", "Sending alerts (cooldown per drone)")
        worker.start()

    def _stop_discord_if_idle(self):
        # a new run may have started during the grace period
        if not self._ingests and self._discord is not None:
            self._discord.stop()

    def _on_discord_failed(self, message: str):
        self.video_bar.discord_light.set_state("red", message)
        self.log(f"Discord alerts stopped: {message}")

    def _on_discord_finished(self):
        self._discord = None
        # keep a failure visible until the next run
        if self.video_bar.discord_light.state != "red":
            self.video_bar.discord_light.set_state("gray", "Off")

    # ------------------------------------------------------------------ log / close

    def log(self, message: str):
        self.log_panel.append(message)

    def closeEvent(self, event):
        if self._ingests:
            answer = QMessageBox.question(self, "Quit", "Feeds are still running. Stop them and quit?")
            if answer != QMessageBox.Yes:
                event.ignore()
                return

        if self.has_unsaved_settings():
            answer = QMessageBox.question(
                self, "Unsaved settings", "Save your settings changes before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
            if answer == QMessageBox.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.Save:
                self.save_settings()

        workers = list(self._ingests.values()) + [w for w in (self._discord, self._feed) if w is not None]
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.wait(5000)
        self.settings_tab.wait_for_jobs()
        self.feed_tab.shutdown()

        # settings are only written by Save changes (demo mode saves to its own settings)
        self.store.setValue("geometry", self.saveGeometry())
        self.store.setValue("sidebar_width", self._sidebar_width)
        self.store.setValue("feed_sidebar_width", self.feed_tab.sidebar_width())
        self.store.setValue("feeds_bar_height", self._bar_height)
        self.store.setValue("log_width", self._log_width)
        super().closeEvent(event)
