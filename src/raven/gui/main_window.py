"""
Main window: Map tab (map + feeds bar + alerts sidebar), Feed tab, Manage tab (search session, drones, accounts),
Settings tab (with the log panel), and the wiring between them.

The window owns the console's Detection Event Bus: drone feeds publish HumanDetected / PipelineStatus on it,
and the alert list, the Detection Log and Discord subscribe to it.
"""
import json
import os
import time

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QLabel, QMainWindow, QMessageBox, QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from raven.backend import accounts as acl
from raven.backend.bus import BackgroundBus
from raven.backend.config import RavenConfig, SearchSession
from raven.backend.detection_log import DetectionLog, read_log
from raven.backend.events import HumanDetected
from raven.gui.alerts import AlertModel, drone_colors
from raven.gui.alerts_panel import STRIP_WIDTH, AlertsSidebar
from raven.gui.feed_tab import FeedTab
from raven.gui.header import HeaderBar
from raven.gui.log_panel import LogPanel
from raven.gui.manage_tab import DroneAssignments, ManageTab, SessionHistory
from raven.gui.map_view import LinkViewer, MapView
from raven.gui.profile import ProfileButton
from raven.gui.settings_tab import SettingsTab
from raven.gui.sign_in import IdleWatcher, LockScreen, UsersEditor
from raven.gui.shortcuts import registry
from raven.gui.theme import ACCENTS, DEFAULT_ACCENT, THEMES, apply_theme, set_accent, system_theme
from raven.gui.video_bar import VideoBar
from raven.gui.workers import AlertFeed, DiscordAlerts, IngestWorker

PAGE_NAMES = ["Map", "Feed", "Manage", "Settings"]

# detection is falling behind (NFR-01) when the model's 90th percentile time or the lag passes these
P90_LIMIT_MS = 500
LAG_LIMIT_S = 2.0

# PipelineStatus values that raise an amber warning, and the kind of problem: a feed warning clears when the feed
# resumes, an analysis warning when a frame is analysed again. frame_skipped alone is only logged (FR-14.1).
WARNINGS = {
    "feed_missing": ("feed", "No video yet, still waiting for the stream"),
    "feed_interrupted": ("feed", "Video interrupted, reconnecting"),
    "feed_unsupported": ("feed", "The video can't be decoded"),
    "persistent_frames_skipped": ("analysis", "Frames keep failing analysis"),
    "model_unavailable": ("analysis", "The detection model is unavailable"),
}
PAGE_ACTIONS = ["page_map", "page_feed", "page_manage", "page_settings"]

DEFAULT_SIDEBAR_WIDTH = 360
DEFAULT_LOG_WIDTH = 520
DEFAULT_BAR_HEIGHT = 0   # 0: the feeds bar's natural height


class MainWindow(QMainWindow):
    def __init__(self, demo=None, user=None, accounts=None):
        """
        demo: a raven.gui.demo.DemoBackend when started with --demo (simulated detector), else None.
        user: the signed-in raven.backend.accounts.User, and accounts the UserStore it came from (app.py signs in
        before the window opens). Without them (scripts and tests) the console runs as an unlocked administrator.
        """
        super().__init__()
        self._demo = demo
        self.accounts = accounts
        self.user = user or acl.User(username="local", name="Local operator", role="admin")
        self.setWindowTitle("RAVEN — Rescue Aerial Vision and Emergency Navigation" + (" DEMO" if demo else ""))
        self.store = QSettings()

        if demo is not None:
            # demo runs always start with no alerts; its settings are kept apart from the real ones
            config = self._load_demo_config(demo)
            self.store.remove("alert_statuses")
        else:
            config = self._load_config()
        self._config = config
        drone_colors.set_feeds([feed.name for feed in config.feeds])
        self._statuses: dict[str, str] = self._load_statuses()

        self._ingests: dict[str, IngestWorker] = {}   # drone name -> running worker
        self._ingest_errors: dict[str, str] = {}
        # set while the window closes, so feeds stopping then don't end the search session
        self._closing = False
        # Yes to the resume prompt: start as soon as the Discord test alert succeeds
        self._resume_pending = False
        # amber warnings: drone -> {kind ("feed", "analysis", "telemetry"): text}
        self._warnings: dict[str, dict[str, str]] = {}

        # the event bus and its subscribers; the detection log's file follows the saved frames folder
        self.bus = BackgroundBus()
        self.detection_log = DetectionLog(config.detection_log_path())
        self._log_path = None
        self.bus.on(HumanDetected, self.detection_log.on_human_detected)
        self.alert_feed = AlertFeed(self.bus, self)
        self.discord = DiscordAlerts(self.bus, self)
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
        self.settings_tab = SettingsTab(config, demo=demo is not None)
        self.feed_tab = FeedTab(self.model, feed_sources=lambda: [feed.source for feed in self._config.feeds])
        self.log_panel = LogPanel()
        # the search session form belongs to the settings, but is shown (and edited) on the Manage tab
        self.assignments = DroneAssignments(self.store)
        self.sessions = SessionHistory(self.store)
        self.sessions.interrupt_running()   # nothing runs yet at launch
        if config.session.created and self.sessions.get(config.session.created) is None:
            # a session from before the list existed
            self.sessions.record(config.session, "interrupted" if self.store.value("active_session", "") else "ended")
        self.manage_tab = ManageTab(self.settings_tab.session_group, self.feed_tab.live, self.warning_text,
                                    self.assignments, self.sessions,
                                    lambda: (self.settings_tab.session_created(), bool(self._ingests)))

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
        self.profile_button = ProfileButton(self.feed_tab.live, self.user, self.assignments)
        self.header.add_trailing(self.profile_button)
        self.profile_button.panel.duty_changed.connect(self._duty_changed)
        self.profile_button.panel.logout_requested.connect(self.log_out)
        self.pages = QStackedWidget()
        for page in (self.splitter, self.feed_tab, self.manage_tab, self.settings_page):
            self.pages.addWidget(page)
        self.header.page_selected.connect(self.show_page)
        self.header.theme_toggle_requested.connect(self.toggle_theme)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.header)
        # feed and analysis problems across every drone, in amber under the header until they clear
        self.warning_notice = QLabel()
        self.warning_notice.setObjectName("warningNotice")
        self.warning_notice.setWordWrap(True)
        self.warning_notice.hide()
        central_layout.addWidget(self.warning_notice)
        central_layout.addWidget(self.pages, 1)

        # the lock screen covers everything while the console is locked or after Log out; feeds keep running
        self.root = QStackedWidget()
        self.root.addWidget(central)
        self.lock_screen = None
        self._logged_out = False
        if accounts is not None:
            self.lock_screen = LockScreen(accounts)
            self.lock_screen.unlocked.connect(self._unlocked)
            self.root.addWidget(self.lock_screen)
        self.setCentralWidget(self.root)
        self.idle = IdleWatcher(parent=self)
        self.idle.idle.connect(self.lock)
        if accounts is not None:
            self.idle.reset()

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
        self.settings_tab.discord_verified_changed.connect(self._on_discord_verified)
        self.settings_tab.session_edited.connect(self._save_session)
        self.manage_tab.show_drone_requested.connect(self._show_drone_on_map)
        self.manage_tab.operator_assigned.connect(lambda *_: self._refresh_users_editor())
        self.manage_tab.new_session_requested.connect(self._new_session)
        self.manage_tab.remove_session_requested.connect(self._remove_session)
        self.manage_tab.operator_assigned.connect(
            lambda drone, name: self.log(f"{drone} assigned to {name}" if name != "Unassigned" else f"{drone} unassigned"))
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
        self.map_view.marker_unselected.connect(lambda _id: self.sidebar.clear_selection())
        self.map_view.link_requested.connect(self._open_link)
        self.link_viewer.back_requested.connect(self._back_to_map)
        self.map_view.log.connect(self.log)

        self.feed_tab.status_changed.connect(self._set_status)
        self.feed_tab.show_on_map_requested.connect(self._show_alert_on_map)
        self.feed_tab.log.connect(self.log)
        self.feed_tab.sidebar.clear_requested.connect(self._clear_alerts)

        self.alert_feed.events.connect(self._on_events)
        self.alert_feed.status.connect(self.log)
        self.alert_feed.pipeline_status.connect(self._on_pipeline_status)
        self.discord.state_changed.connect(self.video_bar.discord_light.set_state)
        self.discord.log.connect(self.log)

        # drone positions for the map legend's go-to links, from each feed's latest frame
        self._positions: dict = {}
        self._position_timer = QTimer(self)
        self._position_timer.setInterval(1000)
        self._position_timer.timeout.connect(self._send_drone_positions)
        self._position_timer.start()

        # show the other folder's alerts shortly after the saved frames folder stops changing
        self._history_timer = QTimer(self)
        self._history_timer.setSingleShot(True)
        self._history_timer.setInterval(600)
        self._history_timer.timeout.connect(self._load_history)

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
        self.log(f"Signed in as {self.user.name} ({self.user.role_label})")
        self.apply_role()
        if demo is not None:
            self.log(f"Demo mode: simulated detector, {len(config.feeds)} generated drone flights in {demo.folder}. "
                     "Press Start to run them.")
        # what's on disk, as the form shows it; the Save changes button lights up when the form differs
        self._saved_settings = self.settings_tab.config().to_dict()
        self._apply_config(self.settings_tab.config())
        self.settings_tab.test_sources()
        # a search that was running when the console last closed can be picked up again (FR-03)
        QTimer.singleShot(0, self._offer_resume)
        QTimer.singleShot(0, self._send_leftover_alerts)

    # ------------------------------------------------------------------ pages / theme / log

    def show_page(self, index: int):
        # pages the signed-in role doesn't have (Manage for reviewers and viewers) stay closed
        if not self.header.is_page_visible(index):
            return
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
        # the webhook is not part of to_dict(): it always comes from .env; the search session saves on its own
        current = self.settings_tab.config().to_dict()
        current.pop("session", None)
        saved = dict(self._saved_settings)
        saved.pop("session", None)
        return current != saved

    def _save_session(self):
        """
        The Manage tab has no Save button, so the session's name and area are kept as soon as they're edited.
        """
        session = self.settings_tab.config().session
        raw = json.dumps({"name": session.name, "area": session.area, "created": session.created})
        if raw != self.store.value("session", ""):
            self.store.setValue("session", raw)
            self.store.sync()
            self.sessions.record(session)   # a renamed session is renamed in the list too
            self.manage_tab.refresh_sessions()
            self.log(f"Search session saved: {session.name or 'unnamed'}" + (f" ({session.area})" if session.area else ""))

    def _saved_session(self, fallback: SearchSession) -> SearchSession:
        raw = self.store.value("session", "")
        try:
            return SearchSession.from_dict(json.loads(raw)) if raw else fallback
        except (ValueError, TypeError):
            return fallback

    def save_settings(self):
        if not self._allowed(acl.EDIT_SETTINGS, "save settings"):
            return
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
        config.session = self._saved_session(config.session)
        return config

    def _load_demo_config(self, demo) -> RavenConfig:
        """
        The generated demo flights, with the demo's own saved choices (feeds, confidence, sample rate, Discord)
        on top. The detector and frames folder always come from this launch.
        """
        config = demo.config()
        raw = self.store.value("config")
        if raw:
            try:
                saved = RavenConfig.from_dict(json.loads(raw))
            except (ValueError, TypeError):
                saved = None
            if saved is not None:
                for name in ("session", "min_confidence", "analysis_fps", "alert_cooldown_s", "discord_enabled", "realtime"):
                    setattr(config, name, getattr(saved, name))
                # saved feeds are used while their files still exist (the demo folder can be regenerated)
                if all(os.path.isfile(f.source) and os.path.isfile(f.telemetry_path)
                       for f in saved.feeds if f.source or f.telemetry_path):
                    config.feeds = saved.feeds
        config.webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
        config.session = self._saved_session(config.session)
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

        if config.detection_log_path() != self._log_path:
            if self._log_path is None:
                # at launch: show the earlier alerts right away
                self._load_history()
            else:
                self._history_timer.start()
        self._update_session_state()

    def _send_drone_positions(self):
        positions = {}
        for name in self.feed_tab.live.drone_names():
            _state, info = self.feed_tab.live.status(name)
            if info and isinstance(info.get("lat"), (int, float)) and isinstance(info.get("lon"), (int, float)):
                positions[name] = {"lat": info["lat"], "lon": info["lon"], "color": drone_colors.color(name)}
        if positions != self._positions:
            self._positions = positions
            self.map_view.set_drone_positions(positions)

    # ------------------------------------------------------------------ alert feed

    def _load_history(self):
        """
        Show the alerts in the current frames folder's detection log, and log new detections to it from now on.
        """
        path = self._config.detection_log_path()
        if path == self._log_path:
            return
        if self._log_path is not None:
            # a different folder holds different alerts
            self._clear_view()
        self._log_path = path
        self.detection_log.path = path

        try:
            records, skipped = read_log(path)
        except OSError as error:
            self.log(f"Could not read the detection log {path}: {error}")
            return
        if skipped:
            self.log(f"Skipped {skipped} unreadable line(s) in {path}")
        if records:
            self._on_events([(record["id"], record) for record in records])

    def _on_events(self, batch):
        new = self.model.add_events(batch, self._statuses)
        if not new:
            return
        self.map_view.add_alerts(new)
        if len(new) <= 3:
            # live alerts: one line each
            for alert in new:
                where = alert.where if alert.has_location else "no coordinates"
                self.log(f"Alert from {alert.drone}: {alert.title} at {alert.when} ({where})")
        else:
            # a replay or burst: one summary line
            missing = sum(1 for alert in new if not alert.has_location)
            note = f", {missing} without coordinates (not on the map)" if missing else ""
            drones = sorted({alert.drone for alert in new})
            self.log(f"Received {len(new)} alerts from {', '.join(drones)}{note}")

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
        if not self._allowed(acl.REVIEW_ALERTS, "review or dismiss alerts"):
            return
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

    def _show_drone_on_map(self, name: str):
        self.show_page(PAGE_NAMES.index("Map"))
        self.map_view.go_to_drone(name)

    def _show_alert_on_map(self, alert_id: str):
        self.show_page(PAGE_NAMES.index("Map"))
        self.sidebar.select_alert(alert_id)
        self._focus_alert(alert_id)

    def _clear_alerts(self):
        if not self._allowed(acl.CLEAR_ALERTS, "clear the alert list"):
            return
        answer = QMessageBox.question(
            self, "Clear alerts",
            "Remove every alert from the list and the map?\n\n"
            "Nothing is deleted from the detection log; restarting the console brings them back.")
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

    def start_blockers(self, config: RavenConfig) -> list[str]:
        """
        Everything that stops Start: the settings' own problems, an unnamed session, and a Discord webhook
        that hasn't passed a test alert this launch (FR-08).
        """
        problems = []
        if not config.session.name.strip():
            problems.append("Name the search session (Manage > Search Sessions).")
        problems.extend(config.problems())
        if config.discord_enabled and config.webhook_url.strip() and not self.settings_tab.discord_verified():
            problems.append("Send a Discord test alert first: the webhook hasn't been tested since the console "
                            "started, or it changed after the last test.")
        return problems

    def _update_session_state(self):
        if not hasattr(self, "header"):
            return
        config = self._config
        if self._ingests:
            self.header.set_session(config.session.name, "running")
            return
        problems = self.start_blockers(config)
        if problems:
            self.header.set_session(config.session.name, "not_ready", "Not ready:\n• " + "\n• ".join(problems))
        else:
            self.header.set_session(config.session.name, "ready", "Ready: press Start to run the feeds")

    def _on_discord_verified(self, verified: bool):
        self._update_session_state()
        if not self._resume_pending:
            return
        self._resume_pending = False
        if verified:
            self.start_ingest()
        else:
            self.log("Resume stopped: the Discord test alert failed. Fix the webhook, then press Start.")
            self.show_page(PAGE_NAMES.index("Settings"))

    # ------------------------------------------------------------------ search session (FR-03)

    def _offer_resume(self):
        name = self.store.value("active_session", "")
        if not name or self._ingests:
            return
        answer = QMessageBox.question(
            self, "Resume search",
            f"Search session \"{name}\" was still running when the console closed.\n\n"
            "Resume search session " + name + "?")
        if answer != QMessageBox.Yes:
            self._end_session("not resumed")
            return
        self.log(f"Resuming search session {name}")
        config = self.settings_tab.config()
        if config.discord_enabled and not self.settings_tab.discord_verified() and config.webhook_url.strip():
            # every launch needs one successful Discord test; send it, then start when it lands
            self._resume_pending = True
            self.log("Sending a Discord test alert before resuming…")
            self.settings_tab.send_test_alert()
        else:
            self.start_ingest()

    def _send_leftover_alerts(self):
        """
        Alerts that hadn't reached Discord when the console last closed are sent now if a webhook is set, else on
        the next Start.
        """
        config = self.settings_tab.config()
        try:
            pending = DiscordAlerts.outbox_for(config).pending()
        except OSError as error:
            self.log(f"Could not read the Discord outbox: {error}")
            return
        if not pending:
            return
        if not config.webhook_url.strip() or self._ingests:
            self.log(f"{len(pending)} Discord alert(s) from earlier weren't sent yet; they'll go out on the next Start")
            return
        self.log(f"Sending {len(pending)} Discord alert(s) that weren't sent before the console closed…")
        self.discord.start(config)
        # turn Discord off again once they're out (or after a minute, keeping the rest for later)
        if not self._ingests:
            self.discord.stop(timeout=60)

    def _begin_session(self, config: RavenConfig):
        """
        Start saves the settings and marks the session active, so it can be resumed after a restart.
        """
        if not config.session.created:
            config.session.created = time.time()
            self.settings_tab.mark_session_created(config.session.created)
        self.save_settings()
        self._save_session()
        self.sessions.record(config.session, "active")
        self.manage_tab.refresh_sessions()
        self.store.setValue("active_session", config.session.name)
        self.store.sync()
        self.log(f"Search session {config.session.name} active"
                 + (f" ({config.session.area})" if config.session.area else ""))

    def _end_session(self, reason: str):
        created = self.settings_tab.session_created()
        if created and self.sessions.get(created) is not None:
            self.sessions.set_status(created, "ended", time.time())
            self.manage_tab.refresh_sessions()
        if self.store.value("active_session", ""):
            self.store.remove("active_session")
            self.store.sync()
            self.log(f"Search session no longer active ({reason})")

    def _new_session(self):
        if self._ingests:
            QMessageBox.information(self, "New session", "Stop the feeds before starting a new search session.")
            return
        self.settings_tab.clear_session()
        self._save_session()
        self.show_page(PAGE_NAMES.index("Manage"))
        self.settings_tab.session_name_edit.setFocus()
        self.log("New search session: name it, then press Start")

    def _remove_session(self, created: float):
        if not self._allowed(acl.REMOVE_SESSIONS, "remove search sessions"):
            return
        entry = self.sessions.get(created)
        if entry is None:
            return
        if self._ingests and created == self.settings_tab.session_created():
            QMessageBox.information(self, "Remove session", "This session is running. Stop the feeds first.")
            return
        name = entry.get("name") or "Unnamed"
        if QMessageBox.question(self, "Remove session",
                                f"Remove the search session \"{name}\" from the list?\n\n"
                                "Its saved frames and detection log are kept.") != QMessageBox.Yes:
            return
        self.sessions.remove(created)
        if created == self.settings_tab.session_created():
            # the form held the removed session: start the next one from scratch
            self.settings_tab.clear_session()
            self._save_session()
            if self.store.value("active_session", ""):
                self.store.remove("active_session")
                self.store.sync()
        self.manage_tab.refresh_sessions(force=True)
        self.log(f"Removed search session {name}")

    def start_ingest(self):
        if self._ingests or not self._allowed(acl.RUN_FEEDS, "start the feeds"):
            return
        config = self.settings_tab.config()
        problems = self.start_blockers(config)
        if not problems:
            # every recorded video has to decode before the run starts (FR-04.2); streams are checked as they run
            from raven.backend.file_ingest import check_decodes

            for feed in config.enabled_feeds():
                if "://" in feed.source:
                    continue
                try:
                    check_decodes(feed.source)
                except RuntimeError as error:
                    problems.append(f"{feed.name}: {error}")
        if problems:
            QMessageBox.warning(self, "Can't start yet", "Fix these first:\n\n• " + "\n• ".join(problems))
            unnamed = any("Name the search session" in problem for problem in problems)
            self.show_page(PAGE_NAMES.index("Manage" if unnamed else "Settings"))
            return
        self._begin_session(config)

        feeds = config.enabled_feeds()
        self._ingest_errors = {}
        for feed in feeds:
            # demo mode swaps the YOLO model for a simulated detector
            worker = IngestWorker(config, feed, self.bus,
                                  model_factory=self._demo.make_detector if self._demo else None)
            worker.progress.connect(self.video_bar.set_stats)
            worker.progress.connect(self._on_progress)
            worker.frame_ready.connect(self.feed_tab.live.on_frame)
            worker.source_info.connect(self.video_bar.set_source_info)
            worker.log.connect(self.log)
            worker.failed.connect(self._on_ingest_failed)
            worker.finished.connect(lambda name=feed.name: self._on_ingest_finished(name))
            self._ingests[feed.name] = worker

        for feed in feeds:
            self._clear_warnings(feed.name)
        self.video_bar.set_running(True, [feed.name for feed in feeds])
        for feed in feeds:
            self.feed_tab.live.set_state(feed.name, "starting")
        self.settings_tab.set_running(True)
        self.log(f"Starting {len(feeds)} feed(s): " + ", ".join(f"{feed.name} ({feed.source})" for feed in feeds))
        for worker in self._ingests.values():
            worker.start()

        if config.discord_enabled:
            self.discord.start(config)
        self._update_session_state()

    def stop_ingest(self):
        if not self._ingests or not self._allowed(acl.RUN_FEEDS, "stop the feeds"):
            return
        self._end_session("stopped by operator")
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
        self._clear_warnings(name)
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
        if not self._closing:
            self._end_session("all feeds finished")
        self._update_session_state()
        # Discord gets the last events and a few seconds to send what is queued, then turns off
        self.discord.stop()

    # ------------------------------------------------------------------ accounts and roles

    def _allowed(self, permission: str, action: str) -> bool:
        if self.user.can(permission):
            return True
        self.log(f"{self.user.name} ({self.user.role_label}) can't {action}")
        QMessageBox.information(self, "Not allowed", f"Your role ({self.user.role_label}) can't {action}. "
                                "Ask an administrator if you need to.")
        return False

    def apply_role(self):
        """
        Grey out what the signed-in user's role can't do (the actions check again when used).
        """
        user = self.user
        self.video_bar.set_allowed(user.can(acl.RUN_FEEDS))
        self.settings_tab.set_read_only(not user.can(acl.EDIT_SETTINGS), user.role_label)
        # Manage: administrators and drone operators (accounts for administrators only)
        manage = PAGE_NAMES.index("Manage")
        self.header.set_page_visible(manage, user.can(acl.MANAGE_SEARCH))
        if not user.can(acl.MANAGE_SEARCH) and self.pages.currentIndex() == manage:
            self.show_page(PAGE_NAMES.index("Map"))
        if user.can(acl.MANAGE_USERS) and self.accounts is not None:
            editor = UsersEditor(self.accounts, user, self.assignments)
            editor.log.connect(self.log)
            editor.changed.connect(self._update_operators)   # new or changed accounts show up in the dropdowns
            self.manage_tab.set_accounts_editor(editor)
            self._users_editor = editor
        else:
            self.manage_tab.set_accounts_editor(None)
            self._users_editor = None
        self._update_operators()
        # viewers only get the live feed, not the footage behind alerts
        self.feed_tab.set_review_allowed(user.can(acl.REVIEW_FOOTAGE))
        self.manage_tab.set_can_remove_sessions(user.can(acl.REMOVE_SESSIONS))
        for sidebar in (self.sidebar, self.feed_tab.sidebar):
            if hasattr(sidebar, "set_review_allowed"):
                sidebar.set_review_allowed(user.can(acl.REVIEW_ALERTS), user.can(acl.CLEAR_ALERTS))

    def _duty_changed(self, key: str):
        self.log(f"Status: {self.profile_button.panel.duty_combo.currentText()}")
        if self.accounts is not None and self.accounts.get(self.user.username) is not None:
            self.accounts.update(self.user.username, duty=key)
        self._refresh_users_editor()

    def _refresh_users_editor(self):
        editor = getattr(self, "_users_editor", None)
        if editor is not None:
            editor.refresh()

    def _update_operators(self):
        """
        Who can be assigned a drone on the Manage tab (everyone who can run feeds); administrators assign them.
        """
        users = self.accounts.users() if self.accounts is not None else [self.user]
        people = [(user.username, user.name) for user in users if user.can(acl.RUN_FEEDS)]
        self.manage_tab.set_people(people, self.user.can(acl.MANAGE_USERS))

    def lock(self):
        """
        Cover the console until the signed-in user enters their password (after IDLE_LOCK_MINUTES idle).
        """
        if self.lock_screen is None or self.root.currentWidget() is self.lock_screen:
            return
        self.profile_button.panel.hide()
        self.lock_screen.show_for(self.user)
        self.root.setCurrentWidget(self.lock_screen)
        self.idle.pause()
        self.log(f"Console locked ({self.user.name})")

    def log_out(self):
        if self.lock_screen is None:
            self.log("Log out needs accounts: start the console with python -m raven.gui")
            return
        self.profile_button.panel.hide()
        self.log(f"{self.user.name} logged out"
                 + (" (feeds keep running until the next sign-in)" if self._ingests else ""))
        self._logged_out = True
        self.lock_screen.show_for(None)
        self.root.setCurrentWidget(self.lock_screen)
        self.idle.pause()

    def _unlocked(self, user):
        # a sign-in after Log out starts on the Map like a fresh launch; unlocking after the idle lock carries on
        signed_in = self._logged_out or user.username.lower() != self.user.username.lower()
        self._logged_out = False
        self.user = user
        if signed_in:
            self.profile_button.panel.set_user(user)
            self.profile_button._draw_icon()
            self.apply_role()
            self.show_page(PAGE_NAMES.index("Map"))
            self.log(f"Signed in as {user.name} ({user.role_label})")
        else:
            self.log(f"Console unlocked ({user.name})")
        self.root.setCurrentIndex(0)
        self.idle.reset()

    # ------------------------------------------------------------------ amber warnings

    def _on_pipeline_status(self, drone: str, status: str, message: str):
        if status == "feed_resumed":
            self._set_warning(drone, "feed", "")
        elif status in WARNINGS:
            kind, text = WARNINGS[status]
            self._set_warning(drone, kind, text)

    def _on_progress(self, name: str, stats: dict):
        # an analysis warning clears once a frame is analysed again; telemetry loss is only reported by live feeds
        if not stats.get("consecutive_failures"):
            self._set_warning(name, "analysis", "")
        self._set_warning(name, "telemetry",
                          "No telemetry for over 2 s, alerts have no location" if stats.get("telemetry_lost") else "")
        p90, lag = stats.get("inference_p90_ms", 0), stats.get("lag_s", 0)
        slow = []
        if p90 > P90_LIMIT_MS:
            slow.append(f"detection takes {p90:.0f} ms (p90)")
        if lag > LAG_LIMIT_S:
            slow.append(f"{lag:.1f} s behind")
        self._set_warning(name, "latency", "Falling behind: " + ", ".join(slow) if slow else "")

    def _set_warning(self, drone: str, kind: str, text: str):
        current = self._warnings.get(drone, {})
        if current.get(kind, "") == text:
            return
        if text:
            self._warnings.setdefault(drone, {})[kind] = text
        else:
            current.pop(kind, None)
            if not current:
                self._warnings.pop(drone, None)
        self._show_warnings(drone)

    def _clear_warnings(self, drone: str):
        if self._warnings.pop(drone, None):
            self._show_warnings(drone)

    def warning_text(self, drone: str) -> str:
        return " · ".join(self._warnings.get(drone, {}).values())

    def _show_warnings(self, drone: str):
        text = self.warning_text(drone)
        self.video_bar.set_warning(drone, text)
        self.feed_tab.live.set_warning(drone, text)
        lines = [f"⚠ {name or 'Feed'}: {self.warning_text(name)}" for name in self._warnings]
        self.warning_notice.setText("    ".join(lines))
        self.warning_notice.setVisible(bool(lines))

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

        unsent = self.discord.unsent()
        if unsent:
            answer = QMessageBox.question(
                self, "Discord alerts not sent",
                f"{unsent} alert(s) haven't reached Discord yet. They're saved and will be sent the next time the "
                "console starts.\n\nQuit anyway?")
            if answer != QMessageBox.Yes:
                event.ignore()
                return

        # quitting mid-run keeps the session active, so the next launch offers to resume it
        self._closing = True
        workers = list(self._ingests.values())
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.wait(5000)
        # the feeds' last events still reach the subscribers, and queued Discord alerts get a few seconds to send
        self.discord.stop(wait=True)
        self.bus.close()
        self.settings_tab.wait_for_jobs()
        self.feed_tab.shutdown()

        # settings are only written by Save changes (demo mode saves to its own settings)
        self.store.setValue("geometry", self.saveGeometry())
        self.store.setValue("sidebar_width", self._sidebar_width)
        self.store.setValue("feed_sidebar_width", self.feed_tab.sidebar_width())
        self.store.setValue("feeds_bar_height", self._bar_height)
        self.store.setValue("log_width", self._log_width)
        super().closeEvent(event)
