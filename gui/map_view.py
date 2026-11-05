"""
Zoomable map (Leaflet + OpenStreetMap inside QtWebEngine) with a pin per alert.
Python calls into the page with runJavaScript; the page calls back through a QWebChannel bridge.

Links clicked on the map (the Leaflet / OpenStreetMap credits) never navigate the map itself: they open in a
separate LinkViewer with a "Go back to map" button, so the map keeps its pins, view and theme.
"""
import json
from pathlib import Path

from PySide6.QtCore import QFile, QIODevice, QObject, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from gui.alerts import Alert

MAP_HTML = Path(__file__).with_name("map.html")

# the page is given an https origin so map tiles are requested with a referrer, as OpenStreetMap requires
PAGE_URL = QUrl("https://raven.local/map")

# page background before tiles load, per theme (matches map.html)
MAP_BACKGROUNDS = {"light": "#dfe3e8", "dark": "#1d2025"}


def _qwebchannel_js() -> str:
    """
    Qt ships the browser half of QWebChannel as a resource; inline it so the https page can use it.
    """
    resource = QFile(":/qtwebchannel/qwebchannel.js")
    if not resource.open(QIODevice.ReadOnly):
        raise RuntimeError("qwebchannel.js is missing from the Qt installation")
    try:
        return bytes(resource.readAll()).decode("utf-8")
    finally:
        resource.close()


class _Bridge(QObject):
    """
    The object the page sees as channel.objects.bridge.
    """
    ready = Signal()
    marker_clicked = Signal(str)

    @Slot()
    def pageReady(self):
        self.ready.emit()

    @Slot(str)
    def markerClicked(self, alert_id):
        self.marker_clicked.emit(alert_id)


class _LinkCatcher(QWebEnginePage):
    """
    Stand-in page for links that ask for a new window (target="_blank"): reports the URL instead of loading it.
    """
    caught = Signal(QUrl)

    def acceptNavigationRequest(self, url, nav_type, is_main_frame):
        self.caught.emit(url)
        self.deleteLater()
        return False


class _MapPage(QWebEnginePage):
    """
    The map's page: forwards JavaScript errors to the log and keeps the map from navigating away.
    """
    console_message = Signal(str)
    link_clicked = Signal(QUrl)

    def javaScriptConsoleMessage(self, level, message, line, source):
        if level != QWebEnginePage.JavaScriptConsoleMessageLevel.InfoMessageLevel:
            self.console_message.emit(f"Map: {message} (line {line})")

    def acceptNavigationRequest(self, url, nav_type, is_main_frame):
        if not is_main_frame:
            return True
        if nav_type == QWebEnginePage.NavigationType.NavigationTypeLinkClicked:
            # open the link elsewhere; the map page itself stays put
            self.link_clicked.emit(url)
            return False
        if nav_type == QWebEnginePage.NavigationType.NavigationTypeBackForward:
            # there is nothing to go back to: leaving would reload the map and lose its state
            return False
        return True

    def createWindow(self, _window_type):
        catcher = _LinkCatcher(self)
        catcher.caught.connect(self.link_clicked)
        return catcher


class MapView(QWebEngineView):
    marker_clicked = Signal(str)
    link_requested = Signal(QUrl)
    log = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        page = _MapPage(self)
        page.console_message.connect(self.log)
        page.link_clicked.connect(self.link_requested)
        self.setPage(page)

        # what the map should show; replayed whenever the page (re)loads so nothing is lost
        self._theme = "light"
        self._alerts: dict[str, dict] = {}
        self._pending_focus: str | None = None
        self._ready = False
        self.loadStarted.connect(self._on_load_started)

        self._bridge = _Bridge(self)
        self._bridge.ready.connect(self._on_ready)
        self._bridge.marker_clicked.connect(self.marker_clicked)
        self._channel = QWebChannel(self.page())
        self._channel.registerObject("bridge", self._bridge)
        self.page().setWebChannel(self._channel)

        html = MAP_HTML.read_text(encoding="utf-8").replace("/*QWEBCHANNEL_JS*/", _qwebchannel_js())
        self.setHtml(html, PAGE_URL)

    def _on_load_started(self):
        self._ready = False

    def _on_ready(self):
        """
        The page and its bridge are up (first load or a reload): restore the theme and every pin.
        """
        self._ready = True
        self._run("setTheme", self._theme)
        if self._alerts:
            self._run("addAlerts", list(self._alerts.values()))
        if self._pending_focus is not None:
            self._run("focusAlert", self._pending_focus, False)
            self._pending_focus = None

    def _run(self, function: str, *args):
        self.page().runJavaScript(f"{function}({', '.join(json.dumps(a) for a in args)});")

    def _call(self, function: str, *args):
        # while the page loads, the state above is enough: _on_ready replays it
        if self._ready:
            self._run(function, *args)

    def add_alerts(self, alerts: list[Alert]):
        located = [alert.map_payload() for alert in alerts if alert.has_location]
        for payload in located:
            self._alerts[payload["id"]] = payload
        if located:
            self._call("addAlerts", located)

    def focus_alert(self, alert_id: str):
        if self._ready:
            self._run("focusAlert", alert_id, False)
        else:
            self._pending_focus = alert_id

    def set_status(self, alert_id: str, status: str):
        if alert_id in self._alerts:
            self._alerts[alert_id]["status"] = status
        self._call("setStatus", alert_id, status)

    def set_drone_colors(self, colors: dict):
        """
        Recolor the pins of these drones ({name: color}), e.g. after feeds were reordered or renamed.
        """
        for payload in self._alerts.values():
            if payload.get("drone") in colors:
                payload["color"] = colors[payload["drone"]]
        self._call("setDroneColors", colors)

    def clear_alerts(self):
        self._alerts.clear()
        self._pending_focus = None
        self._call("clearAlerts")

    def fit_all(self):
        self._call("fitAll")

    def set_theme(self, name: str):
        self._theme = name
        # the page background shows before tiles load, so match it to avoid a white flash in dark mode
        self.page().setBackgroundColor(QColor(MAP_BACKGROUNDS[name]))
        self._call("setTheme", name)


class LinkViewer(QWidget):
    """
    Shows a page opened from the map (e.g. the OpenStreetMap credits) with a "Go back to map" button.
    """
    back_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = QWebEngineView()
        self.view.titleChanged.connect(lambda _: self._update_label())
        self.view.urlChanged.connect(lambda _: self._update_label())

        self.address = QLabel()
        self.address.setObjectName("hint")
        self.address.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.address.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.back_button = QPushButton("Go back to map")
        self.back_button.setObjectName("backToMap")
        self.back_button.setCursor(Qt.PointingHandCursor)
        self.back_button.setToolTip("Return to the map (Esc)")
        self.back_button.clicked.connect(self.back_requested)
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self.back_requested)

        bar = QFrame()
        bar.setObjectName("linkBar")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(10, 6, 10, 6)
        bar_layout.addWidget(self.address, 1)
        bar_layout.addWidget(self.back_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.view, 1)
        layout.addWidget(bar)

    def open(self, url: QUrl):
        self.view.setUrl(url)
        self._update_label()

    def close_page(self):
        # unload the page so it stops using the network (and any audio/video) while hidden
        self.view.setUrl(QUrl("about:blank"))

    def _update_label(self):
        url = self.view.url().toString()
        title = self.view.title()
        text = f"{title} — {url}" if title and title != url else url
        self.address.setText(self.address.fontMetrics().elidedText(text, Qt.ElideRight, max(100, self.address.width())))
        self.address.setToolTip(url)
