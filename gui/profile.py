"""
Operator profile: the button at the right end of the header and the panel it opens, with who is signed in,
the drones they're operating (state, battery, flight time) and their account details, partly hidden.

There's no sign-in yet, so the operator is the sample profile in OPERATOR. Swap it for real account data
once accounts exist.
"""
import time
from datetime import datetime

from PySide6.QtCore import QPoint, QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QToolButton, QVBoxLayout,
    QWidget,
)

from gui.alerts import drone_colors
from gui.alerts_panel import drone_icon
from gui.theme import THEMES, themed_icon

# the signed-in operator: (label, value, how it shows while hidden); rows with a hidden form get an eye button
OPERATOR = {
    "name": "Albert Ross",
    "role": "Drone Operator",
    "account": [
        ("Operator ID", "RVN-OP-4417-2731", "RVN-OP-••••-2731"),
        ("Email", "albert.ross@raven-ops.example", "al•••••••••@raven-ops.example"),
        ("Part 107 certificate", "4823194", "•••3194"),
        ("Certificate expires", "03/2028", None),
        ("Unit", "North Texas SAR · Team 2", None),
    ],
}

# the operator's duty status: key -> (label, color); the header icon carries the same colored dot
DUTY_STATUSES = {
    "on_duty": ("On duty", "#30a46c"),
    "away": ("Away", "#f5b700"),
    "off_duty": ("Off duty", "#8b8d98"),
}
DEFAULT_DUTY = "on_duty"


def _dot(color: str, size: int = 12) -> QIcon:
    pixmap = QPixmap(size * 2, size * 2)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(color))
    painter.drawEllipse(4, 4, size * 2 - 8, size * 2 - 8)
    painter.end()
    return QIcon(pixmap)


# battery levels: at or above each threshold, its word and color
BATTERY_LEVELS = [(50, "Good", "#30a46c"), (25, "Low", "#f5a524"), (0, "Critical", "#e5484d")]

# what a drone's row says for each feed state (the same states as the Live view)
FLIGHT_STATES = {
    "idle": ("On the ground", "#8b8d98"),
    "starting": ("Taking off", "#f5a524"),
    "running": ("Flying", "#30a46c"),
    "finished": ("Landed", "#8b8d98"),
    "stopped": ("Stopped", "#8b8d98"),
    "failed": ("Feed failed", "#e5484d"),
}


def _clock(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, rest = divmod(seconds, 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"{rest // 60}:{rest % 60:02d}"


def battery_level(percent: float) -> tuple[str, str]:
    for threshold, word, color in BATTERY_LEVELS:
        if percent >= threshold:
            return word, color
    return BATTERY_LEVELS[-1][1:]


def _heading(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("profileHeading")
    return label


class DroneRow(QFrame):
    """
    One drone: name and flight state, battery bar, flight time.
    """

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.setObjectName("profileDrone")
        self.name = name
        self.icon = QLabel()
        self.name_label = QLabel(name)
        font = self.name_label.font()
        font.setBold(True)
        self.name_label.setFont(font)
        self.state = QLabel()
        self.state.setObjectName("hint")

        self.battery = QProgressBar()
        self.battery.setRange(0, 100)
        self.battery.setTextVisible(False)
        self.battery.setFixedHeight(8)
        self.battery_label = QLabel()
        self.battery_label.setMinimumWidth(96)
        self.flight_time = QLabel()
        self.flight_time.setObjectName("hint")
        self.flight_time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        top = QHBoxLayout()
        top.setSpacing(6)
        top.addWidget(self.icon)
        top.addWidget(self.name_label)
        top.addStretch(1)
        top.addWidget(self.state)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        bottom.addWidget(QLabel("Battery"))
        bottom.addWidget(self.battery, 1)
        bottom.addWidget(self.battery_label)
        bottom.addWidget(self.flight_time)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        layout.addLayout(top)
        layout.addLayout(bottom)

    def show_status(self, state: str, info: dict | None):
        self.icon.setPixmap(drone_icon(drone_colors.color(self.name)).pixmap(12, 12))
        word, color = FLIGHT_STATES.get(state, FLIGHT_STATES["idle"])
        self.state.setText(f"<span style='color:{color}'>●</span> {word}")

        battery = info.get("battery") if info else None
        if isinstance(battery, (int, float)):
            percent = max(0, min(100, round(battery)))
            level, bar_color = battery_level(percent)
            self.battery.setValue(percent)
            self.battery.setStyleSheet(f"QProgressBar::chunk {{ background: {bar_color}; border-radius: 3px; }}")
            self.battery_label.setText(f"{percent}% · {level}")
            self.battery.setToolTip(f"Battery {percent}% ({level.lower()})")
        else:
            self.battery.setValue(0)
            self.battery_label.setText("No data")
            self.battery.setToolTip("No battery reading yet. It comes from the feed's telemetry "
                                    "(battery_percent column) once the drone is flying.")

        if info is None:
            self.flight_time.setText("Flight time —")
        else:
            self.flight_time.setText(f"Flight time {_clock(info['time'])}")


class ProfilePanel(QWidget):
    """
    The popup under the profile button. Reads each drone's state and latest frame details from the Live view.
    """

    duty_changed = Signal(str)       # a key of DUTY_STATUSES
    logout_requested = Signal()

    def __init__(self, live_view, parent=None):
        super().__init__(parent, Qt.Popup)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.live = live_view
        self._rows: dict[str, DroneRow] = {}
        self.signed_in = datetime.now()

        card = QFrame()
        card.setObjectName("profilePanel")

        # ---- who is signed in
        self.avatar = QLabel()
        self.avatar.setObjectName("profileAvatar")
        self.avatar.setFixedSize(56, 56)
        self.avatar.setAlignment(Qt.AlignCenter)
        name = QLabel(OPERATOR["name"])
        name.setObjectName("profileName")
        role = QLabel(OPERATOR["role"])
        role.setObjectName("hint")
        # duty status, remembered between launches
        self.duty_combo = QComboBox()
        self.duty_combo.setObjectName("dutyCombo")
        self.duty_combo.setToolTip("Your status")
        for key, (label, color) in DUTY_STATUSES.items():
            self.duty_combo.addItem(_dot(color), label, key)
        saved = QSettings().value("duty_status", DEFAULT_DUTY)
        self.duty_combo.setCurrentIndex(max(0, self.duty_combo.findData(saved)))
        self.duty_combo.currentIndexChanged.connect(self._duty_picked)
        self.flying = QLabel()
        self.flying.setObjectName("hint")
        duty_row = QHBoxLayout()
        duty_row.setSpacing(8)
        duty_row.addWidget(self.duty_combo)
        duty_row.addWidget(self.flying, 1)

        who = QVBoxLayout()
        who.setSpacing(2)
        who.addWidget(name)
        who.addWidget(role)
        who.addSpacing(2)
        who.addLayout(duty_row)

        # there's no sign-in yet, so logging out only notes that in the log
        self.logout_button = QPushButton("Log out")
        self.logout_button.setObjectName("logoutButton")
        self.logout_button.setCursor(Qt.PointingHandCursor)
        self.logout_button.setToolTip("Sign out of the console (not available yet: there's no sign-in)")
        self.logout_button.clicked.connect(self._logout)

        top = QHBoxLayout()
        top.setSpacing(12)
        top.addWidget(self.avatar, 0, Qt.AlignTop)
        top.addLayout(who, 1)
        top.addWidget(self.logout_button, 0, Qt.AlignTop)

        # ---- drones being operated
        self.drones_heading = _heading("Drones being operated")
        self.drones_box = QVBoxLayout()
        self.drones_box.setSpacing(6)
        self.no_drones = QLabel("No drone feeds are turned on. Add one in Settings > Drone feeds.")
        self.no_drones.setObjectName("hint")
        self.no_drones.setWordWrap(True)
        self.drones_box.addWidget(self.no_drones)

        # ---- account info
        account = QGridLayout()
        account.setHorizontalSpacing(16)
        account.setVerticalSpacing(5)
        account.setColumnStretch(1, 1)
        self._secrets: list[tuple[QLabel, QToolButton, str, str]] = []
        for row, (label, value, hidden) in enumerate(OPERATOR["account"]):
            key = QLabel(label)
            key.setObjectName("hint")
            shown = QLabel(hidden or value)
            shown.setTextInteractionFlags(Qt.TextSelectableByMouse)
            account.addWidget(key, row, 0)
            account.addWidget(shown, row, 1)
            if hidden is not None:
                eye = QToolButton()
                eye.setObjectName("eyeButton")
                eye.setCheckable(True)
                eye.setIconSize(QSize(15, 15))
                eye.setFixedSize(20, 18)  # no taller than a text row
                eye.setCursor(Qt.PointingHandCursor)
                eye.toggled.connect(lambda _=False, i=len(self._secrets): self._show_secret(i))
                account.addWidget(eye, row, 2)
                self._secrets.append((shown, eye, value, hidden))
        self.signed_in_label = QLabel()
        account.addWidget(_hint_label("Signed in"), len(OPERATOR["account"]), 0)
        account.addWidget(self.signed_in_label, len(OPERATOR["account"]), 1)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addLayout(top)
        layout.addWidget(_line())
        layout.addWidget(self.drones_heading)
        layout.addLayout(self.drones_box)
        layout.addWidget(_line())
        layout.addWidget(_heading("Account info"))
        layout.addLayout(account)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(card)
        self.setFixedWidth(400)

        # flight times and batteries move while the panel is open
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.refresh)
        self.hidden_at = 0.0

    def duty(self) -> str:
        return self.duty_combo.currentData()

    def _duty_picked(self, _index: int):
        QSettings().setValue("duty_status", self.duty())
        self.duty_changed.emit(self.duty())

    def _logout(self):
        self.hide()
        self.logout_requested.emit()

    def set_theme(self, name: str):
        self._theme = name
        self.avatar.setPixmap(themed_icon("profile.svg", name, 64).pixmap(32, 32))
        for i in range(len(self._secrets)):
            self._show_secret(i)

    def _show_secret(self, index: int):
        label, eye, value, hidden = self._secrets[index]
        visible = eye.isChecked()
        label.setText(value if visible else hidden)
        # the icon shows the value's state: closed eye while hidden, open eye while shown
        eye.setIcon(themed_icon("eye-open.svg" if visible else "eye-closed.svg", getattr(self, "_theme", "light")))
        eye.setToolTip("Hide" if visible else "Show")

    def hide_secrets(self):
        for _label, eye, _value, _hidden in self._secrets:
            eye.setChecked(False)

    def refresh(self):
        names = self.live.drone_names()
        for name in list(self._rows):
            if name not in names:
                self._rows.pop(name).deleteLater()
        for i, name in enumerate(names):
            row = self._rows.get(name)
            if row is None:
                row = self._rows[name] = DroneRow(name)
            self.drones_box.insertWidget(i, row)  # keeps them in the feeds' order
            row.show_status(*self.live.status(name))
        self.no_drones.setVisible(not names)

        flying = sum(1 for name in names if self.live.status(name)[0] in ("starting", "running"))
        self.drones_heading.setText(f"Drones being operated ({len(names)})")
        self.flying.setText(f"{flying} of {len(names)} flying")
        self.signed_in_label.setText(f"{self.signed_in:%b %d, %H:%M}")

    def open_under(self, button: QWidget):
        self.refresh()
        self.adjustSize()
        # right edges lined up, just below the button, kept on screen
        corner = button.mapToGlobal(QPoint(button.width(), button.height() + 6))
        x = corner.x() - self.width()
        screen = button.screen().availableGeometry()
        x = max(screen.left() + 8, min(x, screen.right() - self.width() - 8))
        self.move(x, corner.y())
        self.show()
        self._timer.start()

    def hideEvent(self, event):
        self._timer.stop()
        self.hidden_at = time.monotonic()
        self.hide_secrets()  # revealed values are hidden again whenever the panel closes
        super().hideEvent(event)


def _hint_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("hint")
    return label


def _line() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setObjectName("barSeparator")
    return line


class ProfileButton(QToolButton):
    """
    The round profile icon in the header; a click opens the profile panel.
    """
    opened = Signal()

    def __init__(self, live_view, parent=None):
        super().__init__(parent)
        self.setObjectName("themeButton")  # same look as the light/dark toggle beside it
        self.setIconSize(QSize(24, 24))
        self.setCursor(Qt.PointingHandCursor)
        self.panel = ProfilePanel(live_view, self)
        self.panel.duty_changed.connect(lambda _: self._draw_icon())
        self.clicked.connect(self._toggle)
        self._theme = "light"

    def _toggle(self):
        # pressing the button closes the open popup before this click arrives; don't reopen it
        if self.panel.isVisible() or time.monotonic() - self.panel.hidden_at < 0.3:
            self.panel.hide()
        else:
            self.panel.open_under(self)
            self.opened.emit()

    def set_theme(self, name: str):
        self._theme = name
        self.panel.set_theme(name)
        self._draw_icon()

    def _draw_icon(self):
        """
        The profile icon with a dot in the corner for the duty status.
        """
        label, color = DUTY_STATUSES[self.panel.duty()]
        pixmap = themed_icon("profile.svg", self._theme).pixmap(48, 48)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        # a ring in the header color keeps the dot apart from the icon's lines
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(THEMES[self._theme]["header"]))
        painter.drawEllipse(29, 29, 19, 19)
        painter.setBrush(QColor(color))
        painter.drawEllipse(32, 32, 13, 13)
        painter.end()
        self.setIcon(QIcon(pixmap))
        self.setToolTip(f"{OPERATOR['name']} ({label}): profile and drones")
