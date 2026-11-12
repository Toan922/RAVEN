"""
Manage tab: the search sessions (the current one's name and area, and every session created so far with its
status), the drones being flown (flight state, battery, flight time, the operator assigned to
each, and a button to find it on the map) and, for administrators, the accounts.

Only roles that run searches see it: administrators get all three sections, drone operators the session and the
drones. Reviewers and viewers don't have the tab.
"""
import json
from datetime import datetime

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QScrollArea,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from raven.gui.profile import DroneRow

UNASSIGNED = "Unassigned"


# a session's status: key -> (label, color). Interrupted ones were running when the console closed and can be resumed.
SESSION_STATUSES = {
    "active": ("Running", "#30a46c"),
    "interrupted": ("Interrupted", "#f5a524"),
    "ended": ("Ended", "#8b8d98"),
}


def _dot(color: str) -> QIcon:
    pixmap = QPixmap(16, 16)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(color))
    painter.drawEllipse(4, 4, 8, 8)
    painter.end()
    return QIcon(pixmap)


class SessionHistory:
    """
    Every search session started so far: created (its id, seconds since the epoch), name, area, status and when it
    ended. Kept in the console's saved settings (QSettings key "sessions").
    """
    KEY = "sessions"

    def __init__(self, store):
        self.store = store

    def all(self) -> list[dict]:
        try:
            data = json.loads(self.store.value(self.KEY, "") or "[]")
        except (ValueError, TypeError):
            return []
        sessions = [entry for entry in data if isinstance(entry, dict) and entry.get("created")]
        return sorted(sessions, key=lambda entry: entry["created"], reverse=True)

    def _write(self, sessions: list[dict]):
        self.store.setValue(self.KEY, json.dumps(sessions))
        self.store.sync()

    def get(self, created: float) -> dict | None:
        return next((entry for entry in self.all() if entry["created"] == created), None)

    def record(self, session, status: str | None = None):
        """
        Add or update a session (a config.SearchSession that has been started); status None keeps the current one.
        """
        if not session.created:
            return
        sessions = self.all()
        entry = next((e for e in sessions if e["created"] == session.created), None)
        if entry is None:
            entry = {"created": session.created, "status": status or "ended", "ended": 0}
            sessions.append(entry)
        entry["name"], entry["area"] = session.name, session.area
        if status is not None:
            entry["status"] = status
            entry["ended"] = 0 if status == "active" else entry.get("ended") or 0
        self._write(sessions)

    def set_status(self, created: float, status: str, when: float = 0.0):
        sessions = self.all()
        for entry in sessions:
            if entry["created"] == created:
                entry["status"] = status
                if status == "ended":
                    entry["ended"] = when
        self._write(sessions)

    def interrupt_running(self):
        """
        At launch nothing is running yet: sessions still marked running were cut off when the console closed.
        """
        sessions = self.all()
        if any(entry.get("status") == "active" for entry in sessions):
            for entry in sessions:
                if entry.get("status") == "active":
                    entry["status"] = "interrupted"
            self._write(sessions)

    def remove(self, created: float):
        self._write([entry for entry in self.all() if entry["created"] != created])


class DroneAssignments:
    """
    Which operator flies which drone: drone name -> username, kept in the console's saved settings
    (QSettings key "drone_operators") so it survives restarts. Administrators change it on the Manage tab.
    """
    KEY = "drone_operators"

    def __init__(self, store):
        self.store = store

    def all(self) -> dict[str, str]:
        try:
            data = json.loads(self.store.value(self.KEY, "") or "{}")
        except (ValueError, TypeError):
            return {}
        return data if isinstance(data, dict) else {}

    def get(self, drone: str) -> str:
        return self.all().get(drone, "")

    def set(self, drone: str, username: str):
        data = self.all()
        if username:
            data[drone] = username
        else:
            data.pop(drone, None)
        self.store.setValue(self.KEY, json.dumps(data))
        self.store.sync()


class ManagedDrone(QWidget):
    """
    One drone in the Drones section: the profile panel's drone row, its operator (a dropdown for administrators,
    plain text for everyone else), View on map, and the drone's current warning underneath.
    """
    operator_picked = Signal(str, str)   # drone name, username ("" for unassigned)

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.row = DroneRow(name)
        self.warning = QLabel()
        self.warning.setObjectName("warningPill")
        self.warning.hide()
        self.operator_combo = QComboBox()
        self.operator_combo.setMinimumWidth(170)
        self.operator_combo.setToolTip(f"The operator flying {name}. Pick someone else to reassign it.")
        self.operator_combo.activated.connect(
            lambda _i: self.operator_picked.emit(self.name, self.operator_combo.currentData() or ""))
        self.operator_label = QLabel()
        self.operator_label.setObjectName("hint")
        self._people: list[tuple[str, str]] | None = None
        self.map_button = QPushButton("View on map")
        self.map_button.setToolTip(f"Show {name}'s latest position on the map")
        self.map_button.setCursor(Qt.PointingHandCursor)

        # the operator and View on map sit level with the middle of the drone's stats
        line = QHBoxLayout()
        line.setSpacing(8)
        line.addWidget(self.row, 1)
        line.addWidget(self.operator_combo, 0, Qt.AlignVCenter)
        line.addWidget(self.operator_label, 0, Qt.AlignVCenter)
        line.addWidget(self.map_button, 0, Qt.AlignVCenter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(line)
        layout.addWidget(self.warning, 0, Qt.AlignLeft)

    def show_status(self, state: str, info: dict | None, warning: str):
        self.row.show_status(state, info)
        self.warning.setText(f"⚠ {warning}" if warning else "")
        self.warning.setVisible(bool(warning))

    def show_operator(self, people: list[tuple[str, str]], username: str, can_assign: bool):
        """
        people: (username, name) of everyone who can fly a drone; username: who's assigned now ("" for nobody).
        """
        if people != self._people:   # rebuilt only when the accounts change, so an open list isn't disturbed
            self._people = list(people)
            self.operator_combo.blockSignals(True)
            self.operator_combo.clear()
            self.operator_combo.addItem(f"Operator: {UNASSIGNED}", "")
            for user, name in people:
                self.operator_combo.addItem(f"Operator: {name}", user)
            self.operator_combo.blockSignals(False)
        index = self.operator_combo.findData(username) if username else 0
        index = max(index, 0)   # someone no longer allowed to fly shows as unassigned
        if self.operator_combo.currentIndex() != index:
            self.operator_combo.setCurrentIndex(index)
        self.operator_label.setText(self.operator_combo.itemText(index))
        self.operator_combo.setVisible(can_assign)
        self.operator_label.setVisible(not can_assign)


class ManageTab(QWidget):
    show_drone_requested = Signal(str)   # drone name
    operator_assigned = Signal(str, str)   # drone name, the operator's name (or "Unassigned")
    new_session_requested = Signal()
    remove_session_requested = Signal(float)   # the session's created time (its id)

    SESSION_COLUMNS = ["Name", "Area", "Created", "Status"]

    def __init__(self, session_group: QWidget, live_view, warnings, assignments: DroneAssignments,
                 sessions: SessionHistory = None, session_state=None, parent=None):
        """
        session_group: Settings' form for the current search session, shown here. live_view gives each drone's
        state and latest frame details; warnings(name) gives its current amber warning text; assignments who flies
        each drone; sessions every session so far; session_state() gives (the current session's created time,
        whether feeds are running).
        """
        super().__init__(parent)
        self.live = live_view
        self._warnings = warnings
        self.assignments = assignments
        self.sessions = sessions
        self._session_state = session_state or (lambda: (0.0, False))
        self._can_remove_sessions = False
        self._shown_sessions = None
        self._people: list[tuple[str, str]] = []
        self._can_assign = False
        self._rows: dict[str, ManagedDrone] = {}

        content = QWidget()
        content.setMaximumWidth(860)
        column = QVBoxLayout(content)
        column.setSpacing(14)
        column.addWidget(self._build_sessions_group(session_group))

        self.drones_group = QGroupBox("Drones")
        drones = QVBoxLayout(self.drones_group)
        drones.setSpacing(6)
        self.drones_box = QVBoxLayout()
        self.drones_box.setSpacing(6)
        self.no_drones = QLabel("No drone feeds are turned on. Add one in Settings > Drone feeds.")
        self.no_drones.setObjectName("hint")
        self.no_drones.setWordWrap(True)
        drones.addWidget(self.no_drones)
        drones.addLayout(self.drones_box)
        column.addWidget(self.drones_group)

        # filled with the accounts editor for administrators (set_accounts_editor)
        self.accounts_group = QGroupBox("Accounts")
        QVBoxLayout(self.accounts_group)
        self.accounts_group.hide()
        column.addWidget(self.accounts_group)
        column.addStretch(1)

        holder = QWidget()
        holder_layout = QHBoxLayout(holder)
        holder_layout.addWidget(content, 10)
        holder_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(holder)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

        # flight times and batteries move while the tab is open
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.refresh)

    # ------------------------------------------------------------------ search sessions

    def _build_sessions_group(self, session_group: QWidget) -> QGroupBox:
        group = QGroupBox("Search Sessions")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        current = QHBoxLayout()
        current_label = QLabel("Current session")
        font = current_label.font()
        font.setBold(True)
        current_label.setFont(font)
        current.addWidget(current_label)
        current.addStretch(1)
        self.new_session_button = QPushButton("New session")
        self.new_session_button.setToolTip("Clear the form to start a different search session")
        self.new_session_button.clicked.connect(self.new_session_requested)
        current.addWidget(self.new_session_button)
        layout.addLayout(current)
        layout.addWidget(session_group)

        all_label = QLabel("All sessions")
        all_label.setFont(font)
        layout.addSpacing(4)
        layout.addWidget(all_label)
        self.sessions_table = QTableWidget(0, len(self.SESSION_COLUMNS))
        self.sessions_table.setObjectName("accountsTable")   # same look as the accounts table
        self.sessions_table.setHorizontalHeaderLabels(self.SESSION_COLUMNS)
        self.sessions_table.verticalHeader().hide()
        self.sessions_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sessions_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.sessions_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.sessions_table.setShowGrid(False)
        self.sessions_table.setAlternatingRowColors(True)
        self.sessions_table.setMinimumHeight(130)
        header = self.sessions_table.horizontalHeader()
        header.setHighlightSections(False)
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        for i in range(len(self.SESSION_COLUMNS)):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.sessions_table.itemSelectionChanged.connect(self._session_selected)
        layout.addWidget(self.sessions_table)
        self.no_sessions = QLabel("No sessions yet. A session is added here when it's first started.")
        self.no_sessions.setObjectName("hint")
        layout.addWidget(self.no_sessions)

        bottom = QHBoxLayout()
        self.remove_session_button = QPushButton("Remove session")
        self.remove_session_button.setToolTip("Remove the selected session from this list. Its saved frames and "
                                              "detection log are kept.")
        self.remove_session_button.clicked.connect(self._remove_selected_session)
        bottom.addWidget(self.remove_session_button)
        bottom.addStretch(1)
        layout.addLayout(bottom)
        return group

    def set_can_remove_sessions(self, allowed: bool):
        self._can_remove_sessions = allowed
        self.remove_session_button.setVisible(allowed)
        self._session_selected()

    def selected_session(self) -> float | None:
        rows = self.sessions_table.selectionModel().selectedRows()
        item = self.sessions_table.item(rows[0].row(), 0) if rows else None
        return item.data(Qt.UserRole) if item else None

    def select_session(self, created: float):
        for row in range(self.sessions_table.rowCount()):
            if self.sessions_table.item(row, 0).data(Qt.UserRole) == created:
                self.sessions_table.selectRow(row)
                return

    def _session_selected(self):
        current, running = self._session_state()
        selected = self.selected_session()
        # the session that's running can't be removed
        self.remove_session_button.setEnabled(selected is not None and not (running and selected == current))
        self.new_session_button.setEnabled(not running)

    def _remove_selected_session(self):
        selected = self.selected_session()
        if selected is not None and self._can_remove_sessions:
            self.remove_session_requested.emit(selected)

    def refresh_sessions(self, force: bool = False):
        if self.sessions is None:
            return
        current, running = self._session_state()
        sessions = self.sessions.all()
        shown = (json.dumps(sessions), current, running)
        if shown == self._shown_sessions and not force:
            self._session_selected()
            return
        self._shown_sessions = shown
        selected = self.selected_session()
        table = self.sessions_table
        table.blockSignals(True)
        table.setRowCount(len(sessions))
        for row, entry in enumerate(sessions):
            label, color = SESSION_STATUSES.get(entry.get("status"), SESSION_STATUSES["ended"])
            marker = " (current)" if entry["created"] == current else ""
            cells = [f"{entry.get('name') or 'Unnamed'}{marker}", entry.get("area") or "—",
                     datetime.fromtimestamp(entry["created"]).strftime("%d %b %Y, %H:%M"), label]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, entry["created"])
                if column == 3:
                    item.setIcon(_dot(color))
                    if entry.get("status") == "ended" and entry.get("ended"):
                        item.setToolTip("Ended " + datetime.fromtimestamp(entry["ended"]).strftime("%d %b %Y, %H:%M"))
                    elif entry.get("status") == "interrupted":
                        item.setToolTip("Was running when the console closed")
                table.setItem(row, column, item)
            if entry["created"] == selected:
                table.selectRow(row)
        table.blockSignals(False)
        table.setVisible(bool(sessions))
        self.no_sessions.setVisible(not sessions)
        self._session_selected()

    # ------------------------------------------------------------------ accounts

    def set_accounts_editor(self, editor: QWidget | None):
        """
        Show the Accounts section with editor in it (administrators), or hide it (None).
        """
        layout = self.accounts_group.layout()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        if editor is not None:
            layout.addWidget(editor)
        self.accounts_group.setVisible(editor is not None)

    def set_people(self, people: list[tuple[str, str]], can_assign: bool):
        """
        people: (username, name) of the accounts that can fly a drone. can_assign: administrators reassign drones.
        """
        self._people = list(people)
        self._can_assign = can_assign
        self.refresh()

    def _assign(self, drone: str, username: str):
        if not self._can_assign or username == self.assignments.get(drone):
            return
        self.assignments.set(drone, username)
        names = dict(self._people)
        self.operator_assigned.emit(drone, names.get(username, UNASSIGNED))
        self.refresh()

    def refresh(self):
        names = self.live.drone_names()
        for name in list(self._rows):
            if name not in names:
                self._rows.pop(name).deleteLater()
        for i, name in enumerate(names):
            row = self._rows.get(name)
            if row is None:
                row = self._rows[name] = ManagedDrone(name)
                row.map_button.clicked.connect(lambda _=False, n=name: self.show_drone_requested.emit(n))
                row.operator_picked.connect(self._assign)
            self.drones_box.insertWidget(i, row)  # keeps them in the feeds' order
            state, info = self.live.status(name)
            row.show_status(state, info, self._warnings(name))
            row.show_operator(self._people, self.assignments.get(name), self._can_assign)
        self.no_drones.setVisible(not names)
        flying = sum(1 for name in names if self.live.status(name)[0] in ("starting", "running"))
        self.drones_group.setTitle(f"Drones ({flying} of {len(names)} flying)" if names else "Drones")
        self.refresh_sessions()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()
        self.refresh_sessions()
        self._timer.start()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._timer.stop()
