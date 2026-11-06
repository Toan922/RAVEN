"""
Signing in to the console: the first-launch screen that creates the administrator, the sign-in screen, the lock
screen shown after the console sits idle (or after Log out), and the idle timer that triggers it.
"""
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

from raven.backend.accounts import MIN_PASSWORD_LENGTH, ROLES, RUN_FEEDS, AuthError, User, UserStore

# lock the console after this many minutes without mouse or keyboard input
IDLE_LOCK_MINUTES = 10


def _error_label() -> QLabel:
    label = QLabel()
    label.setObjectName("signInError")
    label.setWordWrap(True)
    label.hide()
    return label


class SignInForm(QFrame):
    """
    The card used by every screen here. mode is "setup" (no accounts yet: create the administrator),
    "sign_in", or "unlock" (the signed-in user's password, with a way to switch to someone else).
    """
    signed_in = Signal(object)   # User

    def __init__(self, store: UserStore, mode: str, user: User | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("signInCard")
        self.setFixedWidth(380)
        self.store = store
        self._mode = mode
        self._user = user

        self.title = QLabel()
        self.title.setObjectName("signInTitle")
        self.subtitle = QLabel()
        self.subtitle.setObjectName("hint")
        self.subtitle.setWordWrap(True)

        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("username")
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Albert Ross")
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.confirm_edit = QLineEdit()
        self.confirm_edit.setEchoMode(QLineEdit.Password)
        self.error = _error_label()

        self.form = QFormLayout()
        self.form.addRow("Username", self.username_edit)
        self.form.addRow("Full name", self.name_edit)
        self.form.addRow("Password", self.password_edit)
        self.form.addRow("Confirm password", self.confirm_edit)

        self.submit_button = QPushButton()
        self.submit_button.setObjectName("signInButton")
        self.submit_button.setDefault(True)
        self.submit_button.clicked.connect(self._submit)
        self.switch_button = QPushButton("Sign in as someone else")
        self.switch_button.setFlat(True)
        self.switch_button.setCursor(Qt.PointingHandCursor)
        self.switch_button.clicked.connect(self._switch_user)
        for edit in (self.username_edit, self.name_edit, self.password_edit, self.confirm_edit):
            edit.returnPressed.connect(self._submit)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(10)
        brand = QLabel("RAVEN")
        brand.setObjectName("brand")
        layout.addWidget(brand, 0, Qt.AlignHCenter)
        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)
        layout.addLayout(self.form)
        layout.addWidget(self.error)
        layout.addWidget(self.submit_button)
        layout.addWidget(self.switch_button, 0, Qt.AlignHCenter)
        self.set_mode(mode, user)

    def set_mode(self, mode: str, user: User | None = None):
        self._mode = mode
        if user is not None:
            self._user = user
        setup = mode == "setup"
        unlock = mode == "unlock" and self._user is not None
        for widget in (self.name_edit, self.confirm_edit):
            self.form.setRowVisible(widget, setup)
        self.username_edit.setReadOnly(unlock)
        self.switch_button.setVisible(unlock)
        self.password_edit.clear()
        self.confirm_edit.clear()
        self.error.hide()
        if setup:
            self.title.setText("Create the administrator account")
            self.subtitle.setText(f"No accounts exist yet. This first account can add the rest of the team. "
                                  f"Passwords need at least {MIN_PASSWORD_LENGTH} characters, mixing letters "
                                  "with numbers or symbols.")
            self.submit_button.setText("Create account and sign in")
            self.username_edit.setText("admin")
        elif unlock:
            self.title.setText("Console locked")
            self.subtitle.setText(f"{self._user.name} ({self._user.role_label}). Feeds keep running while the "
                                  "console is locked. Enter your password to continue.")
            self.submit_button.setText("Unlock")
            self.username_edit.setText(self._user.username)
        else:
            self.title.setText("Sign in")
            self.subtitle.setText("Sign in with your RAVEN operator account.")
            self.submit_button.setText("Sign in")
        (self.password_edit if unlock else self.username_edit).setFocus()

    def _switch_user(self):
        self.username_edit.clear()
        self.set_mode("sign_in")

    def focus(self):
        (self.password_edit if self.username_edit.isReadOnly() else self.username_edit).setFocus()

    def _fail(self, message: str):
        self.error.setText(message)
        self.error.show()
        self.password_edit.clear()
        self.password_edit.setFocus()

    def _submit(self):
        username = self.username_edit.text().strip()
        password = self.password_edit.text()
        if not username or not password:
            self._fail("Enter a username and password.")
            return
        try:
            if self._mode == "setup":
                if password != self.confirm_edit.text():
                    self._fail("The passwords don't match.")
                    return
                user = self.store.create(username, self.name_edit.text(), "admin", password)
            else:
                user = self.store.authenticate(username, password)
        except AuthError as error:
            self._fail(str(error))
            return
        self.password_edit.clear()
        self.confirm_edit.clear()
        self.error.hide()
        self.signed_in.emit(user)


class SignInDialog(QDialog):
    """
    The window shown before the console opens: creates the administrator on first launch, else signs in.
    """

    def __init__(self, store: UserStore, demo: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RAVEN — Sign in" + (" (DEMO)" if demo else ""))
        self.user: User | None = None
        self.form = SignInForm(store, "setup" if store.is_empty() else "sign_in")
        self.form.signed_in.connect(self._done)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.addWidget(self.form, 0, Qt.AlignCenter)

    def _done(self, user: User):
        self.user = user
        self.accept()


class LockScreen(QWidget):
    """
    Covers the console while it's locked or nobody is signed in. The window's content is hidden behind it.
    """
    unlocked = Signal(object)   # User

    def __init__(self, store: UserStore, parent=None):
        super().__init__(parent)
        self.setObjectName("lockScreen")
        self.form = SignInForm(store, "sign_in")
        self.form.signed_in.connect(self.unlocked)
        layout = QVBoxLayout(self)
        layout.addStretch(1)
        layout.addWidget(self.form, 0, Qt.AlignHCenter)
        layout.addStretch(2)

    def show_for(self, user: User | None):
        """
        Lock for user (their password unlocks it), or ask anyone to sign in when user is None (after Log out).
        """
        self.form.username_edit.setReadOnly(False)
        if user is None:
            self.form.username_edit.clear()
            self.form.set_mode("sign_in")
        else:
            self.form.set_mode("unlock", user)
        QTimer.singleShot(0, self.form.focus)


class IdleWatcher(QObject):
    """
    Emits idle once the application has had no mouse or keyboard input for minutes.
    """
    idle = Signal()

    INPUT_EVENTS = {QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.KeyPress, QEvent.Wheel, QEvent.TouchBegin}

    def __init__(self, minutes: float = IDLE_LOCK_MINUTES, parent=None):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(int(minutes * 60_000))
        self._timer.timeout.connect(self.idle)
        QApplication.instance().installEventFilter(self)

    def set_minutes(self, minutes: float):
        self._timer.setInterval(int(minutes * 60_000))
        self.reset()

    def reset(self):
        self._timer.start()

    def pause(self):
        self._timer.stop()

    def eventFilter(self, watched, event):
        if event.type() in self.INPUT_EVENTS and self._timer.isActive():
            self._timer.start()
        return False


class UsersEditor(QWidget):
    """
    Manage > Accounts, for administrators: a table of every account (full name, username, role, assigned drones,
    duty status), a role dropdown that changes the selected account's role as soon as one is picked, and buttons
    to add, reset or remove one.
    """
    log = Signal(str)
    changed = Signal()

    COLUMNS = ["Full name", "Username", "Role", "Assigned drone", "Status"]

    def __init__(self, store: UserStore, current: User, assignments=None, parent=None):
        """
        assignments (manage_tab.DroneAssignments) fills the Assigned drone column.
        """
        super().__init__(parent)
        from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget

        self.store = store
        self.current = current
        self.assignments = assignments
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setObjectName("accountsTable")
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(150)
        header = self.table.horizontalHeader()
        header.setHighlightSections(False)
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        for column in range(len(self.COLUMNS)):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        self.add_button = QPushButton("Add account…")
        self.add_button.clicked.connect(self._add)
        self.role_combo = QComboBox()
        for key, (label, what, _perms) in ROLES.items():
            self.role_combo.addItem(label, key)
            self.role_combo.setItemData(self.role_combo.count() - 1, what, Qt.ToolTipRole)
        self.role_combo.setToolTip("The selected account's role. Picking another one changes it right away.")
        self.role_combo.activated.connect(lambda _i: self._change_role())
        self.reset_button = QPushButton("Reset password…")
        self.reset_button.clicked.connect(self._reset)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self._remove)
        self.table.itemSelectionChanged.connect(self._selected)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        for widget in (self.add_button, QLabel("Role"), self.role_combo, self.reset_button, self.remove_button):
            buttons.addWidget(widget)
        buttons.addStretch(1)
        roles = QLabel(" · ".join(f"<b>{label}</b>: {what}" for label, what, _ in ROLES.values()))
        roles.setObjectName("hint")
        roles.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        layout.addLayout(buttons)
        layout.addWidget(roles)
        self.refresh()

    def _drones_of(self, user: User) -> str:
        if self.assignments is None or not user.can(RUN_FEEDS):
            return "—"   # reviewers and viewers watch every feed rather than flying one
        drones = sorted(drone for drone, username in self.assignments.all().items()
                        if username.lower() == user.username.lower())
        return ", ".join(drones) or "None"

    def refresh(self):
        from PySide6.QtWidgets import QTableWidgetItem

        from raven.gui.profile import DEFAULT_DUTY, DUTY_STATUSES, _dot

        selected = self._username()
        users = self.store.users()
        self.table.blockSignals(True)
        self.table.setRowCount(len(users))
        for row, user in enumerate(users):
            you = " (you)" if user.username.lower() == self.current.username.lower() else ""
            duty_label, duty_color = DUTY_STATUSES.get(user.duty, DUTY_STATUSES[DEFAULT_DUTY])
            cells = [f"{user.name}{you}", user.username, user.role_label, self._drones_of(user), duty_label]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, user.username)
                if column == 4:
                    item.setIcon(_dot(duty_color, 8))
                self.table.setItem(row, column, item)
            if user.username == selected:
                self.table.selectRow(row)
        if selected is None or self.store.get(selected) is None:
            self.table.clearSelection()
        self.table.blockSignals(False)
        self._selected()

    def select(self, username: str):
        for row in range(self.table.rowCount()):
            if self.table.item(row, 1).text().lower() == username.lower():
                self.table.selectRow(row)
                return

    def _username(self) -> str | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        item = self.table.item(rows[0].row(), 0) if rows else None
        return item.data(Qt.UserRole) if item else None

    def _selected(self):
        username = self._username()
        user = self.store.get(username) if username else None
        for widget in (self.role_combo, self.reset_button, self.remove_button):
            widget.setEnabled(user is not None)
        if user is not None:
            self.role_combo.setCurrentIndex(max(0, self.role_combo.findData(user.role)))
            # you can't demote or remove yourself while signed in; another administrator can
            yourself = user.username.lower() == self.current.username.lower()
            self.role_combo.setEnabled(not yourself)
            self.remove_button.setEnabled(not yourself)

    def _ask_password(self, title: str) -> str | None:
        from PySide6.QtWidgets import QInputDialog

        password, ok = QInputDialog.getText(self, title, f"New password (at least {MIN_PASSWORD_LENGTH} characters, "
                                            "letters with numbers or symbols):", QLineEdit.Password)
        return password if ok else None

    def _add(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Add account")
        username, name, password = QLineEdit(), QLineEdit(), QLineEdit()
        password.setEchoMode(QLineEdit.Password)
        email, unit = QLineEdit(), QLineEdit()
        role = QComboBox()
        for key, (label, _what, _perms) in ROLES.items():
            role.addItem(label, key)
        role.setCurrentIndex(role.findData("operator"))
        error = _error_label()
        form = QFormLayout(dialog)
        form.addRow("Username", username)
        form.addRow("Full name", name)
        form.addRow("Role", role)
        form.addRow("Email (optional)", email)
        form.addRow("Unit (optional)", unit)
        form.addRow("Temporary password", password)
        form.addRow(error)
        add = QPushButton("Add account")
        add.setDefault(True)
        form.addRow(add)

        def submit():
            try:
                self.store.create(username.text(), name.text(), role.currentData(), password.text(),
                                  email=email.text(), unit=unit.text())
            except AuthError as problem:
                error.setText(str(problem))
                error.show()
                return
            self.log.emit(f"Added account {username.text().strip()} ({role.currentText()})")
            dialog.accept()

        add.clicked.connect(submit)
        if dialog.exec() == QDialog.Accepted:
            self.refresh()
            self.changed.emit()

    def _change_role(self):
        username = self._username()
        user = self.store.get(username) if username else None
        if user is None or user.role == self.role_combo.currentData():
            return
        try:
            user = self.store.update(username, role=self.role_combo.currentData())
        except AuthError as error:
            QMessageBox.warning(self, "Change role", str(error))
            self.refresh()
            return
        self.log.emit(f"{user.username} is now {user.role_label}")
        self.refresh()
        self.changed.emit()

    def _reset(self):
        username = self._username()
        if not username:
            return
        password = self._ask_password(f"Reset password for {username}")
        if password is None:
            return
        try:
            self.store.set_password(username, password)
        except AuthError as error:
            QMessageBox.warning(self, "Reset password", str(error))
            return
        self.log.emit(f"Reset the password for {username}")

    def _remove(self):
        username = self._username()
        if not username:
            return
        if QMessageBox.question(self, "Remove account", f"Remove the account {username}?") != QMessageBox.Yes:
            return
        try:
            self.store.remove(username)
        except AuthError as error:
            QMessageBox.warning(self, "Remove account", str(error))
            return
        self.log.emit(f"Removed account {username}")
        self.refresh()
        self.changed.emit()
