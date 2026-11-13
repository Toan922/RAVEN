"""Fixtures for the operator console tests: one headless window per test, with dialogs answered automatically."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # no screen needed, on CI or locally

import pytest  # noqa: E402

pytest.importorskip("PySide6.QtWebEngineWidgets")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from raven.backend.accounts import User, UserStore  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication(["test"])
    application.setOrganizationName("RAVEN")
    application.setApplicationName("RAVEN Console Tests")
    return application


@pytest.fixture
def dialogs(monkeypatch):
    # every dialog is answered No / OK, and its text kept for the test to check
    shown = []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: shown.append(a[2]) or QMessageBox.No))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: shown.append(a[2]) or QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: shown.append(a[2]) or QMessageBox.Ok))
    return shown


@pytest.fixture
def window(app, dialogs, tmp_path):
    QSettings().clear()
    from raven.gui.main_window import MainWindow

    win = MainWindow()
    config = win.settings_tab.config()
    config.output_dir = str(tmp_path)
    win.settings_tab.set_config(config)
    yield win
    win._closing = True
    win.bus.close()
    QSettings().clear()


@pytest.fixture
def accounts(window, tmp_path):
    """An empty account store in tmp_path, used by the window."""
    store = UserStore(tmp_path / "accounts.dat")
    window.accounts = store
    return store


@pytest.fixture
def sign_in_as(window):
    """sign_in_as(role) switches the window to a user with that role."""
    def sign_in(role):
        window.user = User(username=role, name=role.title(), role=role)
        window.apply_role()
    return sign_in
