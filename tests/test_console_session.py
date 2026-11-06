"""Operator console (headless): search session start rules, the Discord test gate, resume, warnings and roles."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6.QtWebEngineWidgets")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from raven.backend.accounts import User  # noqa: E402

WEBHOOK = "https://discord.com/api/webhooks/12345678901234567/" + "a" * 68


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


def test_start_needs_a_session_name_and_a_tested_webhook(window):
    tab = window.settings_tab
    assert tab.config().discord_enabled                      # required outside demo mode
    assert any("Name the search session" in p for p in window.start_blockers(tab.config()))
    assert window.header.state_label.text() == "Not ready"

    tab.session_name_edit.setText("Cedar Ridge")
    tab.webhook_edit.setText(WEBHOOK)
    assert any("test alert" in p for p in window.start_blockers(tab.config()))
    tab._set_verified(WEBHOOK)
    assert not any("test alert" in p for p in window.start_blockers(tab.config()))
    tab.webhook_edit.setText(WEBHOOK + "b")                  # a changed webhook needs a new test
    assert any("test alert" in p for p in window.start_blockers(tab.config()))


def test_resume_prompt_and_declining_it(window, dialogs):
    window.store.setValue("active_session", "Cedar Ridge")
    window._offer_resume()
    assert "Resume search session Cedar Ridge?" in dialogs[-1]
    assert window.store.value("active_session", "") == ""


def test_warnings_show_and_clear(window):
    window._on_pipeline_status("Drone 1", "feed_interrupted", "lost it")
    assert "Video interrupted" in window.warning_notice.text() and not window.warning_notice.isHidden()
    window._on_progress("Drone 1", {"consecutive_failures": 0, "inference_p90_ms": 800, "lag_s": 0})
    assert "Falling behind" in window.warning_text("Drone 1")
    window._on_pipeline_status("Drone 1", "feed_resumed", "")
    window._on_progress("Drone 1", {"consecutive_failures": 0, "inference_p90_ms": 50, "lag_s": 0})
    assert window.warning_text("Drone 1") == "" and window.warning_notice.isHidden()


def test_viewer_cannot_start_or_review(window, dialogs):
    window.user = User(username="val", name="Val", role="viewer")
    window.apply_role()
    assert not window.video_bar.start_button.isEnabled()
    assert not window.settings_tab._role_sections[0].isEnabled()
    dialogs.clear()
    window.start_ingest()
    window._set_status("missing", "reviewed")
    assert len(dialogs) == 2 and all("Viewer" in text for text in dialogs)


def _sign_in_as(window, role):
    window.user = User(username=role, name=role.title(), role=role)
    window.apply_role()


def test_manage_tab_follows_the_role(window, tmp_path):
    from raven.backend.accounts import UserStore
    from raven.gui.main_window import PAGE_NAMES

    window.accounts = UserStore(tmp_path / "accounts.dat")
    manage = PAGE_NAMES.index("Manage")
    _sign_in_as(window, "admin")
    assert window.header.is_page_visible(manage)
    assert not window.manage_tab.accounts_group.isHidden()
    window.show_page(manage)
    assert window.pages.currentIndex() == manage

    _sign_in_as(window, "operator")
    assert window.header.is_page_visible(manage) and window.manage_tab.accounts_group.isHidden()

    for role in ("reviewer", "viewer"):
        _sign_in_as(window, role)
        assert not window.header.is_page_visible(manage)
        assert window.pages.currentIndex() != manage       # moved off the tab it no longer has
        window.show_page(manage)
        assert window.pages.currentIndex() != manage


def test_viewers_get_live_only(window):
    feed = window.feed_tab
    _sign_in_as(window, "reviewer")
    assert feed.review_allowed() and not feed.review_tab_button.isHidden()

    _sign_in_as(window, "viewer")
    assert feed.review_tab_button.isHidden()
    feed.set_subtab(feed.REVIEW)
    assert feed.subtab() == feed.LIVE


def test_session_edits_save_without_save_changes(window):
    tab = window.settings_tab
    before = window.has_unsaved_settings()
    tab.session_name_edit.setText("Cedar Ridge")
    assert window.has_unsaved_settings() == before      # the session doesn't light up Save changes
    tab.session_edited.emit()
    assert '"Cedar Ridge"' in window.store.value("session", "")


def test_admins_assign_drones_to_operators(window, tmp_path):
    from raven.backend.accounts import UserStore

    store = UserStore(tmp_path / "accounts.dat")
    store.create("ada", "Ada Admin", "admin", "admin-pass-1")
    store.create("otto", "Otto Operator", "operator", "operator-pass-1")
    store.create("val", "Val Viewer", "viewer", "viewer-pass-2")
    window.accounts = store
    window.feed_tab.live.drone_names = lambda: ["Drone 1"]
    window.user = store.get("ada")
    window.apply_role()
    row = window.manage_tab._rows["Drone 1"]
    combo = row.operator_combo
    options = [combo.itemText(i) for i in range(combo.count())]
    assert options == ["Operator: Unassigned", "Operator: Ada Admin", "Operator: Otto Operator"]  # no viewers
    assert not combo.isHidden()

    combo.setCurrentIndex(combo.findData("otto"))
    row.operator_picked.emit("Drone 1", "otto")
    assert window.assignments.get("Drone 1") == "otto"
    assert row.operator_label.text() == "Operator: Otto Operator"

    window.user = store.get("otto")                      # operators see who's assigned but can't change it
    window.apply_role()
    window.profile_button.panel.user = window.user
    assert combo.isHidden() and not row.operator_label.isHidden()
    assert window.profile_button.panel._my_drones() == ["Drone 1"]
    row.operator_picked.emit("Drone 1", "")
    assert window.assignments.get("Drone 1") == "otto"


def test_signing_in_after_log_out_opens_the_map(window, tmp_path):
    from raven.backend.accounts import UserStore
    from raven.gui.main_window import PAGE_NAMES
    from raven.gui.sign_in import LockScreen

    store = UserStore(tmp_path / "accounts.dat")
    ada = store.create("ada", "Ada Admin", "admin", "admin-pass-1")
    window.accounts = store
    window.user = ada
    window.apply_role()
    window.lock_screen = LockScreen(store)
    window.root.addWidget(window.lock_screen)
    window.lock_screen.unlocked.connect(window._unlocked)

    window.show_page(PAGE_NAMES.index("Settings"))
    window.lock()                                    # idle lock: back where they were
    window._unlocked(ada)
    assert window.pages.currentIndex() == PAGE_NAMES.index("Settings")

    window.log_out()                                 # log out and back in (same person): the Map
    window._unlocked(ada)
    assert window.pages.currentIndex() == PAGE_NAMES.index("Map")


def test_only_drone_flyers_have_assigned_drones_in_their_profile(window):
    panel = window.profile_button.panel
    for role, shown in (("admin", True), ("operator", True), ("reviewer", False), ("viewer", False)):
        panel.set_user(User(username=role, name=role.title(), role=role))
        assert panel.drones_section.isHidden() != shown, role


def test_picking_a_role_saves_it_and_the_next_sign_in_uses_it(window, tmp_path):
    from raven.backend.accounts import UserStore
    from raven.gui.main_window import PAGE_NAMES

    path = tmp_path / "accounts.dat"
    store = UserStore(path)
    ada = store.create("ada", "Ada Admin", "admin", "admin-pass-1")
    store.create("val", "Val", "operator", "viewer-pass-2")
    window.accounts = store
    window.user = ada
    window.apply_role()
    editor = window.manage_tab.accounts_group.layout().itemAt(0).widget()
    editor.select("val")
    combo = editor.role_combo
    combo.setCurrentIndex(combo.findData("viewer"))
    combo.activated.emit(combo.currentIndex())             # picking it is enough: no extra button
    assert UserStore(path).get("val").role == "viewer"     # saved to the file

    editor.select("ada")
    assert not combo.isEnabled()                           # can't demote yourself

    window._logged_out = True
    window._unlocked(store.authenticate("val", "viewer-pass-2"))
    assert window.user.role == "viewer"
    assert not window.header.is_page_visible(PAGE_NAMES.index("Manage"))
    assert window.feed_tab.review_tab_button.isHidden()


def test_account_table_columns(window, tmp_path):
    from raven.backend.accounts import UserStore

    path = tmp_path / "accounts.dat"
    store = UserStore(path)
    ada = store.create("ada", "Ada Admin", "admin", "admin-pass-1")
    store.create("otto", "Otto Operator", "operator", "operator-pass-1")
    store.create("rae", "Rae Reviewer", "reviewer", "review-pass-1")
    window.accounts = store
    window.user = ada
    window.profile_button.panel.set_user(ada)
    window.apply_role()
    window.assignments.set("Drone 2", "otto")
    window.assignments.set("Drone 1", "otto")
    window.manage_tab.operator_assigned.emit("Drone 1", "Otto Operator")
    window.profile_button.panel.duty_combo.setCurrentIndex(
        window.profile_button.panel.duty_combo.findData("away"))   # Ada sets herself Away

    editor = window._users_editor
    table = editor.table
    headers = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    assert headers == ["Full name", "Username", "Role", "Assigned drone", "Status"]
    rows = {table.item(r, 1).text(): [table.item(r, c).text() for c in range(5)] for r in range(table.rowCount())}
    assert rows["ada"] == ["Ada Admin (you)", "ada", "Administrator", "None", "Away"]
    assert rows["otto"] == ["Otto Operator", "otto", "Drone Operator", "Drone 1, Drone 2", "On duty"]
    assert rows["rae"][3] == "—"
    assert UserStore(path).get("ada").duty == "away"            # each account keeps its own status


def test_search_sessions_list_new_and_remove(window, dialogs, monkeypatch):
    tab, manage = window.settings_tab, window.manage_tab
    tab.session_name_edit.setText("Cedar Ridge")
    tab.session_area_edit.setText("North trailhead")
    config = tab.config()
    window._begin_session(config)                       # what Start does once the checks pass
    first = tab.session_created()
    assert window.sessions.get(first)["status"] == "active"
    window._end_session("stopped by operator")
    assert window.sessions.get(first)["status"] == "ended"

    window._new_session()                               # a second session
    assert tab.session_created() == 0 and tab.session_name_edit.text() == ""
    tab.session_name_edit.setText("Pine Hollow")
    window._begin_session(tab.config())
    second = tab.session_created()
    manage.refresh_sessions(force=True)
    table = manage.sessions_table
    assert [table.item(r, 0).text() for r in range(table.rowCount())] == ["Pine Hollow (current)", "Cedar Ridge"]
    assert [table.item(r, 3).text() for r in range(table.rowCount())] == ["Running", "Ended"]
    assert table.horizontalHeaderItem(0).text() == "Name"

    window.sessions.interrupt_running()                 # what the next launch sees after a crash
    assert window.sessions.get(second)["status"] == "interrupted"

    window.user = User(username="otto", name="Otto", role="operator")
    window.apply_role()
    assert manage.remove_session_button.isHidden()      # only administrators remove sessions
    window._remove_session(first)
    assert window.sessions.get(first) is not None

    window.user = User(username="ada", name="Ada", role="admin")
    window.apply_role()
    assert not manage.remove_session_button.isHidden()
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    window._remove_session(first)
    assert window.sessions.get(first) is None and window.sessions.get(second) is not None
