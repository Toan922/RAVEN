"""Operator console search sessions: start rules and the Discord test gate, resume, saving, the session list (FR-01 to FR-03)."""
import pytest

pytest.importorskip("PySide6.QtWebEngineWidgets")   # skip (not fail) the console tests without PySide6

from PySide6.QtWidgets import QMessageBox  # noqa: E402

from raven.backend.accounts import User  # noqa: E402


WEBHOOK = "https://discord.com/api/webhooks/12345678901234567/" + "a" * 68


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


def test_session_edits_save_without_save_changes(window):
    tab = window.settings_tab
    before = window.has_unsaved_settings()
    tab.session_name_edit.setText("Cedar Ridge")
    assert window.has_unsaved_settings() == before      # the session doesn't light up Save changes
    tab.session_edited.emit()
    assert '"Cedar Ridge"' in window.store.value("session", "")


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
