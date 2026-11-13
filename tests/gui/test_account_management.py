"""Operator console accounts: roles, the account table, drone assignments, signing in after log out."""
import pytest

pytest.importorskip("PySide6.QtWebEngineWidgets")   # skip (not fail) the console tests without PySide6



def test_picking_a_role_saves_it_and_the_next_sign_in_uses_it(window, accounts):
    from raven.backend.accounts import UserStore
    from raven.gui.main_window import PAGE_NAMES

    ada = accounts.create("ada", "Ada Admin", "admin", "admin-pass-1")
    accounts.create("val", "Val", "operator", "viewer-pass-2")
    window.user = ada
    window.apply_role()
    editor = window.manage_tab.accounts_group.layout().itemAt(0).widget()
    editor.select("val")
    combo = editor.role_combo
    combo.setCurrentIndex(combo.findData("viewer"))
    combo.activated.emit(combo.currentIndex())             # picking it is enough: no extra button
    assert UserStore(accounts.path).get("val").role == "viewer"     # saved to the file

    editor.select("ada")
    assert not combo.isEnabled()                           # can't demote yourself

    window._logged_out = True
    window._unlocked(accounts.authenticate("val", "viewer-pass-2"))
    assert window.user.role == "viewer"
    assert not window.header.is_page_visible(PAGE_NAMES.index("Manage"))
    assert window.feed_tab.review_tab_button.isHidden()


def test_account_table_columns(window, accounts):
    from raven.backend.accounts import UserStore

    ada = accounts.create("ada", "Ada Admin", "admin", "admin-pass-1")
    accounts.create("otto", "Otto Operator", "operator", "operator-pass-1")
    accounts.create("rae", "Rae Reviewer", "reviewer", "review-pass-1")
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
    assert UserStore(accounts.path).get("ada").duty == "away"            # each account keeps its own status


def test_admins_assign_drones_to_operators(window, accounts):
    accounts.create("ada", "Ada Admin", "admin", "admin-pass-1")
    accounts.create("otto", "Otto Operator", "operator", "operator-pass-1")
    accounts.create("val", "Val Viewer", "viewer", "viewer-pass-2")
    window.feed_tab.live.drone_names = lambda: ["Drone 1"]
    window.user = accounts.get("ada")
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

    window.user = accounts.get("otto")                      # operators see who's assigned but can't change it
    window.apply_role()
    window.profile_button.panel.user = window.user
    assert combo.isHidden() and not row.operator_label.isHidden()
    assert window.profile_button.panel._my_drones() == ["Drone 1"]
    row.operator_picked.emit("Drone 1", "")
    assert window.assignments.get("Drone 1") == "otto"


def test_signing_in_after_log_out_opens_the_map(window, accounts):
    from raven.gui.main_window import PAGE_NAMES
    from raven.gui.sign_in import LockScreen

    ada = accounts.create("ada", "Ada Admin", "admin", "admin-pass-1")
    window.user = ada
    window.apply_role()
    window.lock_screen = LockScreen(accounts)
    window.root.addWidget(window.lock_screen)
    window.lock_screen.unlocked.connect(window._unlocked)

    window.show_page(PAGE_NAMES.index("Settings"))
    window.lock()                                    # idle lock: back where they were
    window._unlocked(ada)
    assert window.pages.currentIndex() == PAGE_NAMES.index("Settings")

    window.log_out()                                 # log out and back in (same person): the Map
    window._unlocked(ada)
    assert window.pages.currentIndex() == PAGE_NAMES.index("Map")
