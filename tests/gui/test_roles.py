"""Operator console roles: what each role can see and do (NFR-01)."""
import pytest

pytest.importorskip("PySide6.QtWebEngineWidgets")   # skip (not fail) the console tests without PySide6

from raven.backend.accounts import User  # noqa: E402



def test_viewer_cannot_start_or_review(window, dialogs):
    window.user = User(username="val", name="Val", role="viewer")
    window.apply_role()
    assert not window.video_bar.start_button.isEnabled()
    assert not window.settings_tab._role_sections[0].isEnabled()
    dialogs.clear()
    window.start_ingest()
    window._set_status("missing", "reviewed")
    assert len(dialogs) == 2 and all("Viewer" in text for text in dialogs)


@pytest.mark.usefixtures("accounts")
def test_manage_tab_follows_the_role(window, sign_in_as):
    from raven.gui.main_window import PAGE_NAMES

    manage = PAGE_NAMES.index("Manage")
    sign_in_as("admin")
    assert window.header.is_page_visible(manage)
    assert not window.manage_tab.accounts_group.isHidden()
    window.show_page(manage)
    assert window.pages.currentIndex() == manage

    sign_in_as("operator")
    assert window.header.is_page_visible(manage) and window.manage_tab.accounts_group.isHidden()

    for role in ("reviewer", "viewer"):
        sign_in_as(role)
        assert not window.header.is_page_visible(manage)
        assert window.pages.currentIndex() != manage       # moved off the tab it no longer has
        window.show_page(manage)
        assert window.pages.currentIndex() != manage


def test_viewers_get_live_only(window, sign_in_as):
    feed = window.feed_tab
    sign_in_as("reviewer")
    assert feed.review_allowed() and not feed.review_tab_button.isHidden()

    sign_in_as("viewer")
    assert feed.review_tab_button.isHidden()
    feed.set_subtab(feed.REVIEW)
    assert feed.subtab() == feed.LIVE


def test_only_drone_flyers_have_assigned_drones_in_their_profile(window):
    panel = window.profile_button.panel
    for role, shown in (("admin", True), ("operator", True), ("reviewer", False), ("viewer", False)):
        panel.set_user(User(username=role, name=role.title(), role=role))
        assert panel.drones_section.isHidden() != shown, role
