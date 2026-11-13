"""backend/accounts.py: operator accounts, roles and sign-in lockout (NFR-01)."""
import pytest

from raven.backend.accounts import AuthError, UserStore, check_password


def test_accounts_roles_and_lockout(tmp_path):
    store = UserStore(tmp_path / "accounts.dat")
    assert store.is_empty()
    admin = store.create("admin", "Ada", "admin", "correct-horse-1")
    viewer = store.create("val", "Val", "viewer", "viewer-pass-2")
    assert admin.can("manage_users") and admin.can("run_feeds")
    assert not viewer.can("run_feeds") and not viewer.can("review_alerts")
    assert "correct-horse-1" not in (tmp_path / "accounts.dat").read_text()
    assert UserStore(tmp_path / "accounts.dat").authenticate("ADMIN", "correct-horse-1").name == "Ada"

    with pytest.raises(AuthError):
        store.create("val", "Again", "viewer", "viewer-pass-2")
    with pytest.raises(AuthError, match="only administrator"):
        store.remove("admin")
    for _ in range(4):
        with pytest.raises(AuthError, match="Wrong"):
            store.authenticate("val", "nope")
    with pytest.raises(AuthError, match="locked"):
        store.authenticate("val", "nope")
    with pytest.raises(AuthError, match="Try again"):
        store.authenticate("val", "viewer-pass-2")


def test_password_rules():
    assert check_password("short1") is not None
    assert check_password("lettersonly") is not None
    assert check_password("letters-and-1") is None
