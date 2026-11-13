"""Fixtures for the backend tests."""
import pytest

from raven.backend import vault


@pytest.fixture(autouse=True)
def no_encryption():
    # encryption is a module-wide switch: every test starts and ends with it off
    vault.disable()
    yield
    vault.disable()



