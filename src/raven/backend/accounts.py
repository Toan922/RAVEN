"""
Local operator accounts for the console: who can sign in, and what their role lets them do.

Passwords are stored only as Argon2id hashes. The accounts file sits in the data folder and is encrypted like
everything else there (raven.backend.vault). Five wrong passwords in a row lock that account for a minute.
"""

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from raven.backend import vault
from raven.backend.config import ROOT_DIR

USERS_PATH = ROOT_DIR / "runs" / "accounts.dat"

MIN_PASSWORD_LENGTH = 8
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 60

# what each permission allows
RUN_FEEDS = "run_feeds"            # Start / Stop the drone feeds
EDIT_SETTINGS = "edit_settings"    # change and save Settings
REVIEW_ALERTS = "review_alerts"    # mark alerts reviewed or dismissed
CLEAR_ALERTS = "clear_alerts"      # clear the alert list
MANAGE_USERS = "manage_users"      # add, remove and reset accounts
MANAGE_SEARCH = "manage_search"    # the Manage tab: the search session and the drones being flown
REVIEW_FOOTAGE = "review_footage"  # open an alert's footage in Feed > Review (everyone else gets Live only)
REMOVE_SESSIONS = "remove_sessions"  # remove search sessions from the Manage tab's list

# role key -> (label, what it's for, permissions)
ROLES = {
    "admin": ("Administrator", "Everything, plus managing accounts",
              {RUN_FEEDS, EDIT_SETTINGS, REVIEW_ALERTS, CLEAR_ALERTS, MANAGE_USERS, MANAGE_SEARCH, REVIEW_FOOTAGE,
               REMOVE_SESSIONS}),
    "operator": ("Drone Operator", "Runs the search and the feeds, changes settings and reviews alerts",
                 {RUN_FEEDS, EDIT_SETTINGS, REVIEW_ALERTS, CLEAR_ALERTS, MANAGE_SEARCH, REVIEW_FOOTAGE}),
    "reviewer": ("Reviewer", "Reviews alert footage and marks alerts reviewed or dismissed",
                 {REVIEW_ALERTS, REVIEW_FOOTAGE}),
    "viewer": ("Viewer", "Read only: watches the map, alerts and live feeds", set()),
}


class AuthError(Exception):
    pass


@dataclass
class User:
    username: str
    name: str
    role: str = "operator"
    email: str = ""
    unit: str = ""
    password_hash: str = ""
    duty: str = "on_duty"   # on_duty / away / off_duty, picked in the profile panel
    created: float = field(default_factory=time.time)
    failed_attempts: int = 0
    locked_until: float = 0.0

    @property
    def role_label(self) -> str:
        return ROLES.get(self.role, ROLES["viewer"])[0]

    def can(self, permission: str) -> bool:
        return permission in ROLES.get(self.role, ROLES["viewer"])[2]


def check_password(password: str) -> str | None:
    """
    Why a new password isn't allowed, or None if it's fine.
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Use at least {MIN_PASSWORD_LENGTH} characters."
    if password.strip() != password:
        return "The password can't start or end with a space."
    if password.isalpha() or password.isdigit():
        return "Mix letters with numbers or symbols."
    return None


def _hasher():
    from argon2 import PasswordHasher

    return PasswordHasher()


class UserStore:
    def __init__(self, path=USERS_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._users: dict[str, User] = {}
        self._load()

    # ------------------------------------------------------------------ file

    def _load(self):
        if not self.path.is_file():
            return
        data = json.loads(vault.read_bytes(self.path).decode("utf-8"))
        for entry in data.get("users", []):
            user = User(**{key: value for key, value in entry.items() if key in User.__dataclass_fields__})
            self._users[user.username.lower()] = user

    def _save(self):
        data = {"users": [asdict(user) for user in self._users.values()]}
        vault.write_bytes(self.path, json.dumps(data, indent=1).encode("utf-8"))

    # ------------------------------------------------------------------ accounts

    def is_empty(self) -> bool:
        return not self._users

    def users(self) -> list[User]:
        return sorted(self._users.values(), key=lambda user: user.username.lower())

    def get(self, username: str) -> User | None:
        return self._users.get(username.strip().lower())

    def create(self, username: str, name: str, role: str, password: str, email: str = "", unit: str = "") -> User:
        username = username.strip()
        if not username or not username.replace(".", "").replace("_", "").replace("-", "").isalnum():
            raise AuthError("Usernames use letters, numbers, dots, dashes and underscores.")
        if role not in ROLES:
            raise AuthError(f"Unknown role: {role}")
        problem = check_password(password)
        if problem:
            raise AuthError(problem)
        with self._lock:
            if username.lower() in self._users:
                raise AuthError(f"There's already an account called {username}.")
            user = User(username=username, name=name.strip() or username, role=role, email=email.strip(),
                        unit=unit.strip(), password_hash=_hasher().hash(password))
            self._users[username.lower()] = user
            self._save()
        return user

    def remove(self, username: str):
        with self._lock:
            user = self._users.get(username.lower())
            if user is None:
                return
            if user.role == "admin" and sum(1 for u in self._users.values() if u.role == "admin") == 1:
                raise AuthError("This is the only administrator. Make someone else an administrator first.")
            del self._users[username.lower()]
            self._save()

    def update(self, username: str, *, name=None, role=None, email=None, unit=None, duty=None):
        with self._lock:
            user = self._users.get(username.lower())
            if user is None:
                raise AuthError(f"No account called {username}.")
            if role is not None and role != user.role:
                if role not in ROLES:
                    raise AuthError(f"Unknown role: {role}")
                if user.role == "admin" and sum(1 for u in self._users.values() if u.role == "admin") == 1:
                    raise AuthError("This is the only administrator. Make someone else an administrator first.")
                user.role = role
            if name is not None:
                user.name = name.strip() or user.username
            if email is not None:
                user.email = email.strip()
            if unit is not None:
                user.unit = unit.strip()
            if duty is not None:
                user.duty = duty
            self._save()
            return user

    def set_password(self, username: str, password: str):
        problem = check_password(password)
        if problem:
            raise AuthError(problem)
        with self._lock:
            user = self._users.get(username.lower())
            if user is None:
                raise AuthError(f"No account called {username}.")
            user.password_hash = _hasher().hash(password)
            user.failed_attempts = 0
            user.locked_until = 0.0
            self._save()

    # ------------------------------------------------------------------ signing in

    def authenticate(self, username: str, password: str) -> User:
        """
        The account if the password is right. Raises AuthError with a message for the sign-in screen otherwise.
        The same message is used for an unknown name and a wrong password, so names can't be guessed.
        """
        from argon2.exceptions import InvalidHashError, VerifyMismatchError

        with self._lock:
            user = self._users.get(username.strip().lower())
            if user is None:
                # spend about as long as a real check, so timing doesn't reveal which names exist
                try:
                    _hasher().verify(_hasher().hash("not a real account"), password or " ")
                except VerifyMismatchError:
                    pass
                raise AuthError("Wrong username or password.")
            now = time.time()
            if user.locked_until > now:
                raise AuthError(f"Too many wrong passwords. Try again in {int(user.locked_until - now) + 1} s.")
            try:
                _hasher().verify(user.password_hash, password)
            except (VerifyMismatchError, InvalidHashError):
                user.failed_attempts += 1
                if user.failed_attempts >= MAX_FAILED_ATTEMPTS:
                    user.failed_attempts = 0
                    user.locked_until = now + LOCKOUT_SECONDS
                    self._save()
                    raise AuthError(f"Too many wrong passwords. The account is locked for {LOCKOUT_SECONDS} s.") from None
                self._save()
                raise AuthError("Wrong username or password.") from None
            if user.failed_attempts or user.locked_until:
                user.failed_attempts = 0
                user.locked_until = 0.0
                self._save()
            return user
