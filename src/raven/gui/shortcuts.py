"""
Keyboard shortcuts the operator can rebind in Settings.

Every shortcut in the console is created through bind(), so a new key from Settings applies everywhere at once
(both alerts panels share the same R / D / N keys, for example). Custom keys are saved in QSettings.
"""
import json

from PySide6.QtCore import QObject, QSettings, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut

# (id, group, description, default key); an empty default means no key until the operator sets one
ACTIONS = [
    ("page_map", "Pages", "Go to the Map tab", "Ctrl+1"),
    ("page_feed", "Pages", "Go to the Feed tab", "Ctrl+2"),
    ("page_manage", "Pages", "Go to the Manage tab", "Ctrl+3"),
    ("page_settings", "Pages", "Go to the Settings tab", "Ctrl+4"),
    ("show_log", "Pages", "Show / hide the log", "Ctrl+L"),
    ("toggle_theme", "Pages", "Switch light / dark mode", "Ctrl+Shift+T"),
    ("save_settings", "Run", "Save settings changes", "Ctrl+S"),
    ("start_ingest", "Run", "Start all feeds", "F5"),
    ("stop_ingest", "Run", "Stop all feeds", "Shift+F5"),
    ("alert_reviewed", "Alerts panel", "Mark the selected alert reviewed", "R"),
    ("alert_dismissed", "Alerts panel", "Dismiss the selected alert", "D"),
    ("alert_new", "Alerts panel", "Mark the selected alert new", "N"),
    ("play_pause", "Feed player", "Play / pause", "Space"),
    ("frame_back", "Feed player", "One frame back", "Left"),
    ("frame_forward", "Feed player", "One frame forward", "Right"),
    ("previous_alert", "Feed player", "Jump to the previous alert", "Ctrl+Left"),
    ("next_alert", "Feed player", "Jump to the next alert", "Ctrl+Right"),
]

DEFAULTS = {action_id: default for action_id, _, _, default in ACTIONS}
DESCRIPTIONS = {action_id: description for action_id, _, description, _ in ACTIONS}

class ShortcutRegistry(QObject):
    changed = Signal(str)  # action id whose key changed

    def __init__(self):
        super().__init__()
        self._keys = dict(DEFAULTS)
        self._shortcuts: dict[str, list[QShortcut]] = {action_id: [] for action_id in DEFAULTS}
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            saved = json.loads(QSettings().value("shortcuts", "{}"))
        except (ValueError, TypeError):
            saved = {}
        for action_id, key in saved.items():
            if action_id in self._keys and isinstance(key, str):
                self._keys[action_id] = key

    def key(self, action_id: str) -> str:
        self._load()
        return self._keys[action_id]

    def text(self, action_id: str) -> str:
        """
        The key as shown to the operator, e.g. "Ctrl+1", or "" when unbound.
        """
        return QKeySequence(self.key(action_id)).toString(QKeySequence.NativeText)

    def bind(self, action_id: str, parent, slot, context=Qt.WindowShortcut) -> QShortcut:
        """
        Create a shortcut for an action on parent, keeping it in step with the operator's key for that action.
        """
        shortcut = QShortcut(QKeySequence(self.key(action_id)), parent)
        shortcut.setContext(context)
        shortcut.activated.connect(slot)
        self._shortcuts[action_id].append(shortcut)
        shortcut.destroyed.connect(lambda *_: self._forget(action_id, shortcut))
        return shortcut

    def _forget(self, action_id, shortcut):
        try:
            self._shortcuts[action_id].remove(shortcut)
        except ValueError:
            pass

    def conflict(self, action_id: str, key: str) -> str | None:
        """
        The other action already using this key, or None. Keys are unique across the console, since the
        alerts panel and the Feed player sit on the same page and Qt ignores a key bound twice.
        """
        self._load()
        if not key:
            return None
        wanted = QKeySequence(key)
        for other, other_key in self._keys.items():
            if other == action_id or not other_key:
                continue
            if QKeySequence(other_key) == wanted:
                return other
        return None

    def set_key(self, action_id: str, key: str):
        self._load()
        key = QKeySequence(key).toString(QKeySequence.PortableText) if key else ""
        if self._keys[action_id] == key:
            return
        self._keys[action_id] = key
        for shortcut in self._shortcuts[action_id]:
            shortcut.setKey(QKeySequence(key))
        self._save()
        self.changed.emit(action_id)

    def reset_all(self):
        for action_id, key in DEFAULTS.items():
            self.set_key(action_id, key)

    def _save(self):
        custom = {action_id: key for action_id, key in self._keys.items() if key != DEFAULTS[action_id]}
        QSettings().setValue("shortcuts", json.dumps(custom))


_registry: ShortcutRegistry | None = None


def registry() -> ShortcutRegistry:
    """
    The one registry for the app (created on first use, after the QApplication exists).
    """
    global _registry
    if _registry is None:
        _registry = ShortcutRegistry()
    return _registry
