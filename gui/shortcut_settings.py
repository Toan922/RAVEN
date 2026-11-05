"""
Settings section for rebinding the console's keyboard shortcuts.
"""
from PySide6.QtCore import Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QGridLayout, QGroupBox, QHBoxLayout, QKeySequenceEdit, QLabel, QPushButton, QToolButton, QVBoxLayout, QWidget,
)

from gui.shortcuts import ACTIONS, DEFAULTS, DESCRIPTIONS, registry


class ShortcutSettings(QWidget):
    """
    One row per action: what it does, its key (click and press a new one), clear, and reset to default.
    Changes apply and are saved right away; a key another action already uses is refused.
    """
    log = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._edits: dict[str, QKeySequenceEdit] = {}
        self._resets: dict[str, QToolButton] = {}

        group = QGroupBox()
        grid = QGridLayout(group)
        grid.setColumnStretch(1, 1)
        grid.setVerticalSpacing(4)

        row = 0
        last_group = None
        for action_id, group_name, description, _default in ACTIONS:
            if group_name != last_group:
                heading = QLabel(group_name)
                font = heading.font()
                font.setBold(True)
                heading.setFont(font)
                if row:
                    grid.setRowMinimumHeight(row, 8)
                    row += 1
                grid.addWidget(heading, row, 0, 1, 5)
                row += 1
                last_group = group_name

            edit = QKeySequenceEdit()
            edit.setMaximumSequenceLength(1)
            edit.setClearButtonEnabled(False)
            edit.setToolTip("Click, then press the new key combination")
            edit.setMinimumWidth(150)
            edit.editingFinished.connect(lambda a=action_id: self._edited(a))
            clear = QToolButton()
            clear.setText("Clear")
            clear.setToolTip("Remove this shortcut")
            clear.clicked.connect(lambda _=False, a=action_id: self._apply(a, ""))
            reset = QToolButton()
            reset.setText("Default")
            reset.setToolTip(f"Back to {QKeySequence(DEFAULTS[action_id]).toString(QKeySequence.NativeText) or 'no key'}")
            reset.clicked.connect(lambda _=False, a=action_id: self._apply(a, DEFAULTS[a]))

            grid.addWidget(QLabel(description), row, 0)
            grid.addWidget(edit, row, 2)
            grid.addWidget(clear, row, 3)
            grid.addWidget(reset, row, 4)
            self._edits[action_id] = edit
            self._resets[action_id] = reset
            row += 1

        self.status = QLabel("Click a shortcut and press the new keys. Changes apply and save right away.")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        reset_all = QPushButton("Reset all shortcuts")
        reset_all.clicked.connect(self._reset_all)
        bottom = QHBoxLayout()
        bottom.addWidget(self.status, 1)
        bottom.addWidget(reset_all)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(group)
        layout.addLayout(bottom)

        registry().changed.connect(self._show_key)
        for action_id in self._edits:
            self._show_key(action_id)

    def _show_key(self, action_id: str):
        edit = self._edits[action_id]
        edit.blockSignals(True)
        edit.setKeySequence(QKeySequence(registry().key(action_id)))
        edit.blockSignals(False)
        self._resets[action_id].setEnabled(registry().key(action_id) != DEFAULTS[action_id])

    def _edited(self, action_id: str):
        key = self._edits[action_id].keySequence().toString(QKeySequence.PortableText)
        self._apply(action_id, key)

    def _apply(self, action_id: str, key: str):
        other = registry().conflict(action_id, key)
        if other is not None:
            shown = QKeySequence(key).toString(QKeySequence.NativeText)
            self.status.setText(f"✗ {shown} is already used for “{DESCRIPTIONS[other]}”. "
                                "Change or clear that one first.")
            self._show_key(action_id)  # put the old key back
            return
        registry().set_key(action_id, key)
        shown = registry().text(action_id) or "no key"
        self.status.setText(f"✓ {DESCRIPTIONS[action_id]}: {shown}")
        self.log.emit(f"Shortcut changed: {DESCRIPTIONS[action_id]} → {shown}")

    def _reset_all(self):
        # clear first so swapping keys back (e.g. two actions that traded keys) never collides halfway
        for action_id in DEFAULTS:
            registry().set_key(action_id, "")
        registry().reset_all()
        self.status.setText("✓ All shortcuts are back to their defaults.")
        self.log.emit("Shortcuts reset to defaults")
