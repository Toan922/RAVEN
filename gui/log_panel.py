"""
Log panel: timestamped messages from ingest, the alert feed, Discord and the map.
It opens beside the Settings form (Show log button), and keeps collecting messages while it's closed.
"""
from datetime import datetime

from PySide6.QtCore import Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QFileDialog, QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout

# keep the log from growing without bound on long runs
MAX_LINES = 5000


class LogPanel(QFrame):
    close_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("logPanel")

        title = QLabel("Log")
        font = title.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.25)
        title.setFont(font)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(MAX_LINES)
        self.text.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))

        save = QPushButton("Save log…")
        save.clicked.connect(self._save)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.text.clear)
        close = QPushButton("Close")
        close.setToolTip("Hide the log (it keeps recording)")
        close.clicked.connect(self.close_requested)

        top = QHBoxLayout()
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(save)
        top.addWidget(clear)
        top.addWidget(close)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.addLayout(top)
        layout.addWidget(self.text, 1)

    def append(self, message: str):
        self.text.appendPlainText(f"[{datetime.now():%H:%M:%S}] {message}")

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save log", f"raven_{datetime.now():%Y%m%d_%H%M%S}.log",
                                              "Log files (*.log *.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.text.toPlainText())
