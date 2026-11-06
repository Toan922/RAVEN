"""
Discord outbox for the operator console (FR-11.2, NFR-09): every alert queued for Discord is written here first,
and marked sent once Discord accepts it, so alerts still waiting when the console closes (or crashes) are sent
on the next launch instead of being lost.

The file is JSON Lines: {"queued": <event record>} when an alert is queued, {"sent": <id>} when it's delivered.
Opening the outbox rewrites the file with only the alerts still waiting. Each line is encrypted when the console
turned encryption on (raven.backend.vault).
"""

import json
import threading
from pathlib import Path

from raven.backend import vault

OUTBOX_NAME = "discord_outbox.jsonl"


class Outbox:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._pending: dict[str, dict] = {}
        self._load()

    def _load(self):
        if not self.path.is_file():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(vault.decode_line(line))
                except (ValueError, vault.VaultError):
                    continue  # a line cut short by a crash
                if isinstance(entry.get("queued"), dict) and entry["queued"].get("id"):
                    self._pending[str(entry["queued"]["id"])] = entry["queued"]
                elif entry.get("sent") is not None:
                    self._pending.pop(str(entry["sent"]), None)
        self._rewrite()

    def _rewrite(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [vault.encode_line(json.dumps({"queued": record})) for record in self._pending.values()]
        self.path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")

    def _append(self, entry: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(vault.encode_line(json.dumps(entry)) + "\n")

    def add(self, record: dict):
        """
        An alert was queued for Discord; record is detection_log.event_record(event).
        """
        with self._lock:
            key = str(record["id"])
            if key in self._pending:
                return
            self._pending[key] = record
            self._append({"queued": record})

    def mark_sent(self, alert_id: str):
        with self._lock:
            if self._pending.pop(str(alert_id), None) is not None:
                self._append({"sent": str(alert_id)})

    def pending(self) -> list[dict]:
        """
        Alerts queued but not sent yet, oldest first.
        """
        with self._lock:
            return list(self._pending.values())
