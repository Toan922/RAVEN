"""backend/outbox.py: Discord alerts kept on disk until sent (NFR-09)."""
from raven.backend.outbox import Outbox


def test_outbox_keeps_unsent_alerts_across_a_restart(tmp_path):
    path = tmp_path / "discord_outbox.jsonl"
    outbox = Outbox(path)
    outbox.add({"id": "a", "frame": "1.jpg"})
    outbox.add({"id": "b", "frame": "2.jpg"})
    outbox.add({"id": "a", "frame": "1.jpg"})      # queued twice: kept once
    outbox.mark_sent("a")
    with path.open("a", encoding="utf-8") as f:
        f.write('{"queued": {"id": "c"')            # a line cut short by a crash
    reopened = Outbox(path)
    assert [record["id"] for record in reopened.pending()] == ["b"]
    assert len(path.read_text().splitlines()) == 1  # opening compacts the file
