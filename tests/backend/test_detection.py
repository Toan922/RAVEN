"""backend/detection.py: model loading and the Human Detection filter (FR-07, FR-14)."""
from raven.backend.detection import loadable_path


def test_weights_in_a_folder_with_an_apostrophe_load_from_a_copy(tmp_path):
    # ultralytics drops apostrophes from weight paths, so "Fall '26" would become "Fall 26" and not be found
    folder = tmp_path / "Fall '26"
    folder.mkdir()
    weights = folder / "best.pt"
    weights.write_bytes(b"weights")
    path = loadable_path(weights)
    assert "'" not in path and path.endswith("best.pt")
    assert open(path, "rb").read() == b"weights"
    assert loadable_path(weights) == path                  # copied once, then reused

    plain = tmp_path / "plain.pt"
    plain.write_bytes(b"x")
    assert loadable_path(plain) == str(plain)              # paths without one are used as they are
