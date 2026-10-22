"""python test_curate.py -- stratified, group-aware split & CrossDatasetValidator tests."""
import json
import os
import random
import tempfile
from collections import Counter, defaultdict

from check_unified import CrossDatasetValidator
from curate import (assign_splits, cap_background_frames, quality_bins)
from labels import parse_yolo_label


def test_stratified_group_aware_splits():
    rng = random.Random(1)
    rows = [{"rel_path": f"{i}.png", "keep": 1, "quality_bin": "", "split": "",
             "group_id": i // 3 if i < 300 else "",          # some 3-frame clusters
             "laplacian_var": rng.lognormvariate(4, 1), "std": rng.uniform(5, 60)}
            for i in range(3000)]
    quality_bins(rows, ["laplacian_var", "std"], 3)
    assign_splits(rows, [0.8, 0.1, 0.1], seed=0)

    by_bin = defaultdict(Counter)
    for r in rows:
        by_bin[r["quality_bin"]][r["split"]] += 1
    assert len(by_bin) == 9, by_bin
    for b, c in sorted(by_bin.items()):
        n = sum(c.values())
        print(b, n, {s: round(c[s] / n, 3) for s in ("train", "val", "test")})
        assert abs(c["test"] / n - 0.1) < 0.02, (b, c)

    splits_per_group = defaultdict(set)
    for r in rows:
        if r["group_id"] != "":
            splits_per_group[r["group_id"]].add(r["split"])
    assert all(len(s) == 1 for s in splits_per_group.values())


def test_orphan_detection():
    v = CrossDatasetValidator()
    rep = v.check_orphans(
        image_stems={"img1", "img2", "img_no_lbl"},
        label_stems={"img1", "img2", "ghost_lbl"},
        image_names={"img1.jpg", "img2.jpg", "img_no_lbl.jpg"},
        meta_names={"img1.jpg", "img2.jpg", "img_no_lbl.jpg", "ghost_meta.jpg"},
        coco_names={"img1.jpg", "img2.jpg"},
        coco_orphan_anns=2,
    )
    assert not rep.ok
    assert any("img_no_lbl" in e for e in rep.orphan_errors)
    assert any("ghost_lbl.txt" in e for e in rep.orphan_errors)
    assert any("ghost_meta.jpg" in e for e in rep.orphan_errors)
    assert any("img_no_lbl.jpg" in e for e in rep.orphan_errors)
    assert rep.stats["orphan_annotations"] >= 3


def test_telemetry_synchronization():
    v = CrossDatasetValidator()
    hu_ok = {
        "original_filename": "0_60_30_0_00001.jpg", "source": "HU",
        "split": "train", "is_background": False,
        "time_of_day": 0, "altitude_m": 60, "camera_angle_deg": 30, "weather": 0,
        "laplacian_var": 120.5, "std": 35.2, "entropy": 6.8, "mean": 110.0,
    }
    assert v.check_telemetry_record("HU00001.jpg", hu_ok, "train", False).ok

    ar_ok = {
        "original_filename": "frame_001.png", "source": "AR",
        "split": "val", "is_background": False,
        "session": "2024-02-15", "location": "forest", "weather": "clear",
        "surface": "snow", "season": "winter",
        "laplacian_var": 85.1, "std": 22.0, "entropy": 5.9, "mean": 95.4,
    }
    assert v.check_telemetry_record("AR00001.png", ar_ok, "val", False).ok

    hu_bad = dict(hu_ok, altitude_m=True, camera_angle_deg=120, is_background=True)
    rep_bad = v.check_telemetry_record("HU00002.jpg", hu_bad, "test", False)
    assert not rep_bad.ok and len(rep_bad.telemetry_errors) >= 4

    ar_bad = dict(ar_ok, session="UNKNOWN", location="  ")
    rep_ar_bad = v.check_telemetry_record("AR00002.png", ar_bad)
    assert not rep_ar_bad.ok
    assert any("UNKNOWN" in e for e in rep_ar_bad.telemetry_errors)


def test_bounding_box_validation_and_filtering():
    v = CrossDatasetValidator(n_classes=1, min_box_frac=1e-4)
    raw_boxes = [
        (0, 0.5, 0.5, 0.2, 0.2),       # valid
        (0, 0.5, 0.5, 0.2, 0.2),       # duplicate
        (0, 0.5, 0.5, 0.0, 0.1),       # zero-area
        (0, 0.5, 0.5, 0.001, 0.001),   # microscopic (1e-6 < 1e-4)
        (0, 1.2, 0.5, 0.2, 0.2),       # out of [0, 1]
        (0, 0.95, 0.5, 0.3, 0.2),      # corner out of frame (0.95 + 0.15 > 1.0)
        (2, 0.4, 0.4, 0.1, 0.1),       # invalid class ID
    ]
    rep = v.check_orphans(set(), set())
    valid, issues = v.validate_and_filter_boxes(raw_boxes, name="test.txt", report=rep)
    assert valid == [(0, 0.5, 0.5, 0.2, 0.2)]
    assert rep.stats["zero_area_flagged"] == 1
    assert rep.stats["microscopic_flagged"] == 1
    assert rep.stats["out_of_range_flagged"] == 2
    assert len(issues) == 6

    with tempfile.TemporaryDirectory() as tmp:
        lp = os.path.join(tmp, "lbl.txt")
        with open(lp, "w") as f:
            f.write("0 0.5 0.5 0.1 0.1\n0 0.2 0.2 0.0005 0.0005\n0 0.3 0.3 0.0 0.1\n")
        assert parse_yolo_label(lp, 1, 2e-6, drop_tiny=False)[0] == "invalid"
        st_drop, n_drop, errs_drop, _ = parse_yolo_label(lp, 1, 2e-6, drop_tiny=True)
        assert st_drop == "ok" and n_drop == 1
        assert "tiny(" in errs_drop and "degenerate" in errs_drop


def test_background_ratio_capping_and_leakage():
    rows = []
    for i in range(20):
        rows.append({
            "rel_path": f"pos_{i:02d}.jpg", "status": "ok", "keep": 1,
            "drop_reason": "", "n_boxes": 1, "class_ids": "0",
            "group_id": i + 1, "quality_bin": "0", "split": "",
            "sha256": f"sha_pos_{i}", "laplacian_var": 50.0, "std": 20.0,
            "entropy": 5.5, "mean": 100.0,
        })
    for i in range(15):
        rows.append({
            "rel_path": f"bg_{i:02d}.jpg", "status": "ok", "keep": 1,
            "drop_reason": "", "n_boxes": 0, "class_ids": "",
            "group_id": 100 + i, "quality_bin": "0", "split": "",
            "sha256": f"sha_bg_{i}", "laplacian_var": 50.0, "std": 20.0,
            "entropy": 5.5, "mean": 100.0,
        })

    assert cap_background_frames(rows, max_bg_ratio=0.25, class_map={0: 0}, seed=42) == 10
    assign_splits(rows, [0.8, 0.1, 0.1], seed=42, class_map={0: 0})
    v = CrossDatasetValidator(n_classes=1, max_bg_ratio=0.25, require_telemetry=False)
    rep = v.validate_curation(rows, class_map={0: 0})
    assert rep.ok, rep.summary()

    kept = [r for r in rows if r["keep"]]
    kept[1]["group_id"] = kept[0]["group_id"]
    kept[1]["split"] = "test" if kept[0]["split"] != "test" else "train"
    rep_leak = v.validate_curation(rows, class_map={0: 0})
    assert not rep_leak.ok and rep_leak.stats["leaky_groups"] >= 1


def test_unified_dataset_validation():
    with tempfile.TemporaryDirectory() as tmp:
        ds = os.path.join(tmp, "clean_HITUAV")
        for sub in ("images", "labels", "splits"):
            os.makedirs(os.path.join(ds, sub), exist_ok=True)

        meta = {}
        for idx, (name, sp) in enumerate(
            [("HU00001.jpg", "train"), ("HU00002.jpg", "val"), ("HU00003.jpg", "test")], 1
        ):
            img_p = os.path.join(ds, "images", name)
            lbl_p = os.path.join(ds, "labels", os.path.splitext(name)[0] + ".txt")
            open(img_p, "wb").write(f"fake_image_bytes_{idx}".encode())
            open(lbl_p, "w").write("0 0.500000 0.500000 0.200000 0.200000\n")
            meta[name] = {
                "original_filename": f"0_60_30_0_{idx:05d}.jpg", "source": "HU",
                "split": sp, "is_background": False, "time_of_day": 0,
                "altitude_m": 60, "camera_angle_deg": 30, "weather": 0,
                "laplacian_var": 100.0, "std": 30.0, "entropy": 6.5, "mean": 120.0,
                "quality_bin": "1-1",
            }
            open(os.path.join(ds, "splits", f"{sp}.txt"), "w").write(img_p.replace(os.sep, "/") + "\n")

        json.dump(meta, open(os.path.join(ds, "metadata.json"), "w"))
        yaml_p = os.path.join(tmp, "raven.yaml")
        open(yaml_p, "w").write(
            f"path: {tmp}\ntrain: [clean_HITUAV/splits/train.txt]\n"
            "val: [clean_HITUAV/splits/val.txt]\ntest: [clean_HITUAV/splits/test.txt]\n"
            "names:\n  0: Person\n"
        )

        v = CrossDatasetValidator(n_classes=1, max_bg_ratio=0.0)
        rep = v.validate_unified(yaml_p, check_hashes=True)
        assert rep.ok, rep.summary()

        open(os.path.join(ds, "labels", "HU99999.txt"), "w").write("0 0.5 0.5 0.1 0.1\n")
        rep_orphan = v.validate_unified(yaml_p)
        assert not rep_orphan.ok and any("HU99999.txt" in e for e in rep_orphan.orphan_errors)


if __name__ == "__main__":
    test_stratified_group_aware_splits()
    test_orphan_detection()
    test_telemetry_synchronization()
    test_bounding_box_validation_and_filtering()
    test_background_ratio_capping_and_leakage()
    test_unified_dataset_validation()
    print("ok")
