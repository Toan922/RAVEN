"""python tests/data_pipeline/test_curate.py (or pytest) -- data_pipeline/curate.py: stratified, group-aware splits and background capping."""
import random
from collections import Counter, defaultdict

from raven.data_pipeline.check_unified import CrossDatasetValidator
from raven.data_pipeline.curate import assign_splits, cap_background_frames, quality_bins


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


if __name__ == "__main__":
    test_stratified_group_aware_splits()
    test_background_ratio_capping_and_leakage()
    print("ok")
