"""python test_curate.py -- stratified, group-aware split sanity check."""
import random
from collections import Counter, defaultdict

from curate import assign_splits, quality_bins

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
print("ok")
