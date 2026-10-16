"""Dataset-level decisions made after every image has been inspected: COCO vs
YOLO cross-check, duplicate grouping, keep/drop choices and group-aware
train/val/test splits."""

import os
import random
from collections import defaultdict

import numpy as np

from labels import label_text, read_yolo_boxes

# Boxes are compared against COCO after a round-trip through 6-decimal
# normalised YOLO text, so ~5e-7 of drift is expected; 1e-3 (2 px at 2048)
# separates that from a real conversion error.
COCO_TOL = 1e-3


def near_duplicate_pairs(rows, threshold):
    """Cluster perceptual hashes. Catches consecutive drone-video frames that
    would otherwise leak across the train/test split."""
    idx = [i for i, r in enumerate(rows) if r["status"] == "ok" and r["dhash"] != ""]
    if len(idx) < 2:
        return []
    h = np.array([rows[i]["dhash"] for i in idx], dtype=np.uint64)
    bits = np.unpackbits(h.view(np.uint8).reshape(-1, 8)[:, ::-1], axis=1).astype(np.float32)
    inv = 1.0 - bits
    pairs = []
    chunk = 1024
    for s in range(0, len(idx), chunk):
        block = bits[s:s + chunk]
        binv = inv[s:s + chunk]
        dist = block @ inv.T + binv @ bits.T          # Hamming distance
        for local, row_d in enumerate(dist):
            i = s + local
            hits = np.where(row_d <= threshold)[0]
            for j in hits:
                if j > i:
                    pairs.append((rows[idx[i]]["rel_path"],
                                  rows[idx[j]]["rel_path"], int(row_d[j])))
    return pairs


def enrich_rows(rows, labels_dir, meta, coco):
    """Join session attributes onto every row and diff COCO against the YOLO
    .txt files. A disagreement here means the two ground-truth copies shipped
    out of sync -- the .txt files are what training reads, so that is a defect
    worth dropping the image over."""
    for r in rows:
        name = os.path.basename(r["rel_path"])
        m = meta.get(name)
        if m:
            r["session"] = m["session"]
            for k in ("location", "weather", "surface", "season"):
                r[k] = m[k]
        elif meta:
            r["session"] = "UNKNOWN"

        if not coco:
            r["coco_status"] = "no-coco"
            continue
        rec = coco.get(name)
        if rec is None:
            r["coco_status"], r["coco_note"] = "missing", "not in labels.json"
            continue
        if r["status"] != "ok":
            r["coco_status"] = "skipped"
            continue
        if (rec["width"], rec["height"]) != (r["width"], r["height"]):
            r["coco_status"] = "size-mismatch"
            r["coco_note"] = f'json {rec["width"]}x{rec["height"]}'
            continue
        want = sorted(rec["boxes"])
        got = sorted(read_yolo_boxes(
            os.path.join(labels_dir, os.path.splitext(r["rel_path"])[0] + ".txt")
        )) if labels_dir else []
        if len(want) != len(got):
            r["coco_status"] = "count-mismatch"
            r["coco_note"] = f"json={len(want)} txt={len(got)}"
            continue
        drift = max((max(abs(a - b) for a, b in zip(u, v)) for u, v in zip(want, got)),
                    default=0.0)
        if drift > COCO_TOL:
            r["coco_status"], r["coco_note"] = "coord-mismatch", f"{drift:.2e}"
        else:
            r["coco_status"] = "ok"


class _Union:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def build_groups(rows, pairs):
    """Two levels of grouping:
      sha_group -- byte-identical files, which are redundancy, not data.
      group_id  -- sha groups plus near-duplicate pairs, i.e. frames close
                   enough that letting them straddle a train/test split would
                   leak. Kept as a column so splitting can be group-aware.
    """
    by_rel = {r["rel_path"]: r for r in rows}
    ok = [r for r in rows if r["status"] == "ok"]

    sha_members = defaultdict(list)
    for r in ok:
        if r["sha256"]:
            sha_members[r["sha256"]].append(r["rel_path"])

    u = _Union()
    for r in ok:
        u.find(r["rel_path"])
    for members in sha_members.values():
        for other in members[1:]:
            u.union(members[0], other)
    for a, b, _ in pairs:
        u.union(a, b)

    clusters = defaultdict(list)
    for r in ok:
        clusters[u.find(r["rel_path"])].append(r["rel_path"])
    # stable ids: order clusters by their lexicographically smallest member
    for gid, key in enumerate(sorted(clusters, key=lambda k: min(clusters[k])), 1):
        for rel in clusters[key]:
            by_rel[rel]["group_id"] = gid
            by_rel[rel]["group_size"] = len(clusters[key])

    sha_groups = {}
    for gid, (digest, members) in enumerate(
            sorted(((d, m) for d, m in sha_members.items() if len(m) > 1),
                   key=lambda kv: min(kv[1])), 1):
        for rel in members:
            by_rel[rel]["sha_group"] = gid
        sha_groups[gid] = sorted(members)
    return sha_groups, clusters


def _representative(rows_subset):
    """Richest annotation first, then largest frame, then name -- deterministic."""
    return sorted(rows_subset,
                  key=lambda r: (-int(r["n_boxes"] or 0),
                                 -(int(r["width"] or 0) * int(r["height"] or 0)),
                                 r["rel_path"]))[0]


def choose_keepers(rows, sha_groups, clusters, labels_dir, args):
    """Decide what survives into the clean set. Order matters: hard defects
    first, then redundancy."""
    by_rel = {r["rel_path"]: r for r in rows}
    for r in rows:
        r["keep"], r["drop_reason"] = 1, ""

    def drop(r, why):
        if r["keep"]:
            r["keep"], r["drop_reason"] = 0, why

    for r in rows:
        if r["status"] != "ok":
            drop(r, f'decode:{r["status"]}')
        elif r["label_status"] == "invalid":
            drop(r, f'invalid-label:{r["label_errors"][:60]}')
        elif r["label_status"] == "missing" and labels_dir:
            drop(r, "missing-label")
        elif r["coco_status"] in ("missing", "size-mismatch",
                                  "count-mismatch", "coord-mismatch"):
            drop(r, f'coco:{r["coco_status"]}')
        elif args.drop_negatives and not any(
                int(c) in args.class_map for c in str(r["class_ids"]).split(",") if c):
            drop(r, "background")

    conflicts = 0
    for members in sha_groups.values():
        live = [by_rel[m] for m in members if by_rel[m]["keep"]]
        if len(live) < 2:
            continue
        texts = {label_text(labels_dir, r["rel_path"]) for r in live} if labels_dir \
            else {None}
        if len(texts) > 1:
            # Identical pixels annotated differently: at most one label set can
            # be right and nothing in the data says which. Unresolvable.
            conflicts += 1
            if args.dup_conflict == "drop":
                for r in live:
                    drop(r, f"dup-label-conflict(n={len(live)})")
                continue
        rep = _representative(live)
        for r in live:
            if r is not rep:
                drop(r, f'exact-duplicate-of:{rep["rel_path"]}')

    if args.drop_near_dups:
        for members in clusters.values():
            live = [by_rel[m] for m in members if by_rel[m]["keep"]]
            if len(live) < 2:
                continue
            rep = _representative(live)
            for r in live:
                if r is not rep:
                    drop(r, f'near-duplicate-of:{rep["rel_path"]}')
    return conflicts


def quality_bins(rows, keys, n):
    """Quantile-bin each metric over the kept frames into n bins and tag each
    row with its stratum, e.g. "0-2" = lowest laplacian_var bin, highest std
    bin. Quantiles are per dataset, so each source is binned on its own scale."""
    kept = [r for r in rows if r["keep"]]
    cols = []
    for k in keys:
        v = np.array([float(r[k]) for r in kept])
        edges = np.quantile(v, np.linspace(0, 1, n + 1)[1:-1])
        cols.append(np.searchsorted(edges, v, side="right"))
    for r, *b in zip(kept, *cols):
        r["quality_bin"] = "-".join(map(str, b))


def assign_splits(rows, fractions, seed):
    """Group-aware: a near-duplicate cluster lands wholly in one split, so
    near-identical frames cannot straddle train and test. Stratified: groups
    are walked bin by bin and each goes to the split furthest behind its
    running quota, so every quality_bin is spread across splits in proportion."""
    kept = [r for r in rows if r["keep"]]
    groups = defaultdict(list)
    for r in kept:
        groups[r["group_id"] or r["rel_path"]].append(r)
    names = ["train", "val", "test"][:len(fractions)]
    counts = [0] * len(names)
    keys = sorted(groups, key=str)
    random.Random(seed).shuffle(keys)
    # ponytail: a group is binned by its first frame; fine while groups are small
    # within a bin, largest groups first so later small ones can even out the quota
    keys.sort(key=lambda k: (groups[k][0]["quality_bin"], -len(groups[k])))
    seen = 0
    for key in keys:
        seen += len(groups[key])
        i = max(range(len(names)), key=lambda j: fractions[j] * seen - counts[j])
        for r in groups[key]:
            r["split"] = names[i]
        counts[i] += len(groups[key])
    return dict(zip(names, counts))
