#!/usr/bin/env python3
"""
AIResQ dataset validator and cleaner.

Stage 1 (always, read-only): walks an image directory, decodes every file,
measures integrity, resolution, IR-likelihood, quality stats and (optionally)
YOLO label validity, then writes a per-image manifest CSV plus a summary.

Stage 2 (read-only on the source): joins AIResQ's two native sidecars --
`metadata.json` (per-session capture conditions) and `labels.json` (COCO
ground truth) -- onto the manifest, cross-checks COCO against the YOLO .txt
files, and clusters exact/near duplicates into leakage groups.

Stage 3 (opt-in, --clean-out): emits a cleaned copy of the dataset in a NEW
directory. The source tree is never written to; images are hardlinked when
the filesystem allows and copied otherwise.

Usage:
    # validate only
    python validate_dataset.py --images AIResQ/images --labels AIResQ/labels \
        --out manifest.csv

    # validate + emit a cleaned dataset
    python validate_dataset.py --images AIResQ/images --labels AIResQ/labels \
        --out manifest.csv --clean-out clean_AIResQ

Deps: pillow, numpy
"""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageFile

# Fail loudly on truncated files instead of silently padding them with grey.
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = 200_000_000
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SINGLE_CHANNEL_MODES = {"L", "I", "I;16", "I;16B", "I;16L", "F"}

FIELDS = [
    "path", "rel_path", "bytes", "sha256", "status", "error",
    "format", "mode", "bit_depth", "width", "height", "megapixels", "aspect",
    "ir_verdict", "ir_reason", "exif_make", "exif_model", "flir_segment",
    "unique_colors", "gray_frac", "mean", "std", "clipped_low_frac",
    "clipped_high_frac", "entropy", "laplacian_var", "dhash",
    "label_status", "n_boxes", "label_errors", "class_ids",
    # stage 2 -- AIResQ sidecar join, COCO cross-check, duplicate grouping
    "session", "location", "weather", "surface", "season",
    "coco_status", "coco_note", "sha_group", "group_id", "group_size",
    # stage 3 -- cleaning decision
    "keep", "drop_reason", "split",
]

# Boxes are compared against COCO after a round-trip through 6-decimal
# normalised YOLO text, so ~5e-7 of drift is expected; 1e-3 (2 px at 2048)
# separates that from a real conversion error.
COCO_TOL = 1e-3


# --------------------------------------------------------------------------
# per-image work
# --------------------------------------------------------------------------

def sha256_and_head(path, head_bytes=65536):
    h = hashlib.sha256()
    head = b""
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            if not head:
                head = chunk[:head_bytes]
            h.update(chunk)
    return h.hexdigest(), head


def to_uint8_gray(arr):
    """Normalise any bit depth / channel count to an 8-bit grayscale array."""
    a = arr.astype(np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return (((a - lo) / (hi - lo)) * 255.0).astype(np.uint8)


def dhash64(gray_u8):
    im = Image.fromarray(gray_u8).resize((9, 8), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    out = np.uint64(0)
    for b in bits:
        out = np.uint64(out << np.uint64(1)) | np.uint64(bool(b))
    return int(out)


def laplacian_var(gray_u8):
    """Blur proxy. Low variance of the Laplacian => soft / out of focus."""
    a = gray_u8.astype(np.float32)
    if min(a.shape) < 3:
        return 0.0
    lap = (
        -4.0 * a[1:-1, 1:-1]
        + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
    )
    return float(lap.var())


def shannon_entropy(gray_u8):
    counts = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def classify_ir(mode, arr, exif_make, exif_model, flir_segment, filename):
    """
    Decide whether this frame is plausibly infrared.

    Three independent signals, cheapest first:
      1. Provenance -- FLIR APP1 segment, thermal camera model, DJI _T suffix.
      2. Channel structure -- 16-bit single channel or 8-bit grayscale.
      3. Colour manifold -- a false-coloured thermal frame is a palette lookup,
         so its pixels lie on a 1-D curve through RGB space and it has very
         few unique colours. A natural visible-light photo has thousands.
    Returns (verdict, reason, unique_colors, gray_frac).
    """
    meta = " ".join(x for x in (exif_make or "", exif_model or "") if x).lower()
    thermal_model = any(
        k in meta for k in ("flir", "boson", "lepton", "tau2", "xt2", "h20t",
                            "m2ea", "m30t", "m3t", "thermal", "zh20t")
    )
    name_thermal = os.path.splitext(filename)[0].upper().endswith(("_T", "_R"))

    unique_colors = -1
    gray_frac = -1.0

    if mode in SINGLE_CHANNEL_MODES:
        verdict, reason = "ir", f"single-channel ({mode})"
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        # subsample so this stays cheap on multi-megapixel frames
        step = max(1, int(max(arr.shape[:2]) // 512))
        sub = arr[::step, ::step, :3].astype(np.int32)
        spread = sub.max(axis=2) - sub.min(axis=2)
        gray_frac = float((spread <= 2).mean())
        packed = (sub[..., 0] << 16) | (sub[..., 1] << 8) | sub[..., 2]
        unique_colors = int(np.unique(packed).size)

        if gray_frac >= 0.99:
            verdict, reason = "ir", "grayscale stored as RGB"
        elif unique_colors <= 1024:
            verdict, reason = "ir", f"palette-mapped ({unique_colors} colors)"
        elif unique_colors <= 8192:
            verdict, reason = "uncertain", f"{unique_colors} colors"
        else:
            verdict, reason = "not_ir", f"broadband RGB ({unique_colors} colors)"
    else:
        verdict, reason = "uncertain", f"unhandled mode {mode}"

    if flir_segment or thermal_model or name_thermal:
        src = ("flir-segment" if flir_segment
               else "thermal-model" if thermal_model else "filename-suffix")
        if verdict == "not_ir":
            verdict, reason = "conflict", f"{src} but {reason}"
        else:
            verdict, reason = "ir", f"{src}; {reason}"

    return verdict, reason, unique_colors, gray_frac


def parse_yolo_label(label_path, n_classes, min_box_frac):
    if not os.path.exists(label_path):
        return "missing", 0, "no label file", ""
    errs, boxes, cls = [], [], []
    with open(label_path, "r", errors="replace") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                errs.append(f"L{i}:fields={len(parts)}")
                continue
            try:
                c = int(float(parts[0]))
                x, y, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                errs.append(f"L{i}:unparseable")
                continue
            if not (0 <= c < n_classes):
                errs.append(f"L{i}:class={c}")
            if w <= 0 or h <= 0:
                errs.append(f"L{i}:degenerate")
                continue
            if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                errs.append(f"L{i}:not-normalized")
            if x - w / 2 < -1e-3 or x + w / 2 > 1 + 1e-3 or \
               y - h / 2 < -1e-3 or y + h / 2 > 1 + 1e-3:
                errs.append(f"L{i}:out-of-frame")
            if w * h < min_box_frac:
                errs.append(f"L{i}:tiny({w*h:.2e})")
            boxes.append((c, round(x, 5), round(y, 5), round(w, 5), round(h, 5)))
            cls.append(c)
    dupes = len(boxes) - len(set(boxes))
    if dupes:
        errs.append(f"duplicate-boxes={dupes}")
    if not boxes and not errs:
        status = "empty"          # legitimate negative sample, but count it
    elif errs:
        status = "invalid"
    else:
        status = "ok"
    return status, len(boxes), ";".join(errs[:8]), ",".join(map(str, sorted(set(cls))))


def read_yolo_boxes(label_path):
    """Coordinates only, no validation -- used by the COCO cross-check and by
    the cleaner. parse_yolo_label() stays the authority on validity."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                boxes.append((int(float(parts[0])), *(float(v) for v in parts[1:5])))
            except ValueError:
                continue
    return boxes


def inspect(job):
    path, root, labels_dir, n_classes, min_box_frac = job
    rel = os.path.relpath(path, root)
    row = {k: "" for k in FIELDS}
    row["path"] = path
    row["rel_path"] = rel
    try:
        row["bytes"] = os.path.getsize(path)
    except OSError as e:
        row.update(status="unreadable", error=str(e))
        return row

    if row["bytes"] == 0:
        row.update(status="corrupt", error="zero-byte file")
        return row

    try:
        digest, head = sha256_and_head(path)
        row["sha256"] = digest
        row["flir_segment"] = int(b"FLIR" in head)

        # Pass 1: structural verification (header/checksum level).
        with Image.open(path) as im:
            im.verify()
        # Pass 2: full decode -- catches truncated pixel data that verify() misses.
        with Image.open(path) as im:
            im.load()
            row["format"] = im.format or ""
            row["mode"] = im.mode
            row["width"], row["height"] = im.size
            exif = im.getexif()
            row["exif_make"] = str(exif.get(271, "") or "")[:40]
            row["exif_model"] = str(exif.get(272, "") or "")[:40]
            arr = np.asarray(im)
    except Exception as e:
        row.update(status="corrupt", error=f"{type(e).__name__}: {e}"[:200])
        return row

    row["status"] = "ok"
    row["bit_depth"] = int(arr.dtype.itemsize * 8)
    w, h = row["width"], row["height"]
    row["megapixels"] = round(w * h / 1e6, 3)
    row["aspect"] = round(w / h, 4) if h else 0

    verdict, reason, uniq, grayf = classify_ir(
        row["mode"], arr, row["exif_make"], row["exif_model"],
        row["flir_segment"], os.path.basename(path)
    )
    row["ir_verdict"], row["ir_reason"] = verdict, reason
    row["unique_colors"] = uniq
    row["gray_frac"] = round(grayf, 4) if grayf >= 0 else ""

    gray = to_uint8_gray(arr)
    flat = arr.astype(np.float32)
    dmax = float(np.iinfo(arr.dtype).max) if np.issubdtype(arr.dtype, np.integer) else 1.0
    row["mean"] = round(float(flat.mean()), 2)
    row["std"] = round(float(flat.std()), 2)
    row["clipped_low_frac"] = round(float((flat <= 0).mean()), 4)
    row["clipped_high_frac"] = round(float((flat >= dmax).mean()), 4)
    row["entropy"] = round(shannon_entropy(gray), 3)
    row["laplacian_var"] = round(laplacian_var(gray), 2)
    row["dhash"] = dhash64(gray)

    if labels_dir:
        lp = os.path.join(labels_dir, os.path.splitext(rel)[0] + ".txt")
        st, n, errs, cls = parse_yolo_label(lp, n_classes, min_box_frac)
        row["label_status"], row["n_boxes"] = st, n
        row["label_errors"], row["class_ids"] = errs, cls
    return row


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# AIResQ native sidecars: metadata.json (sessions) + labels.json (COCO)
# --------------------------------------------------------------------------

def load_session_metadata(path):
    """metadata.json maps a capture date -> {filenames, location, weather,
    surface, season}. Invert it to filename -> session attributes."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for session, blk in raw.items():
        attrs = {k: blk.get(k, "") for k in ("location", "weather", "surface", "season")}
        for fn in blk.get("filenames", []):
            out[fn] = dict(attrs, session=session)
    return out


def load_coco(path):
    """labels.json is COCO detection format. Return filename -> record with
    boxes already converted to normalised YOLO (class ids shifted to 0-based),
    plus the category table."""
    if not path or not os.path.exists(path):
        return {}, []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    by_id = {}
    for im in raw.get("images", []):
        by_id[im["id"]] = {
            "file_name": im["file_name"],
            "width": im.get("width", 0),
            "height": im.get("height", 0),
            "boxes": [],
        }
    for a in raw.get("annotations", []):
        rec = by_id.get(a["image_id"])
        if rec is None:
            continue
        x, y, w, h = a["bbox"]
        W, H = rec["width"] or 1, rec["height"] or 1
        rec["boxes"].append((int(a["category_id"]) - 1,
                             (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
    return {r["file_name"]: r for r in by_id.values()}, raw.get("categories", [])


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


# --------------------------------------------------------------------------
# duplicate grouping and cleaning decisions
# --------------------------------------------------------------------------

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


def label_text(labels_dir, rel_path):
    p = os.path.join(labels_dir, os.path.splitext(rel_path)[0] + ".txt")
    if not os.path.exists(p):
        return None
    with open(p, "r", errors="replace") as f:
        return "\n".join(sorted(ln.strip() for ln in f if ln.strip()))


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


def clean_boxes(boxes, min_box_frac, drop_tiny):
    """Clamp to the frame, drop degenerate//tiny boxes, de-duplicate, sort."""
    out, dropped = [], 0
    for c, x, y, w, h in boxes:
        x0, y0 = max(0.0, x - w / 2), max(0.0, y - h / 2)
        x1, y1 = min(1.0, x + w / 2), min(1.0, y + h / 2)
        nw, nh = x1 - x0, y1 - y0
        if nw <= 0 or nh <= 0 or (drop_tiny and nw * nh < min_box_frac):
            dropped += 1
            continue
        out.append((c, round(x0 + nw / 2, 6), round(y0 + nh / 2, 6),
                    round(nw, 6), round(nh, 6)))
    deduped = sorted(set(out))
    dropped += len(out) - len(deduped)
    return deduped, dropped


def assign_splits(rows, fractions, seed):
    """Group-aware: a near-duplicate cluster lands wholly in one split, so
    near-identical frames cannot straddle train and test."""
    kept = [r for r in rows if r["keep"]]
    groups = defaultdict(list)
    for r in kept:
        groups[r["group_id"] or r["rel_path"]].append(r)
    names = ["train", "val", "test"][:len(fractions)]
    targets = [f * len(kept) for f in fractions]
    counts = [0] * len(names)
    keys = sorted(groups, key=str)
    random.Random(seed).shuffle(keys)
    # largest groups first so they cannot overshoot a small split at the end
    for key in sorted(keys, key=lambda k: -len(groups[k])):
        i = max(range(len(names)), key=lambda j: targets[j] - counts[j])
        for r in groups[key]:
            r["split"] = names[i]
        counts[i] += len(groups[key])
    return dict(zip(names, counts))


# --------------------------------------------------------------------------
# clean dataset emission
# --------------------------------------------------------------------------

def materialize(src, dst, mode):
    if os.path.lexists(dst):
        os.remove(dst)
    if mode in ("auto", "hard"):
        try:
            os.link(src, dst)
            return "hard"
        except OSError:
            if mode == "hard":
                raise
    if mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
        return "symlink"
    shutil.copy2(src, dst)
    return "copy"


def write_clean_dataset(rows, labels_dir, meta, categories, args):
    out = args.clean_out
    src_root = os.path.abspath(args.images)
    if os.path.abspath(out) == src_root or \
       os.path.abspath(out).startswith(src_root + os.sep):
        sys.exit(f"--clean-out must be outside the source image tree ({src_root})")

    img_dir, lab_dir = os.path.join(out, "images"), os.path.join(out, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)

    kept = [r for r in rows if r["keep"]]
    kept.sort(key=lambda r: r["rel_path"])
    link_kinds = Counter()
    boxes_written = boxes_dropped = 0
    coco_images, coco_anns = [], []

    for img_id, r in enumerate(kept, 1):
        rel = r["rel_path"]
        dst = os.path.join(img_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        link_kinds[materialize(r["path"], dst, args.link)] += 1

        boxes = read_yolo_boxes(
            os.path.join(labels_dir, os.path.splitext(rel)[0] + ".txt")
        ) if labels_dir else []
        boxes, nd = clean_boxes(boxes, args.min_box_frac, args.drop_tiny_boxes)
        boxes_dropped += nd
        boxes_written += len(boxes)

        lp = os.path.join(lab_dir, os.path.splitext(rel)[0] + ".txt")
        os.makedirs(os.path.dirname(lp), exist_ok=True)
        with open(lp, "w", newline="\n") as f:
            for c, x, y, w, h in boxes:
                f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        W, H = int(r["width"]), int(r["height"])
        coco_images.append({"id": img_id, "file_name": os.path.basename(rel),
                            "width": W, "height": H})
        for c, x, y, w, h in boxes:
            bw, bh = w * W, h * H
            coco_anns.append({
                "id": len(coco_anns) + 1, "image_id": img_id,
                "category_id": c + 1,
                "bbox": [round((x - w / 2) * W, 3), round((y - h / 2) * H, 3),
                         round(bw, 3), round(bh, 3)],
                "area": round(bw * bh, 3), "iscrowd": 0,
            })

    # labels.json regenerated FROM the cleaned .txt files so the two copies of
    # the ground truth cannot drift apart again.
    with open(os.path.join(out, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"images": coco_images, "annotations": coco_anns,
                   "categories": categories or
                   [{"id": 1, "name": "person", "supercategory": ""}]}, f)

    if meta:
        kept_names = {os.path.basename(r["rel_path"]) for r in kept}
        sessions = defaultdict(lambda: {"filenames": []})
        for name in sorted(kept_names):
            m = meta.get(name)
            if not m:
                continue
            blk = sessions[m["session"]]
            blk["filenames"].append(name)
            for k in ("location", "weather", "surface", "season"):
                blk[k] = m[k]
        with open(os.path.join(out, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(dict(sorted(sessions.items())), f, indent=1)

    with open(os.path.join(out, "manifest.csv"), "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=FIELDS)
        wri.writeheader()
        wri.writerows(rows)
    dropped = [r for r in rows if not r["keep"]]
    with open(os.path.join(out, "dropped.csv"), "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=FIELDS)
        wri.writeheader()
        wri.writerows(dropped)

    if args.splits:
        os.makedirs(os.path.join(out, "splits"), exist_ok=True)
        for name in ("train", "val", "test"):
            sel = [r for r in kept if r["split"] == name]
            if not sel:
                continue
            with open(os.path.join(out, "splits", f"{name}.txt"), "w") as f:
                for r in sel:
                    f.write(f'./images/{r["rel_path"]}\n')
        names = [n for n in ("train", "val", "test")
                 if any(r["split"] == n for r in kept)]
        with open(os.path.join(out, "dataset.yaml"), "w") as f:
            f.write("path: .\n")
            for n in names:
                f.write(f"{n}: splits/{n}.txt\n")
            f.write("names:\n")
            for cat in (categories or [{"id": 1, "name": "person"}]):
                f.write(f'  {cat["id"] - 1}: {cat["name"]}\n')

    return kept, dropped, link_kinds, boxes_written, boxes_dropped


def write_clean_report(path, rows, kept, dropped, link_kinds,
                       boxes_written, boxes_dropped, conflicts, args, split_counts):
    lines = []
    add = lines.append
    add("clean_AIResQ -- generated by validate_dataset.py")
    add(f"source images : {os.path.abspath(args.images)}")
    add(f"source labels : {os.path.abspath(args.labels) if args.labels else '(none)'}")
    add("the source tree was opened read-only; nothing in it was modified.")
    add("")
    add(f"input images  : {len(rows)}")
    add(f"kept          : {len(kept)}")
    add(f"dropped       : {len(dropped)}")
    add("")
    add("drop reasons:")
    for reason, n in Counter(r["drop_reason"].split("(")[0].split(":")[0]
                             for r in dropped).most_common():
        add(f"  {reason:<28} {n:>6}")
    add("")
    add(f"boxes written : {boxes_written}")
    add(f"boxes removed : {boxes_dropped} (degenerate / duplicate"
        f'{" / tiny" if args.drop_tiny_boxes else ""} after clamping)')
    add(f"dup groups with conflicting labels: {conflicts} "
        f"(policy: --dup-conflict {args.dup_conflict})")
    if args.drop_near_dups:
        add("near-duplicate removal: on (one frame kept per cluster)")
    else:
        add("near-duplicate removal: off -- clusters preserved; use the "
            "group_id column for leakage-safe splits")
    add(f"image materialisation: {dict(link_kinds)}")
    if link_kinds.get("hard"):
        add("  NOTE: hardlinks share one inode with the source. Deleting or")
        add("  adding files here is safe; editing an image in place would also")
        add("  change the AIResQ copy. Re-run with --link copy to decouple.")
    if split_counts:
        add(f"splits (group-aware, seed {args.seed}): {split_counts}")
    add("")
    add("layout:")
    add("  images/        kept frames")
    add("  labels/        YOLO .txt, clamped to frame, de-duplicated, sorted")
    add("  labels.json    COCO, regenerated from the cleaned .txt files")
    add("  metadata.json  session attributes, filtered to kept frames")
    add("  manifest.csv   every input row incl. keep/drop_reason/group_id")
    add("  dropped.csv    the dropped rows only")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def summarize(rows, pairs, out_prefix, blur_pct):
    n = len(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"\n{'='*64}\nAIResQ validation summary  --  {n} files\n{'='*64}")

    bad = Counter(r["status"] for r in rows if r["status"] != "ok")
    print(f"decoded cleanly      : {len(ok)}/{n}")
    for k, v in bad.items():
        print(f"  {k:<18} : {v}")

    res = Counter((r["width"], r["height"]) for r in ok)
    print("\nresolutions (top 6):")
    for (w, h), c in res.most_common(6):
        print(f"  {w}x{h:<8} {c:>6}")
    if len(res) > 6:
        print(f"  ...{len(res)-6} more distinct resolutions")

    print("\nIR verdict:")
    for k, v in Counter(r["ir_verdict"] for r in ok).most_common():
        print(f"  {k:<12} {v:>6}")

    depth = Counter(f'{r["mode"]}/{r["bit_depth"]}bit' for r in ok)
    print("\nmode/bit depth:")
    for k, v in depth.most_common(6):
        print(f"  {k:<14} {v:>6}")

    dupes = Counter(r["sha256"] for r in ok if r["sha256"])
    exact = sum(c - 1 for c in dupes.values() if c > 1)
    print(f"\nexact duplicates (sha256) : {exact} redundant files")
    print(f"near-duplicate pairs      : {len(pairs)}")

    if any(r["label_status"] for r in rows):
        print("\nlabels:")
        for k, v in Counter(r["label_status"] for r in rows if r["label_status"]).most_common():
            print(f"  {k:<12} {v:>6}")
        cls = Counter()
        for r in ok:
            if r["class_ids"]:
                for c in str(r["class_ids"]).split(","):
                    cls[c] += 1
        if cls:
            print(f"  images per class id: {dict(sorted(cls.items()))}")
        print(f"  total boxes: {sum(int(r['n_boxes'] or 0) for r in rows)}")

    if any(r["session"] for r in rows):
        sess = Counter(r["session"] for r in ok if r["session"])
        print(f"\nsessions: {len(sess)}  "
              f"(min {min(sess.values())} / max {max(sess.values())} frames)")
        for k in ("season", "weather", "surface"):
            print(f"  {k:<8} {dict(Counter(r[k] for r in ok if r[k]).most_common())}")

    cs = Counter(r["coco_status"] for r in rows if r["coco_status"])
    if cs and set(cs) != {"no-coco"}:
        print("\nlabels.json vs YOLO .txt:")
        for k, v in cs.most_common():
            print(f"  {k:<16} {v:>6}")

    lap = sorted(float(r["laplacian_var"]) for r in ok)
    if lap:
        cut = lap[max(0, int(len(lap) * blur_pct / 100) - 1)]
        print(f"\nblur: laplacian-var p{blur_pct} = {cut:.1f} "
              f"(median {lap[len(lap)//2]:.1f}) -- review the tail by hand")

    flagged = [r for r in rows if r["status"] != "ok"
               or r["ir_verdict"] in ("not_ir", "conflict", "uncertain")
               or r["label_status"] in ("missing", "invalid")]
    path = f"{out_prefix}_flagged.csv"
    with open(path, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=FIELDS)
        wri.writeheader()
        wri.writerows(flagged)
    print(f"\n{len(flagged)} files need a human look -> {path}")

    if pairs:
        p = f"{out_prefix}_nearpairs.csv"
        with open(p, "w", newline="") as f:
            wri = csv.writer(f)
            wri.writerow(["a", "b", "hamming"])
            wri.writerows(pairs)
        print(f"near-duplicate pairs -> {p}")


def main():
    ap = argparse.ArgumentParser(
        description="Validate and optionally clean the AIResQ image dataset.")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", default=None, help="YOLO .txt label dir (optional)")
    ap.add_argument("--classes", type=int, default=None,
                    help="default: derived from labels.json, else 2")
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--near-dup-hamming", type=int, default=4,
                    help="0 disables near-duplicate search")
    ap.add_argument("--min-box-frac", type=float, default=2e-6,
                    help="flag boxes smaller than this fraction of frame area")
    ap.add_argument("--blur-pct", type=int, default=5)
    # -- AIResQ sidecars: default to the files that sit next to the image dir
    ap.add_argument("--coco", default=None, help="labels.json (COCO ground truth)")
    ap.add_argument("--metadata", default=None, help="metadata.json (session attrs)")
    # -- cleaning
    ap.add_argument("--clean-out", default=None,
                    help="write a cleaned copy of the dataset to this NEW dir")
    ap.add_argument("--link", choices=("auto", "hard", "copy", "symlink"),
                    default="auto", help="how to materialise kept images")
    ap.add_argument("--dup-conflict", choices=("drop", "keep-first"), default="drop",
                    help="byte-identical images whose labels disagree")
    ap.add_argument("--drop-near-dups", action="store_true",
                    help="keep only one frame per near-duplicate cluster")
    ap.add_argument("--drop-tiny-boxes", action="store_true",
                    help="remove boxes below --min-box-frac instead of flagging")
    ap.add_argument("--splits", default="",
                    help='e.g. "0.8,0.1,0.1" -- group-aware train/val/test lists')
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    parent = os.path.dirname(os.path.abspath(args.images))
    for attr, default in (("coco", "labels.json"), ("metadata", "metadata.json")):
        if getattr(args, attr) is None:
            cand = os.path.join(parent, default)
            setattr(args, attr, cand if os.path.exists(cand) else None)

    meta = load_session_metadata(args.metadata)
    coco, categories = load_coco(args.coco)
    if args.classes is None:
        args.classes = max((c["id"] for c in categories), default=2)
    print(f"sidecars: metadata={len(meta)} filenames, "
          f"coco={len(coco)} images, classes={args.classes}")

    fractions = [float(x) for x in args.splits.split(",")] if args.splits else []
    if fractions and abs(sum(fractions) - 1.0) > 1e-6:
        sys.exit(f"--splits must sum to 1.0, got {sum(fractions)}")

    files = []
    for dirpath, _, names in os.walk(args.images):
        for nm in names:
            if os.path.splitext(nm)[1].lower() in IMG_EXTS:
                files.append(os.path.join(dirpath, nm))
    files.sort()
    if not files:
        sys.exit(f"no images found under {args.images}")
    print(f"scanning {len(files)} files with {args.workers} workers...")

    jobs = [(p, args.images, args.labels, args.classes, args.min_box_frac)
            for p in files]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, row in enumerate(ex.map(inspect, jobs, chunksize=16), 1):
            rows.append(row)
            if i % 500 == 0:
                print(f"  {i}/{len(files)}", flush=True)

    enrich_rows(rows, args.labels, meta, coco)

    pairs = near_duplicate_pairs(rows, args.near_dup_hamming) \
        if args.near_dup_hamming > 0 else []
    sha_groups, clusters = build_groups(rows, pairs)
    conflicts = choose_keepers(rows, sha_groups, clusters, args.labels, args)
    split_counts = assign_splits(rows, fractions, args.seed) if fractions else {}

    with open(args.out, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=FIELDS)
        wri.writeheader()
        wri.writerows(rows)
    print(f"manifest -> {args.out}")

    summarize(rows, pairs, os.path.splitext(args.out)[0], args.blur_pct)

    dropped = [r for r in rows if not r["keep"]]
    print(f"\ncleaning decision: keep {len(rows) - len(dropped)}, "
          f"drop {len(dropped)}")
    for reason, n in Counter(r["drop_reason"].split("(")[0].split(":")[0]
                             for r in dropped).most_common():
        print(f"  {reason:<28} {n:>6}")

    if args.clean_out:
        kept, dropped, kinds, bw, bd = write_clean_dataset(
            rows, args.labels, meta, categories, args)
        report = write_clean_report(
            os.path.join(args.clean_out, "clean_report.txt"), rows, kept, dropped,
            kinds, bw, bd, conflicts, args, split_counts)
        print(f"\n{'='*64}\n{report}")
        print(f"{'='*64}\nclean dataset -> {args.clean_out}")
    elif dropped:
        print("\n(re-run with --clean-out clean_AIResQ to emit the cleaned copy)")


if __name__ == "__main__":
    main()