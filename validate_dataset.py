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
import os
import sys
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageFile

# Sibling modules resolve relative to this file, not the caller's cwd (and not
# via sys.path[0], which Windows leaves empty for forward-slash script paths).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_unified import CrossDatasetValidator
from curate import (assign_splits, build_groups, choose_keepers, enrich_rows,
                    near_duplicate_pairs, quality_bins)
from image_stats import (QUALITY_KEYS, classify_ir, dhash64, quality_metrics, sha256_and_head,
                         to_uint8_gray)
from labels import (load_coco, load_session_metadata, load_yolo_names,
                    parse_yolo_label)
from writers import (FIELDS, write_clean_dataset, write_clean_report,
                     write_hituav_dataset)

# Fail loudly on truncated files instead of silently padding them with grey.
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = 200_000_000
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def inspect(job):
    path, root, labels_dir, n_classes, min_box_frac, *rest = job
    drop_tiny = rest[0] if rest else False
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
    row["clipped_low_frac"] = round(float((flat <= 0).mean()), 4)
    row["clipped_high_frac"] = round(float((flat >= dmax).mean()), 4)
    row.update(quality_metrics(arr))
    row["dhash"] = dhash64(gray)

    if labels_dir:
        lp = os.path.join(labels_dir, os.path.splitext(rel)[0] + ".txt")
        st, n, errs, cls = parse_yolo_label(lp, n_classes, min_box_frac, drop_tiny=drop_tiny)
        row["label_status"], row["n_boxes"] = st, n
        row["label_errors"], row["class_ids"] = errs, cls
    return row


def summarize(rows, pairs, out_prefix, blur_pct):
    n = len(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"\n{'='*64}\ndataset validation summary  --  {n} files\n{'='*64}")

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
    spans = defaultdict(set)
    for r in ok:
        if r["split"] and r["group_size"] and int(r["group_size"]) > 1:
            spans[r["group_id"]].add(r["split"])
    if spans:
        leaky = sum(len(s) > 1 for s in spans.values())
        print(f"duplicate groups crossing splits: {leaky}  (train/test leakage)")

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
        wri = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
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
                    help="default: derived from dataset.yaml or labels.json, else 2")
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
    ap.add_argument("--yaml", default=None, help="YOLO dataset.yaml (class names)")
    # -- cleaning
    ap.add_argument("--clean-out", default=None,
                    help="write a cleaned copy of the dataset to this NEW dir")
    ap.add_argument("--link", choices=("auto", "hard", "copy", "symlink"),
                    default="auto", help="how to materialise kept images")
    ap.add_argument("--dup-conflict", choices=("drop", "keep-first"), default="drop",
                    help="byte-identical images whose labels disagree")
    ap.add_argument("--drop-near-dups", action="store_true",
                    help="keep only one frame per near-duplicate cluster")
    ap.add_argument("--keep-classes", default="",
                    help='e.g. "person" -- cleaned copy keeps only these classes, '
                         "renumbered from 0; other boxes are removed")
    ap.add_argument("--drop-negatives", action="store_true",
                    help="drop images with no box of a kept class (background)")
    ap.add_argument("--max-bg-ratio", type=float, default=None,
                    help="cap empty background frames at this ratio of positive target frames")
    ap.add_argument("--layout", choices=("default", "hituav"), default="default",
                    help="hituav: the team's clean_HITUAV folder structure")
    ap.add_argument("--prefix", default="HU", help="renamed file prefix (--layout hituav)")
    ap.add_argument("--drop-tiny-boxes", action="store_true",
                    help="remove boxes below --min-box-frac instead of flagging")
    ap.add_argument("--splits", default="",
                    help='e.g. "0.8,0.1,0.1" -- group-aware train/val/test lists')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stratify", default="",
                    help='e.g. "laplacian_var,std" -- quantile-bin these quality '
                         "metrics and spread every bin evenly across --splits")
    ap.add_argument("--bins", type=int, default=3, help="quantile bins per --stratify metric")
    args = ap.parse_args()

    parent = os.path.dirname(os.path.abspath(args.images))
    for attr, default in (("coco", "labels.json"), ("metadata", "metadata.json"),
                          ("yaml", "dataset.yaml")):
        if getattr(args, attr) is None:
            cand = os.path.join(parent, default)
            setattr(args, attr, cand if os.path.exists(cand) else None)

    meta = load_session_metadata(args.metadata)
    coco, categories = load_coco(args.coco)
    class_names = load_yolo_names(args.yaml) or [c["name"] for c in categories]
    if args.classes is None:
        args.classes = len(class_names) or 2
    if len(class_names) < args.classes:
        class_names += [f"class{i}" for i in range(len(class_names), args.classes)]
    keep_ids = list(range(len(class_names)))
    if args.keep_classes:
        want = [w.strip().lower() for w in args.keep_classes.split(",") if w.strip()]
        lower = [n.lower() for n in class_names]
        unknown = [w for w in want if w not in lower]
        if unknown:
            sys.exit(f"--keep-classes {unknown} not in {class_names}")
        keep_ids = [lower.index(w) for w in want]
    # source class id -> id in the cleaned copy (renumbered from 0)
    args.class_map = {old: new for new, old in enumerate(keep_ids)}
    out_names = [class_names[i] for i in keep_ids]
    print(f"sidecars: metadata={len(meta)} filenames, coco={len(coco)} images, "
          f"classes={args.classes} {class_names}")

    fractions = [float(x) for x in args.splits.split(",")] if args.splits else []
    strat = [k.strip() for k in args.stratify.split(",") if k.strip()]
    if set(strat) - set(QUALITY_KEYS):
        sys.exit(f"--stratify {sorted(set(strat) - set(QUALITY_KEYS))} not in {QUALITY_KEYS}")
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

    jobs = [(p, args.images, args.labels, args.classes, args.min_box_frac, args.drop_tiny_boxes)
            for p in files]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, row in enumerate(ex.map(inspect, jobs, chunksize=16), 1):
            rows.append(row)
            if i % 500 == 0:
                print(f"  {i}/{len(files)}", flush=True)

    enrich_rows(rows, args.labels, meta, coco)
    # Datasets shipped pre-split (images/train/..., e.g. HIT-UAV) keep that
    # split unless --splits asks for a fresh one.
    for r in rows:
        top = r["rel_path"].split(os.sep)[0]
        if top in ("train", "val", "test"):
            r["split"] = top

    pairs = near_duplicate_pairs(rows, args.near_dup_hamming) \
        if args.near_dup_hamming > 0 else []
    sha_groups, clusters = build_groups(rows, pairs)
    conflicts = choose_keepers(rows, sha_groups, clusters, args.labels, args)
    if strat:
        quality_bins(rows, strat, args.bins)
    split_counts = assign_splits(rows, fractions, args.seed, class_map=args.class_map) \
        if fractions else dict(Counter(r["split"] for r in rows if r["keep"] and r["split"]))

    # Run the 4-stage Cross-Dataset Integrity Check over the curated manifest
    eff_bg_ratio = 0.0 if args.drop_negatives else args.max_bg_ratio
    validator = CrossDatasetValidator(
        n_classes=args.classes,
        min_box_frac=args.min_box_frac,
        max_bg_ratio=eff_bg_ratio,
    )
    integrity_report = validator.validate_curation(
        rows, labels_dir=args.labels, meta=meta, coco=coco,
        coco_path=args.coco, class_map=args.class_map
    )

    with open(args.out, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        wri.writeheader()
        wri.writerows(rows)
    print(f"manifest -> {args.out}")

    summarize(rows, pairs, os.path.splitext(args.out)[0], args.blur_pct)
    print(f"\n{integrity_report.summary()}")

    dropped = [r for r in rows if not r["keep"]]
    print(f"\ncleaning decision: keep {len(rows) - len(dropped)}, "
          f"drop {len(dropped)}")
    for reason, n in Counter(r["drop_reason"].split("(")[0].split(":")[0]
                             for r in dropped).most_common():
        print(f"  {reason:<28} {n:>6}")

    if args.clean_out and args.layout == "hituav":
        report = write_hituav_dataset(rows, args.labels, out_names, args)
        print(f"\n{report}\nclean dataset -> {args.clean_out}")
    elif args.clean_out:
        kept, dropped, kinds, bw, bd, bf, neg = write_clean_dataset(
            rows, args.labels, meta, out_names, args)
        report = write_clean_report(
            os.path.join(args.clean_out, "clean_report.txt"), rows, kept, dropped,
            kinds, bw, bd, bf, neg, conflicts, args, split_counts, out_names)
        print(f"\n{'='*64}\n{report}")
        print(f"{'='*64}\nclean dataset -> {args.clean_out}")
    elif dropped:
        print("\n(re-run with --clean-out clean_AIResQ to emit the cleaned copy)")


if __name__ == "__main__":
    main()
