"""Output emission: the manifest column schema and the two cleaned-dataset
layouts (default AIResQ-style and the team's clean_HITUAV), plus reports."""

import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict

from image_stats import quality_of
from labels import clean_boxes, read_yolo_boxes, write_yolo_txt

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
    "keep", "drop_reason", "quality_bin", "split",
]


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


def cleaned_label(r, labels_dir, args):
    """Source YOLO boxes -> kept classes only, renumbered, clamped, de-duped.
    Returns (boxes, n_other_class_removed, n_degenerate_removed)."""
    boxes = read_yolo_boxes(
        os.path.join(labels_dir, os.path.splitext(r["rel_path"])[0] + ".txt")
    ) if labels_dir else []
    n_in = len(boxes)
    boxes = [(args.class_map[c], *xywh) for c, *xywh in boxes if c in args.class_map]
    filtered = n_in - len(boxes)
    boxes, nd = clean_boxes(boxes, args.min_box_frac, args.drop_tiny_boxes)
    return boxes, filtered, nd


def check_out_dir(args):
    out, src_root = os.path.abspath(args.clean_out), os.path.abspath(args.images)
    if out == src_root or out.startswith(src_root + os.sep):
        sys.exit(f"--clean-out must be outside the source image tree ({src_root})")


def write_clean_dataset(rows, labels_dir, meta, class_names, args):
    out = args.clean_out
    check_out_dir(args)

    img_dir, lab_dir = os.path.join(out, "images"), os.path.join(out, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)

    kept = [r for r in rows if r["keep"]]
    kept.sort(key=lambda r: r["rel_path"])
    link_kinds = Counter()
    boxes_written = boxes_dropped = boxes_filtered = negatives = 0
    coco_images, coco_anns = [], []

    for img_id, r in enumerate(kept, 1):
        rel = r["rel_path"]
        dst = os.path.join(img_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        link_kinds[materialize(r["path"], dst, args.link)] += 1

        boxes, nf, nd = cleaned_label(r, labels_dir, args)
        boxes_filtered += nf
        boxes_dropped += nd
        boxes_written += len(boxes)
        negatives += not boxes

        lp = os.path.join(lab_dir, os.path.splitext(rel)[0] + ".txt")
        os.makedirs(os.path.dirname(lp), exist_ok=True)
        write_yolo_txt(lp, boxes)

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
                   "categories": [{"id": i + 1, "name": n, "supercategory": ""}
                                  for i, n in enumerate(class_names)]}, f)

    if meta:
        kept_by_name = {os.path.basename(r["rel_path"]): r for r in kept}
        sessions = defaultdict(lambda: {"filenames": [], "quality": {}})
        for name in sorted(kept_by_name):
            m = meta.get(name)
            if not m:
                continue
            blk = sessions[m["session"]]
            blk["filenames"].append(name)
            blk["quality"][name] = quality_of(kept_by_name[name])
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

    names = [n for n in ("train", "val", "test")
             if any(r["split"] == n for r in kept)]
    if names:
        # Lists live at the dataset root: Ultralytics resolves "./" in a list
        # file relative to that file's own directory.
        for name in names:
            with open(os.path.join(out, f"{name}.txt"), "w", newline="\n") as f:
                for r in kept:
                    if r["split"] == name:
                        f.write(f'./images/{r["rel_path"].replace(os.sep, "/")}\n')
        # Absolute path: Ultralytics resolves a relative `path:` against its
        # global datasets dir, not the yaml's location. Re-run to relocate.
        with open(os.path.join(out, "dataset.yaml"), "w", newline="\n") as f:
            f.write(f"path: {os.path.abspath(out)}\n")
            for n in names:
                f.write(f"{n}: {n}.txt\n")
            f.write(f"nc: {len(class_names)}\nnames:\n")
            for i, n in enumerate(class_names):
                f.write(f"  {i}: {n}\n")

    return kept, dropped, link_kinds, boxes_written, boxes_dropped, boxes_filtered, negatives


def parse_hituav_name(name):
    """HIT-UAV filenames are <time_of_day>_<altitude_m>_<camera_angle_deg>_
    <weather>_<index>.jpg; returns {} for anything else."""
    parts = os.path.splitext(name)[0].split("_")
    if len(parts) != 5 or not all(p.isdigit() for p in parts):
        return {}
    return dict(zip(("time_of_day", "altitude_m", "camera_angle_deg", "weather"),
                    map(int, parts[:4])))


def write_hituav_dataset(rows, labels_dir, class_names, args):
    """The team's clean_HITUAV layout: flat images/ + labels/ renamed
    HU00001.. (train, then val, then test, by original name), per-image
    metadata.json, splits/*.txt, dataset.yaml, dropped.csv, clean_report.txt."""
    out = args.clean_out
    check_out_dir(args)
    for d in ("images", "labels", "splits"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    order = {"train": 0, "val": 1, "test": 2}
    kept = sorted((r for r in rows if r["keep"]),
                  key=lambda r: (order.get(r["split"], 3), os.path.basename(r["rel_path"])))
    if any(not r["split"] for r in kept):
        sys.exit("--layout hituav needs splits: images/{train,val,test}/ or --splits")

    meta, lists = {}, defaultdict(list)
    n_boxes = n_filtered = 0
    for i, r in enumerate(kept, 1):
        stem = f"{args.prefix}{i:05d}"
        name = stem + os.path.splitext(r["rel_path"])[1].lower()
        dst = os.path.join(out, "images", name)
        materialize(r["path"], dst, args.link)
        boxes, nf, _ = cleaned_label(r, labels_dir, args)
        n_boxes += len(boxes)
        n_filtered += nf
        write_yolo_txt(os.path.join(out, "labels", stem + ".txt"), boxes)
        orig = os.path.basename(r["rel_path"])
        # shared core keys first, then whatever the source knows about the frame:
        # AIResQ session attrs (joined in enrich_rows) or HIT-UAV filename fields
        session = {k: r[k] for k in ("session", "location", "weather", "surface", "season")
                   if r.get(k)}
        meta[name] = {"original_filename": orig, "source": args.prefix,
                      "split": r["split"], "is_background": not boxes,
                      **session, **parse_hituav_name(orig), **quality_of(r),
                      "quality_bin": r["quality_bin"]}
        # absolute: Ultralytics resolves "./x" in a list file against the
        # list's own folder (splits/), so the old ./images/.. lines broke
        lists[r["split"]].append(os.path.abspath(dst).replace(os.sep, "/"))

    with open(os.path.join(out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    for split, paths in lists.items():
        with open(os.path.join(out, "splits", f"{split}.txt"), "w", newline="\n") as f:
            f.write("\n".join(paths) + "\n")
    with open(os.path.join(out, "dataset.yaml"), "w", newline="\n") as f:
        f.write(f"path: {os.path.abspath(out)}\n")
        for split in sorted(lists, key=order.get):
            f.write(f"{split}: splits/{split}.txt\n")
        f.write("\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(class_names)))

    dropped = [r for r in rows if not r["keep"]]
    with open(os.path.join(out, "dropped.csv"), "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["Original_Filename", "Split", "Reason"])
        for r in sorted(dropped, key=lambda r: (order.get(r["split"], 3), r["rel_path"])):
            wri.writerow([os.path.basename(r["rel_path"]), r["split"],
                          f'Dropped: {r["drop_reason"]}'])

    # other-class boxes across the whole source, not just the kept frames
    n_filtered += sum(cleaned_label(r, labels_dir, args)[1]
                      for r in dropped if r["status"] == "ok")
    reasons = Counter(r["drop_reason"].split("(")[0].split(":")[0] for r in dropped)
    sizes = Counter(r["split"] for r in kept)
    bar = "=" * 52
    title = os.path.basename(os.path.normpath(out)).upper()
    lines = [bar, f"      {title} DATASET PREPROCESSING REPORT",
             f"      Generated: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M:%S}",
             bar, "", "OVERVIEW",
             f"Total Original Images Processed: {len(rows)}",
             f"Total Images Retained:           {len(kept)}",
             f"Retention Rate:                  {100 * len(kept) / max(1, len(rows)):.2f}%",
             "", "ANNOTATION METRICS",
             f"Target {'/'.join(map(repr, class_names))} Bounding Boxes Kept:  {n_boxes}",
             f"Other-Class Annotations Dropped: {n_filtered}",
             "", "DROPPED IMAGES"]
    lines += [f"{k:<32} {v}" for k, v in reasons.most_common()] or ["none"]
    lines += ["", "SPLIT DISTRIBUTION",
              f"Training Set:   {sizes['train']} images",
              f"Validation Set: {sizes['val']} images",
              f"Test Set:       {sizes['test']} images"]
    report = "\n".join(lines)
    with open(os.path.join(out, "clean_report.txt"), "w") as f:
        f.write(report + "\n")
    return report


def write_clean_report(path, rows, kept, dropped, link_kinds, boxes_written,
                       boxes_dropped, boxes_filtered, negatives, conflicts, args,
                       split_counts, out_names):
    lines = []
    add = lines.append
    add(f"{os.path.basename(os.path.abspath(args.clean_out))} -- generated by validate_dataset.py")
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
    if args.keep_classes:
        add(f"classes kept  : {out_names} -- {boxes_filtered} other-class boxes removed")
    add(f"images with no boxes (negatives): {negatives}")
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
        how = f"group-aware, seed {args.seed}" if args.splits else "from source folders"
        add(f"splits ({how}): {split_counts}")
    add("")
    add("layout:")
    add("  images/        kept frames")
    add("  labels/        YOLO .txt, clamped to frame, de-duplicated, sorted")
    add("  labels.json    COCO, regenerated from the cleaned .txt files")
    add("  metadata.json  session attributes + per-frame quality metrics "
        "(laplacian_var/std/entropy/mean), kept frames only")
    add("  manifest.csv   every input row incl. keep/drop_reason/group_id")
    add("  dropped.csv    the dropped rows only")
    if split_counts:
        add("  {train,val,test}.txt + dataset.yaml   YOLO/Ultralytics entry point")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)
