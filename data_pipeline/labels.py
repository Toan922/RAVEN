"""Ground-truth handling: YOLO .txt parsing/cleaning and the AIResQ sidecars
(labels.json COCO, metadata.json sessions, dataset.yaml class names)."""

import json
import os


def parse_hituav_name(name):
    """HIT-UAV filenames are <time_of_day>_<altitude_m>_<camera_angle_deg>_
    <weather>_<index>.jpg; returns {} for anything else."""
    parts = os.path.splitext(name)[0].split("_")
    if len(parts) != 5 or not all(p.isdigit() for p in parts):
        return {}
    # Extract integer flight telemetry fields from the 5-part filename stem
    return dict(zip(("time_of_day", "altitude_m", "camera_angle_deg", "weather"),
                    map(int, parts[:4])))


def parse_yolo_label(label_path, n_classes, min_box_frac, drop_tiny=False):
    if not os.path.exists(label_path):
        return "missing", 0, "no label file", ""
    # Separate audit flags (errs) from fatal label defects (hard_errs)
    errs, hard_errs, boxes, cls = [], [], [], []
    with open(label_path, "r", errors="replace") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                msg = f"L{i}:fields={len(parts)}"
                errs.append(msg)
                hard_errs.append(msg)
                continue
            try:
                c = int(float(parts[0]))
                x, y, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                msg = f"L{i}:unparseable"
                errs.append(msg)
                hard_errs.append(msg)
                continue
            # Validate class ID falls within [0, n_classes)
            if not (0 <= c < n_classes):
                msg = f"L{i}:class={c}"
                errs.append(msg)
                hard_errs.append(msg)
            # Filter out zero-area / degenerate boxes
            if w <= 0 or h <= 0:
                msg = f"L{i}:degenerate"
                errs.append(msg)
                if not drop_tiny:
                    hard_errs.append(msg)
                continue
            # Ensure center and dimensions fall within normalized [0.0, 1.0] range
            if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                msg = f"L{i}:not-normalized"
                errs.append(msg)
                hard_errs.append(msg)
            # Ensure box corners do not extend outside the image frame
            if x - w / 2 < -1e-3 or x + w / 2 > 1 + 1e-3 or \
               y - h / 2 < -1e-3 or y + h / 2 > 1 + 1e-3:
                msg = f"L{i}:out-of-frame"
                errs.append(msg)
                hard_errs.append(msg)
            # Flag and optionally filter out microscopic box artifacts
            if w * h < min_box_frac:
                msg = f"L{i}:tiny({w*h:.2e})"
                errs.append(msg)
                if drop_tiny:
                    continue
                hard_errs.append(msg)
            boxes.append((c, round(x, 5), round(y, 5), round(w, 5), round(h, 5)))
            cls.append(c)
    # Flag duplicate bounding boxes within the same frame
    dupes = len(boxes) - len(set(boxes))
    if dupes:
        msg = f"duplicate-boxes={dupes}"
        errs.append(msg)
        if not drop_tiny:
            hard_errs.append(msg)
    if hard_errs:
        status = "invalid"
    elif not boxes:
        status = "empty"          # legitimate negative sample, but count it
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


def label_text(labels_dir, rel_path):
    p = os.path.join(labels_dir, os.path.splitext(rel_path)[0] + ".txt")
    if not os.path.exists(p):
        return None
    with open(p, "r", errors="replace") as f:
        return "\n".join(sorted(ln.strip() for ln in f if ln.strip()))


def clean_boxes(boxes, min_box_frac, drop_tiny):
    """Clamp to the frame, drop degenerate//tiny boxes, de-duplicate, sort."""
    out, dropped = [], 0
    for c, x, y, w, h in boxes:
        # Clamp box corners to [0.0, 1.0]
        x0, y0 = max(0.0, x - w / 2), max(0.0, y - h / 2)
        x1, y1 = min(1.0, x + w / 2), min(1.0, y + h / 2)
        nw, nh = x1 - x0, y1 - y0
        # Filter out zero-area boxes and (when enabled) microscopic artifacts
        if nw <= 0 or nh <= 0 or (drop_tiny and nw * nh < min_box_frac):
            dropped += 1
            continue
        out.append((c, round(x0 + nw / 2, 6), round(y0 + nh / 2, 6),
                    round(nw, 6), round(nh, 6)))
    # De-duplicate identical boxes in the same frame
    deduped = sorted(set(out))
    dropped += len(out) - len(deduped)
    return deduped, dropped


def write_yolo_txt(path, boxes):
    with open(path, "w", newline="\n") as f:
        for c, x, y, w, h in boxes:
            f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


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


def load_yolo_names(path):
    """Class names from a YOLO dataset.yaml, `names:` as an indented mapping
    or an inline list. Hand-parsed to avoid a PyYAML dependency."""
    if not path or not os.path.exists(path):
        return []
    names, in_names = {}, False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.split("#")[0].rstrip()
            if s.startswith("names:"):
                rest = s[len("names:"):].strip()
                if rest.startswith("["):
                    return [x.strip().strip("'\"") for x in rest.strip("[]").split(",")
                            if x.strip()]
                in_names = True
            elif in_names and s[:1] in (" ", "\t") and ":" in s:
                k, v = s.split(":", 1)
                names[int(k)] = v.strip().strip("'\"")
            elif s.strip():
                in_names = False
    return [names[k] for k in sorted(names)]


def load_coco(path):
    """labels.json is COCO detection format. Return filename -> record with
    boxes already converted to normalised YOLO (category ids remapped to
    0..n-1 in id order, so 0- and 1-based files both work), plus the
    category table."""
    if not path or not os.path.exists(path):
        return {}, []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cat_index = {c["id"]: i for i, c in
                 enumerate(sorted(raw.get("categories", []), key=lambda c: c["id"]))}
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
        rec["boxes"].append((cat_index.get(a["category_id"], int(a["category_id"]) - 1),
                             (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
    return {r["file_name"]: r for r in by_id.values()}, \
        sorted(raw.get("categories", []), key=lambda c: c["id"])
