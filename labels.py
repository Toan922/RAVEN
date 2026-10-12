"""Ground-truth handling: YOLO .txt parsing/cleaning and the AIResQ sidecars
(labels.json COCO, metadata.json sessions, dataset.yaml class names)."""

import json
import os


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
