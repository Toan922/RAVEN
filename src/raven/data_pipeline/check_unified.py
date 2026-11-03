"""Cross-Dataset Integrity Check for the RAVEN SAR data pipeline.

    python -m raven.data_pipeline.check_unified raven.yaml
    python -m raven.data_pipeline.check_unified raven.yaml --max-bg-ratio 0.25 --check-hashes
"""

import argparse
import json
import math
import os
from collections import defaultdict

from raven.data_pipeline.curate import is_positive_row
from raven.data_pipeline.image_stats import QUALITY_KEYS, sha256_and_head
from raven.data_pipeline.labels import clean_boxes, load_yolo_names, parse_hituav_name

# raven.yaml lives at the repo root, three levels above this folder (src/raven/data_pipeline)
DEFAULT_YAML = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "raven.yaml")

# Shared core + quality fields required on every frame in metadata.json
CORE_SCHEMA = {
    "original_filename": str, "source": str, "split": str, "is_background": bool,
    "laplacian_var": (int, float), "std": (int, float),
    "entropy": (int, float), "mean": (int, float),
}
CORE = tuple(CORE_SCHEMA.keys())

# Source-specific telemetry schemas for AIResQ sessions and HIT-UAV flight logs
AIRESQ_TELEMETRY_SCHEMA = {
    "session": str, "location": str, "weather": str, "surface": str, "season": str,
}
HITUAV_TELEMETRY_SCHEMA = {
    "time_of_day": int, "altitude_m": int, "camera_angle_deg": int, "weather": int,
}


def _is_valid_type(val, expected):
    """Strict type check that rejects bool masquerading as int or float."""
    if expected is bool:
        return isinstance(val, bool)
    return not isinstance(val, bool) and isinstance(val, expected)


def split_lists(yaml_path):
    """Parse `{split: [list files]}` from a merged YOLO yaml."""
    base_dir = os.path.dirname(os.path.abspath(yaml_path))
    root, out = base_dir, {}
    with open(yaml_path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line.startswith("path:"):
                cand = line[5:].strip()
                # Fall back to the yaml's own folder if path points to another machine
                root = cand if os.path.exists(cand) else base_dir
            for s in ("train", "val", "test"):
                if line.startswith(s + ":"):
                    items = [p.strip() for p in line[len(s) + 1:].strip(" []").split(",") if p.strip()]
                    resolved = []
                    for p in items:
                        cand = os.path.join(root, p)
                        if not os.path.exists(cand) and os.path.exists(os.path.join(base_dir, p)):
                            cand = os.path.join(base_dir, p)
                        resolved.append(cand)
                    out[s] = resolved
    return out


class IntegrityReport:
    """Structured results from CrossDatasetValidator across the 4 integrity pillars."""

    def __init__(self):
        self.orphan_errors = []
        self.telemetry_errors = []
        self.box_errors = []
        self.split_ratio_errors = []
        self.stats = {
            "total_frames": 0, "positive_frames": 0, "background_frames": 0,
            "bg_to_pos_ratio": 0.0, "boxes_validated": 0, "zero_area_flagged": 0,
            "microscopic_flagged": 0, "out_of_range_flagged": 0, "orphan_images": 0,
            "orphan_annotations": 0, "orphan_metadata": 0, "split_counts": {},
            "split_bg_ratios": {}, "leaky_groups": 0,
        }

    @property
    def ok(self):
        return not (self.orphan_errors or self.telemetry_errors
                    or self.box_errors or self.split_ratio_errors)

    @property
    def errors(self):
        return (self.orphan_errors + self.telemetry_errors
                + self.box_errors + self.split_ratio_errors)

    def summary(self):
        s = self.stats
        lines = [
            f"Cross-Dataset Integrity Check: {'PASSED' if self.ok else 'FAILED'}",
            f"  1. Orphan Detection       : {len(self.orphan_errors)} errors "
            f"(missing_labels={s['orphan_images']}, orphan_labels={s['orphan_annotations']}, "
            f"orphan_meta={s['orphan_metadata']})",
            f"  2. Telemetry Sync/Schema  : {len(self.telemetry_errors)} errors",
            f"  3. Bounding Box Geometry  : {len(self.box_errors)} errors "
            f"(boxes={s['boxes_validated']}, zero_area={s['zero_area_flagged']}, "
            f"microscopic={s['microscopic_flagged']}, out_of_range={s['out_of_range_flagged']})",
            f"  4. Splits & BG Ratio Cap  : {len(self.split_ratio_errors)} errors "
            f"(pos={s['positive_frames']}, bg={s['background_frames']}, "
            f"ratio={s['bg_to_pos_ratio']:.4f}, leaky_groups={s['leaky_groups']})",
        ]
        if s["split_counts"]:
            lines.append(f"     Split distribution     : {s['split_counts']}")
        if s["split_bg_ratios"]:
            lines.append(f"     Split BG/Pos ratios    : {s['split_bg_ratios']}")
        for cat, errs in (("Orphan", self.orphan_errors), ("Telemetry", self.telemetry_errors),
                          ("BoundingBox", self.box_errors), ("Split/Ratio", self.split_ratio_errors)):
            lines.extend(f"  [{cat}] {e}" for e in errs[:10])
            if len(errs) > 10:
                lines.append(f"  [{cat}] ... and {len(errs) - 10} more")
        return "\n".join(lines)

    def raise_for_errors(self):
        if not self.ok:
            raise AssertionError(self.summary())


class CrossDatasetValidator:
    """Cross-Dataset Integrity Check for the RAVEN SAR data pipeline."""

    def __init__(self, n_classes=1, min_box_frac=2e-6, max_bg_ratio=None,
                 require_telemetry=True):
        self.n_classes = n_classes
        self.min_box_frac = min_box_frac
        self.max_bg_ratio = max_bg_ratio
        self.require_telemetry = require_telemetry

    # 1. Orphan Record Detection
    def check_orphans(self, image_stems, label_stems, meta_names=None,
                      coco_names=None, image_names=None, split_names=None,
                      coco_orphan_anns=0, report=None):
        rep = report or IntegrityReport()
        img_stems, lbl_stems = set(image_stems), set(label_stems)
        # Flag images without annotations and annotations pointing to missing images
        missing_labels = sorted(img_stems - lbl_stems)
        orphan_labels = sorted(lbl_stems - img_stems)
        rep.stats["orphan_images"] += len(missing_labels)
        rep.stats["orphan_annotations"] += len(orphan_labels) + int(coco_orphan_anns)
        rep.orphan_errors.extend(f"image missing annotation file: {st}" for st in missing_labels)
        rep.orphan_errors.extend(f"orphan annotation pointing to missing image: {st}.txt" for st in orphan_labels)
        if coco_orphan_anns:
            rep.orphan_errors.append(f"COCO labels.json has {coco_orphan_anns} annotations with missing image_id")

        # Flag mismatches against COCO labels.json, metadata.json, and split lists
        names_set = set(image_names) if image_names is not None else img_stems
        if coco_names is not None:
            coco_set = set(coco_names)
            for fn in sorted(names_set - coco_set):
                rep.stats["orphan_images"] += 1
                rep.orphan_errors.append(f"image missing from COCO labels.json: {fn}")
            for fn in sorted(coco_set - names_set):
                rep.stats["orphan_annotations"] += 1
                rep.orphan_errors.append(f"orphan COCO labels.json entry pointing to missing image: {fn}")

        if meta_names is not None:
            meta_set = set(meta_names)
            for fn in sorted(names_set - meta_set):
                rep.stats["orphan_metadata"] += 1
                rep.orphan_errors.append(f"image missing from metadata.json: {fn}")
            for fn in sorted(meta_set - names_set):
                rep.stats["orphan_metadata"] += 1
                rep.orphan_errors.append(f"orphan metadata.json entry pointing to missing image: {fn}")

        if split_names is not None:
            split_set = set(split_names)
            for fn in sorted(split_set - names_set):
                rep.stats["orphan_images"] += 1
                rep.orphan_errors.append(f"split list entry points to missing image: {fn}")
            for fn in sorted(names_set - split_set):
                rep.stats["orphan_images"] += 1
                rep.orphan_errors.append(f"unreferenced image on disk not in any split: {fn}")
        return rep

    @staticmethod
    def scan_label_stems(labels_dir):
        """Collect all .txt label stems on disk (ignoring classes.txt)."""
        if not labels_dir or not os.path.isdir(labels_dir):
            return set()
        return {
            os.path.splitext(os.path.relpath(os.path.join(dp, nm), labels_dir))[0]
            for dp, _, files in os.walk(labels_dir)
            for nm in files if nm.lower().endswith(".txt") and nm.lower() != "classes.txt"
        }

    @staticmethod
    def scan_coco_orphans(coco_path):
        """Count COCO annotations whose image_id does not exist in COCO images."""
        if not coco_path or not os.path.exists(coco_path):
            return 0
        raw = json.load(open(coco_path, "r", encoding="utf-8"))
        valid_ids = {im["id"] for im in raw.get("images", []) if "id" in im}
        return sum(1 for a in raw.get("annotations", []) if a.get("image_id") not in valid_ids)

    # 2. Telemetry Synchronization & Schema Validation
    def check_telemetry_record(self, name, rec, expected_split=None,
                               actual_is_background=None, report=None):
        rep = report or IntegrityReport()
        if not isinstance(rec, dict):
            rep.telemetry_errors.append(f"{name}: telemetry record is not a dict")
            return rep

        # Validate shared core fields, strict types, and numeric bounds
        for key, expected_type in CORE_SCHEMA.items():
            if key not in rec or rec[key] is None or rec[key] == "":
                rep.telemetry_errors.append(f"{name}: missing core field '{key}'")
                continue
            val = rec[key]
            if not _is_valid_type(val, expected_type):
                rep.telemetry_errors.append(f"{name}: field '{key}' has invalid type {type(val).__name__}")
                continue
            if expected_type is str and not val.strip():
                rep.telemetry_errors.append(f"{name}: field '{key}' is empty")
            elif expected_type == (int, float):
                fval = float(val)
                if not math.isfinite(fval):
                    rep.telemetry_errors.append(f"{name}: field '{key}' is not finite ({val})")
                elif key == "mean" and not (0.0 <= fval <= 255.0):
                    rep.telemetry_errors.append(f"{name}: 'mean' out of [0, 255] range ({fval})")
                elif key in ("laplacian_var", "std", "entropy") and fval < 0.0:
                    rep.telemetry_errors.append(f"{name}: '{key}' cannot be negative ({fval})")

        # Ensure metadata split and background flag match the split file and label boxes
        split_val = rec.get("split")
        if isinstance(split_val, str) and split_val not in ("train", "val", "test"):
            rep.telemetry_errors.append(f"{name}: invalid split '{split_val}'")
        if expected_split is not None and split_val != expected_split:
            rep.telemetry_errors.append(f"{name}: split mismatch ({split_val} != {expected_split})")
        if actual_is_background is not None and rec.get("is_background") != actual_is_background:
            rep.telemetry_errors.append(f"{name}: is_background={rec.get('is_background')} != {actual_is_background}")

        # Validate source-specific telemetry fields (HIT-UAV flight vs AIResQ session)
        orig, src = str(rec.get("original_filename", "")), str(rec.get("source", "")).upper()
        hu_from_name = parse_hituav_name(orig) if orig else {}
        has_hu_keys = any(k in rec for k in ("time_of_day", "altitude_m", "camera_angle_deg"))
        has_ar_keys = any(k in rec for k in ("session", "location", "surface", "season"))

        if hu_from_name or has_hu_keys or (src in ("HU", "HITUAV", "HIT-UAV") and not has_ar_keys):
            for key, expected_type in HITUAV_TELEMETRY_SCHEMA.items():
                if key not in rec or rec[key] is None or rec[key] == "":
                    rep.telemetry_errors.append(f"{name}: missing HIT-UAV telemetry '{key}'")
                    continue
                val = rec[key]
                if not _is_valid_type(val, expected_type):
                    rep.telemetry_errors.append(f"{name}: HIT-UAV '{key}' must be int, got {type(val).__name__}")
                    continue
                if key == "altitude_m" and val <= 0:
                    rep.telemetry_errors.append(f"{name}: invalid altitude_m={val} (must be > 0)")
                elif key == "camera_angle_deg" and not (0 <= val <= 90):
                    rep.telemetry_errors.append(f"{name}: invalid camera_angle_deg={val} (must be in [0, 90])")
                elif key in ("time_of_day", "weather") and val < 0:
                    rep.telemetry_errors.append(f"{name}: invalid {key}={val} (must be >= 0)")
                # Cross-check telemetry values against the original HIT-UAV filename
                if hu_from_name and key in hu_from_name and val != hu_from_name[key]:
                    rep.telemetry_errors.append(f"{name}: telemetry '{key}'={val} != filename '{orig}'")
        elif has_ar_keys or src in ("AR", "AIRESQ"):
            for key, expected_type in AIRESQ_TELEMETRY_SCHEMA.items():
                val = rec.get(key)
                if val is None or not _is_valid_type(val, expected_type) or not val.strip():
                    rep.telemetry_errors.append(f"{name}: AIResQ telemetry '{key}' must be non-empty str")
                elif key == "session" and val.strip().upper() == "UNKNOWN":
                    rep.telemetry_errors.append(f"{name}: AIResQ session is 'UNKNOWN'")
        elif self.require_telemetry:
            rep.telemetry_errors.append(f"{name}: missing source telemetry fields")
        return rep

    # 3. Bounding Box Coordinate Validation & Filtering
    def validate_and_filter_boxes(self, boxes_or_path, n_classes=None,
                                  min_box_frac=None, name="", report=None):
        rep = report or IntegrityReport()
        nc = self.n_classes if n_classes is None else n_classes
        min_frac = self.min_box_frac if min_box_frac is None else min_box_frac
        prefix = f"{name}: " if name else ""
        issues, raw_boxes = [], []

        # Parse raw YOLO lines from a .txt file path or an in-memory sequence
        if isinstance(boxes_or_path, (str, os.PathLike)):
            if not os.path.exists(boxes_or_path):
                msg = f"{prefix}missing label file {boxes_or_path}"
                issues.append(msg)
                rep.box_errors.append(msg)
                return [], issues
            with open(boxes_or_path, "r", errors="replace") as f:
                for line_no, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if len(parts) != 5:
                        msg = f"{prefix}L{line_no}: expected 5 fields, got {len(parts)}"
                        issues.append(msg)
                        rep.box_errors.append(msg)
                        continue
                    try:
                        c_raw = float(parts[0])
                        c = int(c_raw)
                        if c != c_raw:
                            raise ValueError("non-integer class id")
                        raw_boxes.append((line_no, c, *(float(v) for v in parts[1:5])))
                    except ValueError:
                        msg = f"{prefix}L{line_no}: non-numeric box tokens '{line.strip()}'"
                        issues.append(msg)
                        rep.box_errors.append(msg)
        else:
            for line_no, box in enumerate(boxes_or_path, 1):
                if len(box) < 5:
                    msg = f"{prefix}L{line_no}: expected 5 elements, got {len(box)}"
                    issues.append(msg)
                    rep.box_errors.append(msg)
                else:
                    raw_boxes.append((line_no, int(box[0]), *(float(v) for v in box[1:5])))

        valid_boxes, seen = [], set()
        for line_no, c, x, y, w, h in raw_boxes:
            rep.stats["boxes_validated"] += 1
            box_ok = True
            # Check class ID bounds
            if not (0 <= c < nc):
                msg = f"{prefix}L{line_no}: class id {c} outside [0, {nc})"
                issues.append(msg)
                rep.box_errors.append(msg)
                box_ok = False
            # Reject NaN or infinite coordinates
            if not all(math.isfinite(v) for v in (x, y, w, h)):
                msg = f"{prefix}L{line_no}: non-finite coordinates ({x}, {y}, {w}, {h})"
                issues.append(msg)
                rep.box_errors.append(msg)
                continue
            # Filter out zero-area / degenerate boxes
            if w <= 0.0 or h <= 0.0:
                rep.stats["zero_area_flagged"] += 1
                msg = f"{prefix}L{line_no}: zero-area/degenerate box (w={w}, h={h})"
                issues.append(msg)
                rep.box_errors.append(msg)
                continue
            # Validate normalized [0, 1] coordinates and in-frame box corners
            if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                rep.stats["out_of_range_flagged"] += 1
                msg = f"{prefix}L{line_no}: coordinates outside normalized [0, 1]"
                issues.append(msg)
                rep.box_errors.append(msg)
                box_ok = False
            elif (x - w / 2 < -1e-3 or x + w / 2 > 1.0 + 1e-3
                  or y - h / 2 < -1e-3 or y + h / 2 > 1.0 + 1e-3):
                rep.stats["out_of_range_flagged"] += 1
                msg = f"{prefix}L{line_no}: box corners extend outside [0, 1] frame"
                issues.append(msg)
                rep.box_errors.append(msg)
                box_ok = False
            # Filter out microscopic artifacts below min_box_frac
            if w * h < min_frac:
                rep.stats["microscopic_flagged"] += 1
                msg = f"{prefix}L{line_no}: microscopic artifact (area={w*h:.2e} < {min_frac:.2e})"
                issues.append(msg)
                rep.box_errors.append(msg)
                continue
            if not box_ok:
                continue
            # Clamp valid box and filter out intra-frame duplicates
            clamped, _ = clean_boxes([(c, x, y, w, h)], min_frac, drop_tiny=True)
            if not clamped:
                rep.stats["zero_area_flagged"] += 1
                continue
            norm_box = clamped[0]
            if norm_box in seen:
                msg = f"{prefix}L{line_no}: duplicate bounding box {norm_box}"
                issues.append(msg)
                rep.box_errors.append(msg)
                continue
            seen.add(norm_box)
            valid_boxes.append(norm_box)
        return valid_boxes, issues

    # 4. Background-to-Positive Ratio & Cross-Split Leakage Validation
    def check_splits_and_ratios(self, records, report=None):
        rep = report or IntegrityReport()
        rep.stats["total_frames"] = len(records)
        pos_total = sum(1 for r in records if not r.get("is_background", False))
        bg_total = sum(1 for r in records if r.get("is_background", False))
        rep.stats["positive_frames"], rep.stats["background_frames"] = pos_total, bg_total
        ratio = bg_total / pos_total if pos_total > 0 else (float("inf") if bg_total > 0 else 0.0)
        rep.stats["bg_to_pos_ratio"] = round(ratio, 6) if math.isfinite(ratio) else ratio

        # Verify overall background-to-positive ratio respects max_bg_ratio
        if self.max_bg_ratio is not None and bg_total > int(self.max_bg_ratio * pos_total):
            rep.split_ratio_errors.append(
                f"overall background ratio {bg_total}/{pos_total} ({ratio:.4f}) > cap {self.max_bg_ratio:.4f}"
            )

        by_split = defaultdict(list)
        seen_name, seen_orig, seen_sha = {}, {}, {}
        group_splits = defaultdict(set)

        # Check for data leakage across splits (filenames, source records, SHA-256, group_id)
        for r in records:
            sp = r.get("split", "")
            if not sp:
                continue
            by_split[sp].append(r)
            name = r.get("name") or os.path.basename(r.get("rel_path", ""))
            if name:
                if name in seen_name and seen_name[name] != sp:
                    rep.stats["leaky_groups"] += 1
                    rep.split_ratio_errors.append(f"filename leakage: '{name}' in '{seen_name[name]}' and '{sp}'")
                seen_name[name] = sp
            orig, src = r.get("original_filename", ""), r.get("source", "")
            if orig:
                key = (src, orig)
                if key in seen_orig and seen_orig[key] != sp:
                    rep.stats["leaky_groups"] += 1
                    rep.split_ratio_errors.append(f"source leakage: {key} in '{seen_orig[key]}' and '{sp}'")
                seen_orig[key] = sp
            sha = r.get("sha256", "")
            if sha:
                if sha in seen_sha and seen_sha[sha] != sp:
                    rep.stats["leaky_groups"] += 1
                    rep.split_ratio_errors.append(f"SHA-256 leakage: {sha[:12]} in '{seen_sha[sha]}' and '{sp}'")
                seen_sha[sha] = sp
            gid = r.get("group_id", "")
            if gid != "" and gid is not None:
                group_splits[gid].add(sp)

        for gid, splits in sorted(group_splits.items(), key=lambda kv: str(kv[0])):
            if len(splits) > 1:
                rep.stats["leaky_groups"] += 1
                rep.split_ratio_errors.append(f"group_id={gid} leaks across splits: {sorted(splits)}")

        # Verify per-split background-to-positive ratios
        for sp, s_recs in sorted(by_split.items()):
            s_pos = sum(1 for r in s_recs if not r.get("is_background", False))
            s_bg = sum(1 for r in s_recs if r.get("is_background", False))
            rep.stats["split_counts"][sp] = len(s_recs)
            s_ratio = s_bg / s_pos if s_pos > 0 else (float("inf") if s_bg > 0 else 0.0)
            rep.stats["split_bg_ratios"][sp] = round(s_ratio, 4) if math.isfinite(s_ratio) else s_ratio
            if self.max_bg_ratio is not None:
                s_limit = 0 if self.max_bg_ratio == 0 else math.ceil(self.max_bg_ratio * s_pos)
                if s_bg > s_limit:
                    rep.split_ratio_errors.append(
                        f"split '{sp}' background ratio {s_bg}/{s_pos} ({s_ratio:.4f}) > cap {self.max_bg_ratio:.4f}"
                    )
        return rep

    def validate_curation(self, rows, labels_dir=None, meta=None, coco=None,
                          coco_path=None, class_map=None):
        """Run all 4 integrity checks over a curated manifest and its sidecars."""
        rep = IntegrityReport()
        kept = [r for r in rows if r.get("keep")]
        all_stems = {os.path.splitext(r["rel_path"])[0] for r in rows}
        all_names = {os.path.basename(r["rel_path"]) for r in rows}

        # Check for reverse-orphan label files, COCO entries, and metadata keys
        if labels_dir and os.path.isdir(labels_dir):
            orphan_lbls = sorted(self.scan_label_stems(labels_dir) - all_stems)
            rep.stats["orphan_annotations"] += len(orphan_lbls)
            rep.orphan_errors.extend(f"orphan annotation pointing to missing image: {st}.txt" for st in orphan_lbls)
        for r in kept:
            if labels_dir and r.get("label_status") == "missing":
                rep.stats["orphan_images"] += 1
                rep.orphan_errors.append(f"kept image missing annotation file: {r['rel_path']}")

        if coco:
            orphan_coco = sorted(set(coco.keys()) - all_names)
            rep.stats["orphan_annotations"] += len(orphan_coco)
            rep.orphan_errors.extend(f"orphan COCO entry pointing to missing image: {fn}" for fn in orphan_coco)
        if coco_path:
            n_orphan_anns = self.scan_coco_orphans(coco_path)
            if n_orphan_anns:
                rep.stats["orphan_annotations"] += n_orphan_anns
                rep.orphan_errors.append(f"COCO labels.json has {n_orphan_anns} annotations with missing image_id")
        if meta:
            orphan_meta = sorted(set(meta.keys()) - all_names)
            rep.stats["orphan_metadata"] += len(orphan_meta)
            rep.orphan_errors.extend(f"orphan metadata.json entry pointing to missing image: {fn}" for fn in orphan_meta)

        # Validate telemetry and box status across all kept frames
        split_records = []
        for r in kept:
            name = os.path.basename(r["rel_path"])
            is_pos = is_positive_row(r, class_map)
            split_val = r.get("split") or "train"
            hu = parse_hituav_name(name)
            session_fields = {k: r[k] for k in ("session", "location", "weather", "surface", "season") if r.get(k)}
            rec = {
                "original_filename": name,
                "source": "AR" if session_fields else ("HU" if hu else "SRC"),
                "split": split_val, "is_background": not is_pos,
                **{k: float(r[k]) for k in QUALITY_KEYS if r.get(k) != "" and r.get(k) is not None},
                **session_fields, **hu,
            }
            prev_req = self.require_telemetry
            self.require_telemetry = prev_req and bool(meta or hu)
            self.check_telemetry_record(name, rec, expected_split=split_val, report=rep)
            self.require_telemetry = prev_req

            if r.get("label_status") == "invalid":
                rep.box_errors.append(f"{r['rel_path']}: invalid label ({r.get('label_errors', '')})")
            rep.stats["boxes_validated"] += int(r.get("n_boxes") or 0)
            split_records.append({
                "name": r["rel_path"], "rel_path": r["rel_path"], "original_filename": name,
                "source": rec["source"], "split": r.get("split", ""), "is_background": not is_pos,
                "sha256": r.get("sha256", ""), "group_id": r.get("group_id", ""),
            })

        # Tally box anomalies flagged across the entire input manifest
        for r in rows:
            errs_str = str(r.get("label_errors", ""))
            rep.stats["zero_area_flagged"] += errs_str.count("degenerate")
            rep.stats["microscopic_flagged"] += errs_str.count("tiny(")
            rep.stats["out_of_range_flagged"] += errs_str.count("not-normalized") + errs_str.count("out-of-frame")

        self.check_splits_and_ratios(split_records, report=rep)
        return rep

    def validate_unified(self, yaml_path, check_hashes=False):
        """Run all 4 integrity checks across a merged dataset YAML (e.g. raven.yaml)."""
        rep = IntegrityReport()
        names = load_yolo_names(yaml_path)
        nc = len(names) if names else self.n_classes
        splits = split_lists(yaml_path)
        if not splits:
            rep.orphan_errors.append(f"no train/val/test splits found in {yaml_path}")
            return rep

        ds_split_images = defaultdict(set)
        ds_metas, split_records = {}, []

        # Walk every split list file and verify images, labels, and metadata entries
        for split, list_files in splits.items():
            for lst in list_files:
                if not os.path.exists(lst):
                    rep.orphan_errors.append(f"missing split list file: {lst}")
                    continue
                ds = os.path.dirname(os.path.dirname(os.path.abspath(lst)))
                if os.path.basename(os.path.dirname(os.path.abspath(lst))) != "splits":
                    ds = os.path.dirname(os.path.abspath(lst))

                if ds not in ds_metas:
                    meta_path = os.path.join(ds, "metadata.json")
                    if os.path.exists(meta_path):
                        ds_metas[ds] = json.load(open(meta_path, "r", encoding="utf-8"))
                    else:
                        ds_metas[ds] = {}
                        rep.orphan_errors.append(f"missing metadata.json in {ds}")
                meta = ds_metas[ds]

                for img_entry in (ln.strip() for ln in open(lst, "r", encoding="utf-8") if ln.strip()):
                    img_path = img_entry if os.path.isabs(img_entry) else os.path.normpath(os.path.join(ds, img_entry))
                    if not os.path.exists(img_path):
                        alt = os.path.join(ds, "images", os.path.basename(img_path))
                        if os.path.exists(alt):
                            img_path = alt

                    name = os.path.basename(img_path)
                    stem = os.path.splitext(name)[0]
                    ds_split_images[ds].add(name)

                    if not os.path.exists(img_path):
                        rep.stats["orphan_images"] += 1
                        rep.orphan_errors.append(f"split '{split}' references missing image: {img_entry}")

                    lbl_path = os.path.join(ds, "labels", stem + ".txt")
                    if not os.path.exists(lbl_path):
                        rep.stats["orphan_images"] += 1
                        rep.orphan_errors.append(f"image '{name}' missing annotation file: {lbl_path}")
                        boxes = []
                    else:
                        boxes, _ = self.validate_and_filter_boxes(
                            lbl_path, n_classes=nc, min_box_frac=self.min_box_frac,
                            name=f"{os.path.basename(ds)}/{name}", report=rep
                        )

                    is_bg_actual = len(boxes) == 0
                    m = meta.get(name)
                    if m is None:
                        rep.stats["orphan_metadata"] += 1
                        rep.orphan_errors.append(f"image '{name}' missing from metadata.json")
                        m = {}
                    else:
                        self.check_telemetry_record(
                            f"{os.path.basename(ds)}/{name}", m,
                            expected_split=split, actual_is_background=is_bg_actual, report=rep
                        )

                    digest = sha256_and_head(img_path)[0] if (check_hashes and os.path.exists(img_path)) else ""
                    split_records.append({
                        "name": name, "rel_path": f"{os.path.basename(ds)}/{name}",
                        "original_filename": m.get("original_filename", name),
                        "source": m.get("source", os.path.basename(ds)),
                        "split": split, "is_background": is_bg_actual,
                        "sha256": m.get("sha256", digest), "group_id": m.get("group_id", ""),
                    })

        # Scan each constituent dataset folder for reverse orphans on disk
        for ds, split_img_names in ds_split_images.items():
            img_dir, lbl_dir = os.path.join(ds, "images"), os.path.join(ds, "labels")
            disk_imgs = {
                fn for _, _, files in os.walk(img_dir) for fn in files
                if os.path.splitext(fn)[1].lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
            } if os.path.isdir(img_dir) else set()
            disk_img_stems = {os.path.splitext(fn)[0] for fn in disk_imgs}
            disk_lbl_stems = self.scan_label_stems(lbl_dir)

            for st in sorted(disk_lbl_stems - disk_img_stems):
                rep.stats["orphan_annotations"] += 1
                rep.orphan_errors.append(f"{os.path.basename(ds)}: orphan label '{st}.txt' has no matching image")
            for st in sorted(disk_img_stems - disk_lbl_stems):
                rep.stats["orphan_images"] += 1
                rep.orphan_errors.append(f"{os.path.basename(ds)}: image stem '{st}' has no matching .txt label")
            for fn in sorted(disk_imgs - split_img_names):
                rep.stats["orphan_images"] += 1
                rep.orphan_errors.append(f"{os.path.basename(ds)}: image '{fn}' on disk not in any split")
            meta = ds_metas.get(ds, {})
            if isinstance(meta, dict):
                for fn in sorted(set(meta.keys()) - disk_imgs):
                    rep.stats["orphan_metadata"] += 1
                    rep.orphan_errors.append(f"{os.path.basename(ds)}: orphan metadata.json key '{fn}' has no image")

        self.check_splits_and_ratios(split_records, report=rep)
        return rep


def main(yaml_path=DEFAULT_YAML, max_bg_ratio=0.0, min_box_frac=2e-6, check_hashes=False):
    validator = CrossDatasetValidator(
        n_classes=1, min_box_frac=min_box_frac, max_bg_ratio=max_bg_ratio, require_telemetry=True,
    )
    report = validator.validate_unified(yaml_path, check_hashes=check_hashes)
    for split, count in report.stats["split_counts"].items():
        print(f"{split:<5} ok ({count} images)")
    print(report.summary())
    report.raise_for_errors()
    print(f"{report.stats['total_frames']} images, schema and cross-dataset integrity ok")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the Cross-Dataset Integrity Check over a merged YOLO yaml.")
    ap.add_argument("yaml", nargs="?", default=DEFAULT_YAML, help="path to merged YOLO yaml (default: raven.yaml)")
    ap.add_argument("--max-bg-ratio", type=float, default=0.0, help="max allowed background-to-positive ratio")
    ap.add_argument("--min-box-frac", type=float, default=2e-6, help="min normalized box area before microscopic flag")
    ap.add_argument("--check-hashes", action="store_true", help="hash image files to check cross-split SHA-256 leakage")
    args = ap.parse_args()
    main(args.yaml, max_bg_ratio=args.max_bg_ratio, min_box_frac=args.min_box_frac, check_hashes=args.check_hashes)
