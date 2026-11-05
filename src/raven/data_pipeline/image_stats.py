"""Per-image measurements: hashes, grayscale conversion, quality metrics and the
IR-likelihood classifier. Pure functions over decoded arrays -- no dataset
layout, labels or CLI -- so they can be imported on their own (e.g. to score
live frames) without pulling in the cleaning pipeline.
"""

import hashlib
import os

import numpy as np
from PIL import Image

SINGLE_CHANNEL_MODES = {"L", "I", "I;16", "I;16B", "I;16L", "F"}


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


QUALITY_KEYS = ("laplacian_var", "std", "entropy", "mean")


def to_quality_gray(arr):
    """8-bit gray for the quality metrics. 8-bit frames are used as shot: the
    per-image min-max stretch in to_uint8_gray would force every frame to span
    0..255 and make mean/std/laplacian incomparable between images. Higher bit
    depths have no canonical scale, so those still go through the stretch."""
    if arr.dtype != np.uint8:
        return to_uint8_gray(arr)
    if arr.ndim == 2:
        return arr
    n = 1 if arr.shape[2] == 2 else min(3, arr.shape[2])   # LA -> L, RGB(A) -> RGB
    return arr[..., :n].mean(axis=2).astype(np.uint8)


def quality_metrics(arr):
    """Sharpness (laplacian_var), contrast (std), detail (entropy) and
    brightness (mean) of a decoded image array, on a 0..255 scale. Works on any
    dataset, so HIT-UAV frames get the same numbers as AIResQ ones."""
    g = to_quality_gray(arr)
    return {
        "laplacian_var": round(laplacian_var(g), 2),
        "std": round(float(g.std()), 2),
        "entropy": round(shannon_entropy(g), 3) + 0.0,   # +0.0 turns -0.0 into 0.0
        "mean": round(float(g.mean()), 2),
    }


def quality_of(row):
    """The four quality fields of a manifest row, for the metadata.json files."""
    return {k: row[k] for k in QUALITY_KEYS}


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
