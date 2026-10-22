"""Assert the merged dataset matches the shared schema before training.

    python check_unified.py raven.yaml
"""
import json
import os
import sys

CORE = ("original_filename", "source", "split", "is_background",
        "laplacian_var", "std", "entropy", "mean")


def split_lists(yaml_path):
    """{split: [list files]} from the merged yaml's `split: [a, b]` lines."""
    root, out = os.path.dirname(os.path.abspath(yaml_path)), {}
    for line in open(yaml_path, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if line.startswith("path:"):
            root = line[5:].strip()
        for s in ("train", "val", "test"):
            if line.startswith(s + ":"):
                out[s] = [os.path.join(root, p.strip())
                          for p in line[len(s) + 1:].strip(" []").split(",")]
    return out


def main(yaml_path):
    seen = {}  # basename -> split
    for split, lists in split_lists(yaml_path).items():
        for lst in lists:
            ds = os.path.dirname(os.path.dirname(lst))
            meta = json.load(open(os.path.join(ds, "metadata.json"), encoding="utf-8"))
            for img in open(lst, encoding="utf-8").read().split():
                name = os.path.basename(img)
                assert os.path.exists(img), f"missing image {img}"
                assert name not in seen, f"{name} in {seen.get(name)} and {split}"
                seen[name] = split
                m = meta[name]
                assert all(k in m for k in CORE), f"{name} missing core keys"
                assert m["split"] == split and not m["is_background"], name
                lbl = os.path.join(ds, "labels", os.path.splitext(name)[0] + ".txt")
                rows = [r.split() for r in open(lbl).read().splitlines() if r.strip()]
                assert rows, f"empty label {lbl}"
                for r in rows:
                    assert len(r) == 5 and r[0] == "0", f"{lbl}: {r}"
                    assert all(0 <= float(v) <= 1 for v in r[1:]), f"{lbl}: {r}"
        print(f"{split:<5} ok")
    print(f"{len(seen)} images, schema ok")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "raven.yaml")
