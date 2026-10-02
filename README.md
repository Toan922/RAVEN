# RAVEN
RAVEN (Rescue Aerial Vision and Emergency Navigation): an ML system that detects people in live IR drone video and posts geotagged alerts to Discord, helping SAR teams find missing persons without manual footage review.

## Dataset validation (`validate_dataset.py`)

Checks an image dataset before training and can write a cleaned copy. Datasets are not stored in this repo; point the script at your local copy.

```bash
pip install -r requirements.txt

# validate only (read-only)
python validate_dataset.py --images <dataset>/images --labels <dataset>/labels --out manifest.csv

# validate + write a cleaned copy with group-aware 80/10/10 splits
python validate_dataset.py --images <dataset>/images --labels <dataset>/labels \
    --out manifest.csv --clean-out clean_<dataset> --splits 0.8,0.1,0.1
```

`--labels` takes YOLO `.txt` files that mirror the image tree. If `labels.json` (COCO) or `metadata.json` (capture sessions) sit next to the `images/` folder, they are picked up automatically; otherwise pass `--coco` / `--metadata`, or leave them out.

**What it checks**
- Integrity: unreadable, zero-byte, corrupt or truncated images
- IR-likelihood: whether each frame looks thermal (`ir` / `uncertain` / `not_ir` / `conflict`)
- Quality: brightness (`mean`), contrast (`std`), detail (`entropy`), sharpness (`laplacian_var`), clipped pixels
- Labels: missing/invalid YOLO boxes, wrong class ids, out-of-frame or tiny boxes, COCO vs YOLO mismatches
- Duplicates: exact (sha256) and near-duplicates (perceptual hash), grouped so they never straddle train/test

**Outputs**
- `manifest.csv`: one row per image with every metric and the keep/drop decision
- `manifest_flagged.csv`: files worth a human look
- `manifest_nearpairs.csv`: near-duplicate pairs
- `--clean-out` dir: kept images, cleaned YOLO labels, regenerated `labels.json`, `dropped.csv`, `clean_report.txt`, and with `--splits`, `splits/*.txt` and a YOLO `dataset.yaml`

The source dataset is never modified. With the default `--link auto`, kept images are hardlinked; use `--link copy` if you plan to edit images in the cleaned copy. Run `python validate_dataset.py -h` for all options (near-dup threshold, `--drop-near-dups`, `--drop-tiny-boxes`, `--classes`, etc.).
