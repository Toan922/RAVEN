# RAVEN
RAVEN (Rescue Aerial Vision and Emergency Navigation): an ML system that detects people in live IR drone video and posts geotagged alerts to Discord, helping SAR teams find missing persons without manual footage review.

## Dataset validation (`validate_dataset.py`)

Checks an image dataset before training and can write a cleaned copy. Datasets are not stored in this repo; point the script at your local copy.

```bash
pip install -r requirements.txt

# validate only (read-only)
python validate_dataset.py --images <dataset>/images --labels <dataset>/labels --out manifest.csv

# validate + write a cleaned copy with group-aware 80/10/10 splits and a 25% background cap
python validate_dataset.py --images <dataset>/images --labels <dataset>/labels \
    --out manifest.csv --clean-out clean_<dataset> --splits 0.8,0.1,0.1 --max-bg-ratio 0.25
```

`--labels` takes YOLO `.txt` files that mirror the image tree. Sidecars next to the `images/` folder are picked up automatically (or pass the flag):
- `dataset.yaml` (`--yaml`): YOLO class names, which also set the class count
- `labels.json` (`--coco`): COCO ground truth, cross-checked against the `.txt` files
- `metadata.json` (`--metadata`): AIResQ capture sessions

If images sit in `images/train`, `images/val`, `images/test` (e.g. HIT-UAV), that split is kept; `--splits` replaces it with a fresh group-aware one.

Add `--stratify laplacian_var,std --bins 3` to quantile-bin those quality metrics (per dataset) and spread every bin across train/val/test in proportion; each frame's bin is written as `quality_bin` (manifest and `--layout hituav` metadata). Fewer metrics × bins keeps bins large enough to split evenly.

**HIT-UAV**
```bash
python validate_dataset.py --images hit-uav/images --labels hit-uav/labels \
    --out hit_manifest.csv --clean-out clean_hit --keep-classes person
```
`--keep-classes person` keeps only Person boxes (renumbered to class 0) in the cleaned copy; images with no people stay as negative samples (use `--max-bg-ratio <float>` to cap the background-to-positive ratio, or `--drop-negatives` to drop all empty backgrounds). Validation still checks every class in the source.

To produce the team's `clean_HITUAV` structure (persons only, no background images):
```bash
python validate_dataset.py --images hit-uav/images --labels hit-uav/labels \
    --out hit_manifest.csv --clean-out clean_HITUAV \
    --keep-classes person --drop-negatives --layout hituav
```
```
clean_HITUAV/
  images/HU00001.jpg ...   renamed: train, then val, then test, by original name
  labels/HU00001.txt ...
  splits/{train,val,test}.txt
  metadata.json            per image: original_filename, time_of_day, altitude_m,
                           camera_angle_deg, weather, split, is_background
  dataset.yaml  dropped.csv  clean_report.txt
```
Split lists hold absolute paths (Ultralytics resolves `./` against the list file's folder), so re-run if you move the folder.

**Cross-Dataset Integrity Check (`check_unified.py` / `curate.py`)**
```bash
# verify merged dataset (raven.yaml) before training
python check_unified.py raven.yaml
python check_unified.py raven.yaml --max-bg-ratio 0.25 --check-hashes
```
`CrossDatasetValidator` in `check_unified.py` runs during both curation (`validate_dataset.py`) and merged verification (`check_unified.py`), checking:
1. **Orphan records**: images missing annotations/metadata, and reverse-orphan annotations or sidecar entries pointing to missing images.
2. **Telemetry synchronization**: core + quality schemas for every frame, plus source-specific telemetry schemas (HIT-UAV flight telemetry vs. AIResQ session attributes) with strict typing and range checks.
3. **Bounding box geometry**: normalized `[0, 1]` coordinates and in-frame corners, flagging and filtering zero-area degenerate boxes and microscopic artifacts (`--min-box-frac`, `--drop-tiny-boxes`).
4. **Background capping & split leakage**: enforces `--max-bg-ratio` across overall and per-split pools, and verifies zero cross-split leakage across filenames, source records, SHA-256 digests, and near-duplicate `group_id` clusters.

**Training / testing** — the `--clean-out` dir is a ready Ultralytics dataset:
```bash
yolo detect train data=clean_hit/dataset.yaml model=yolov8n.pt
yolo detect val   data=clean_hit/dataset.yaml model=runs/detect/train/weights/best.pt split=test
```
`dataset.yaml` stores an absolute path, so re-run the script if you move the folder.

**What it checks**
- Integrity: unreadable, zero-byte, corrupt or truncated images
- IR-likelihood: whether each frame looks thermal (`ir` / `uncertain` / `not_ir` / `conflict`)
- Quality: brightness (`mean`), contrast (`std`), detail (`entropy`), sharpness (`laplacian_var`), clipped pixels
- Labels: missing/invalid YOLO boxes, wrong class ids, out-of-frame or tiny boxes, COCO vs YOLO mismatches, reverse-orphan label files
- Duplicates: exact (sha256) and near-duplicates (perceptual hash), grouped so `--splits` never puts one group in both train and test; for pre-split data it drops/reports groups that cross splits

**Outputs**
- `manifest.csv`: one row per image with every metric and the keep/drop decision
- `manifest_flagged.csv`: files worth a human look
- `manifest_nearpairs.csv`: near-duplicate pairs
- `--clean-out` dir: kept images, cleaned YOLO labels, regenerated `labels.json`, `dropped.csv`, `clean_report.txt`, and with splits (native or `--splits`), `train.txt`/`val.txt`/`test.txt` and a YOLO `dataset.yaml`

The source dataset is never modified. With the default `--link auto`, kept images are hardlinked; use `--link copy` if you plan to edit images in the cleaned copy. Run `python validate_dataset.py -h` for all options (near-dup threshold, `--drop-near-dups`, `--drop-tiny-boxes`, `--max-bg-ratio`, `--classes`, etc.).
