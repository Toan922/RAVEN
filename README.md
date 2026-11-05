# RAVEN
RAVEN (Rescue Aerial Vision and Emergency Navigation): an ML system that detects people in live IR drone video and posts geotagged alerts to Discord, helping SAR teams find missing persons without manual footage review.

## Operator console (`gui/`)

A PySide6 desktop app for running the detection pipeline and reviewing alerts from up to 3 drones at once. It only takes operator inputs (each drone's video file or stream link and telemetry CSV, sampling and confidence settings); nothing connects to the drones.

```bash
pip install -r requirements.txt
python -m gui           # real Redis server + YOLO model
python -m gui --demo    # no Redis, model or dataset needed
```

**Demo mode** (`--demo`) starts a fake Redis server inside the app (fakeredis, on a free local port), generates three 30 s synthetic thermal flights in `runs/demo/` (Drone 1 over Dallas, Drone 2 in the Big Bend desert, Drone 3 in the Piney Woods, each with its own telemetry), and swaps the YOLO model for a simulated detector that marks the warm bodies in that video (hotter people get higher confidence, so the confidence slider visibly changes what's detected). Press **Start**; everything after that (ingest, the Redis stream, alerts, the map, Discord if you turn it on) is the real code path. Demo mode saves its settings separately, so they never overwrite your real ones.

The header row has the Map / Feed / Settings pages (`Ctrl+1`–`3`) and a light/dark mode toggle on the right. Every shortcut can be changed in Settings.

- **Map** tab: a zoomable OpenStreetMap with a pin per alert at the drone's position (needs internet for map tiles). Pins and clusters are colored per drone (legend bottom-left); the pin's ring shows its status. A selected alert's pin stays visible at any zoom; click a cluster to zoom in or spread its pins out, and click it again to fold it back. The alerts sidebar is open at launch and collapses into the right edge: each card names its drone, a drone filter appears once alerts come from more than one drone, click a card to find its pin, double-click to review its footage, `R`/`D`/`N` to mark reviewed / dismissed / new. The feeds bar underneath has a row per drone (source, progress, counts, status, Browse), Start / Stop for all feeds (`F5` / `Shift+F5`), and the Redis / Discord status lights.
- **Profile** (icon at the top right, next to the light/dark toggle, with a dot for the operator's status): the signed-in operator (a sample profile, Albert Ross, until sign-in exists) with a status picker (on duty, away, off duty; remembered between launches) and a Log out button that does nothing until sign-in exists, the drones they are operating with each one's flight state, battery and flight time, and their account details with sensitive values hidden (the eye button at the end of a line shows it; they hide again when the panel closes). Battery comes from an optional `battery_percent` column in the telemetry CSV; the demo flights include one.
- Panels are resizable: drag the line between the map and the feeds bar, between either tab and its alerts panel, or between Settings and the log. Panels keep the sizes you set while alerts come in, and the sizes are remembered for the next launch.
- **Feed** tab, with *Live* and *Review* subtabs on the left of its banner:
  - *Live*: the feed of one drone (pick it from the dropdown) as it's being processed, with the latest detections, the drone's position and a shortcut to review its newest alert.
  - *Review*: the footage behind an alert (double-click any alert). Play/pause (`Space`), speed 0.25×–4×, one frame at a time (`←`/`→`), jump to the previous/next alert (`Ctrl+←`/`→`), and a timeline with a red tick per alert. Detection boxes are solid on the frames the detector checked and dashed while carried to the next check. It has its own alerts panel (same list and statuses as the Map tab) plus Reviewed / Dismiss / Show on map. Footage comes from the video recorded in each event's `source`; live streams aren't recorded, so their alerts show the saved frame only.
- **Settings** tab:
  - *Drone feeds* (up to 3): per drone a name (tags its alerts, Discord titles and saved-frames folder), on/off, video file or stream link (with *Test source*) and telemetry CSV (checked as soon as it's picked). Start runs every ticked feed in parallel.
  - Shared by all feeds: confidence threshold, frame sample rate (shown as checks per second), model weights, *Play video files at real speed* (so a recorded flight behaves like a live drone; demo mode turns it on), Discord alerts (webhook, cooldown per drone, *Send test alert*), and under *Advanced* the Redis host/port, stream name and saved-frames folder.
  - *Appearance*: whether the console starts in the system, light or dark theme (the header's sun/moon button switches for the current session only), and the accent color: blue, violet, teal, orange or rose. Both apply and save right away.
  - *Keyboard shortcuts*: click a shortcut and press new keys; changes apply and save right away, and a key that's already taken is refused.
  - *Save changes* (`Ctrl+S`) keeps the settings for the next launch; they can also be saved as JSON presets.
  - *Show log* (`Ctrl+L` from anywhere) opens the log beside the form: what ingest, Redis and Discord are doing, and why something failed.

Feeds can also be run without the console, one process per drone, all publishing to the same stream:
```bash
python backend/ingest.py --drone "Drone 1" --video videos/flight1.mp4 --telemetry videos/flight1_telemetry.csv
python backend/ingest.py --drone "Drone 2" --video rtsp://host:554/stream --telemetry videos/flight2_telemetry.csv
```

Needs a running Redis server (`localhost:6379` by default; on Windows run it in WSL with `sudo apt install redis-server && redis-server`, or use Docker / Memurai) and `pip install ultralytics` for the model. The Discord webhook is read from `DISCORD_WEBHOOK_URL` in `.env` (gitignored) and is never written to presets. The console reads the whole `raven:detections` stream on launch, so earlier alerts reappear after a restart; reviewed/dismissed marks are remembered.

## Dataset validation (`validate_dataset.py`)

Checks an image dataset before training and can write a cleaned copy. Datasets are not stored in this repo; point the script at your local copy.

```bash
pip install -r requirements.txt

# validate only (read-only)
python data_pipeline/validate_dataset.py --images <dataset>/images --labels <dataset>/labels --out manifests/manifest.csv

# validate + write a cleaned copy with group-aware 80/10/10 splits and a 25% background cap
python data_pipeline/validate_dataset.py --images <dataset>/images --labels <dataset>/labels \
    --out manifests/manifest.csv --clean-out clean_<dataset> --splits 0.8,0.1,0.1 --max-bg-ratio 0.25
```

`--labels` takes YOLO `.txt` files that mirror the image tree. Sidecars next to the `images/` folder are picked up automatically (or pass the flag):
- `dataset.yaml` (`--yaml`): YOLO class names, which also set the class count
- `labels.json` (`--coco`): COCO ground truth, cross-checked against the `.txt` files
- `metadata.json` (`--metadata`): AIResQ capture sessions

If images sit in `images/train`, `images/val`, `images/test` (e.g. HIT-UAV), that split is kept; `--splits` replaces it with a fresh group-aware one.

Add `--stratify laplacian_var,std --bins 3` to quantile-bin those quality metrics (per dataset) and spread every bin across train/val/test in proportion; each frame's bin is written as `quality_bin` (manifest and `--layout hituav` metadata). Fewer metrics × bins keeps bins large enough to split evenly.

**HIT-UAV**
```bash
python data_pipeline/validate_dataset.py --images hit-uav/images --labels hit-uav/labels \
    --out manifests/hit_manifest.csv --clean-out clean_hit --keep-classes person
```
`--keep-classes person` keeps only Person boxes (renumbered to class 0) in the cleaned copy; images with no people stay as negative samples (use `--max-bg-ratio <float>` to cap the background-to-positive ratio, or `--drop-negatives` to drop all empty backgrounds). Validation still checks every class in the source.

To produce the team's `clean_HITUAV` structure (persons only, no background images):
```bash
python data_pipeline/validate_dataset.py --images hit-uav/images --labels hit-uav/labels \
    --out manifests/hit_manifest.csv --clean-out clean_HITUAV \
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
python data_pipeline/check_unified.py raven.yaml
python data_pipeline/check_unified.py raven.yaml --max-bg-ratio 0.25 --check-hashes
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

The source dataset is never modified. With the default `--link auto`, kept images are hardlinked; use `--link copy` if you plan to edit images in the cleaned copy. Run `python data_pipeline/validate_dataset.py -h` for all options (near-dup threshold, `--drop-near-dups`, `--drop-tiny-boxes`, `--max-bg-ratio`, `--classes`, etc.).
