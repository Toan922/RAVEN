# RAVEN
RAVEN (Rescue Aerial Vision and Emergency Navigation): an ML system that detects people in live IR drone video and posts geotagged alerts to Discord, helping SAR teams find missing persons without manual footage review.

## Getting started

Run these from the repo root, in this order. Commands assume WSL / Ubuntu with Python 3.10 or newer (the team uses 3.12).

**1. Clone and switch to the working branch**
```bash
git clone <repo-url> RAVEN
cd RAVEN
git checkout RAVEN-toan-video
```

**2. Create and activate a virtual environment** (activate it again in every new terminal)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install RAVEN and its Python dependencies**
```bash
pip install -e .
```
This installs everything in `requirements.txt` and makes the `raven` package importable, so modules run with `python -m raven.<package>.<module>`. Code edits take effect without reinstalling.

**4. Install ffmpeg** (a system program, not a pip package, so it does not go in the venv)
```bash
sudo apt install ffmpeg
ffmpeg -version   # should print a version
```

**5. Get the datasets** (not stored in Git). Download them from the team Google Drive and put them at the repo root:
```
AIResQ/images/ ...     hit-uav/images/ ...
```
For the mock Discord alert (step 9) you also need the cleaned copies, `clean_HITUAV/` and `clean_AIResQ/`: download them from the Drive too, or rebuild them with the commands under [Dataset validation](#dataset-validation-validate_datasetpy).

**6. Build the demo videos**
```bash
python -m raven.videos.splice_video
```
Writes `videos/hit-uav.mp4` and `videos/AIResQ.mp4` (110 s at 10 fps each). The simulated ground station plays `hit-uav.mp4`.

**7. Add your Discord credentials** in a file named `.env` at the repo root (it is gitignored, never commit it):
```
DISCORD_WEBHOOK_URL=...
DISCORD_TOKEN=...
DISCORD_CHANNEL_ID=...
```
| Variable | Used by | Where to get it |
|---|---|---|
| `DISCORD_WEBHOOK_URL` | alert subscriber (`discord_bot/subscriber.py`) | In Discord: alerts channel → Edit Channel → Integrations → Webhooks → New Webhook → Copy Webhook URL |
| `DISCORD_TOKEN` | mock alert (`discord_bot/mock_alert.py`) | [Discord Developer Portal](https://discord.com/developers/applications) → your application → Bot → Reset Token. Invite the bot to the server with Send Messages and Attach Files permissions |
| `DISCORD_CHANNEL_ID` | mock alert | In Discord: Settings → Advanced → Developer Mode on, then right-click the alerts channel → Copy Channel ID |

Ask the team lead for the team's values instead of making your own, so alerts land in the shared channel.

**8. Start the simulated drone ground station** (terminal 1, leave it running, stop with Ctrl+C)
```bash
python -m raven.sim.ground_station
```
It sends:
- **video** to `udp://127.0.0.1:5000`: `videos/hit-uav.mp4` played in real time on a loop
- **telemetry** to UDP port `5001`: one JSON row about 10 times a second (`latitude`, `longitude`, `altitude_agl_m`, `heading_deg`, `camera_depression_deg`), for a drone flying a straight line at 5 m/s

Check the feeds (terminal 2): `ffplay udp://127.0.0.1:5000` shows the video and `nc -ul 5001` prints the telemetry. Close both before running RAVEN, since only one program can listen on each port.

**9. Optional: send a mock alert to Discord** to check your credentials
```bash
python -m raven.discord_bot.mock_alert
```
Posts one alert built from a random test frame to `DISCORD_CHANNEL_ID`.

**Current status:** the backend stages (frame sampling, live telemetry, detection) are built, but the entry point that connects them to Discord (`pipeline.py`) is not written yet, so there is no single command that runs detection end to end. The Redis-based `discord_bot/subscriber.py` is being replaced.

**Tests**
```bash
python tests/test_curate.py   # or: pip install pytest && pytest
```

## Layout

```
src/raven/
  backend/        frame sampling, live telemetry, detection, events
  sim/            simulated drone ground station (video + telemetry)
  data_pipeline/  dataset cleaning, splitting, integrity checks
  discord_bot/    alert formatting and Discord delivery
  videos/         builds demo videos from dataset frames
tests/            tests
```

## Dataset validation (`validate_dataset.py`)

Checks an image dataset before training and can write a cleaned copy. Datasets are not stored in this repo; point the script at your local copy.

```bash
# validate only (read-only)
python -m raven.data_pipeline.validate_dataset --images <dataset>/images --labels <dataset>/labels --out manifests/manifest.csv

# validate + write a cleaned copy with group-aware 80/10/10 splits and a 25% background cap
python -m raven.data_pipeline.validate_dataset --images <dataset>/images --labels <dataset>/labels \
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
python -m raven.data_pipeline.validate_dataset --images hit-uav/images --labels hit-uav/labels \
    --out manifests/hit_manifest.csv --clean-out clean_hit --keep-classes person
```
`--keep-classes person` keeps only Person boxes (renumbered to class 0) in the cleaned copy; images with no people stay as negative samples (use `--max-bg-ratio <float>` to cap the background-to-positive ratio, or `--drop-negatives` to drop all empty backgrounds). Validation still checks every class in the source.

To produce the team's `clean_HITUAV` structure (persons only, no background images):
```bash
python -m raven.data_pipeline.validate_dataset --images hit-uav/images --labels hit-uav/labels \
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
python -m raven.data_pipeline.check_unified raven.yaml
python -m raven.data_pipeline.check_unified raven.yaml --max-bg-ratio 0.25 --check-hashes
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

The source dataset is never modified. With the default `--link auto`, kept images are hardlinked; use `--link copy` if you plan to edit images in the cleaned copy. Run `python -m raven.data_pipeline.validate_dataset -h` for all options (near-dup threshold, `--drop-near-dups`, `--drop-tiny-boxes`, `--max-bg-ratio`, `--classes`, etc.).
