# RAVEN
RAVEN (Rescue Aerial Vision and Emergency Navigation): an ML system that detects people in live IR drone video and posts geotagged alerts to Discord, helping SAR teams find missing persons without manual footage review.

## Getting started

Run these from the repo root, in this order. Commands assume WSL / Ubuntu with Python 3.10 or newer (the team uses 3.12).

**1. Clone the repo**
```bash
git clone <repo-url> RAVEN
cd RAVEN
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

**6. Build the demo videos**
```bash
python -m raven.videos.splice_video
```
Writes `videos/hit-uav.mp4` and `videos/AIResQ.mp4` (110 s at 10 fps each). The simulated ground station plays `hit-uav.mp4`.

**7. Add your Discord webhook** in a file named `.env` at the repo root (it is gitignored, never commit it):
```
DISCORD_WEBHOOK_URL=...
```
To get one in Discord: alerts channel → Edit Channel → Integrations → Webhooks → New Webhook → Copy Webhook URL.

Ask the team lead for the team's value instead of making your own, so alerts land in the shared channel.

**8. Start the simulated drone ground station** (terminal 1, leave it running, stop with Ctrl+C)
```bash
python -m raven.sim.ground_station
```
It sends:
- **video** to `udp://127.0.0.1:5000`: `videos/hit-uav.mp4` played in real time on a loop
- **telemetry** to UDP port `5001`: one JSON row about 10 times a second (`latitude`, `longitude`, `altitude_agl_m`, `heading_deg`, `camera_depression_deg`), for a drone flying a straight line at 5 m/s

Check the feeds (terminal 2): `ffplay udp://127.0.0.1:5000` shows the video and `nc -ul 5001` prints the telemetry. Close both before running RAVEN, since only one program can listen on each port.

**9. Run RAVEN** (terminal 2, stop with Ctrl+C)
```bash
python -m raven.pipeline
```
Reads the stream, runs detection, and posts alerts to the channel in `DISCORD_WEBHOOK_URL`, at most one every 10 s. Feed and detection problems print to the terminal.

**Current status:** detection runs end to end and posts to Discord. Alerts show the drone's position, not the person's, because per-person location isn't built yet. See `docs/TODO.md`. The operator console (below) runs the same detector on recorded flights with a map, alert review and a detection log.

**Tests**
```bash
python tests/test_curate.py   # or: pip install pytest && pytest
```

## Layout

```
src/raven/
  backend/        frame sampling, live telemetry, detection, events, event bus helpers,
                  detection log, file ingest and settings for the operator console
  sim/            simulated drone ground station (video + telemetry)
  data_pipeline/  dataset cleaning, splitting, integrity checks
  discord_bot/    alert formatting and Discord delivery
  gui/            operator console (PySide6)
  videos/         builds demo videos from dataset frames
tests/            tests
```

## Operator console (`src/raven/gui/`)

A PySide6 desktop app for running the detection pipeline and reviewing alerts from up to 3 drones at once. It only takes operator inputs (each drone's video file or stream link, its telemetry as a CSV or a live `udp://host:port`, sampling and confidence settings); nothing connects to the drones.

```bash
pip install -e .                 # once, see Getting started
python -m raven.gui              # the YOLO model on the feeds set in Settings
python -m raven.gui --demo       # no model or dataset needed
```

**Signing in.** The console opens with a sign-in screen. The first launch creates the administrator account; the administrator adds everyone else in Manage > Accounts. Roles: *Administrator* (everything, plus accounts), *Drone Operator* (runs the search and the feeds, changes settings, reviews alerts), *Reviewer* (reviews alert footage and marks alerts reviewed or dismissed; no Manage tab) and *Viewer* (read only: map, alerts and the Live feed; no Manage tab and no footage review). Passwords are stored as Argon2 hashes; five wrong tries lock an account for a minute. The console locks after 10 minutes without input, and *Log out* in the profile panel hands it to the next person; feeds keep running in both cases.

**Encrypted data.** Everything the console writes to the saved-frames folder (frames, `detections.jsonl`, the Discord outbox) and the accounts file (`runs/accounts.dat`) is encrypted with a key kept in the Windows Credential Manager ("RAVEN operator console"), and decrypted in memory for display and Discord. Files from before encryption are still read. Deleting that credential makes the existing data unreadable.

Inside the console everything goes through RAVEN's Detection Event Bus (bubus), the same events `raven.pipeline` uses: each drone feed publishes `HumanDetected` and `PipelineStatus` events, and the alert list, the Detection Log and Discord alerts subscribe to them. Nothing else needs to be running.

**Demo mode** (`--demo`) generates three 30 s synthetic thermal flights in `runs/demo/` (Drone 1 over Dallas, Drone 2 in the Big Bend desert, Drone 3 in the Piney Woods, each with its own telemetry), and swaps the YOLO model for a simulated detector that marks the warm bodies in that video (hotter people get higher confidence, so the confidence slider visibly changes what's detected). Press **Start**; everything after that (ingest, the event bus, alerts, the detection log, the map, Discord if you turn it on) is the real code path. Each demo launch starts with no alerts. Demo mode saves its settings separately, so they never overwrite your real ones.

The header row has the Map / Feed / Manage / Settings pages (`Ctrl+1`–`4`; Manage is shown to administrators and drone operators only) and a light/dark mode toggle on the right. Every shortcut can be changed in Settings.

- **Header**: the search session's name and state: *Not ready* (hover to see what's missing), *Ready for footage*, or *Running*. Feed and detection problems (no stream yet, interrupted, undecodable video, frames failing, no telemetry, detection falling behind) show in amber under the header, on the drone's row in the feeds bar and in the Live view, and clear once the problem does.
- **Map** tab: a zoomable OpenStreetMap with a pin per alert at the person's estimated ground position, worked out from the drone's position, altitude, heading and camera angle (needs internet for map tiles). When the telemetry has too little for that, the pin sits at the drone's position with a dashed ring and its card says "≈ approx."; Discord labels those "Approximate (drone position)". Pins and clusters are colored per drone (legend bottom-left; click a drone there to jump to its latest position, or to its pins when no position is known); the pin's ring shows its status. A selected alert's pin stays visible at any zoom; click a cluster to zoom in or spread its pins out, and click it again to fold it back. The alerts sidebar is open at launch and collapses into the right edge: each card names its drone, a drone filter appears once alerts come from more than one drone, click a card to find its pin, double-click to review its footage, `R`/`D`/`N` to mark reviewed / dismissed / new. The feeds bar underneath has a row per drone (source, progress, counts, status, Browse), Start / Stop for all feeds (`F5` / `Shift+F5`), and the Discord status light.
- **Profile** (icon at the top right, next to the light/dark toggle, with a dot for the operator's status): the signed-in account and role with a status picker (on duty, away, off duty; remembered between launches) and Log out, the drones they are operating with each one's flight state, battery and flight time, and their account details with sensitive values hidden (the eye button at the end of a line shows it; they hide again when the panel closes). Battery comes from an optional `battery_percent` column in the telemetry CSV; the demo flights include one.
- Panels are resizable: drag the line between the map and the feeds bar, between either tab and its alerts panel, or between Settings and the log. Panels keep the sizes you set while alerts come in, and the sizes are remembered for the next launch.
- **Feed** tab, with *Live* and *Review* subtabs on the left of its banner:
  - *Live*: the feed of one drone (pick it from the dropdown) as it's being processed, with the latest detections, the drone's position and a shortcut to review its newest alert.
  - *Review*: the footage behind an alert (double-click any alert). Viewers don't have this subtab: double-clicking an alert shows that drone on *Live* instead. Play/pause (`Space`), speed 0.25×–4×, one frame at a time (`←`/`→`), jump to the previous/next alert (`Ctrl+←`/`→`), and a timeline with a red tick per alert. Detection boxes are solid on the frames the detector checked and dashed while carried to the next check. It has its own alerts panel (same list and statuses as the Map tab) plus Reviewed / Dismiss / Show on map. Footage comes from the video recorded in each event's `source`; live streams aren't recorded, so their alerts show the saved frame only.
- **Manage** tab (administrators and drone operators):
  - *Search Sessions*: the current session's name (required before Start) and the area being searched; edits save on their own. Start marks the session running; if the console closes mid-run, the next launch offers to resume it. *New session* clears the form for the next search. Below it, every session started so far with its area, when it was created and its status (*Running*, *Interrupted* if the console closed mid-run, *Ended*). Administrators can remove a session from the list (not while it's running); its saved frames and detection log are kept.
  - *Drones*: each drone's flight state, battery and flight time (as in the profile panel), any feed warning, its assigned operator, and *View on map* to jump to its latest position. Administrators reassign a drone from its operator dropdown (administrators and drone operators can be assigned); the profile panel lists only the drones assigned to whoever is signed in.
  - *Accounts* (administrators only): a table of every account (full name, username, role, assigned drone, duty status). Pick a role in the *Role* dropdown to change the selected account's role right away; add, reset and remove accounts with the buttons. Each person's duty status is the one they set in their profile panel and is kept with their account.
- **Settings** tab:
  - *Drone feeds* (up to 3): per drone a name (tags its alerts, Discord titles and saved-frames folder), on/off, video file or stream link (with *Test source*) and telemetry: a CSV (checked as soon as it's picked) or `udp://host:port` for live JSON telemetry from the ground station (*Test telemetry* listens for a message). Live UDP telemetry needs a stream link, and every UDP port can only be used once. Start runs every ticked feed in parallel; recorded videos must decode before Start.
  - Shared by all feeds: confidence threshold (0.70 by default), analysis rate (1–2 frames per second, FR-05), model weights, *Play video files at real speed* (so a recorded flight behaves like a live drone; demo mode turns it on), Discord alerts (required outside demo mode; webhook, cooldown per drone, *Send test alert*, which has to succeed once per launch and after every webhook change before Start), and under *Advanced* the saved-frames folder (which also holds the detection log).
  - *Appearance*: whether the console starts in the system, light or dark theme (the header's sun/moon button switches for the current session only), and the accent color: blue, violet, teal, orange or rose. Both apply and save right away.
  - *Keyboard shortcuts*: click a shortcut and press new keys; changes apply and save right away, and a key that's already taken is refused.
  - *Save changes* (`Ctrl+S`) keeps the settings for the next launch; they can also be saved as JSON presets.
  - *Show log* (`Ctrl+L` from anywhere) opens the log beside the form: what ingest and Discord are doing, feed and detection problems (`PipelineStatus`), and why something failed.

Use the model's weights at the default path (`runs/detect/runs/train/yolov8n_raven/weights/best.pt`) or pick others in Settings. The Discord webhook is read from `DISCORD_WEBHOOK_URL` in `.env` (gitignored) and is never written to presets; failed Discord sends are retried, and the Discord light turns amber while retrying. Alerts not yet sent are kept in `discord_outbox.jsonl` and sent at the next launch; quitting with alerts still waiting asks first. The feeds bar shows each feed's detection time (p90) and lag. Every detection is appended to `detections.jsonl` in the saved-frames folder (`runs/ingest/` by default), so earlier alerts reappear after a restart; reviewed/dismissed marks are remembered.

For the live UDP stream from the ground station, set a feed's video to `udp://127.0.0.1:5000` and its telemetry to `udp://127.0.0.1:5001` (the simulator in `raven.sim.ground_station` sends both), or run `python -m raven.pipeline` (Getting started, step 9).

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
