"""
Demo mode (python -m raven.gui --demo): run the whole console without the model or a dataset.

- three 30 s synthetic "thermal" drone flights (video + telemetry CSV each) are generated in runs/demo/,
  spread across Texas: Dallas, the Big Bend desert and the Piney Woods
- a simulated detector finds the warm blobs in that video instead of the YOLO model

Everything else (ingest, the event bus, the alert list, the detection log, the map, Discord if enabled) is the
real code path. Saved frames and the detection log go in runs/demo/frames/, which starts empty every launch.
"""
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

from raven.backend.config import ROOT_DIR, FeedConfig, RavenConfig

DEMO_DIR = ROOT_DIR / "runs" / "demo"
DETECTOR_NOTE = DEMO_DIR / "simulated_detector.txt"
FRAMES_DIR = DEMO_DIR / "frames"

# bump when the generated files change so old copies are rebuilt
DEMO_VERSION = "2"

WIDTH, HEIGHT, FPS, SECONDS = 640, 512, 10, 30

# where each demo drone searches, spread across Texas: a city campus and two rural areas
DALLAS = (32.98580, -96.75010)        # UT Dallas campus, Richardson (urban)
BIG_BEND = (29.30200, -103.27500)     # Chihuahuan Desert below the Chisos Mountains, Big Bend (remote)
PINEY_WOODS = (30.55600, -95.43300)   # Sam Houston National Forest, East Texas (forest)

# pixels hotter than this count as a warm body for the simulated detector
HEAT_THRESHOLD = 190

# simulated people: peak brightness (sets the detector's confidence), visible frames, start, velocity (px/frame)
PEOPLE = [
    {"heat": 245, "frames": (30, 150), "start": (180, 120), "velocity": (1.2, 0.6)},
    {"heat": 214, "frames": (110, 230), "start": (470, 90), "velocity": (-0.8, 0.9)},
    {"heat": 236, "frames": (190, 290), "start": (250, 60), "velocity": (0.4, 1.1)},
    {"heat": 226, "frames": (196, 290), "start": (280, 70), "velocity": (0.5, 1.0)},
]

# three drones searching far-apart areas: each has its own ground (seed), people, start point and heading
FLIGHTS = [
    {"name": "Drone 1", "file": "demo_flight_1", "seed": 7, "people": PEOPLE,
     "start": DALLAS, "step": (0.000042, 0.000015), "battery": 92},
    {"name": "Drone 2", "file": "demo_flight_2", "seed": 11,
     "people": [
         {"heat": 232, "frames": (20, 140), "start": (420, 80), "velocity": (-1.0, 0.7)},
         {"heat": 220, "frames": (160, 280), "start": (150, 100), "velocity": (0.9, 0.5)},
     ],
     "start": BIG_BEND, "step": (0.000030, -0.000032), "battery": 61},
    {"name": "Drone 3", "file": "demo_flight_3", "seed": 23,
     "people": [
         {"heat": 240, "frames": (60, 200), "start": (320, 60), "velocity": (0.2, 0.8)},
         {"heat": 228, "frames": (70, 210), "start": (345, 75), "velocity": (0.3, 0.8)},
         {"heat": 212, "frames": (220, 295), "start": (520, 120), "velocity": (-1.1, 0.4)},
     ],
     "start": PINEY_WOODS, "step": (0.000016, 0.000040), "battery": 33},
]


def video_path(flight) -> Path:
    return DEMO_DIR / f"{flight['file']}.mp4"


def telemetry_path(flight) -> Path:
    return DEMO_DIR / f"{flight['file']}_telemetry.csv"


def _heat_to_confidence(peak: float, frame_index: int) -> float:
    """
    Hotter blobs get higher confidence (214 -> ~0.67, 245 -> ~0.88), with a small wobble per frame.
    """
    confidence = 0.5 + (peak - HEAT_THRESHOLD) / 65 * 0.45
    return float(min(0.97, max(0.05, confidence + 0.03 * math.sin(frame_index * 0.7))))


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = xyxy
        self.conf = conf


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class SimulatedDetector:
    """
    Stands in for ultralytics.YOLO: called with a frame, returns [result] where result.boxes has .xyxy and .conf,
    which is all raven.backend.detection.run_model uses.
    """

    def __init__(self):
        self._frame_index = 0

    def __call__(self, frame, conf=0.5, verbose=False):
        self._frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, hot = cv2.threshold(gray, HEAT_THRESHOLD, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        xyxy, scores = [], []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < 20:
                continue
            confidence = _heat_to_confidence(float(gray[y:y + h, x:x + w].max()), self._frame_index)
            if confidence >= conf:
                xyxy.append([x - 3.0, y - 3.0, x + w + 3.0, y + h + 3.0])
                scores.append(confidence)
        return [_Result(_Boxes(xyxy, scores))]


def _make_video(path: Path, seed: int, people: list):
    """
    Write a grayscale-looking IR flight: cool, slowly scrolling ground with warm rocks, and walking people.
    """
    rng = np.random.default_rng(seed)
    total = FPS * SECONDS

    # ground texture taller than the frame so it can scroll as the drone flies
    ground = rng.normal(0, 1, (HEIGHT + total, WIDTH)).astype(np.float32)
    ground = cv2.GaussianBlur(ground, (0, 0), 9)
    ground = 60 + 18 * ground / (np.abs(ground).max() + 1e-6)
    for _ in range(60):  # sun-warmed rocks: warm, but under the detection threshold
        cx, cy = int(rng.integers(0, WIDTH)), int(rng.integers(0, HEIGHT + total))
        cv2.circle(ground, (cx, cy), int(rng.integers(4, 12)), float(rng.integers(120, 170)), -1)
    ground = cv2.GaussianBlur(ground, (0, 0), 1.5)

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for i in range(total):
        # the drone flies north, so the ground moves down the frame one pixel per frame
        top = total - i
        frame = ground[top:top + HEIGHT].copy()

        for person in people:
            first, last = person["frames"]
            if not first <= i < last:
                continue
            step = i - first
            x = person["start"][0] + person["velocity"][0] * step
            y = person["start"][1] + person["velocity"][1] * step + step  # own walk + ground drift
            if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
                continue
            body = np.zeros_like(frame)
            cv2.ellipse(body, (int(x), int(y)), (5, 9), 0, 0, 360, 1.0, -1)
            body = cv2.GaussianBlur(body, (0, 0), 1.2)
            frame = np.maximum(frame, body / body.max() * person["heat"])

        noisy = frame + rng.normal(0, 2.5, frame.shape)
        gray = np.clip(noisy, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    writer.release()


BATTERY_DRAIN = 0.3  # percent per second, sped up so the 30 s demo flights visibly use some


def _make_telemetry(path: Path, start: tuple, step: tuple, battery: float):
    """
    One row per second: the drone flies a straight line at ~5 m/s, 80 m above ground, its battery draining.
    """
    lines = ["t,latitude,longitude,altitude_agl_m,camera_depression_deg,heading_deg,speed_mps,battery_percent"]
    heading = math.degrees(math.atan2(step[1] * math.cos(math.radians(start[0])), step[0])) % 360
    for t in range(SECONDS + 1):
        lat = start[0] + t * step[0]
        lon = start[1] + t * step[1]
        charge = max(0.0, battery - t * BATTERY_DRAIN)
        lines.append(f"{t},{lat:.6f},{lon:.6f},80,45,{heading + 3 * math.sin(t / 5):.1f},5.0,{charge:.1f}")
    path.write_text("\n".join(lines) + "\n")


def ensure_demo_files():
    """
    Generate the demo flights and detector note if missing or out of date, and clear old frames.
    """
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    version_file = DEMO_DIR / "version.txt"
    current = version_file.read_text().strip() if version_file.exists() else ""
    missing = any(not video_path(f).exists() for f in FLIGHTS)
    if current != DEMO_VERSION or missing:
        # files from version 1 (one flight)
        for old in ("demo_flight.mp4", "demo_flight_telemetry.csv"):
            try:
                (DEMO_DIR / old).unlink(missing_ok=True)
            except OSError:  # still open in another console; it's unused now, so leave it
                pass
        for flight in FLIGHTS:
            _make_video(video_path(flight), flight["seed"], flight["people"])
        DETECTOR_NOTE.write_text(
            "Demo mode does not load a model. A simulated detector marks warm blobs in the generated video;\n"
            "hotter blobs get higher confidence, so raising the confidence threshold hides the cooler people.\n")
        version_file.write_text(DEMO_VERSION)

    # telemetry is quick to write, so it's always rewritten (moving a flight doesn't need new videos)
    for flight in FLIGHTS:
        _make_telemetry(telemetry_path(flight), flight["start"], flight["step"], flight["battery"])

    # every demo launch starts with no alerts, so frames and the detection log from earlier demo runs go
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)


class DemoBackend:
    folder = DEMO_DIR

    def start(self):
        ensure_demo_files()

    def config(self) -> RavenConfig:
        return RavenConfig(
            feeds=[FeedConfig(name=f["name"], source=str(video_path(f)), telemetry_path=str(telemetry_path(f)))
                   for f in FLIGHTS],
            model_path=str(DETECTOR_NOTE),
            min_confidence=0.5,
            # the most FR-05 allows, so the demo shows plenty of alerts
            analysis_fps=2.0,
            # play the flights at their real 30 s, so the Live view looks like a drone feed
            realtime=True,
            output_dir=str(FRAMES_DIR),
        )

    def make_detector(self, _model_path):
        return SimulatedDetector()
