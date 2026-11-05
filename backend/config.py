"""
Operator settings for a RAVEN run, shared by the GUI, presets and the command line.
"""
import json
import re
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path

# get the root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent

# the trained person detector
DEFAULT_MODEL_PATH = ROOT_DIR / "runs/detect/runs/train/yolov8n_raven/weights/best.pt"

# save the frames with detections here, one subfolder per video
DEFAULT_OUTPUT_DIR = ROOT_DIR / "runs" / "ingest"

# the redis stream ingest publishes to and the GUI / Discord bot read from
DEFAULT_STREAM_NAME = "raven:detections"

# settings that are secrets and never written to presets
SECRET_FIELDS = {"webhook_url"}

# how many drone feeds can run at once
MAX_FEEDS = 3


def drone_folder(name: str) -> str:
    """
    A file-system safe folder name for a drone, so each drone's saved frames stay apart.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("_") or "drone"


@dataclass
class FeedConfig:
    """
    One drone feed: its name (tags every alert), video file or stream link, and telemetry CSV.
    """
    name: str = "Drone 1"

    # a video file path or a stream link (rtsp://, http://, ...)
    source: str = ""

    # telemetry CSV with at least t, latitude, longitude columns
    telemetry_path: str = ""

    # turned-off feeds stay in the list but don't run
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "FeedConfig":
        feed = cls()
        for name in ("name", "source", "telemetry_path"):
            if name in data:
                setattr(feed, name, str(data[name]))
        if "enabled" in data:
            value = data["enabled"]
            feed.enabled = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")
        return feed

    def problems(self) -> list[str]:
        problems = []
        label = self.name.strip() or "A feed"
        if not self.name.strip():
            problems.append("A feed has no drone name.")
        if not self.source.strip():
            problems.append(f"{label}: no video source or stream link is set.")
        elif "://" not in self.source and not Path(self.source).is_file():
            problems.append(f"{label}: video file not found: {self.source}")
        if not self.telemetry_path.strip():
            problems.append(f"{label}: no telemetry CSV is set.")
        elif not Path(self.telemetry_path).is_file():
            problems.append(f"{label}: telemetry file not found: {self.telemetry_path}")
        return problems


@dataclass
class RavenConfig:
    # up to MAX_FEEDS drone feeds; confidence, sample rate and everything below are shared by all of them
    feeds: list = field(default_factory=lambda: [FeedConfig()])

    # detection
    min_confidence: float = 0.5
    frame_stride: int = 5
    model_path: str = str(DEFAULT_MODEL_PATH)

    # read video files at their real speed, like a live drone (streams always arrive in real time)
    realtime: bool = False

    # alerts
    alert_cooldown_s: float = 10.0
    discord_enabled: bool = False
    webhook_url: str = ""

    # backend
    redis_host: str = "localhost"
    redis_port: int = 6379
    stream_name: str = DEFAULT_STREAM_NAME
    output_dir: str = str(DEFAULT_OUTPUT_DIR)

    def to_dict(self, include_secrets=False) -> dict:
        """
        Return the settings as a plain dict, leaving out secrets unless asked for them.
        """
        data = asdict(self)
        if not include_secrets:
            for name in SECRET_FIELDS:
                data.pop(name, None)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "RavenConfig":
        """
        Build a config from a dict, ignoring unknown keys and keeping defaults for missing ones.
        Values are converted to each field's type so a preset with "5" still gives an int.
        """
        config = cls()
        if isinstance(data.get("feeds"), list) and data["feeds"]:
            config.feeds = [FeedConfig.from_dict(feed) for feed in data["feeds"][:MAX_FEEDS]
                            if isinstance(feed, dict)] or [FeedConfig()]
        elif data.get("source") or data.get("telemetry_path"):
            # settings saved before multiple feeds existed had a single source
            config.feeds = [FeedConfig(source=str(data.get("source", "")),
                                       telemetry_path=str(data.get("telemetry_path", "")))]
        for f in fields(cls):
            if f.name == "feeds":
                continue
            if f.name in data:
                default = getattr(config, f.name)
                value = data[f.name]
                if isinstance(default, bool):
                    value = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")
                else:
                    value = type(default)(value)
                setattr(config, f.name, value)
        return config

    def save(self, path) -> None:
        """
        Save the settings to a JSON preset file, without secrets.
        """
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "RavenConfig":
        """
        Load the settings from a JSON preset file.
        """
        return cls.from_dict(json.loads(Path(path).read_text()))

    def problems(self) -> list[str]:
        """
        Return a list of readable problems that would stop a run, empty if the settings are usable.
        """
        problems = []
        enabled = self.enabled_feeds()
        if not enabled:
            problems.append("No drone feed is turned on.")
        for feed in enabled:
            problems.extend(feed.problems())
        names = [feed.name.strip().lower() for feed in self.feeds]
        folders = [drone_folder(feed.name).lower() for feed in self.feeds]
        if len(set(names)) != len(names):
            problems.append("Two feeds have the same drone name. Each drone needs its own name.")
        elif len(set(folders)) != len(folders):
            problems.append("Two drone names only differ by punctuation, so their saved frames would mix. Rename one.")
        if len(self.feeds) > MAX_FEEDS:
            problems.append(f"At most {MAX_FEEDS} feeds can run at once.")
        if not Path(self.model_path).is_file():
            problems.append(f"Model weights not found: {self.model_path}")
        if not 0 < self.min_confidence < 1:
            problems.append("Confidence threshold must be between 0 and 1.")
        if self.frame_stride < 1:
            problems.append("Frame sample rate must be at least 1.")
        if self.discord_enabled and not self.webhook_url.strip():
            problems.append("Discord alerts are on but no webhook URL is set.")
        return problems

    def enabled_feeds(self) -> list[FeedConfig]:
        return [feed for feed in self.feeds if feed.enabled]
