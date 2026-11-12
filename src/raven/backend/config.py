"""
Operator settings for a RAVEN run, shared by the GUI, presets and the command line.
"""
import json
import re
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path

# get the root directory of the project
ROOT_DIR = Path(__file__).resolve().parents[3]

# the trained person detector
DEFAULT_MODEL_PATH = ROOT_DIR / "runs/detect/runs/train/yolov8n_raven/weights/best.pt"

# save the frames with detections here, one subfolder per video
DEFAULT_OUTPUT_DIR = ROOT_DIR / "runs" / "ingest"

# every HumanDetected event is appended to this file in the saved frames folder, so the console can show
# earlier alerts again after a restart
DETECTION_LOG_NAME = "detections.jsonl"

# settings that are secrets and never written to presets
SECRET_FIELDS = {"webhook_url"}

# how many drone feeds can run at once
MAX_FEEDS = 3

# frames analysed per second of video (FR-05 asks for 1 to 2)
MIN_ANALYSIS_FPS = 1.0
MAX_ANALYSIS_FPS = 2.0


def is_udp_address(text: str) -> bool:
    return text.strip().lower().startswith("udp://")


def parse_udp_address(text: str) -> tuple[str, int]:
    """
    Split "udp://host:port" into (host, port). Raises ValueError with a readable message if it isn't one.
    """
    rest = text.strip()[len("udp://"):] if is_udp_address(text) else ""
    host, _, port = rest.rpartition(":")
    host = host.strip("[]") or ""
    if not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError(f"not a udp://host:port address: {text.strip()}")
    return host, int(port)


def udp_port(text: str) -> int | None:
    """
    The port of a udp:// link, or None for anything else (files, rtsp://, a malformed link).
    """
    try:
        return parse_udp_address(text)[1]
    except ValueError:
        return None


def stride_for(video_fps: float, analysis_fps: float) -> int:
    """
    How many frames to step between checks so about analysis_fps frames are analysed each second.
    """
    if not video_fps or video_fps <= 0 or not analysis_fps or analysis_fps <= 0:
        return 1
    return max(1, round(video_fps / analysis_fps))


def drone_folder(name: str) -> str:
    """
    A file-system safe folder name for a drone, so each drone's saved frames stay apart.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("_") or "drone"


@dataclass
class FeedConfig:
    """
    One drone feed: its name (tags every alert), video file or stream link, and telemetry: a CSV file, or
    udp://host:port for live telemetry from the ground station.
    """
    name: str = "Drone 1"

    # a video file path or a stream link (rtsp://, http://, ...)
    source: str = ""

    # telemetry CSV with at least t, latitude, longitude columns, or udp://host:port for live JSON telemetry
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
            problems.append(f"{label}: no telemetry is set (a CSV file or udp://host:port).")
        elif is_udp_address(self.telemetry_path):
            try:
                parse_udp_address(self.telemetry_path)
            except ValueError as error:
                problems.append(f"{label}: telemetry {error}")
            if self.source.strip() and "://" not in self.source:
                problems.append(f"{label}: live UDP telemetry needs a stream link, not a video file. "
                                "Use the flight's telemetry CSV with a recorded video.")
        elif not Path(self.telemetry_path).is_file():
            problems.append(f"{label}: telemetry file not found: {self.telemetry_path}")
        return problems

    def is_live(self) -> bool:
        """
        True when telemetry comes over UDP from a ground station, so the video is a live stream too.
        """
        return is_udp_address(self.telemetry_path)


def port_clashes(feeds: list[FeedConfig]) -> list[str]:
    """
    Every UDP port the console listens on (live telemetry, and udp:// video) can only be used once.
    """
    users: dict[int, list[str]] = {}
    for feed in feeds:
        label = feed.name.strip() or "A feed"
        for kind, link in (("telemetry", feed.telemetry_path), ("video", feed.source)):
            port = udp_port(link)
            if port is not None:
                users.setdefault(port, []).append(f"{label} {kind}")
    return [f"UDP port {port} is used by more than one input ({', '.join(names)}). Give each its own port."
            for port, names in users.items() if len(names) > 1]


@dataclass
class SearchSession:
    """
    The search this run belongs to: a name for it, the area being searched, and when it was created.
    """
    name: str = ""
    area: str = ""

    # seconds since the epoch, 0 until the session is first started
    created: float = 0.0

    @classmethod
    def from_dict(cls, data) -> "SearchSession":
        session = cls()
        if not isinstance(data, dict):
            return session
        session.name = str(data.get("name", ""))
        session.area = str(data.get("area", ""))
        try:
            session.created = float(data.get("created", 0) or 0)
        except (TypeError, ValueError):
            session.created = 0.0
        return session


@dataclass
class RavenConfig:
    # up to MAX_FEEDS drone feeds; confidence, sample rate and everything below are shared by all of them
    feeds: list = field(default_factory=lambda: [FeedConfig()])

    # the search session (name and area) this run belongs to
    session: SearchSession = field(default_factory=SearchSession)

    # detection
    min_confidence: float = 0.7

    # frames analysed per second of video, between MIN_ANALYSIS_FPS and MAX_ANALYSIS_FPS
    analysis_fps: float = 1.0
    model_path: str = str(DEFAULT_MODEL_PATH)

    # read video files at their real speed, like a live drone (streams always arrive in real time)
    realtime: bool = False

    # alerts
    alert_cooldown_s: float = 2.0
    discord_enabled: bool = False
    webhook_url: str = ""

    # saved frames (one subfolder per drone) and the detection log
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
        config.session = SearchSession.from_dict(data.get("session"))
        for f in fields(cls):
            if f.name in ("feeds", "session"):
                continue
            if f.name in data:
                default = getattr(config, f.name)
                value = data[f.name]
                if isinstance(default, bool):
                    value = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")
                else:
                    value = type(default)(value)
                setattr(config, f.name, value)
        # presets saved before analysis_fps existed had a frame_stride instead; they keep the 1 fps default
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
        problems.extend(port_clashes(enabled))
        if len(self.feeds) > MAX_FEEDS:
            problems.append(f"At most {MAX_FEEDS} feeds can run at once.")
        if not Path(self.model_path).is_file():
            problems.append(f"Model weights not found: {self.model_path}")
        if not 0 < self.min_confidence < 1:
            problems.append("Confidence threshold must be between 0 and 1.")
        if not MIN_ANALYSIS_FPS <= self.analysis_fps <= MAX_ANALYSIS_FPS:
            problems.append(f"Analysis rate must be between {MIN_ANALYSIS_FPS:g} and {MAX_ANALYSIS_FPS:g} "
                            "frames per second.")
        if self.discord_enabled and not self.webhook_url.strip():
            problems.append("Discord alerts are on but no webhook URL is set.")
        return problems

    def enabled_feeds(self) -> list[FeedConfig]:
        return [feed for feed in self.feeds if feed.enabled]

    def detection_log_path(self) -> Path:
        return Path(self.output_dir) / DETECTION_LOG_NAME
