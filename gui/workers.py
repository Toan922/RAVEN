"""
Background threads for the GUI, so model inference and Redis reads never freeze the window.
Each worker reports back to the GUI thread through Qt signals.
"""
import json
import threading

import redis
from PySide6.QtCore import QObject, QThread, Signal
from redis.backoff import ExponentialBackoff
from redis.retry import Retry

from backend.config import FeedConfig, RavenConfig


def redis_client(config: RavenConfig, retries: int = 0) -> redis.Redis:
    """
    Create a Redis client from the operator settings, with short timeouts so failures surface quickly.
    redis-py's default retries make an unreachable server take ~50 s to report, so they are off unless asked for.
    """
    return redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=5,
        retry=Retry(ExponentialBackoff(cap=2), retries),
    )


def redis_unreachable_message(config: RavenConfig, error: Exception) -> str:
    return (f"Redis is not reachable at {config.redis_host}:{config.redis_port} ({error}). "
            "Start the Redis server and try again.")


class FunctionWorker(QThread):
    """
    Run one function off the GUI thread and report its return value or error message.
    """
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            self.succeeded.emit(self._fn())
        except Exception as error:  # report every failure to the operator instead of crashing the thread
            self.failed.emit(str(error) or type(error).__name__)


class JobRunner(QObject):
    """
    Keep short background jobs alive until they finish, and let the window wait for them on close.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._jobs = set()

    def run(self, fn, on_success=None, on_failure=None):
        job = FunctionWorker(fn)
        if on_success is not None:
            job.succeeded.connect(on_success)
        if on_failure is not None:
            job.failed.connect(on_failure)
        job.finished.connect(lambda: self._jobs.discard(job))
        self._jobs.add(job)
        job.start()
        return job

    def wait_all(self, timeout_ms=3000):
        for job in list(self._jobs):
            job.wait(timeout_ms)


class AlertFeed(QThread):
    """
    Read every detection event from the Redis stream, oldest first, then keep waiting for new ones.

    It uses plain XREAD instead of a consumer group: it never acknowledges or takes messages from the
    Discord bot's group, and every launch replays the whole stream so the alert list is rebuilt.
    Reconnects on its own if Redis goes away.
    """
    events = Signal(list)                   # list of (message_id, event dict), oldest first
    connection_changed = Signal(bool, str)  # connected?, detail
    log = Signal(str)

    def __init__(self, config: RavenConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        last_id = "0"
        client = None
        connected = None

        while not self._stop.is_set():
            try:
                if client is None:
                    client = redis_client(self._config)
                    client.ping()
                if connected is not True:
                    connected = True
                    self.connection_changed.emit(True, f"{self._config.redis_host}:{self._config.redis_port}")

                # wait up to a second for new events so a stop request is noticed quickly
                reply = client.xread({self._config.stream_name: last_id}, count=500, block=1000)

                batch = []
                for _, entries in reply or []:
                    for message_id, data in entries:
                        last_id = message_id
                        try:
                            batch.append((message_id, json.loads(data["event"])))
                        except (KeyError, ValueError):
                            self.log.emit(f"Skipped a malformed event ({message_id}) on the stream")
                if batch:
                    self.events.emit(batch)

            except redis.exceptions.RedisError as error:
                client = None
                if connected is not False:
                    connected = False
                    self.connection_changed.emit(False, str(error))
                # retry every two seconds until Redis is back or the feed is stopped
                self._stop.wait(2)


class IngestWorker(QThread):
    """
    Run backend.ingest on one drone feed with the shared settings: load telemetry, open the source, load the
    model, then process the video until it ends or the operator presses Stop. Every signal carries the drone name.
    """
    progress = Signal(str, dict)          # drone, stats
    source_info = Signal(str, str, dict)  # drone, source, info
    log = Signal(str)
    failed = Signal(str, str)             # drone, message

    def __init__(self, config: RavenConfig, feed: FeedConfig, model_factory=None, parent=None):
        """
        model_factory(model_path) returns the detector; None loads the YOLO model with ultralytics.
        """
        super().__init__(parent)
        self._config = config
        self.feed = feed
        self.name = feed.name
        self._model_factory = model_factory
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped_by_operator(self) -> bool:
        return self._stop.is_set()

    def _log(self, message: str):
        self.log.emit(f"[{self.name}] {message}")

    def run(self):
        from backend import ingest

        config, feed = self._config, self.feed
        try:
            telemetry = ingest.load_telemetry(feed.telemetry_path)
            self._log(f"Loaded {len(telemetry)} telemetry rows from {feed.telemetry_path}")

            info = ingest.probe_source(feed.source)
            self.source_info.emit(self.name, feed.source, info)
            self._log(f"Opened {feed.source} ({info['width']}x{info['height']}, {info['fps']:.1f} fps)")

            # check reachability fast, then publish with a few retries so a brief hiccup doesn't end the run
            try:
                redis_client(config).ping()
            except redis.exceptions.RedisError as error:
                raise RuntimeError(redis_unreachable_message(config, error)) from None
            client = redis_client(config, retries=3)

            self._log(f"Loading model {config.model_path} ...")
            if self._model_factory is not None:
                model = self._model_factory(config.model_path)
            else:
                try:
                    from ultralytics import YOLO
                except ImportError:
                    raise RuntimeError("The ultralytics package is not installed. Run: pip install ultralytics") from None
                # each feed loads its own copy: a YOLO model isn't safe to share between threads
                model = YOLO(config.model_path)

            if self._stop.is_set():
                self._log("Ingest cancelled before it started")
                return

            self._log(f"Ingest started: confidence >= {config.min_confidence:.2f}, "
                      f"checking every {config.frame_stride} frame(s)")
            stats = ingest.ingest_video(
                feed.source, telemetry, model, client,
                config.min_confidence, config.frame_stride,
                stop_event=self._stop,
                on_progress=lambda stats: self.progress.emit(self.name, stats),
                output_dir=config.output_dir,
                stream_name=config.stream_name,
                drone=self.name,
            )

            reason = "stopped by operator" if self._stop.is_set() else "end of video"
            self._log(f"Ingest finished ({reason}): {stats['frames_read']} frames read, "
                      f"{stats['frames_checked']} checked, {stats['events_published']} alerts published")

        except Exception as error:  # surface every failure in the GUI instead of killing the thread silently
            self.failed.emit(self.name, str(error) or type(error).__name__)


class DiscordWorker(QThread):
    """
    Run the Discord bot's subscriber (discord_bot/subscriber.py) with the operator's webhook and cooldown.
    """
    log = Signal(str)
    failed = Signal(str)

    def __init__(self, config: RavenConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        config = self._config
        try:
            import discord
            from discord_bot import subscriber

            webhook = discord.SyncWebhook.from_url(config.webhook_url)
            client = redis_client(config)
            subscriber.create_group(client, config.stream_name)
            self.log.emit(f"Discord alerts on (cooldown {config.alert_cooldown_s:g} s of video time, per drone)")

            def on_alert(event):
                people = len(event.get("bounding_boxes", []))
                drone = event.get("drone") or "unnamed drone"
                self.log.emit(f"Sent Discord alert for {drone}: {people} person(s) at {event.get('time', 0):.1f} s")

            subscriber.listen(client, webhook, cooldown=config.alert_cooldown_s,
                              stream_name=config.stream_name, stop_event=self._stop, on_alert=on_alert)
            self.log.emit("Discord alerts off")

        except Exception as error:  # a failed send stops the bot; tell the operator why
            self.failed.emit(str(error) or type(error).__name__)


def send_test_alert(webhook_url: str) -> str:
    """
    Post a short test message to the Discord webhook. Raises if it can't be sent.
    """
    import discord

    webhook = discord.SyncWebhook.from_url(webhook_url)
    webhook.send(embed=discord.Embed(
        title="RAVEN test alert",
        description="The operator console can post to this channel.",
        color=discord.Color.blue(),
    ))
    return "Test alert sent"
