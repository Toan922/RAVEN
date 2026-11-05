"""
Background work for the GUI, so model inference and the event bus never freeze the window.

The console runs one Detection Event Bus (a bubus EventBus on its own thread, see raven.backend.bus):
each drone feed's IngestWorker publishes HumanDetected and PipelineStatus events on it, and the
subscribers here (the alert list, Discord) plus the Detection Log react to them.
Everything reports back to the GUI thread through Qt signals, which are safe to emit from any thread.
"""
import asyncio
import threading
import time

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QImage

from raven.backend.config import FeedConfig, RavenConfig
from raven.backend.detection_log import event_record
from raven.backend.events import HumanDetected, PipelineStatus

# how long stopping Discord waits for queued alerts to send, the same limit raven.pipeline uses at shutdown
DISCORD_SHUTDOWN_TIMEOUT_S = 5.0


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


class AlertFeed(QObject):
    """
    The operator console's subscriber on the event bus: passes every HumanDetected event to the alert list,
    and every PipelineStatus (feed and detection problems) to the log.
    """
    events = Signal(list)   # list of (event id, event record dict), oldest first, see detection_log.event_record
    status = Signal(str)    # a readable PipelineStatus line for the log

    def __init__(self, bus, parent=None):
        super().__init__(parent)
        bus.on(HumanDetected, self._on_human_detected)
        bus.on(PipelineStatus, self._on_status)

    def _on_human_detected(self, event):
        record = event_record(event)
        self.events.emit([(record["id"], record)])

    def _on_status(self, event):
        drone = f"[{event.drone}] " if event.drone else ""
        self.status.emit(f"{drone}{event.status.replace('_', ' ')} ({event.source}): {event.message}")


# how many frames a second each feed sends to the Live view
LIVE_FPS = 15


class IngestWorker(QThread):
    """
    Run raven.backend.file_ingest on one drone feed with the shared settings: load telemetry, open the source,
    load the model, then process the video until it ends or the operator presses Stop. Detections are published
    on the event bus; every signal here carries the drone name.
    """
    progress = Signal(str, dict)          # drone, stats
    frame_ready = Signal(str, object, dict)  # drone, QImage, {"frame_index", "time", "fps", "total_frames", ...}
    source_info = Signal(str, str, dict)  # drone, source, info
    log = Signal(str)
    failed = Signal(str, str)             # drone, message

    def __init__(self, config: RavenConfig, feed: FeedConfig, bus, model_factory=None, parent=None):
        """
        bus: the console's event bus (a ThreadSafeBus, so this thread can publish to it).
        model_factory(model_path) returns the detector; None loads the YOLO model with raven.backend.detection.
        """
        super().__init__(parent)
        self._config = config
        self.feed = feed
        self.name = feed.name
        self._bus = bus
        self._model_factory = model_factory
        self._stop = threading.Event()
        self._last_frame_sent = 0.0

    def stop(self):
        self._stop.set()

    def stopped_by_operator(self) -> bool:
        return self._stop.is_set()

    def _send_frame(self, telemetry, info, frame_index, frame, boxes, checked):
        """
        Pass a frame to the live view, at most LIVE_FPS times a second (a checked frame always goes through
        when it's been at least half that long, so new detections show up promptly).
        """
        now = time.monotonic()
        gap = now - self._last_frame_sent
        if gap < 1 / LIVE_FPS and not (checked and gap >= 0.5 / LIVE_FPS):
            return
        self._last_frame_sent = now
        from raven.backend.file_ingest import telemetry_at_time

        fps = info["fps"] or 30.0
        seconds = frame_index / fps
        row = telemetry_at_time(telemetry, seconds)
        height, width = frame.shape[:2]
        image = QImage(frame.data, width, height, frame.strides[0], QImage.Format_BGR888).copy()
        self.frame_ready.emit(self.name, image, {
            "frame_index": frame_index, "time": seconds, "fps": fps, "total_frames": info["frame_count"],
            "boxes": boxes, "checked": checked, "lat": row.get("latitude"), "lon": row.get("longitude"),
            "battery": row.get("battery_percent"), "source": self.feed.source,
        })

    def _log(self, message: str):
        self.log.emit(f"[{self.name}] {message}")

    def _load_model(self):
        config = self._config
        if self._model_factory is not None:
            return self._model_factory(config.model_path)
        try:
            import ultralytics  # noqa: F401  (only checks that it's installed)
        except ImportError:
            raise RuntimeError("The ultralytics package is not installed. Run: pip install ultralytics") from None
        from raven.backend.detection import load_model

        # each feed loads its own copy: a YOLO model isn't safe to share between threads
        return load_model(self._bus, config.model_path)

    def run(self):
        from raven.backend import file_ingest

        config, feed = self._config, self.feed
        try:
            telemetry = file_ingest.load_telemetry(feed.telemetry_path)
            self._log(f"Loaded {len(telemetry)} telemetry rows from {feed.telemetry_path}")

            info = file_ingest.probe_source(feed.source)
            self.source_info.emit(self.name, feed.source, info)
            self._log(f"Opened {feed.source} ({info['width']}x{info['height']}, {info['fps']:.1f} fps)")

            self._log(f"Loading model {config.model_path} ...")
            model = self._load_model()

            if self._stop.is_set():
                self._log("Ingest cancelled before it started")
                return

            self._log(f"Ingest started: confidence >= {config.min_confidence:.2f}, "
                      f"checking every {config.frame_stride} frame(s)")
            stats = file_ingest.ingest_video(
                feed.source, telemetry, model, self._bus,
                config.min_confidence, config.frame_stride, config.output_dir,
                stop_event=self._stop,
                on_progress=lambda stats: self.progress.emit(self.name, stats),
                drone=self.name,
                on_frame=lambda *args: self._send_frame(telemetry, info, *args),
                realtime=config.realtime,
            )

            reason = "stopped by operator" if self._stop.is_set() else "end of video"
            self._log(f"Ingest finished ({reason}): {stats['frames_read']} frames read, "
                      f"{stats['frames_checked']} checked, {stats['events_published']} alerts published")

        except Exception as error:  # surface every failure in the GUI instead of killing the thread silently
            self.failed.emit(self.name, str(error) or type(error).__name__)


class DiscordAlerts(QObject):
    """
    Discord alerts for the console: while on, raven.discord_bot.subscriber.DiscordSubscriber receives HumanDetected
    from the bus with the operator's webhook and cooldown, and sends alerts from its own task on the bus thread,
    retrying failed sends until they go through.
    """
    state_changed = Signal(str, str)   # status light color, detail
    log = Signal(str)

    def __init__(self, bus, parent=None):
        super().__init__(parent)
        self._bus = bus
        self._running = False
        # only used on the bus thread
        self._subscriber = None
        self._task = None
        # bubus has no way to remove a handler, so one handler stays registered and forwards while Discord is on
        bus.on(HumanDetected, self._on_human_detected)

    async def _on_human_detected(self, event):
        subscriber = self._subscriber
        if subscriber is not None:
            await subscriber.on_human_detected(event)

    def is_running(self) -> bool:
        return self._running

    def start(self, config: RavenConfig):
        if self._running:
            return
        import discord
        from raven.discord_bot.subscriber import DiscordSubscriber

        try:
            webhook = discord.SyncWebhook.from_url(config.webhook_url)
        except Exception as error:  # a malformed URL
            message = str(error) or type(error).__name__
            self.state_changed.emit("red", message)
            self.log.emit(f"Discord alerts not started: {message}")
            return

        def sent(event):
            people = len(event.bounding_boxes)
            drone = event.drone or "unnamed drone"
            when = f"video {event.video_time:.1f} s" if event.video_time is not None else time.strftime(
                "%H:%M:%S", time.localtime(event.time))
            self.state_changed.emit("green", "Sending alerts (cooldown per drone)")
            self.log.emit(f"Sent Discord alert for {drone}: {people} person(s) at {when}")

        def failed(event, error):
            self.state_changed.emit("amber", f"Retrying: {error}")
            self.log.emit(f"Discord alert for {event.drone or 'unnamed drone'} failed ({error}), retrying")

        def install():
            self._subscriber = DiscordSubscriber(webhook, cooldown=config.alert_cooldown_s, on_sent=sent, on_error=failed)
            self._task = asyncio.get_running_loop().create_task(self._subscriber.deliver_alerts())

        self._bus.call(install)
        self._running = True
        self.state_changed.emit("green", "Sending alerts (cooldown per drone)")
        self.log.emit(f"Discord alerts on (cooldown {config.alert_cooldown_s:g} s per drone)")

    def stop(self, timeout=DISCORD_SHUTDOWN_TIMEOUT_S, wait=False):
        """
        Hand the events still on the bus to Discord, give queued alerts up to timeout seconds to send, then turn
        Discord off. Returns right away unless wait is True; the log says which alerts could not be sent.
        """
        if not self._running:
            return
        self._running = False

        async def shut_down():
            subscriber, task = self._subscriber, self._task
            try:
                await self._bus.event_bus.wait_until_idle(timeout)
                await asyncio.wait_for(subscriber.alerts.join(), timeout)
            except (TimeoutError, asyncio.TimeoutError):
                pass
            unsent = []
            while not subscriber.alerts.empty():
                unsent.append(subscriber.alerts.get_nowait())
            task.cancel()
            # a new run may have turned Discord back on meanwhile, with its own subscriber
            if self._subscriber is subscriber:
                self._subscriber = None
                self._task = None
            return unsent

        def finished(future):
            try:
                unsent = future.result()
            except Exception as error:  # report it instead of losing it in the bus thread
                self.log.emit(f"Stopping Discord alerts failed: {error}")
                unsent = []
            for event in unsent:
                self.log.emit(f"Discord alert not sent: {event.drone or 'unnamed drone'}, {event.frame}")
            if not self._running:
                self.state_changed.emit("gray", "Off")
                self.log.emit("Discord alerts off")

        future = self._bus.submit(shut_down())
        future.add_done_callback(finished)
        if wait:
            try:
                future.result(timeout * 2 + 2)
            except Exception:
                pass


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
