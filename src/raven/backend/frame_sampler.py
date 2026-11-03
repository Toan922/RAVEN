"""
Video Ingestion and Frame Sampling filters (FR-04, FR-05, FR-06, FR-13).
Connects to the IR video stream from the ground station, waits for it if it is not up yet,
reconnects if it drops, keeps frames at the configured analysis rate, and pairs each kept frame
with the drone's live telemetry for that moment.
"""

import cv2
import time
from raven.backend.events import PipelineStatus

# how long to wait for the stream to open before reporting feed_missing (FR-13.1)
STREAM_TIMEOUT_MSEC = 10000 # milliseconds

# how long a stream can be stalled for (FR-13.2)
STREAM_STALL_TIMEOUT_MSEC = 3000 # milliseconds

def open_stream(stream_url):
    """
    Open the video stream at stream_url with FFmpeg, giving up after STREAM_TIMEOUT_MSEC if it doesn't open,
    and making each read() give up after STREAM_STALL_TIMEOUT_MSEC if no frame arrives.
    Returns a cv2.VideoCapture; check isOpened() to see whether it connected.
    """
    return cv2.VideoCapture(
        str(stream_url),
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, STREAM_TIMEOUT_MSEC,
         cv2.CAP_PROP_READ_TIMEOUT_MSEC, STREAM_STALL_TIMEOUT_MSEC],
    )

def sample_frame(stream_url, telemetry_receiver, frames_per_second_rate, bus, stop):
    """
    Read the video stream (e.g. udp://127.0.0.1:5000) frame by frame and keep about frames_per_second_rate
    frames per second (FR-05), each matched to the newest live telemetry row from telemetry_receiver (FR-06).
    If the stream doesn't open within STREAM_TIMEOUT_MSEC, publishes one feed_missing PipelineStatus on the bus
    and keeps retrying until it opens (FR-13.1).
    If the feed stalls for STREAM_STALL_TIMEOUT_MSEC, publishes feed_interrupted and reconnects (FR-13.2).
    After either problem, publishes one feed_resumed once the stream is back.
    Yields (frame, frame_index, time, telemetry_row) one sampled frame at a time, where time is the wall-clock
    time (time.time()) the frame was read, and telemetry_row is None if no telemetry arrived within 2 seconds of it.
    Checks stop (a threading.Event) on every frame read and every reconnect attempt, and returns once it is set:
    within about one frame while streaming, STREAM_STALL_TIMEOUT_MSEC if the feed is stalled, and
    STREAM_TIMEOUT_MSEC while waiting for it to open. The stream is released whenever the generator ends.
    """

    # check if there has been a stream issue reported
    problem_reported = False

    while True:
        # open the video
        capture = open_stream(stream_url)

        # keep retrying until the stream opens, reporting feed_missing only on the first failure
        while not capture.isOpened():
            if not problem_reported:
                bus.dispatch(PipelineStatus(source="video_ingestion", status="feed_missing", message=f"No feed from {stream_url} within {STREAM_TIMEOUT_MSEC // 1000} seconds, still waiting"))
                problem_reported = True

            # check if the stop event has been set
            if stop.is_set():
                return

            capture = open_stream(stream_url)

        # publish feed_resumed if a feed problem (missing or interrupted) was reported
        if problem_reported:
            bus.dispatch(PipelineStatus(source="video_ingestion", status="feed_resumed", message=f"Feed from {stream_url} is now available"))
            problem_reported = False

        # frame counter, starts at -1 so the first frame read is index 0
        frame_index = -1

        # get the frames per second of the video, some streams report 0 so fall back to 30 to avoid dividing by zero
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

        # keep every frame_stride-th frame so about frames_per_second_rate frames are analysed each second (FR-05),
        # never below 1 so a rate higher than the stream's fps keeps every frame instead of dividing by zero
        frame_stride = max(1, round(fps / frames_per_second_rate))

        try:
            while True:
                # Read a frame from the video
                ok, frame = capture.read()

                # check if the stop event has been set
                if stop.is_set():
                    return

                # If the frame was not read successfully, break the loop
                if not ok:
                    break

                # count every frame read, including the ones we skip, so sampling keeps every frame_stride-th frame
                frame_index += 1

                # only keep every frame_stride-th frame
                if frame_index % frame_stride != 0:
                    continue

                # stamp the frame with RAVEN's clock, the same clock the telemetry receiver stamps rows with,
                # and get the newest telemetry row for that moment
                current_time = time.time()
                telemetry_row = telemetry_receiver.at_time(current_time)

                # yield the frame along with its index, timestamp, and corresponding telemetry row
                yield frame, frame_index, current_time, telemetry_row

        finally:
            capture.release()

        # only reach here if there is a failure to read frames from the video stream
        bus.dispatch(PipelineStatus(source="video_ingestion", status="feed_interrupted", message=f"Failed to read frames from {stream_url}"))
        problem_reported = True

