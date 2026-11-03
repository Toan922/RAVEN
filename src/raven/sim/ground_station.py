"""
Simulated IR drone ground station for development.
Streams the demo video over UDP with ffmpeg and sends live drone telemetry as JSON over a second UDP port,
the two feeds RAVEN receives from a real ground station (C-02, FR-04, FR-06).
"""

import json
import math
import socket
import subprocess
import time
from pathlib import Path

# get the root directory
ROOT_DIR = Path(__file__).resolve().parents[3]

# get the demo video path
VIDEO_PATH = ROOT_DIR / "videos" / "hit-uav.mp4"

# set up for where RAVEN will listen for the video and telemetry streams
VIDEO_URL = "udp://127.0.0.1:5000"
TELEMETRY_ADDRESS = ("127.0.0.1", 5001)

# configure the telemetry send rate
TELEMETRY_INTERVAL_SECONDS = 0.1  # send telemetry every 0.1 seconds (10 Hz is average transmission rate)

# configure the starting telemetry
START_LATITUDE = 47.4110
START_LONGITUDE = 10.2820
ALTITUDE_AGL_M = 85 # in meters
HEADING_DEG = 95
CAMERA_DEPRESSION_DEG = 50
SPEED_MPS = 5 # meters per second

def start_video_stream():
    """
    Streams VIDEO_PATH to VIDEO_URL in real time with ffmpeg, looping forever.
    Returns the running process so the caller can stop it with terminate().
    """

    return subprocess.Popen([
        "ffmpeg", "-loglevel", "error",
        "-re", "-stream_loop", "-1",
        "-i", str(VIDEO_PATH),
        "-c:v", "libx264", "-tune", "zerolatency",
        "-f", "mpegts", VIDEO_URL,
    ])

def send_telemetry():
    """
    Send one JSON telemetry row to TELEMETRY_ADDRESS every TELEMETRY_INTERVAL_SECONDS (about 10 Hz),
    flying in a straight line along HEADING_DEG at SPEED_MPS from START_LATITUDE, START_LONGITUDE.
    Each row has the same fields as the telemetry CSV, without "t": RAVEN stamps rows with its own clock on arrival.
    Runs until interrupted (Ctrl+C).
    """

    # open a UDP socket for sending telemetry
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # keep track of the drone's position
    latitude = START_LATITUDE
    longitude = START_LONGITUDE

    # instantiate the telemetry data
    row = {
        "latitude": START_LATITUDE,
        "longitude": START_LONGITUDE,
        "altitude_agl_m": ALTITUDE_AGL_M,
        "heading_deg": HEADING_DEG,
        "camera_depression_deg": CAMERA_DEPRESSION_DEG
    }

    # get the distance the drone travels per send interval
    distance_m = SPEED_MPS * TELEMETRY_INTERVAL_SECONDS

    while True:
        # send the telemetry data over UDP
        sock.sendto(json.dumps(row).encode(), TELEMETRY_ADDRESS)

        # update the drone's position based on the distance traveled
        latitude += distance_m * math.cos(math.radians(HEADING_DEG)) / 111_320
        longitude += distance_m * math.sin(math.radians(HEADING_DEG)) / (111_320 * math.cos(math.radians(latitude)))

        # update the telemetry row with the new position
        row["latitude"] = latitude
        row["longitude"] = longitude
        time.sleep(TELEMETRY_INTERVAL_SECONDS)

if __name__ == "__main__":
    proc = start_video_stream()
    
    # send the telemetry data over UDP
    try:
        send_telemetry()
    finally:
        proc.terminate()
        proc.wait()
