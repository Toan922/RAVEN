"""
Drone telemetry for geotagging detections (FR-06).
TelemetryReceiver listens for live telemetry from the ground station over UDP and hands each sampled frame
the newest row, if it is at most 2 seconds old.
"""

import json
import socket
import time

# define the maximum allowed time difference between a frame and its corresponding telemetry row
MAX_TIME_DIFFERENCE = 2  # seconds

class TelemetryReceiver:
    """
    Listens for live telemetry over UDP and keeps the newest row
    Call run() in background thread and at_time() from the sampler
    """

    def __init__(self, address):
        """
        Binds the UDP socket to the given address.
        No telemetry rows have been received yet.
        """
        

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(address)
        
        # represent that no rows have been received yet
        self.latest = None

        # set by close(), so run() knows the socket was closed on purpose
        self._closed = False

    def run(self):
        """
        Loops forever, receive, check, stamp telemetry with arrival time, and
        store as latest
        Bad messages are skipped
        """
        
        while True:
            # wait for the next message; close() makes recv fail, which ends the loop
            try:
                data = self.sock.recv(4096)
            except OSError:
                if self._closed:
                    return
                # Windows reports an earlier send to a closed port as an error on the next recv, skip it
                continue
            
            try:
                # parse the json message
                row = json.loads(data)

                # ensure the row has latitude and longitude
                if "latitude" not in row or "longitude" not in row:
                    # raise an error if the row is missing latitude or longitude
                    raise ValueError(f"Telemetry row is missing latitude or longitude: {row}")

                # convert row to float
                for key in row:
                    row[key] = float(row[key])


            except (ValueError, TypeError):
                # skip bad messages
                continue

            # stamp based on RAVEN's clock
            row["t"] = time.time()

            # update the latest row with the new timestamp
            self.latest = row

    def close(self):
        """
        Stop receiving: run() returns and the port is freed for the next run.
        """
        self._closed = True
        try:
            self.sock.close()
        except OSError:
            pass

    def at_time(self, frame_time):
        """
        Return the newest telemetry row if it arrived at most MAX_TIME_DIFFERENCE seconds before frame_time (FR-06).
        Return None if no row has been received yet, or the newest one is older than that (telemetry lost).
        """

        # read self.latest once, so the check and the return use the same row even if run() replaces it in between
        latest = self.latest.copy() if self.latest is not None else None

        if latest is None or frame_time - latest["t"] > MAX_TIME_DIFFERENCE:
            return None
        return latest


def listen_once(address, timeout=3.0):
    """
    Wait up to timeout seconds for one valid telemetry message on address (host, port) and return it as a dict.
    Raises TimeoutError if none arrives, OSError if the port can't be opened (for example it is already in use).
    Used by the console's Test button.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(address)
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"No telemetry on udp://{address[0]}:{address[1]} within {timeout:g} s")
            sock.settimeout(left)
            try:
                data = sock.recv(4096)
            except socket.timeout:
                continue
            except ConnectionResetError:
                continue
            try:
                row = json.loads(data)
                if "latitude" in row and "longitude" in row:
                    return {key: float(value) for key, value in row.items()}
            except (ValueError, TypeError, AttributeError):
                continue
    finally:
        sock.close()
