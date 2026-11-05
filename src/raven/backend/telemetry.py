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
    
    def run(self):
        """
        Loops forever, receive, check, stamp telemetry with arrival time, and
        store as latest
        Bad messages are skipped
        """
        
        while True:
            # wait for the next message
            data = self.sock.recv(4096)
            
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
