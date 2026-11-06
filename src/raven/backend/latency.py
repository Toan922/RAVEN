"""
Detection latency for the operator console (NFR-01): how long the model takes per frame, as the 90th percentile
of the most recent frames, so one slow frame doesn't raise a warning but a steady slowdown does.
"""

from collections import deque

# how many of the newest inference times the percentile is taken over
WINDOW = 50


class LatencyTracker:
    def __init__(self, window=WINDOW):
        self._times = deque(maxlen=window)

    def record(self, seconds: float):
        self._times.append(seconds)

    def p90_ms(self) -> float:
        """
        The 90th percentile of the recent inference times in milliseconds, 0 before the first frame.
        """
        if not self._times:
            return 0.0
        ordered = sorted(self._times)
        return 1000 * ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
