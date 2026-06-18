import logging
from collections import deque
import time

logger = logging.getLogger("latency_monitor")


class LatencyMonitor:
    def __init__(self, window_size=5, threshold_ms=500):
        self.history = deque(maxlen=window_size)
        self.threshold_ms = threshold_ms
        self.in_fallback_mode = False

    def record(self, total_latency_ms):
        self.history.append(total_latency_ms)
        self._evaluate()

    def _evaluate(self):
        if len(self.history) == self.history.maxlen:
            # If all recent requests exceeded threshold
            if all(lat > self.threshold_ms for lat in self.history):
                if not self.in_fallback_mode:
                    logger.warning(
                        "LATENCY ALERT: 5 consecutive requests > 500ms. Enabling fallback mode."
                    )
                    self.in_fallback_mode = True
            else:
                if self.in_fallback_mode:
                    logger.info("Latency recovered. Disabling fallback mode.")
                    self.in_fallback_mode = False

    def should_degrade(self):
        return self.in_fallback_mode
