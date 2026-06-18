import logging
from collections import deque

logger = logging.getLogger("production_monitor")


class ProductionModelMonitor:
    def __init__(self, window_sizes=[100, 500]):
        self.window_sizes = window_sizes
        self.history = deque(maxlen=max(window_sizes))
        self.baseline_mae = 3.5

    def log_actual_outcome(self, incident_id: str, predicted_val: float, actual_val: float):
        """
        Record the ground truth outcome once an incident clears to measure drift.
        """
        error = abs(predicted_val - actual_val)
        self.history.append({"incident_id": incident_id, "error": error})

        self._evaluate_drift()

    def _evaluate_drift(self):
        for window in self.window_sizes:
            if len(self.history) >= window:
                # get last N items
                recent = list(self.history)[-window:]
                avg_error = sum(item["error"] for item in recent) / window

                # Check for 5% degradation against baseline
                if avg_error > self.baseline_mae * 1.05:
                    logger.error(
                        f"MODEL DRIFT DETECTED: MAE over last {window} predictions is {avg_error:.2f} (Baseline: {self.baseline_mae:.2f})"
                    )
                    # In production, this fires a PagerDuty or Slack alert
