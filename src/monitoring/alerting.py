import logging
import time
from collections import defaultdict

logger = logging.getLogger("alerting")


class AlertingSystem:
    def __init__(self):
        self.last_alert_times = defaultdict(float)
        self.dedup_window_sec = 15 * 60  # 15 minutes

    def trigger_alert(self, alert_id: str, severity: str, message: str):
        now = time.time()

        # Deduplication
        if now - self.last_alert_times[alert_id] < self.dedup_window_sec:
            return  # Skip spam

        self.last_alert_times[alert_id] = now

        # Log to stderr (acting as our mock PagerDuty/Slack routing)
        if severity == "CRITICAL":
            logger.critical(f"[ALERT: {alert_id}] {message} - ESCALATING TO ON-CALL")
        else:
            logger.error(f"[ALERT: {alert_id}] {message}")


alert_system = AlertingSystem()
