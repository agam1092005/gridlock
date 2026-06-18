import logging
from .alerting import alert_system

logger = logging.getLogger("data_quality")


class DataQualityMonitor:
    def __init__(self):
        self.total_records = 0
        self.valid_records = 0
        self.missing_counts = {}

    def observe_batch(self, total: int, valid: int, missing_fields: dict):
        self.total_records += total
        self.valid_records += valid

        for k, v in missing_fields.items():
            self.missing_counts[k] = self.missing_counts.get(k, 0) + v

        self._check_thresholds()

    def _check_thresholds(self):
        if self.total_records > 100:
            pass_rate = self.valid_records / self.total_records
            if pass_rate < 0.90:
                alert_system.trigger_alert(
                    "DQ_LOW_PASS_RATE",
                    "WARNING",
                    f"Validation pass rate dropped to {pass_rate*100:.1f}%",
                )

            for field, missing in self.missing_counts.items():
                missing_rate = missing / self.total_records
                if missing_rate > 0.40:
                    alert_system.trigger_alert(
                        f"DQ_HIGH_MISSING_{field}",
                        "WARNING",
                        f"Field {field} missing in {missing_rate*100:.1f}% of records",
                    )


dq_monitor = DataQualityMonitor()
