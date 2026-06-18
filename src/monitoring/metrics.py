from collections import defaultdict
import threading


class MetricsRegistry:
    def __init__(self):
        self.counters = defaultdict(int)
        self.histograms = defaultdict(list)
        self.lock = threading.Lock()

    def inc_counter(self, name: str, labels: str = ""):
        key = f"{name}{{{labels}}}" if labels else name
        with self.lock:
            self.counters[key] += 1

    def observe_histogram(self, name: str, value: float, labels: str = ""):
        key = f"{name}{{{labels}}}" if labels else name
        with self.lock:
            self.histograms[key].append(value)
            # Cap histogram history to prevent memory leak in mock
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]

    def render_prometheus_text(self):
        lines = []
        with self.lock:
            for key, val in self.counters.items():
                name = key.split("{")[0]
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{key} {val}")

            for key, vals in self.histograms.items():
                if not vals:
                    continue
                name = key.split("{")[0]
                lines.append(f"# TYPE {name} histogram")
                # Basic mock buckets
                b100 = sum(1 for v in vals if v <= 100)
                b500 = sum(1 for v in vals if v <= 500)
                inf = len(vals)
                lines.append(f'{name}_bucket{{le="100"}} {b100}')
                lines.append(f'{name}_bucket{{le="500"}} {b500}')
                lines.append(f'{name}_bucket{{le="+Inf"}} {inf}')
                lines.append(f"{name}_sum {sum(vals)}")
                lines.append(f"{name}_count {inf}")

        return "\n".join(lines)


metrics_registry = MetricsRegistry()
