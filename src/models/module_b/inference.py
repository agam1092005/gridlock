import torch
import time
import logging
from .stgcn_model import STGCNModel
from .graph_loader import GraphLoader

logger = logging.getLogger(__name__)


class ModuleBPredictor:
    def __init__(self, model_path, graph_path=None):
        self.graph_loader = GraphLoader(osm_path=graph_path)
        self.graph = self.graph_loader.load_osm_graph()
        self.num_nodes = self.graph.num_nodes

        self.model = STGCNModel(num_nodes=self.num_nodes)

        import os

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        else:
            logger.warning(f"Model path {model_path} not found. Using uninitialized weights.")

        self.model.eval()

    def generate_heatmap_geojson(self, node_predictions):
        """
        Converts node predictions to a GeoJSON format for the dashboard.
        """
        features = []
        for i, val in enumerate(node_predictions):
            val_float = float(val)
            if val_float < 5.0:  # filter out low noise
                continue
            coords = self.graph_loader.node_coords.get(i, [0.0, 0.0])
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": coords},
                    "properties": {
                        "node_id": i,
                        "weight": val_float / 100.0,
                        "congestion_level": val_float,
                    },
                }
            )

        return {"type": "FeatureCollection", "features": features}

    def predict(self, incident_context):
        """
        End-to-end inference for STGCN with incident context.
        """
        latencies = {}
        start_time = time.time()

        # 1. Fetch historical state (Mocked)
        t0 = time.time()
        # [batch, seq_len, num_nodes, in_features]
        x_history = torch.rand((1, 24, self.num_nodes, 3))
        latencies["history_fetch_ms"] = (time.time() - t0) * 1000

        # 2. Inject incident context
        t0 = time.time()
        lat = float(incident_context.get("latitude", 12.9))
        lon = float(incident_context.get("longitude", 77.5))
        severity = float(incident_context.get("severity_score", 50))

        # Find closest node
        min_dist = float("inf")
        incident_node = 0
        for node_id, coords in self.graph_loader.node_coords.items():
            dist = (coords[0] - lon) ** 2 + (coords[1] - lat) ** 2
            if dist < min_dist:
                min_dist = dist
                incident_node = node_id

        x_history[:, -1, incident_node, 0] = min(1.0, severity / 100.0 + 0.2)
        latencies["incident_injection_ms"] = (time.time() - t0) * 1000

        # 3. Model Inference
        t0 = time.time()
        with torch.no_grad():
            preds = self.model(x_history, self.graph.edge_index)

        # preds: [1, pred_len, num_nodes] -> Take horizon +30m (index 5)
        pred_30m = preds[0, 5, :]
        latencies["inference_ms"] = (time.time() - t0) * 1000

        # 4. Generate output
        t0 = time.time()
        geojson = self.generate_heatmap_geojson(pred_30m)
        latencies["geojson_prep_ms"] = (time.time() - t0) * 1000

        total_time_ms = (time.time() - start_time) * 1000
        latencies["total_ms"] = total_time_ms

        if total_time_ms > 250:
            logger.warning(f"Module B latency exceeded budget: {total_time_ms:.2f}ms")

        return {"geojson": geojson, "latency_ms": latencies}
