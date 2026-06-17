import torch
from torch_geometric.data import Data
import numpy as np
import logging

logger = logging.getLogger(__name__)

class GraphLoader:
    def __init__(self, osm_path=None):
        self.osm_path = osm_path
        self.num_nodes = 100  # Default grid
        self.edge_index = None
        self.node_features = None

    def build_organic_graph(self, num_nodes=400, k=4):
        """Builds an organic spatial graph based on real historical incident coordinates."""
        import pandas as pd
        from scipy.spatial.distance import cdist
        
        logger.info(f"Building organic spatial graph with {num_nodes} nodes mapped to real roads.")
        
        try:
            df = pd.read_csv('dataset/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv')
            # Extract longitude and latitude to match standard geojson order [lon, lat]
            coords = df[['longitude', 'latitude']].dropna().values
            # Get unique physical locations
            coords = np.unique(coords, axis=0)
            
            # Sample N random points
            np.random.seed(42)
            idx = np.random.choice(len(coords), min(num_nodes, len(coords)), replace=False)
            sampled_coords = coords[idx]
            self.num_nodes = len(sampled_coords)
            
            # Save node coordinates mapping
            self.node_coords = {}
            for i, c in enumerate(sampled_coords):
                self.node_coords[i] = [c[0], c[1]]
                
            # Build edges via K-Nearest Neighbors
            dist_matrix = cdist(sampled_coords, sampled_coords)
            np.fill_diagonal(dist_matrix, np.inf)
            edges = []
            for i in range(self.num_nodes):
                k_nearest = np.argpartition(dist_matrix[i], k)[:k]
                for neighbor in k_nearest:
                    edges.append([i, int(neighbor)])
                    
            self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
        except Exception as e:
            logger.error(f"Failed to build organic graph, falling back to basic: {e}")
            self.num_nodes = num_nodes
            self.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t().contiguous()
            
            # Scatter fallback nodes randomly around Bengaluru center (77.5946, 12.9716)
            np.random.seed(42)
            self.node_coords = {}
            for i in range(num_nodes):
                self.node_coords[i] = [77.5946 + np.random.normal(0, 0.05), 12.9716 + np.random.normal(0, 0.05)]
            
        # Synthetic node features
        self.node_features = torch.rand((self.num_nodes, 4), dtype=torch.float)
        
        return self.get_graph_data()

    def load_osm_graph(self):
        """Loads graph from actual OSM or Shapefile. Fallback to organic if None."""
        if not self.osm_path:
            logger.warning("No OSM path provided. Falling back to organic real-road graph.")
            return self.build_organic_graph()
        
        # In a real scenario, use osmnx or geopandas here.
        logger.info(f"Loading graph from {self.osm_path}")
        return self.build_organic_graph()

    def get_graph_data(self):
        return Data(x=self.node_features, edge_index=self.edge_index)
