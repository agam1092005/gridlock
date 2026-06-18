import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from src.data_pipeline.feature_encoder import FeatureEncoder
from src.data_pipeline.embedding_engine import EmbeddingEngine
import logging

logger = logging.getLogger(__name__)


class AstramDataset:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.feature_encoder = FeatureEncoder()
        self.embedding_engine = EmbeddingEngine()

    def load_data(self):
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} records.")
        return self.df

    def preprocess(self):
        if self.df is None:
            self.load_data()

        df = self.df.copy()

        # Parse datetime
        df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce")
        df["end_datetime"] = pd.to_datetime(df["end_datetime"], errors="coerce")
        df["closed_datetime"] = pd.to_datetime(df["closed_datetime"], errors="coerce")
        df["resolved_datetime"] = pd.to_datetime(df["resolved_datetime"], errors="coerce")

        # Calculate duration
        # If end_datetime is missing, try to use resolved_datetime or closed_datetime
        # If all are missing, duration is considered censored (handled later in survival analysis)
        end_times = df["end_datetime"].fillna(df["resolved_datetime"]).fillna(df["closed_datetime"])
        df["duration_minutes"] = (end_times - df["start_datetime"]).dt.total_seconds() / 60.0
        df["is_censored"] = df["duration_minutes"].isna()

        # For censored events that are 'ongoing', duration might be time since start until 'now'.
        # For simplicity in this dataset prep, we leave missing durations as NaN or fill with median for baseline.

        # Create a proxy for severity_score (0-100) if not present
        if "severity_score" not in df.columns:
            logger.info(
                "Generating proxy severity_score based on Priority, Road Closure, and Type."
            )
            severity = np.zeros(len(df))
            severity += (df["priority"].str.lower() == "high").astype(int) * 40
            severity += (df["requires_road_closure"] == True).astype(int) * 30

            # Additional score based on event_cause
            cause_scores = {
                "accident": 20,
                "tree_fall": 15,
                "water_logging": 15,
                "vehicle_breakdown": 10,
                "pot_holes": 5,
                "congestion": 15,
            }
            cause_add = df["event_cause"].map(cause_scores).fillna(5)
            severity += cause_add

            df["severity_score"] = np.clip(severity, 0, 100)

        # Clean features
        df["event_cause"] = df["event_cause"].fillna("unknown")
        df["description"] = df["description"].fillna("")

        self.df = df
        return self.df

    def create_splits(self, output_dir: str):
        """Splits the data into train/val/test and saves them."""
        logger.info("Creating splits...")

        # Filter rows with missing critical features
        df_clean = self.df.dropna(subset=["latitude", "longitude", "start_datetime"]).copy()

        # Stratify by event_cause if possible
        # Some event_cause might have too few samples, handle it
        counts = df_clean["event_cause"].value_counts()
        valid_causes = counts[counts > 5].index
        df_clean = df_clean[df_clean["event_cause"].isin(valid_causes)]

        train_val, test_df = train_test_split(
            df_clean, test_size=0.15, stratify=df_clean["event_cause"], random_state=42
        )
        train_df, val_df = train_test_split(
            train_val, test_size=0.15 / 0.85, stratify=train_val["event_cause"], random_state=42
        )

        logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        os.makedirs(output_dir, exist_ok=True)

        # Save as pickle
        train_df.to_pickle(os.path.join(output_dir, "train.pkl"))
        val_df.to_pickle(os.path.join(output_dir, "val.pkl"))
        test_df.to_pickle(os.path.join(output_dir, "test.pkl"))

        # Save metadata
        metadata = {
            "num_samples": len(df_clean),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "incident_type_distribution": df_clean["event_cause"].value_counts().to_dict(),
        }

        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            import json

            json.dump(metadata, f, indent=4)

        return train_df, val_df, test_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_path = "dataset/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
    output_dir = "models/artifacts/dataset/"

    dataset = AstramDataset(data_path)
    dataset.preprocess()
    dataset.create_splits(output_dir)
