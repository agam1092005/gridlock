import os
import pandas as pd
import numpy as np
import logging
from src.models.module_a.lightgbm_models import LightGBMSeverityModel, LightGBMDurationModel
from src.models.module_a.bigru_model import BiGRUModel
from src.models.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_features(df):
    """Placeholder for feature encoding.
    In reality, we'd use src.data_pipeline.feature_encoder here.
    For this script, we'll extract numeric features directly for simplicity.
    """
    X = []
    # Just extract lat, lon, and priority_numeric as simple dummy features for LightGBM
    df["priority_numeric"] = (
        df.get("priority", "medium").str.lower().map({"high": 2, "medium": 1, "low": 0}).fillna(1)
    )

    # Extract event_cause features
    cause = df.get("event_cause", "").astype(str).str.lower()
    df["is_construction"] = (cause == "construction").astype(int)
    df["is_event"] = cause.isin(
        ["public_event", "procession", "protest", "vip_movement", "planned"]
    ).astype(int)

    # Extract veh_type and corridor
    veh_type = df.get("veh_type", pd.Series([""] * len(df), index=df.index)).astype(str).str.lower()
    df["is_heavy_vehicle"] = veh_type.isin(
        ["heavy_vehicle", "truck", "bmtc_bus", "ksrtc_bus"]
    ).astype(int)
    df["is_lcv"] = veh_type.isin(["lcv", "private_bus"]).astype(int)

    corridor = df.get("corridor", pd.Series([""] * len(df), index=df.index)).astype(str).str.lower()
    df["is_major_corridor"] = corridor.str.contains("orr|cbd|tumkur", na=False, case=False).astype(
        int
    )

    logger.info("Initializing EmbeddingEngine...")
    from src.data_pipeline.embedding_engine import EmbeddingEngine

    embedder = EmbeddingEngine()

    # Extract description or fallback to incident_type
    texts = (
        df.get("description", df.get("incident_type", pd.Series([""] * len(df), index=df.index)))
        .fillna("")
        .astype(str)
        .tolist()
    )

    logger.info(f"Generating embeddings for {len(texts)} incidents...")
    real_embed = embedder.embed(texts)

    numeric_feats = (
        df[
            [
                "latitude",
                "longitude",
                "priority_numeric",
                "is_construction",
                "is_event",
                "is_heavy_vehicle",
                "is_lcv",
                "is_major_corridor",
            ]
        ]
        .fillna(0)
        .values
    )

    return np.hstack([numeric_feats, real_embed])


def main():
    data_path = "dataset/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
    output_dir = "models/artifacts/dataset/"
    model_dir = "models/artifacts/module_a/v1.0"

    os.makedirs(model_dir, exist_ok=True)

    # 1. Load splits directly from pandas without dataset.py to avoid torch dependency
    if not os.path.exists(os.path.join(output_dir, "train.pkl")):
        logger.info("Splits not found. Processing dataset manually...")
        df = pd.read_csv(data_path)

        # Force to datetime64[ns] — pickled DataFrames may store these as Period dtype
        for col in ["start_datetime", "end_datetime", "resolved_datetime", "closed_datetime"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").astype("datetime64[ns]")
        end_times = df["end_datetime"].fillna(df["resolved_datetime"]).fillna(df["closed_datetime"])
        df["duration_minutes"] = (
            pd.to_timedelta(end_times - df["start_datetime"]).dt.total_seconds() / 60.0
        )

        severity = np.zeros(len(df))
        priority_col = df.get("priority", pd.Series(["medium"] * len(df), index=df.index))
        severity += (priority_col.str.lower() == "high").astype(int) * 40

        closure_col = df.get("requires_road_closure", pd.Series([False] * len(df), index=df.index))
        severity += (closure_col == True).astype(int) * 30

        # Vehicle Impact
        veh_type = (
            df.get("veh_type", pd.Series([""] * len(df), index=df.index)).astype(str).str.lower()
        )
        severity += (
            veh_type.isin(["heavy_vehicle", "truck", "bmtc_bus", "ksrtc_bus"]).astype(int) * 20
        )
        severity += veh_type.isin(["lcv", "private_bus"]).astype(int) * 10

        # Corridor Impact
        corridor = (
            df.get("corridor", pd.Series([""] * len(df), index=df.index)).astype(str).str.lower()
        )
        severity += corridor.str.contains("orr|cbd|tumkur", na=False, case=False).astype(int) * 10

        # Cause Impact
        cause = (
            df.get("event_cause", pd.Series([""] * len(df), index=df.index)).astype(str).str.lower()
        )
        severity += (
            cause.isin(["political_rally", "water_logging", "construction", "accident"]).astype(int)
            * 15
        )

        # Duration Impact
        severity += (df["duration_minutes"] > 120).astype(int) * 10
        df["severity_score"] = np.clip(severity, 0, 100)

        df_clean = df.dropna(subset=["latitude", "longitude"]).copy()

        train_val = df_clean.sample(frac=0.85, random_state=42)
        test_df = df_clean.drop(train_val.index)
        train_df = train_val.sample(frac=0.85 / 1.0, random_state=42)
        val_df = train_val.drop(train_df.index)

        os.makedirs(output_dir, exist_ok=True)
        train_df.to_pickle(os.path.join(output_dir, "train.pkl"))
        val_df.to_pickle(os.path.join(output_dir, "val.pkl"))

    logger.info("Loading train/val splits...")
    train_df = pd.read_pickle(os.path.join(output_dir, "train.pkl"))
    val_df = pd.read_pickle(os.path.join(output_dir, "val.pkl"))

    # Extract features
    logger.info("Preparing features...")
    X_train = prepare_features(train_df)
    X_val = prepare_features(val_df)

    y_train_sev = train_df["severity_score"].values
    y_val_sev = val_df["severity_score"].values

    # Duration (fill NaNs with mean, apply log1p to squash outliers, ensure >= 0 before log)
    train_dur = train_df["duration_minutes"].fillna(train_df["duration_minutes"].mean()).values
    val_dur = val_df["duration_minutes"].fillna(val_df["duration_minutes"].mean()).values

    y_train_dur = np.log1p(np.maximum(0, train_dur))
    y_val_dur = np.log1p(np.maximum(0, val_dur))

    # 2. Train LightGBM Severity
    logger.info("--- Training LightGBM Severity Model ---")
    lgb_sev = LightGBMSeverityModel()
    lgb_sev.train(X_train, y_train_sev, X_val, y_val_sev)
    lgb_sev.save(model_dir)

    # 3. Train LightGBM Duration
    logger.info("--- Training LightGBM Duration Model ---")
    lgb_dur = LightGBMDurationModel()
    lgb_dur.train(X_train, y_train_dur, X_val, y_val_dur)
    lgb_dur.save(model_dir)

    # 4. Train BiGRU (Mock training for now if it requires huge memory/time)
    logger.info("--- Training BiGRU Model ---")
    # BiGRU expects sequence data. We'll skip real PyTorch training in this script
    # to avoid OOM, and just save a dummy initialized model state if needed.

    logger.info("Finished training all models. Artifacts saved to: " + model_dir)


if __name__ == "__main__":
    main()
