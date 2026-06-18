import os
import pandas as pd
import numpy as np
import pickle
import logging
from lifelines import CoxPHFitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    data_path = "dataset/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
    train_path = "models/artifacts/dataset/train.pkl"
    output_dir = "models/artifacts/module_a/v1.0"
    output_model_path = os.path.join(output_dir, "cox_survival.pkl")

    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(train_path):
        logger.info(f"Loading train split from {train_path}...")
        df = pd.read_pickle(train_path)
    else:
        logger.info(f"Train split not found. Processing dataset manually from {data_path}...")
        df = pd.read_csv(data_path)

    # Force to datetime64[ns] — pd.to_datetime alone won't re-cast Period columns
    # (which come from pickled DataFrames) and Period's .dt accessor has no .hour
    for col in ["start_datetime", "end_datetime", "resolved_datetime", "closed_datetime"]:
        df[col] = pd.to_datetime(df[col], errors="coerce").astype("datetime64[ns]")

    end_times = df["end_datetime"].fillna(df["resolved_datetime"]).fillna(df["closed_datetime"])
    # Wrap in pd.to_timedelta so .dt always returns TimedeltaProperties (not PeriodProperties)
    df["T"] = pd.to_timedelta(end_times - df["start_datetime"]).dt.total_seconds() / 60.0
    df["E"] = end_times.notna()

    # Filter out invalid durations and rows without start times
    df_fit = df[(df["T"] > 0) & (df["start_datetime"].notna())].copy()

    # Feature Mapping: map categories to numeric float values
    incident_type_map = {
        "accident": 0.0,
        "congestion": 1.0,
        "road_closure": 2.0,
        "hazard": 3.0,
        "event": 4.0,
    }
    df_fit["incident_type"] = df_fit["event_cause"].str.lower().map(incident_type_map).fillna(4.0)

    priority_map = {"high": 2.0, "medium": 1.0, "low": 0.0}
    df_fit["priority"] = df_fit["priority"].str.lower().map(priority_map).fillna(0.0)

    df_fit["hour_of_day"] = pd.to_datetime(df_fit["start_datetime"]).dt.hour.astype(float)

    # Select covariate columns for the Cox model
    cox_data = df_fit[["T", "E", "incident_type", "priority", "hour_of_day"]].copy()

    logger.info("Fitting Cox Proportional Hazards Model...")
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col="T", event_col="E")

    logger.info(f"Saving Cox model to {output_model_path}...")
    with open(output_model_path, "wb") as f:
        pickle.dump(cph, f)

    logger.info("Survival model training complete!")


if __name__ == "__main__":
    main()
