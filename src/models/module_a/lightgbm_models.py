import lightgbm as lgb
import numpy as np
import pickle
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


class LightGBMSeverityModel:
    def __init__(self, version="1.0"):
        self.version = version
        self.models = {}  # Store quantile models
        self.params = {
            "objective": "regression",
            "learning_rate": 0.05,
            "max_depth": 10,
            "num_leaves": 50,
            "verbosity": -1,
        }
        self.quantiles = [0.025, 0.5, 0.975]

    def train(self, X_train, y_train, X_val, y_val):
        logger.info("Training LightGBMSeverityModel...")

        for q in self.quantiles:
            logger.info(f"Training quantile {q}")
            params = self.params.copy()
            params["objective"] = "quantile"
            params["alpha"] = q

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[val_data])
            self.models[q] = model

        # Compute metrics on median prediction (q=0.5)
        preds = self.models[0.5].predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        metrics = {"rmse": rmse, "mae": mae, "r2": r2}
        logger.info(f"Validation metrics: {metrics}")
        return metrics

    def predict(self, X):
        predictions = {}
        for q in self.quantiles:
            predictions[q] = self.models[q].predict(X)

        return {
            "score": predictions[0.5],
            "ci_lower": predictions[0.025],
            "ci_upper": predictions[0.975],
        }

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        for q, model in self.models.items():
            model_path = os.path.join(directory, f"lightgbm_severity_q_{q}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

    def load(self, directory):
        for q in self.quantiles:
            model_path = os.path.join(directory, f"lightgbm_severity_q_{q}.pkl")
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    self.models[q] = pickle.load(f)


class LightGBMDurationModel:
    def __init__(self, version="1.0"):
        self.version = version
        self.model = None
        self.params = {
            "objective": "regression",
            "learning_rate": 0.05,
            "max_depth": 10,
            "num_leaves": 50,
            "verbosity": -1,
        }

    def train(self, X_train, y_train, X_val, y_val, censor_weights=None):
        logger.info("Training LightGBMDurationModel...")

        train_data = lgb.Dataset(X_train, label=y_train, weight=censor_weights)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        self.model = lgb.train(self.params, train_data, num_boost_round=100, valid_sets=[val_data])

        preds = np.asarray(self.model.predict(X_val))
        # Note: True survival metrics like C-index should be used here instead of standard regression metrics
        # For simplicity, calculating standard RMSE on available y_val
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        metrics = {"rmse": rmse}
        logger.info(f"Validation metrics: {metrics}")
        return metrics

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained.")
        median_pred = self.model.predict(X)
        # In a real Weibull setup, we'd output parameters for a distribution
        # For this prototype, we simulate CI bounds
        return {
            "estimate": median_pred,
            "ci_lower": median_pred * 0.8,
            "ci_upper": median_pred * 1.2,
        }

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, "lightgbm_duration.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, directory):
        model_path = os.path.join(directory, "lightgbm_duration.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
