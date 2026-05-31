import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import cross_val_score

from src.data.clean import get_train_test_data
from src.data.features import FeatureEngineer


MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    ),
}

def train_model(model_name: str = "random_forest",
                experiment_name: str = "titanic-survival",
                tracking_uri: str = "http://mlflow:5000",) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X_train, X_test, y_train, y_test = get_train_test_data()

    fe = FeatureEngineer()
    X_train_fe = fe.fit_transform(X_train)
    X_test_fe = fe.transform(X_test)

    model = MODELS[model_name]

    with mlflow.start_run() as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(model.get_params())

        cv_scores = cross_val_score(
            model,
            X_train_fe,
            y_train,
            cv=5,
            scoring="roc_auc"
        )
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_scores.std())

        model.fit(X_train_fe, y_train)
        y_pred = model.predict(X_test_fe)
        y_proba = model.predict_proba(X_test_fe)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact("src/data/clean.py", "code")
        mlflow.log_artifact("src/data/features.py", "code")

    return run.info.run_id
