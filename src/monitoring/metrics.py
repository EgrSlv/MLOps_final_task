import numpy as np
from prometheus_client import Gauge


PREDICTION_DISTRIBUTION = Gauge("prediction_positive_rate", "Доля положительных предсказаний (выжил)")
FEATURE_MEAN_AGE = Gauge("feature_mean_age", "Средний возраст пассажиров")
FEATURE_MEAN_FARE = Gauge("feature_mean_fare", "Средняя стоимость билета")
DATA_DRIFT_PSI = Gauge("data_drift_psi", "Индекс стабильности популяции для дрейфа признаков")

def update_monitoring_metrics(predictions: list[int], probabilities: list[float], features: dict):
    if predictions:
        PREDICTION_DISTRIBUTION.set(np.mean(predictions))
    if "Age" in features:
        FEATURE_MEAN_AGE.set(np.mean(features["Age"]))
    if "Fare" in features:
        FEATURE_MEAN_FARE.set(np.mean(features["Fare"]))
