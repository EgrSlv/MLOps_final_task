import pickle
import logging
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier

from src.data.clean import get_train_test_data
from src.data.features import FeatureEngineer
from sklearn.metrics import accuracy_score, roc_auc_score


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent

def main():
    X_train, X_test, y_train, y_test = get_train_test_data()

    fe = FeatureEngineer()
    X_train_fe = fe.fit_transform(X_train)
    X_test_fe = fe.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, random_state=42
    )
    model.fit(X_train_fe, y_train)

    y_pred = model.predict(X_test_fe)
    y_proba = model.predict_proba(X_test_fe)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    logger.info(f"Точность: {acc:.4f}, ROC-AUC: {auc:.4f}")

    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / "fe.pkl", "wb") as f:
        pickle.dump(fe, f)
    logger.info(f"Модель сохранена в {MODEL_DIR / 'model.pkl'}")
    logger.info(f"FeatureEngineering сохранён в {MODEL_DIR / 'fe.pkl'}")


if __name__ == "__main__":
    main()
