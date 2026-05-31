import os
import pickle
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.features import FeatureEngineer

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "model.pkl"
FE_PATH = MODEL_DIR / "fe.pkl"


class ModelPredictor:
    def __init__(self, model_path: str | Path = MODEL_PATH,
                 fe_path: str | Path = FE_PATH):
        self.model = None
        self.fe = FeatureEngineer()
        self._load_local(model_path, fe_path)

    def _load_local(self, model_path: str | Path, fe_path: str | Path):
        if os.path.exists(model_path) and os.path.exists(fe_path):
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(fe_path, "rb") as f:
                self.fe = pickle.load(f)
            logger.info(f"Модель загружена из {model_path}")
        else:
            logger.warning(f"Модель не найдена по пути {model_path}.",
                           "Обучите: python -m src.models.train_cli")
            self.model = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.fe.fitted

    def _prepare_input(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.fe.label_encoders:
            if col in df.columns:
                df[col] = df[col].astype(str).astype("category")
        return df

    def predict(self, features: dict) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Модель не загружена")
        df = self._prepare_input(pd.DataFrame([features]))
        df_fe = self.fe.transform(df)
        pred = int(self.model.predict(df_fe)[0])
        proba = float(self.model.predict_proba(df_fe)[0, 1])
        return {"предсказание": pred, "вероятность": proba}

    def predict_batch(self, records: list[dict]) -> list[dict]:
        if not self.is_loaded:
            raise RuntimeError("Модель не загружена")
        df = self._prepare_input(pd.DataFrame(records))
        df_fe = self.fe.transform(df)
        preds = self.model.predict(df_fe).tolist()
        probas = self.model.predict_proba(df_fe)[:, 1].tolist()
        return [
            {"предсказание": int(p), "вероятность": float(pr)}
            for p, pr in zip(preds, probas)
        ]
