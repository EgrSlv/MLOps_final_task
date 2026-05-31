import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.fitted = False

    def fit(self, X: pd.DataFrame):
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            self.scaler.fit(X[numeric_cols])
        categ_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
        for col in categ_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("FeatureEngineer не обучен. Вызовите fit() сначала.")
        result = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            result[numeric_cols] = self.scaler.transform(result[numeric_cols])
        for col, le in self.label_encoders.items():
            if col in result.columns:
                result[col] = le.transform(result[col].astype(str))
        return result

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)
