from pydantic import BaseModel, Field
from typing import List


class PassengerFeatures(BaseModel):
    Pclass: int = Field(ge=1, le=3, description="Класс пассажира (1=первый, 2=второй, 3=третий)")
    Sex: str = Field(description="Пол: male или female")
    Age: float = Field(ge=0, le=120, description="Возраст в годах")
    SibSp: int = Field(ge=0, description="Кол-во братьев/супругов на борту")
    Parch: int = Field(ge=0, description="Кол-во родителей/детей на борту")
    Fare: float = Field(ge=0, description="Стоимость билета в фунтах")
    Embarked: str = Field(default="S", description="Порт посадки (C=Шербур, Q=Квинстаун, S=Саутгемптон)")


class PredictionResult(BaseModel):
    предсказание: int = Field(description="0 = Не выжил, 1 = Выжил")
    вероятность: float = Field(ge=0, le=1, description="Вероятность выживания")


class BatchPredictionResult(BaseModel):
    пассажиры: List[PredictionResult]


class HealthStatus(BaseModel):
    status: str = "healthy"
    model_loaded: bool
