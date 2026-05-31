import logging
from contextlib import asynccontextmanager

from typing import List

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.api.schemas import PassengerFeatures, PredictionResult, BatchPredictionResult, HealthStatus
from src.models.predict import ModelPredictor


logger = logging.getLogger(__name__)

PREDICT_COUNTER = Counter("predictions_total", "Всего предсказаний", ["status"])
PREDICT_LATENCY = Histogram("prediction_latency_seconds", "Задержка предсказания в секундах")


predictor: ModelPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    try:
        predictor = ModelPredictor()
        if predictor.is_loaded:
            logger.info("Модель загружена из локального файла")
        else:
            logger.warning("Файл модели не найден")
    except Exception as e:
        logger.warning(f"Ошибка загрузки модели при запуске: {e}")
        predictor = None
    yield


app = FastAPI(
    title="Titanic Survival Prediction API",
    description="ML API для предсказания выживаемости пассажиров Титаника",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthStatus)
def health():
    return HealthStatus(status="healthy", model_loaded=predictor is not None)


@app.get("/ready")
def ready():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResult)
async def predict(features: PassengerFeatures):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Модель недоступна")
    with PREDICT_LATENCY.time():
        try:
            result = predictor.predict(features.model_dump())
            PREDICT_COUNTER.labels(status="success").inc()
            return PredictionResult(**result)
        except Exception as e:
            PREDICT_COUNTER.labels(status="error").inc()
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResult)
async def predict_batch(features_list: List[PassengerFeatures]):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Модель недоступна")
    try:
        results = predictor.predict_batch([f.model_dump() for f in features_list])
        PREDICT_COUNTER.labels(status="success").inc(len(results))
        return BatchPredictionResult(пассажиры=[PredictionResult(**r) for r in results])
    except Exception as e:
        PREDICT_COUNTER.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
