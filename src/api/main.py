import logging
from contextlib import asynccontextmanager

from typing import List

from starlette.responses import Response

from fastapi import FastAPI, HTTPException

from prometheus_client import (Counter,
                               Histogram,
                               generate_latest,
                               CONTENT_TYPE_LATEST)

from src.api.schemas import (PassengerFeatures,
                             PredictionResult,
                             BatchPredictionResult,
                             HealthStatus)
from src.models.predict import ModelPredictor


logger = logging.getLogger(__name__)

PREDICT_COUNTER = Counter("predictions_total", "Всего предсказаний", ["status"])
PREDICT_LATENCY = Histogram("prediction_latency_seconds",
                            "Задержка предсказания в секундах")


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


@app.get("/")
def root():
    html = """
    <html>
    <head><title>Titanic ML API</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }
        h1 { color: #1a5276; }
        .endpoint { background: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0; }
        code { background: #e8e8e8; padding: 2px 6px; border-radius: 3px; }
        a { color: #2980b9; }
    </style></head>
    <body>
        <h1>ML-система предсказания выживаемости на Титанике</h1>
        <p>Уровень зрелости: <strong>2</strong> (ML Automation)</p>
        <p>Модель: GradientBoostingClassifier</p>
        <h3>Эндпоинты:</h3>
        <div class="endpoint">GET <code>/health</code> — статус сервиса</div>
        <div class="endpoint">GET <code>/ready</code> — готовность модели</div>
        <div class="endpoint">POST <code>/predict</code> — предсказание</div>
        <div class="endpoint">POST <code>/predict/batch</code> — массовые предсказания</div>
        <div class="endpoint">GET <code>/metrics</code> — Prometheus метрики</div>
        <div class="endpoint">GET <code>/docs</code> — Swagger документация</div>
        <hr>
        <p>GitHub: <a href="https://github.com/EgrSlv/MLOps_final_task">EgrSlv/MLOps_final_task</a></p>
    </body></html>
    """
    return Response(content=html, media_type="text/html")


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
