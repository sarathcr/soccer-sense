import time

from src.football_ai.predictor import FootballPredictor
from src.football_ai.train import MODEL_PATH, train_model


def test_prediction_finishes_under_30_seconds():
    if not MODEL_PATH.exists():
        train_model()

    predictor = FootballPredictor()
    started = time.perf_counter()
    predictor.predict("Argentina", "Brazil")
    elapsed = time.perf_counter() - started

    assert elapsed < 30
