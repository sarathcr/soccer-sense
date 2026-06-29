import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier, DummyRegressor
from src.football_ai.train import make_pure_python_pipeline, fit_robust_classifier, fit_robust_regressor

def test_dummy_classifier_fallback():
    # Test fallback DummyClassifier behavior (e.g., single class)
    x = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
    y = pd.Series([1, 1, 1])  # Only one class, triggers DummyClassifier
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", DummyClassifier(strategy="most_frequent"))
    ])
    
    fitted_pipeline = fit_robust_classifier(pipeline, x, y)
    pure_pipeline = make_pure_python_pipeline(fitted_pipeline)
    
    # Check that serialization worked and can predict
    assert pure_pipeline.params["type"] == "dummy_classifier"
    assert pure_pipeline.params["constant"] == 1
    
    preds = pure_pipeline.predict(x)
    assert np.all(preds == 1)
    
    probs = pure_pipeline.predict_proba(x)
    assert probs.shape == (3, 1)
    assert np.all(probs[:, 0] == 1.0)

def test_dummy_regressor_fallback():
    # Test DummyRegressor behavior (e.g., single target value)
    x = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
    y = pd.Series([2.5, 2.5, 2.5])  # Only one target value
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", DummyRegressor(strategy="mean"))
    ])
    
    fitted_pipeline = fit_robust_regressor(pipeline, x, y)
    pure_pipeline = make_pure_python_pipeline(fitted_pipeline)
    
    # Check that serialization worked and can predict
    assert pure_pipeline.params["type"] == "dummy_regressor"
    constant = pure_pipeline.params["constant"]
    val = constant[0][0] if isinstance(constant, list) and isinstance(constant[0], list) else (constant[0] if isinstance(constant, list) else constant)
    assert abs(val - 2.5) < 1e-5
    
    preds = pure_pipeline.predict(x)
    assert np.allclose(preds, 2.5)
