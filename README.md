# ⚽ SoccerSense — Football AI Match Prediction

An end-to-end football match prediction system that combines **ELO-rated team strength**, **Poisson regression for expected goals**, **Bayesian player scoring probability**, and a **FastAPI REST backend** with a full browser-based UI. The system can be trained on live data scraped from any URL or via file upload, and serves predictions covering match outcome, scoreline, both-teams-to-score, clean sheets, and per-player goal probability.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Libraries Used](#libraries-used)
- [Algorithms & Models](#algorithms--models)
- [Feature Engineering](#feature-engineering)
- [Training Pipeline](#training-pipeline)
- [Prediction Engine](#prediction-engine)
- [API Endpoints](#api-endpoints)
- [Data Formats](#data-formats)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [Training the Model](#training-the-model)
- [CLI Usage](#cli-usage)
- [Running Tests](#running-tests)
- [Deployment](#deployment)
- [Model Artifact](#model-artifact)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Browser UI (index.html)                    │
│  Upload CSV/Excel │ Scrape URL │ Predict Match │ View Score Matrix  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP REST
┌──────────────────────────────▼──────────────────────────────────────┐
│                        FastAPI Backend  (api.py)                    │
│  /predict  /train-from-upload  /train-from-url  /teams  /reset ...  │
└──────────┬────────────────────────┬────────────────────────────────-┘
           │                        │
    ┌──────▼──────┐         ┌───────▼──────────┐
    │  predictor  │         │   crawler.py      │
    │  .py        │         │  (URL scraping &  │
    │  FootballPre│         │   data enrichment)│
    │  dictor     │         └───────┬───────────┘
    └──────┬──────┘                 │
           │                ┌───────▼───────────┐
    ┌──────▼──────┐         │   train.py         │
    │  soccer_    │         │  (Model training,  │
    │  sense.pkl  │◄────────│   ELO calibration, │
    │  (artifact) │         │   grid search,     │
    └─────────────┘         │   artifact saving) │
                            └───────┬────────────┘
                                    │
                            ┌───────▼────────────┐
                            │   features.py       │
                            │  (TeamProfile,      │
                            │   feature matrix,   │
                            │   ELO/GK features)  │
                            └────────────────────-┘
```

### Data Flow

1. **Input Data** — Historical match results (with ELO ratings) + player statistics CSV/Excel/JSON
2. **Feature Engineering** — ELO ratings, ELO rank, ELO differential, rank differential, goalkeeper save ratio, goalkeeper goals-prevented-per-90
3. **Model Training** — Chronological 75/25 split → grid search for `C` and `alpha` → two-stage training (dev model for metrics + prod model on 100% data)
4. **Serialization** — Models serialized as `PurePythonPipeline` (sklearn-free at inference time) via custom pickle bytecode
5. **Prediction** — Poisson regression λ → joint PMF matrix → win/draw/loss/BTTS/clean sheet/first scorer probabilities
6. **Player Predictions** — Bayesian shrinkage + xG/shots blending for per-player goal probability

---

## Project Structure

```
football_ai_prediction/
├── src/
│   └── football_ai/
│       ├── __init__.py
│       ├── api.py            # FastAPI app — all REST endpoints
│       ├── crawler.py        # Web scraping, data enrichment, ELO computation
│       ├── features.py       # Feature engineering, TeamProfile dataclass
│       ├── pipeline.py       # CLI pipeline runner (config-file or URL-crawl mode)
│       ├── predictor.py      # FootballPredictor class — inference logic
│       ├── train.py          # Full training pipeline, PurePythonPipeline, PredictorWrapper
│       └── static/
│           └── index.html    # Full browser UI (single-file SPA)
├── api/
│   └── index.py              # Vercel serverless entry point
├── data/
│   ├── sample_matches.csv    # Historical match data (date, teams, goals, ELO, rank)
│   ├── sample_players.csv    # Player stats (name, team, position, xG, goals, SOT …)
│   ├── raw_matches_1.csv     # Additional raw match data
│   ├── sample_matches.csv.bak
│   └── sample_players.csv.bak
├── models/
│   ├── soccersense.pkl           # Production model artifact
│   ├── soccersense_v1.pkl        # Versioned production model
│   └── dev_model.pkl             # Dev model (trained on 75% of data)
├── tests/
│   ├── conftest.py
│   ├── test_api_upload.py
│   ├── test_crawler.py
│   ├── test_dummy_fallback.py
│   ├── test_player_validation.py
│   ├── test_prediction_contract.py
│   ├── test_prediction_speed.py
│   └── test_unavailable_data.py
├── pipeline_config.json      # Data source mapping configuration
├── requirements.txt
├── Dockerfile
├── vercel.json
└── .python-version
```

---

## Libraries Used

| Library              | Version       | Purpose                                                                                             |
| -------------------- | ------------- | --------------------------------------------------------------------------------------------------- |
| **pandas**           | ≥ 2.0         | Data loading, manipulation, DataFrame operations                                                    |
| **numpy**            | ≥ 1.24        | Numerical computation, PMF arrays, matrix operations                                                |
| **scikit-learn**     | ≥ 1.3         | `LogisticRegression`, `PoissonRegressor`, `StandardScaler`, `Pipeline`, `DummyClassifier/Regressor` |
| **scipy**            | ≥ 1.10        | `scipy.stats.poisson` — PMF computation for score distributions                                     |
| **joblib**           | ≥ 1.3         | Model loading (legacy support)                                                                      |
| **fastapi**          | ≥ 0.110       | REST API framework — all HTTP endpoints                                                             |
| **uvicorn**          | ≥ 0.27        | ASGI server for running FastAPI                                                                     |
| **pydantic**         | (via fastapi) | Request/response validation (`MatchInput`, `CrawlTrainInput`)                                       |
| **requests**         | ≥ 2.31        | HTTP client for web crawling                                                                        |
| **beautifulsoup4**   | ≥ 4.12        | HTML parsing for web-scraped match/player tables                                                    |
| **cloudpickle**      | ≥ 3.0         | Advanced serialization (dependency for artifact format)                                             |
| **python-multipart** | ≥ 0.0.12      | Multipart file upload support for FastAPI                                                           |
| **pytest**           | ≥ 8.0         | Test framework                                                                                      |

**Python version:** 3.12 (see `.python-version`)

---

## Algorithms & Models

### 1. Match Outcome — Multinomial Logistic Regression

- **Purpose:** Predict result class (`home_win` / `draw` / `away_win`)
- **Implementation:** `sklearn.linear_model.LogisticRegression` (multi-class softmax)
- **Hyperparameter:** Regularization `C` — grid-searched over `[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]`
- **Preprocessing:** `StandardScaler` in a `Pipeline`

> **Note:** The match model is trained for completeness but match outcome probabilities are **derived from the Poisson joint distribution** at inference time for greater accuracy and consistency with scoreline predictions.

---

### 2. Expected Goals (λ) — Poisson Regression

- **Purpose:** Predict the expected goals rate (λ) for home and away teams independently
- **Implementation:** `sklearn.linear_model.PoissonRegressor` (log-link GLM)
- **Hyperparameter:** Regularization `alpha` — grid-searched jointly with `C`
- **Two models:** `home_goals_model` and `away_goals_model`, each a `StandardScaler → PoissonRegressor` pipeline

**Neutral venue adjustment:**  
For fairness (e.g., FIFA World Cup neutral grounds), predictions are symmetrized:

```
λ_home = (model_home(A as home) + model_away(A swapped to home)) / 2
λ_away = (model_away(B as away) + model_home(B swapped to away)) / 2
```

---

### 3. ELO-Calibrated Strength Boost

After the Poisson regressors produce base λ values (which regress toward the mean due to regularization), an **ELO multiplicative boost** is applied:

```
units = (home_elo - away_elo) / 400
raw_boost = 1.0 + 0.15 × units

home_multiplier = clip(raw_boost, 0.55, 1.75)
away_multiplier = clip(2.0 - raw_boost, 0.55, 1.75)

λ_home_final = λ_home × home_multiplier
λ_away_final = λ_away × away_multiplier
```

- Every **400 ELO points** of difference → **±15% adjustment**
- Clamped between **×0.55** (suppress) and **×1.75** (boost) for realism

---

### 4. Outcome Probability — Bivariate Poisson (Joint PMF)

Given λ_home and λ_away, outcome probabilities are computed from the **joint Poisson distribution**:

```
P(H=h, A=a) = Poisson(h, λ_home) × Poisson(a, λ_away)

P(home_win) = Σ P(H > A)     [lower triangle of joint matrix]
P(draw)     = Σ P(H = A)     [trace of joint matrix]
P(away_win) = Σ P(H < A)     [upper triangle of joint matrix]
```

PMF truncated at 15 goals and re-normalized to sum to 1.

---

### 5. Auxiliary Binary Classifiers (Logistic Regression)

All use the same features and are trained in parallel with the same `C` hyperparameter:

| Model                    | Target                                       |
| ------------------------ | -------------------------------------------- |
| `btts_model`             | Both Teams to Score (binary: 0/1)            |
| `first_goal_model`       | First goal scored by home team (binary: 0/1) |
| `home_clean_sheet_model` | Home team keeps clean sheet                  |
| `away_clean_sheet_model` | Away team keeps clean sheet                  |

> At inference time, BTTS and clean sheet probabilities are **re-derived from the Poisson distribution** using analytical formulas for consistency:
>
> - `P(BTTS) = (1 − e^−λ_home) × (1 − e^−λ_away)`
> - `P(home CS) = e^−λ_away`
> - `P(away CS) = e^−λ_home`
> - `P(first goal = home) = λ_home / (λ_home + λ_away)`

---

### 6. Player Goal Probability — Bayesian Shrinkage + xG Blend

For each outfield player, expected goals per match (λ_player) is estimated as:

**Step 1 — Bayesian shrinkage** (low-minute penalty):

```
shrinkage = mins / (mins + 90)
```

**Step 2 — xG / actual goals blend** (for players with ≥90 min):

```
weight   = min(0.5, mins / 900)
base_λ   = (1 - weight) × xG_90 + weight × Goals_90
```

**Step 3 — Shot conversion adjustment** (if shots-on-target and conversion% available):

```
shot_derived = SOT_90 × conversion_decimal
base_λ       = 0.7 × base_λ + 0.3 × shot_derived
```

**Step 4 — Final expected goals:**

```
expected_goals = base_λ × start_probability × shrinkage
```

**Goal probabilities (Poisson):**

```
P(≥1 goal) = 1 − e^−expected_goals   (capped at 85%)
P(≥2 goals) ≈ P(≥1) × expected_goals / 2   (capped at 50%)
```

---

### 7. Grid Search & Hyperparameter Tuning

A manual grid search is run over `(C, alpha)` pairs before final model training:

- **Small datasets (< 5,000 rows):** 7×7 grid = 49 combinations
- **Medium datasets (5,000–15,000 rows):** 3×3 grid on a random 5,000-row sample
- **Large datasets (> 15,000 rows):** Skip grid, use defaults `C=0.1`, `alpha=1.0`

Best pair is chosen by highest **validation accuracy**, with **log-loss** as tiebreaker.

---

### 8. Time-Decay Sample Weights

Older matches receive lower weight during training using exponential decay:

```
weight = exp(−ln(2) / 365 × days_since_match)
```

Half-life = 365 days — a match from one year ago is weighted at 50% of a current match.

---

### 9. Two-Stage Training (Dev + Production)

| Stage                | Data Used                           | Purpose                                      |
| -------------------- | ----------------------------------- | -------------------------------------------- |
| **Dev model**        | First 75% of data (chronologically) | Validation metrics, hyperparameter selection |
| **Production model** | 100% of data                        | Maximum accuracy for deployment              |

Both models are serialized as `PurePythonPipeline` — a pure-Python re-implementation of `StandardScaler → Logistic/Poisson` that does **not require scikit-learn at inference time**.

---

## Feature Engineering

All 10 features used for training and inference:

| Feature                    | Description                                        |
| -------------------------- | -------------------------------------------------- |
| `home_elo`                 | Home team ELO rating (most recent match)           |
| `away_elo`                 | Away team ELO rating (most recent match)           |
| `home_elo_rank`            | Home team ELO-based world ranking                  |
| `away_elo_rank`            | Away team ELO-based world ranking                  |
| `elo_diff`                 | `home_elo − away_elo`                              |
| `rank_diff`                | `away_elo_rank − home_elo_rank`                    |
| `home_gk_save_ratio`       | Primary GK's save percentage (fraction, e.g. 0.72) |
| `away_gk_save_ratio`       | Primary GK's save percentage                       |
| `home_gk_prevented_per_90` | Goals prevented per 90 minutes (primary GK)        |
| `away_gk_prevented_per_90` | Goals prevented per 90 minutes (primary GK)        |

**Team profiles** are built by grouping all historical matches and taking the **most recent ELO** for each team. Goalkeeper features are sourced from the player statistics file.

---

## Training Pipeline

```
matches CSV + players CSV
        │
        ▼
validate_and_standardize_players()     ← position inference, column normalization
        │
        ▼
build_team_profiles()                  ← ELO snapshot + GK metrics per team
        │
        ▼
build_training_frame()                 ← result labels, BTTS, clean sheets, first goal
        │
        ▼
Chronological 75/25 split
        │
        ├── Time-decay weights
        │
        ├── Grid search (C × alpha)
        │        └── Validation: accuracy + log-loss on Poisson-derived probs
        │
        ├── Dev model fit (75% data)
        │        └── Save → models/dev_model.pkl
        │
        └── Production model fit (100% data)
                 └── Save → models/soccersense.pkl
                         → models/soccersense_v{version}.pkl
```

---

## Prediction Engine

```
POST /predict  {home_team, away_team}
        │
        ▼
normalize team names (Title Case)
        │
        ▼
lookup team_profiles → ELO + GK stats
        │ (fallback: league-average ELO if unknown team)
        ▼
build_inference_features()   → 10-feature DataFrame
        │
        ▼
Neutral venue symmetrization
        │
        ▼
home_goals_model.predict() + away_goals_model.predict()
        │
        ▼
ELO strength boost (±15% per 400 ELO points)
        │
        ▼
Bivariate Poisson joint PMF (15×15 matrix)
        │
        ├── P(home win) / P(draw) / P(away win)
        ├── Most likely scoreline within predicted outcome class
        ├── P(BTTS) = (1−e^−λh)(1−e^−λa)
        ├── P(home CS) = e^−λa
        ├── P(away CS) = e^−λh
        └── P(first goal home) = λh / (λh + λa)
        │
        ▼
Per-player Bayesian goal probability (top 3 attackers per team)
        │
        ▼
JSON response
```

---

## API Endpoints

| Method | Endpoint                               | Description                                        |
| ------ | -------------------------------------- | -------------------------------------------------- |
| `GET`  | `/`                                    | Serves the browser UI (`index.html`)               |
| `GET`  | `/health`                              | Health check — returns `{"status":"ok"}`           |
| `POST` | `/predict`                             | Predict match outcome for `{home_team, away_team}` |
| `GET`  | `/teams`                               | List all teams in the loaded model                 |
| `GET`  | `/team-profiles`                       | Full team profile data (ELO, GK stats)             |
| `GET`  | `/model-version`                       | Current model version string                       |
| `POST` | `/train-from-upload`                   | Upload CSV/Excel files and retrain model           |
| `POST` | `/train-from-url`                      | Scrape URLs and retrain model                      |
| `POST` | `/update-version`                      | Update model version tag                           |
| `POST` | `/reset`                               | Reset to default sample data and retrain           |
| `GET`  | `/download/matches`                    | Download processed matches CSV                     |
| `GET`  | `/download/players`                    | Download processed players CSV                     |
| `GET`  | `/download/dev_model`                  | Download dev model pickle                          |
| `GET`  | `/download/production_model`           | Download production model pickle                   |
| `GET`  | `/download/production_model_versioned` | Download versioned production model                |

### Example Prediction Request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Argentina", "away_team": "Brazil"}'
```

### Example Prediction Response

```json
{
  "output": {
    "match_prediction": {
      "win_probabilities": {
        "home_team": { "team": "Argentina", "probability": 45 },
        "draw":      { "probability": 28 },
        "away_team": { "team": "Brazil",    "probability": 27 }
      }
    },
    "score_prediction": {
      "predicted_scoreline": {
        "home_team": "Argentina", "home_goals": 1,
        "away_team": "Brazil",    "away_goals": 0
      },
      "total_goals": 1
    },
    "goal_insights": {
      "first_team_to_score": { "team": "Argentina", "probability": 57 },
      "both_teams_to_score": { "prediction": false, "probability": 38 }
    },
    "player_prediction": {
      "home_team": {
        "team": "Argentina",
        "goal": [
          { "name": "L. Messi", "predictions": [{"goal_count":1,"probability":42},{"goal_count":2,"probability":9}] }
        ],
        "clean_sheet_prediction": { "goalkeeper": "E. Martinez", "prediction": true, "probability": 62 }
      },
      "away_team": { "..." }
    },
    "explanation_steps": {
      "step1_profiles": { "..." },
      "step2_expected_goals": { "home_lambda": 1.423, "away_lambda": 0.981 },
      "step3_joint_distribution": { "score_matrix_6x6": [[...]] },
      "step4_insights_math": { "btts": { "..." }, "clean_sheets": { "..." } }
    }
  }
}
```

---

## Data Formats

### Matches CSV (`sample_matches.csv`)

Required columns:

| Column          | Type          | Description                       |
| --------------- | ------------- | --------------------------------- |
| `date`          | string / date | Match date (ISO format preferred) |
| `home_team`     | string        | Home team name                    |
| `away_team`     | string        | Away team name                    |
| `home_goals`    | int           | Goals scored by home team         |
| `away_goals`    | int           | Goals scored by away team         |
| `home_elo`      | float         | Home team ELO rating              |
| `away_elo`      | float         | Away team ELO rating              |
| `home_elo_rank` | float         | Home team ELO rank                |
| `away_elo_rank` | float         | Away team ELO rank                |

> If ELO columns are missing, `enrich_match_data()` in `crawler.py` **automatically computes** them from historical match sequences using a rolling ELO algorithm (K=32, starting at 1500).

### Players CSV (`sample_players.csv`)

Useful columns (all optional — missing fields are inferred or defaulted):

| Column                       | Description                                          |
| ---------------------------- | ---------------------------------------------------- |
| `name`                       | Player name                                          |
| `team`                       | National team                                        |
| `position`                   | `GK` / `DF` / `MF` / `FW` (auto-inferred if missing) |
| `mins`                       | Total minutes played                                 |
| `goals`                      | Goals scored                                         |
| `xg`                         | Total expected goals                                 |
| `xg_x90` / `xg_per_90`       | xG per 90 minutes                                    |
| `goals_x90` / `goals_per_90` | Goals per 90 minutes                                 |
| `sot_x90`                    | Shots on target per 90                               |
| `conv_pct`                   | Conversion percentage                                |
| `start_probability`          | Probability of starting (0–1)                        |
| `save_perc`                  | Goalkeeper save percentage                           |
| `goals_prevented`            | Goalkeeper goals prevented                           |

### Pipeline Config (`pipeline_config.json`)

Defines column mappings from raw source files to internal format:

```json
{
  "matches": {
    "output": "data/sample_matches.csv",
    "sources": [{
      "file": "data/raw_matches_1.csv",
      "mapping": { "tournament": "competition", "home_elo": "home_elo" },
      "filters": {},
      "defaults": {}
    }]
  },
  "players": {
    "output": "data/sample_players.csv",
    "sources": [{ "..." }]
  }
}
```

---

## Setup & Installation

### Prerequisites

- Python 3.12+
- pip

### Local Setup

```bash
# Clone / navigate to the project
cd football_ai_prediction

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

### Start the API Server

```bash
uvicorn src.football_ai.api:app --reload --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser to use the full UI.

For production (no auto-reload):

```bash
uvicorn src.football_ai.api:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build
docker build -t soccersense .

# Run
docker run -p 8000:8000 soccersense
```

---

## Training the Model

### Using Default Sample Data

```bash
python -m src.football_ai.train
```

Output:

```
Optimal parameters selected: LogisticRegression C=0.1, PoissonRegressor alpha=1.0 (Validation Accuracy: 0.542)
Saved model to models/soccersense.pkl
Validation metrics:
  result_accuracy: 0.542
  result_log_loss: 0.981
  home_goals_mae: 0.823
  away_goals_mae: 0.714
```

### Using the Pipeline (Config-Based)

```bash
python -m src.football_ai.pipeline --config pipeline_config.json
```

### Using URL Scraping

```bash
python -m src.football_ai.pipeline \
  --matches-url "https://example.com/international-results" \
  --players-url "https://example.com/squad-stats"
```

### Retrain After Upload (via API)

```bash
curl -X POST http://localhost:8000/train-from-upload \
  -F "matches_files=@data/my_matches.csv" \
  -F "players_files=@data/my_players.csv"
```

---

## CLI Usage

### Predict a Match

```bash
python -m src.football_ai.predictor Argentina Brazil
```

### Pipeline + Predict

```bash
python -m src.football_ai.pipeline \
  --config pipeline_config.json \
  --predict Argentina Brazil
```

---

## Running Tests

```bash
pytest
```

Run specific test modules:

```bash
pytest tests/test_prediction_contract.py -v
pytest tests/test_prediction_speed.py -v
pytest tests/test_api_upload.py -v
```

**Test coverage:**

| Test File                     | What It Covers                                        |
| ----------------------------- | ----------------------------------------------------- |
| `test_prediction_contract.py` | JSON output schema validation                         |
| `test_prediction_speed.py`    | Inference must complete in < 5 seconds                |
| `test_api_upload.py`          | File upload and training via API                      |
| `test_crawler.py`             | Web crawling and data parsing                         |
| `test_player_validation.py`   | Player data standardization and position inference    |
| `test_dummy_fallback.py`      | DummyClassifier/Regressor fallback for small datasets |
| `test_unavailable_data.py`    | "Data Unavailable" response for unknown teams         |

---

## Deployment

### Vercel (Serverless)

The `api/index.py` and `vercel.json` configure Vercel deployment:

```json
{
  "rewrites": [{ "source": "/(.*)", "destination": "/api/index.py" }]
}
```

Deploy:

```bash
vercel deploy
```

### Docker / Cloud

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.football_ai.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Model Artifact

The `.pkl` files in `models/` are **self-contained executable pickle files** — they embed the full `PredictorWrapper` class definition and re-define `PurePythonPipeline` internally using custom pickle bytecode that calls Python's `eval()`. This means:

- **No scikit-learn dependency** required at inference time
- **Portable** — the model runs in any Python 3.x environment with only `numpy`, `pandas`, and `scipy`
- Two model files are always saved:
  - `dev_model.pkl` — trained on 75% of data, used for metric reporting
  - `soccer_sense.pkl` — trained on 100% of data, used for predictions
  - `soccer_sense_v{version}.pkl` — versioned copy for rollback

### Artifact Contents

```python
{
  "version": "1.0.0",
  "team_profiles": { "Argentina": {"elo": 2048.5, "elo_rank": 1, ...}, ... },
  "player_profiles": [ {"name": "L. Messi", "team": "Argentina", ...}, ... ],
  "feature_columns": ["home_elo", "away_elo", ...],  # 10 features
  "metrics": {
    "result_accuracy": 0.542,
    "result_log_loss": 0.981,
    "home_goals_mae": 0.823,
    "away_goals_mae": 0.714,
  },
  "match_model": PurePythonPipeline(...),
  "home_goals_model": PurePythonPipeline(...),
  "away_goals_model": PurePythonPipeline(...),
  "btts_model": PurePythonPipeline(...),
  "first_goal_model": PurePythonPipeline(...),
  "home_clean_sheet_model": PurePythonPipeline(...),
  "away_clean_sheet_model": PurePythonPipeline(...),
}
```
