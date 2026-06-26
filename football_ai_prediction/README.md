# Football AI Match Prediction

Beginner-friendly AI application for the Goalgorithm FIFA AI Match Prediction Challenge.

The app accepts only two countries:

```json
{
  "home_team": "Argentina",
  "away_team": "Brazil"
}
```

It returns the required match prediction, score prediction, goal insights, and player prediction JSON.

## What This Project Includes

- `src/football_ai/train.py` trains the local ML models.
- `src/football_ai/predictor.py` loads `production_model.pkl` and returns the required JSON.
- `src/football_ai/api.py` exposes a FastAPI `/predict` endpoint.
- `data/sample_matches.csv` and `data/sample_players.csv` let you run the project immediately.
- `tests/` verifies the JSON contract and inference speed.

## Setup

```bash
cd football_ai_prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you already have `pandas`, `scikit-learn`, and `pytest`, you can run the core model without creating a virtual environment.

## Train Model

```bash
python3 -m src.football_ai.train
```

This creates:

```text
models/production_model.pkl
```

That `.pkl` file is the challenge submission model artifact.

## Predict From CLI

```bash
python3 -m src.football_ai.predictor Argentina Brazil
```

## Run API

```bash
uvicorn src.football_ai.api:app --reload
```

Then call:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team":"Argentina","away_team":"Brazil"}'
```

## Test

```bash
pytest
```

## How To Improve Accuracy

Replace the sample CSVs with real data:

- Historical international matches
- FIFA rankings
- Elo ratings
- Recent team form
- xG and xGA
- Player xG/xA per 90
- Injuries and suspensions
- Expected starting lineups

Then retrain before each model freeze:

```bash
python3 -m src.football_ai.train
pytest
```
