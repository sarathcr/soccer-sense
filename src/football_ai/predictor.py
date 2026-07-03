from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .features import build_inference_features, normalize_team_name


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "soccersense.pkl"


_PLAYER_NAME_TRANSLATE_TABLE = str.maketrans({
    "Ø": "O", "ø": "o",
    "Æ": "AE", "æ": "ae",
    "ß": "ss",
    "Ð": "D", "ð": "d",
    "Þ": "TH", "þ": "th"
})


def normalize_player_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    import unicodedata
    name = name.translate(_PLAYER_NAME_TRANSLATE_TABLE)
    nfkd_form = unicodedata.normalize('NFKD', name)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).strip()


class FootballPredictor:
    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)

        # If the default path is requested, prefer a model that was written to
        # /tmp by a training run in the current session (Vercel or local).
        if Path(model_path) == DEFAULT_MODEL_PATH:
            import tempfile
            tmp_model_path = Path(tempfile.gettempdir()) / "soccersense" / "models" / "soccersense.pkl"
            if tmp_model_path.exists():
                self.model_path = tmp_model_path

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. Run `python3 -m src.football_ai.train` first."
            )
        import pickle
        with open(self.model_path, "rb") as f:
            self.artifact = pickle.load(f)

    def predict(self, home_team: str, away_team: str) -> dict[str, Any]:
        return self.artifact.predict(home_team, away_team)


def predict(home_team: str, away_team: str) -> dict[str, Any]:
    return FootballPredictor().predict(home_team, away_team)


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python3 -m src.football_ai.predictor Argentina Brazil")
        raise SystemExit(2)
    result = predict(sys.argv[1], sys.argv[2])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
