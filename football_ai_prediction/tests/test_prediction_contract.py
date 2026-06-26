from src.football_ai.predictor import FootballPredictor
from src.football_ai.train import MODEL_PATH, train_model


def test_prediction_matches_required_json_contract():
    if not MODEL_PATH.exists():
        train_model()

    result = FootballPredictor().predict("Argentina", "Brazil")
    output = result["output"]

    assert "match_prediction" in output
    assert "score_prediction" in output
    assert "goal_insights" in output
    assert "player_prediction" in output

    win_probabilities = output["match_prediction"]["win_probabilities"]
    assert win_probabilities["home_team"]["team"] == "Argentina"
    assert win_probabilities["away_team"]["team"] == "Brazil"
    assert (
        win_probabilities["home_team"]["probability"]
        + win_probabilities["draw"]["probability"]
        + win_probabilities["away_team"]["probability"]
        == 100
    )

    scoreline = output["score_prediction"]["predicted_scoreline"]
    assert isinstance(scoreline["home_goals"], int)
    assert isinstance(scoreline["away_goals"], int)

    assert isinstance(output["goal_insights"]["both_teams_to_score"]["prediction"], bool)
    assert output["player_prediction"]["home_team"]["goal"]
    assert output["player_prediction"]["away_team"]["goal"]
