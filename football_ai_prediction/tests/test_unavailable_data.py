from src.football_ai.predictor import FootballPredictor
from src.football_ai.train import MODEL_PATH, train_model


def test_predictor_unavailable_team():
    if not MODEL_PATH.exists():
        train_model()

    predictor = FootballPredictor()
    
    # "Atlantis" is not present in profiles
    result = predictor.predict("Atlantis", "Brazil")
    assert "output" in result
    output = result["output"]

    assert "match_prediction" in output
    assert "score_prediction" in output
    assert "goal_insights" in output
    assert "player_prediction" in output
    assert "explanation_steps" in output

    win_probabilities = output["match_prediction"]["win_probabilities"]
    assert win_probabilities["home_team"]["team"] == "Atlantis"
    assert win_probabilities["home_team"]["probability"] == "Data Unavailable"
    assert win_probabilities["draw"]["probability"] == "Data Unavailable"
    assert win_probabilities["away_team"]["team"] == "Brazil"
    assert win_probabilities["away_team"]["probability"] == "Data Unavailable"

    scoreline = output["score_prediction"]["predicted_scoreline"]
    assert scoreline["home_team"] == "Atlantis"
    assert scoreline["home_goals"] == "Data Unavailable"
    assert scoreline["away_team"] == "Brazil"
    assert scoreline["away_goals"] == "Data Unavailable"
    assert output["score_prediction"]["total_goals"] == "Data Unavailable"

    goal_insights = output["goal_insights"]
    assert goal_insights["first_team_to_score"]["team"] == "Data Unavailable"
    assert goal_insights["first_team_to_score"]["probability"] == "Data Unavailable"
    assert goal_insights["both_teams_to_score"]["prediction"] == "Data Unavailable"
    assert goal_insights["both_teams_to_score"]["probability"] == "Data Unavailable"

    player_prediction = output["player_prediction"]
    assert player_prediction["home_team"]["team"] == "Atlantis"
    assert player_prediction["home_team"]["goal"] == "Data Unavailable"
    assert player_prediction["home_team"]["clean_sheet_prediction"]["goalkeeper"] == "Data Unavailable"
    assert player_prediction["home_team"]["clean_sheet_prediction"]["prediction"] == "Data Unavailable"
    assert player_prediction["home_team"]["clean_sheet_prediction"]["probability"] == "Data Unavailable"

    assert player_prediction["away_team"]["team"] == "Brazil"
    assert player_prediction["away_team"]["goal"] == "Data Unavailable"
    assert player_prediction["away_team"]["clean_sheet_prediction"]["goalkeeper"] == "Data Unavailable"
    assert player_prediction["away_team"]["clean_sheet_prediction"]["prediction"] == "Data Unavailable"
    assert player_prediction["away_team"]["clean_sheet_prediction"]["probability"] == "Data Unavailable"

    explanation = output["explanation_steps"]
    assert isinstance(explanation, dict)
    assert explanation["step1_profiles"]["home"]["elo"] == "Data Unavailable"
    assert explanation["step2_expected_goals"]["home_lambda"] == "Data Unavailable"
    assert explanation["step3_joint_distribution"]["home_win_raw"] == "Data Unavailable"
    assert explanation["step4_insights_math"]["btts"]["result"] == "Data Unavailable"
