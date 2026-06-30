import pandas as pd
import pytest
from src.football_ai.crawler import (
    clean_column_name,
    parse_score,
    enrich_match_data,
    detect_and_parse_matches,
    detect_and_parse_players,
)


def test_clean_column_name():
    assert clean_column_name("Home Team") == "home_team"
    assert clean_column_name("Goals (Home)") == "goals_home"
    assert clean_column_name("Caps/Apps") == "capsapps"
    assert clean_column_name("  Pos.  ") == "pos"


def test_parse_score():
    assert parse_score("2-1") == (2, 1)
    assert parse_score("1 - 0") == (1, 0)
    assert parse_score("3 : 2 (a.e.t.)") == (3, 2)
    assert parse_score("0–0") == (0, 0)
    assert parse_score("Invalid") is None


def test_detect_and_parse_matches():
    mock_html = """
    <table>
        <tr>
            <th>Date</th>
            <th>Home Team</th>
            <th>Away Team</th>
            <th>Result</th>
        </tr>
        <tr>
            <td>2024-06-01</td>
            <td>Argentina</td>
            <td>Brazil</td>
            <td>2-1</td>
        </tr>
        <tr>
            <td>2024-06-02</td>
            <td>France</td>
            <td>England</td>
            <td>0-0</td>
        </tr>
    </table>
    """
    df = detect_and_parse_matches(mock_html)
    assert len(df) == 2
    assert list(df.columns) == ["home_team", "away_team", "home_goals", "away_goals", "date", "competition"]
    assert df.loc[0, "home_team"] == "Argentina"
    assert df.loc[0, "away_team"] == "Brazil"
    assert df.loc[0, "home_goals"] == 2
    assert df.loc[0, "away_goals"] == 1


def test_detect_and_parse_players():
    mock_html = """
    <h3>Argentina</h3>
    <table>
        <tr>
            <th>Pos.</th>
            <th>Player</th>
            <th>Goals</th>
            <th>Caps</th>
        </tr>
        <tr>
            <td>FW</td>
            <td>Lionel Messi</td>
            <td>108</td>
            <td>182</td>
        </tr>
        <tr>
            <td>GK</td>
            <td>Emiliano Martínez</td>
            <td>0</td>
            <td>45</td>
        </tr>
    </table>
    """
    df = detect_and_parse_players(mock_html)
    assert len(df) == 2
    assert df.loc[0, "team"] == "Argentina"
    assert df.loc[0, "name"] == "Lionel Messi"
    assert df.loc[0, "position"] == "FW"
    assert df.loc[1, "name"] == "Emiliano Martinez"
    assert df.loc[1, "position"] == "GK"


def test_enrich_match_data():
    raw_df = pd.DataFrame([
        {"home_team": "Argentina", "away_team": "Brazil", "home_goals": 2, "away_goals": 1, "date": "2024-06-01", "competition": "Copa America"},
        {"home_team": "Brazil", "away_team": "Uruguay", "home_goals": 0, "away_goals": 1, "date": "2024-06-05", "competition": "Copa America"},
    ])
    
    enriched = enrich_match_data(raw_df)
    assert len(enriched) == 2
    # Verify Elo columns exist
    assert "home_elo" in enriched.columns
    assert "away_elo" in enriched.columns
    # Verify Elo rank exists
    assert "home_elo_rank" in enriched.columns
    assert "away_elo_rank" in enriched.columns
