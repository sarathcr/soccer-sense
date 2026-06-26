import io
import pandas as pd
import numpy as np
from src.football_ai.crawler import (
    validate_and_standardize_players,
    normalize_player_name,
    infer_position_from_stats
)

def test_unicode_normalization():
    assert normalize_player_name("Arda Güler") == "Arda Guler"
    assert normalize_player_name("Kerem Aktürkoglu") == "Kerem Akturkoglu"
    assert normalize_player_name("Deniz Gül") == "Deniz Gul"
    assert normalize_player_name("Álex Zendejas") == "Alex Zendejas"

def test_position_inference():
    # GK by name
    gk_row = {"name": "Mert Günok", "xg_x90": 0.0, "shots_x90": 0.0}
    assert infer_position_from_stats(gk_row) == "GK"

    # FW by stats
    fw_row = {"name": "Arda Güler", "xg_x90": 0.35, "shots_x90": 4.0}
    assert infer_position_from_stats(fw_row) == "FW"

    # DF by stats
    df_row = {"name": "Tim Ream", "xg_x90": 0.0, "shots_x90": 0.0}
    assert infer_position_from_stats(df_row) == "DF"

def test_player_validation_full():
    # Conforming CSV text
    csv_data = (
        "name,nationality,apps,mins,goals,xg,Goals vs xG,shots,sot,conv %,xG per Shot,goals x90,xg x90,Goals vs xG x90,shots x90,sot x90\n"
        "Deniz Gül,Türkiye,2,35,0,0.59,-0.59,2,1,0,0.29,0,1.52,-1.52,5.14,2.57\n"
        "Arda Güler,Türkiye,3,270,1,1.04,-0.04,12,4,8.33,0.09,0.33,0.35,-0.01,4,1.33\n"
    )
    df = pd.read_csv(io.StringIO(csv_data))
    standardized, missing = validate_and_standardize_players(df)
    
    assert len(missing) == 0
    assert len(standardized) == 2
    assert "team" in standardized.columns
    assert list(standardized["team"]) == ["Türkiye", "Türkiye"]
    assert list(standardized["position"]) == ["FW", "FW"]

def test_player_validation_missing_cols():
    # Non-conforming CSV (missing sot x90, conv %, shots)
    csv_data = (
        "name,nationality,apps,mins,goals,xg\n"
        "Tyler Adams,United States,2,180,0,0\n"
    )
    df = pd.read_csv(io.StringIO(csv_data))
    standardized, missing = validate_and_standardize_players(df)
    
    assert len(missing) > 0
    assert "sot x90" in missing
    assert "conv %" in missing
    assert "shots" in missing
    assert len(standardized) == 1
    assert standardized.loc[0, "name"] == "Tyler Adams"
    assert standardized.loc[0, "sot_x90"] == 0.0
