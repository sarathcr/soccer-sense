import io
import shutil
from pathlib import Path
from fastapi import UploadFile
import src.football_ai.api as api
import src.football_ai.train as train
from src.football_ai.api import train_from_upload, get_teams


def test_api_train_from_upload(tmp_path, monkeypatch):
    # Redirect DATA_DIR and MODEL_PATH to a temporary test directory
    test_data_dir = tmp_path / "data"
    test_models_dir = tmp_path / "models"
    test_data_dir.mkdir()
    test_models_dir.mkdir()
    
    project_root = Path(__file__).resolve().parents[1]
    shutil.copy(project_root / "data" / "sample_matches.csv", test_data_dir / "sample_matches.csv")
    shutil.copy(project_root / "data" / "sample_players.csv", test_data_dir / "sample_players.csv")
    
    monkeypatch.setattr(api, "DATA_DIR", test_data_dir)
    monkeypatch.setattr(train, "DATA_DIR", test_data_dir)
    monkeypatch.setattr(api, "MODEL_PATH", test_models_dir / "soccer_sense.pkl")
    monkeypatch.setattr(train, "MODEL_PATH", test_models_dir / "soccer_sense.pkl")

    # Create mock CSV contents
    matches_csv = (
        "date,home_team,away_team,home_goals,away_goals,competition\n"
        "2024-06-01,TestHome,TestAway,2,1,World Cup\n"
        "2024-06-02,TestHome,TestAway2,0,0,World Cup\n"
        "2024-06-03,TestAway,TestAway2,1,0,World Cup\n"
        "2024-06-04,TestHome,TestAway,3,1,World Cup\n"
        "2024-06-05,TestHome,TestAway2,1,0,World Cup\n"
        "2024-06-06,TestAway,TestAway2,0,2,World Cup\n"
    )
    players_csv = (
        "team,name,position,xg_per_90,goals_per_90,xa_per_90,assists_per_90,start_probability,status\n"
        "TestHome,Player One,FW,0.5,0.4,0.2,0.1,0.9,active\n"
        "TestHome,Player Two,GK,0.0,0.0,0.0,0.0,1.0,active\n"
        "TestAway,Player Three,FW,0.4,0.3,0.1,0.1,0.8,active\n"
    )
    
    # Wrap in FastAPI UploadFile mock objects
    matches_file = UploadFile(
        filename="matches.csv",
        file=io.BytesIO(matches_csv.encode())
    )
    players_file = UploadFile(
        filename="players.csv",
        file=io.BytesIO(players_csv.encode())
    )
    
    # Run endpoint function directly
    res = train_from_upload(matches_files=[matches_file], players_files=[players_file])
    
    assert res["status"] == "success"
    assert res["uploaded_matches"] == 6
    assert res["uploaded_players"] == 3
    assert "metrics" in res
    assert "teams" in res
    
    # Check that teams endpoint reflects new teams
    teams = get_teams()
    assert "Testhome" in teams
    assert "Testaway" in teams
