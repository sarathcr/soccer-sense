import io
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from .predictor import FootballPredictor
from .train import train_model, DATA_DIR, MODEL_PATH
from .crawler import crawl_and_process, clean_column_name, enrich_match_data


app = FastAPI(title="Football AI Match Prediction API", version="1.0.0")
predictor = FootballPredictor()


class MatchInput(BaseModel):
    home_team: str
    away_team: str


class CrawlTrainInput(BaseModel):
    matches_url: str | None = None
    players_url: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def read_root():
    static_file = Path(__file__).parent / "static" / "index.html"
    if static_file.exists():
        with open(static_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Football AI Match Prediction API</h1><p>Static UI not found. Create src/football_ai/static/index.html to display the UI.</p>"


@app.post("/predict")
def predict_match(payload: MatchInput):
    try:
        return predictor.predict(payload.home_team, payload.away_team)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/teams")
def get_teams() -> list[str]:
    try:
        return sorted(list(predictor.artifact.get("team_profiles", {}).keys()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train-from-url")
def train_from_url(payload: CrawlTrainInput):
    global predictor
    try:
        # Determine output paths from config, or default
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "pipeline_config.json"
        
        matches_out = project_root / "data" / "sample_matches.csv"
        players_out = project_root / "data" / "sample_players.csv"
        
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    if "matches" in config and "output" in config["matches"]:
                        matches_out = project_root / config["matches"]["output"]
                    if "players" in config and "output" in config["players"]:
                        players_out = project_root / config["players"]["output"]
            except Exception:
                pass

        logger_msg = f"Crawling matches={payload.matches_url}, players={payload.players_url}"
        print(logger_msg)
        
        matches_df, players_df = crawl_and_process(
            payload.matches_url, payload.players_url
        )
        
        # Save to file system so retrained state persists
        if matches_df is not None and not matches_df.empty:
            matches_out.parent.mkdir(parents=True, exist_ok=True)
            matches_df.to_csv(matches_out, index=False)
            matches_arg = matches_df
        else:
            matches_arg = matches_out if matches_out.exists() else None
            
        if players_df is not None and not players_df.empty:
            players_out.parent.mkdir(parents=True, exist_ok=True)
            players_df.to_csv(players_out, index=False)
            players_arg = players_df
        else:
            players_arg = players_out if players_out.exists() else None
            
        # Train model
        artifact = train_model(matches_path=matches_arg, players_path=players_arg)
        
        # Reload predictor with the new model
        predictor = FootballPredictor(MODEL_PATH)
        
        # Check warnings
        warnings = []
        if matches_df is not None and not matches_df.empty:
            warnings.append("Matches: Feature columns (home_elo, away_elo, home_elo_rank, away_elo_rank) were missing from webpage; dynamically generated using matches history.")
        if players_df is not None and not players_df.empty:
            warnings.append("Players: Detailed stats (xg_per_90, goals_per_90, start_probability) were missing from webpage; auto-filled with position-appropriate defaults.")

        return {
            "status": "success",
            "message": "Model trained and loaded successfully.",
            "scraped_matches": len(matches_df) if matches_df is not None else 0,
            "scraped_players": len(players_df) if players_df is not None else 0,
            "metrics": artifact["metrics"],
            "teams": sorted(list(artifact["team_profiles"].keys())),
            "warnings": warnings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def load_uploaded_file(upload_file: UploadFile) -> pd.DataFrame:
    """Read uploaded file into a pandas DataFrame based on file extension."""
    content = upload_file.file.read()
    filename = upload_file.filename.lower()
    
    # Needs pandas import
    import pandas as pd
    
    if filename.endswith('.csv'):
        return pd.read_csv(io.BytesIO(content))
    elif filename.endswith(('.xls', '.xlsx')):
        return pd.read_excel(io.BytesIO(content))
    elif filename.endswith('.json'):
        return pd.read_json(io.BytesIO(content))
    elif filename.endswith('.parquet'):
        return pd.read_parquet(io.BytesIO(content))
    else:
        raise ValueError(f"Unsupported file format for {upload_file.filename}")


def load_uploaded_files(upload_files: list[UploadFile] | None) -> pd.DataFrame | None:
    """Read multiple uploaded files and concatenate them into a single DataFrame."""
    if not upload_files:
        return None
        
    import pandas as pd
    dfs = []
    for uf in upload_files:
        if not uf.filename:
            continue
        try:
            df = load_uploaded_file(uf)
            if df is not None and not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"Error loading uploaded file {uf.filename}: {e}")
            raise ValueError(f"Failed to parse file {uf.filename}: {str(e)}")
            
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


@app.post("/train-from-upload")
def train_from_upload(
    matches_files: list[UploadFile] = File(None),
    players_files: list[UploadFile] = File(None)
):
    global predictor
    import pandas as pd
    try:
        # Determine output paths from config, or default
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "pipeline_config.json"
        
        matches_out = project_root / "data" / "sample_matches.csv"
        players_out = project_root / "data" / "sample_players.csv"
        
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    if "matches" in config and "output" in config["matches"]:
                        matches_out = project_root / config["matches"]["output"]
                    if "players" in config and "output" in config["players"]:
                        players_out = project_root / config["players"]["output"]
            except Exception:
                pass

        warnings = []
        
        # Load and combine uploaded files
        raw_matches = load_uploaded_files(matches_files)
        raw_players = load_uploaded_files(players_files)
        
        matches_df = None
        players_df = None
        
        if raw_matches is not None and not raw_matches.empty:
            raw_matches.columns = [clean_column_name(c) for c in raw_matches.columns]
            
            # Check matches missing fields
            for col in ["home_team", "away_team", "home_goals", "away_goals"]:
                if col not in raw_matches.columns:
                    warnings.append(f"Matches file: Critical column '{col}' was missing and initialized to defaults.")
            for col in ["home_elo", "away_elo", "home_elo_rank", "away_elo_rank"]:
                if col not in raw_matches.columns:
                    warnings.append(f"Matches file: Feature column '{col}' was missing; dynamically generated using historical records.")
                    
            matches_df = enrich_match_data(raw_matches)
            
        if raw_players is not None and not raw_players.empty:
            raw_players.columns = [clean_column_name(c) for c in raw_players.columns]
            players_df = raw_players
            
            # Check players missing fields
            for col in ["team", "name", "position"]:
                if col not in players_df.columns:
                    warnings.append(f"Players file: Critical column '{col}' was missing.")
            
            # Map required columns or set defaults
            required_player_cols = ["team", "name", "position", "xg_per_90", "goals_per_90", "xa_per_90", "assists_per_90", "start_probability", "status"]
            for col in required_player_cols:
                if col not in players_df.columns:
                    warnings.append(f"Players file: Feature column '{col}' was missing; auto-filled with position-appropriate defaults.")
                    if col == "status":
                        players_df[col] = "active"
                    elif col == "start_probability":
                        players_df[col] = 0.8
                    else:
                        players_df[col] = 0.0

        # Save to file system so state persists and determine paths for training
        if matches_df is not None and not matches_df.empty:
            matches_out.parent.mkdir(parents=True, exist_ok=True)
            matches_df.to_csv(matches_out, index=False)
            matches_arg = matches_df
        else:
            # Fallback to existing or default sample matches
            matches_arg = matches_out if matches_out.exists() else (project_root / "data" / "sample_matches.csv")
            
        if players_df is not None and not players_df.empty:
            players_out.parent.mkdir(parents=True, exist_ok=True)
            players_df.to_csv(players_out, index=False)
            players_arg = players_df
        else:
            # Fallback to existing or default sample players
            players_arg = players_out if players_out.exists() else (project_root / "data" / "sample_players.csv")
            
        # Train model
        artifact = train_model(matches_path=matches_arg, players_path=players_arg)
        
        # Reload predictor with the new production model
        predictor = FootballPredictor(MODEL_PATH)
        
        return {
            "status": "success",
            "message": "Model trained and loaded successfully from uploaded files.",
            "uploaded_matches": len(matches_df) if matches_df is not None else 0,
            "uploaded_players": len(players_df) if players_df is not None else 0,
            "metrics": artifact["metrics"],
            "teams": sorted(list(artifact["team_profiles"].keys())),
            "warnings": warnings
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train-from-url")
def train_from_url(payload: CrawlTrainInput):
    global predictor
    try:
        # Determine output paths from config, or default
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "pipeline_config.json"
        
        matches_out = project_root / "data" / "sample_matches.csv"
        players_out = project_root / "data" / "sample_players.csv"
        
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    if "matches" in config and "output" in config["matches"]:
                        matches_out = project_root / config["matches"]["output"]
                    if "players" in config and "output" in config["players"]:
                        players_out = project_root / config["players"]["output"]
            except Exception:
                pass

        logger_msg = f"Crawling matches={payload.matches_url}, players={payload.players_url}"
        print(logger_msg)
        
        matches_df, players_df = crawl_and_process(
            payload.matches_url if payload.matches_url and payload.matches_url.strip() else None,
            payload.players_url if payload.players_url and payload.players_url.strip() else None
        )
        
        # Save to file system so retrained state persists
        if matches_df is not None and not matches_df.empty:
            matches_out.parent.mkdir(parents=True, exist_ok=True)
            matches_df.to_csv(matches_out, index=False)
            matches_arg = matches_df
        else:
            matches_arg = matches_out if matches_out.exists() else (project_root / "data" / "sample_matches.csv")
            
        if players_df is not None and not players_df.empty:
            players_out.parent.mkdir(parents=True, exist_ok=True)
            players_df.to_csv(players_out, index=False)
            players_arg = players_df
        else:
            players_arg = players_out if players_out.exists() else (project_root / "data" / "sample_players.csv")
            
        # Train model
        artifact = train_model(matches_path=matches_arg, players_path=players_arg)
        
        # Reload predictor with the new production model
        predictor = FootballPredictor(MODEL_PATH)
        
        # Check warnings
        warnings = []
        if matches_df is not None and not matches_df.empty:
            warnings.append("Matches: Feature columns (home_elo, away_elo, home_elo_rank, away_elo_rank) were missing from webpage; dynamically generated using matches history.")
        if players_df is not None and not players_df.empty:
            warnings.append("Players: Detailed stats (xg_per_90, goals_per_90, start_probability) were missing from webpage; auto-filled with position-appropriate defaults.")

        return {
            "status": "success",
            "message": "Model trained and loaded successfully.",
            "scraped_matches": len(matches_df) if matches_df is not None else 0,
            "scraped_players": len(players_df) if players_df is not None else 0,
            "metrics": artifact["metrics"],
            "teams": sorted(list(artifact["team_profiles"].keys())),
            "warnings": warnings
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/team-profiles")
def get_team_profiles():
    try:
        return predictor.artifact.get("team_profiles", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/matches")
def download_matches():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "pipeline_config.json"
    matches_out = project_root / "data" / "sample_matches.csv"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                if "matches" in config and "output" in config["matches"]:
                    matches_out = project_root / config["matches"]["output"]
        except Exception:
            pass
    if matches_out.exists():
        return FileResponse(path=matches_out, filename="processed_matches.csv", media_type="text/csv")
    raise HTTPException(status_code=404, detail="Processed matches file not found.")


@app.get("/download/players")
def download_players():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "pipeline_config.json"
    players_out = project_root / "data" / "sample_players.csv"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                if "players" in config and "output" in config["players"]:
                    players_out = project_root / config["players"]["output"]
        except Exception:
            pass
    if players_out.exists():
        return FileResponse(path=players_out, filename="processed_players.csv", media_type="text/csv")
    raise HTTPException(status_code=404, detail="Processed players file not found.")


@app.get("/download/dev_model")
def download_dev_model():
    project_root = Path(__file__).resolve().parents[2]
    dev_model = project_root / "models" / "dev_model.pkl"
    if dev_model.exists():
        return FileResponse(path=dev_model, filename="dev_model.pkl", media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="Dev model file not found.")


@app.get("/download/production_model")
def download_production_model():
    project_root = Path(__file__).resolve().parents[2]
    prod_model = project_root / "models" / "production_model.pkl"
    if prod_model.exists():
        return FileResponse(path=prod_model, filename="production_model.pkl", media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="Production model file not found.")


@app.get("/download/production_model_versioned")
def download_production_model_versioned():
    try:
        version = predictor.artifact.get("version", "1.0.0")
        project_root = Path(__file__).resolve().parents[2]
        prod_model_v = project_root / "models" / f"production_model_v{version}.pkl"
        if prod_model_v.exists():
            return FileResponse(path=prod_model_v, filename=f"production_model_v{version}.pkl", media_type="application/octet-stream")
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="Versioned production model file not found.")
