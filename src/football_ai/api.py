import io
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from .predictor import FootballPredictor
from .train import train_model, DATA_DIR, MODEL_PATH, get_writable_path
from .crawler import crawl_and_process, clean_column_name, enrich_match_data, validate_and_standardize_players


app = FastAPI(title="Soccer Sense Match Prediction API", version="1.0.0")
predictor = FootballPredictor()


class MatchInput(BaseModel):
    home_team: str
    away_team: str


class CrawlTrainInput(BaseModel):
    matches_url: str | None = None
    players_url: str | None = None


class UpdateVersionInput(BaseModel):
    version: str


@app.get("/health")
def health() -> dict:
    from .blob_storage import get_blob_status
    return {"status": "ok", "blob": get_blob_status()}


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
        data = predictor.predict(payload.home_team, payload.away_team)
        return Response(content=json.dumps(data), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/teams")
def get_teams() -> list[str]:
    try:
        return sorted(list(predictor.artifact.get("team_profiles", {}).keys()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def load_uploaded_file(upload_file: UploadFile) -> pd.DataFrame:
    """Read uploaded file into a pandas DataFrame based on file extension."""
    content = upload_file.file.read()
    filename = upload_file.filename.lower()
    
    # Needs pandas import
    import pandas as pd
    
    if filename.endswith('.csv'):
        return pd.read_csv(io.BytesIO(content), encoding='utf-8-sig', sep=None, engine='python')
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
    
    import tempfile
    debug_log_path = Path(tempfile.gettempdir()) / "soccer_sense" / "debug_upload.log"
    debug_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(debug_log_path, "a", encoding="utf-8") as f:
        f.write("\n--- Individual Files Columns ---\n")
        
        for uf in upload_files:
            if not uf.filename:
                continue
            try:
                uf.file.seek(0)
                df = load_uploaded_file(uf)
                if df is not None and not df.empty:
                    dfs.append(df)
                    f.write(f"File: {uf.filename} | Shape: {df.shape} | Columns: {list(df.columns)}\n")
            except Exception as e:
                f.write(f"Error reading file {uf.filename}: {e}\n")
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
        
        matches_out = DATA_DIR / "sample_matches.csv"
        players_out = DATA_DIR / "sample_players.csv"
        
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    if "matches" in config and "output" in config["matches"]:
                        matches_out = get_writable_path(DATA_DIR / Path(config["matches"]["output"]).name)
                    if "players" in config and "output" in config["players"]:
                        players_out = get_writable_path(DATA_DIR / Path(config["players"]["output"]).name)
            except Exception:
                pass

        warnings = []
        
        # Load and combine uploaded files
        raw_matches = load_uploaded_files(matches_files)
        raw_players = load_uploaded_files(players_files)
        
        matches_df = None
        players_df = None
        
        # Write debug log of uploaded files and columns
        try:
            import tempfile
            debug_log_path = Path(tempfile.gettempdir()) / "soccer_sense" / "debug_upload.log"
            debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_log_path, "w", encoding="utf-8") as f:
                f.write("=== Upload Debug ===\n")
                if matches_files:
                    f.write(f"Uploaded matches files: {[uf.filename for uf in matches_files]}\n")
                if players_files:
                    f.write(f"Uploaded players files: {[uf.filename for uf in players_files]}\n")
                if raw_matches is not None:
                    f.write(f"Parsed raw_matches shape: {raw_matches.shape}\n")
                    f.write(f"Parsed raw_matches columns: {list(raw_matches.columns)}\n")
                else:
                    f.write("raw_matches is None\n")
        except Exception as ex:
            print(f"Failed to write debug upload log: {ex}")
        
        if raw_matches is not None and not raw_matches.empty:
            raw_matches.columns = [clean_column_name(c) for c in raw_matches.columns]
            
            # Check matches missing fields
            for col in ["home_team", "away_team", "home_goals", "away_goals"]:
                if col not in raw_matches.columns:
                    warnings.append(f"Matches file: Critical column '{col}' was missing and initialized to defaults.")

            # enrich_match_data() preserves ELO columns if present (or aliased) in the upload;
            # otherwise it computes them automatically from match history — this is normal
            # expected behaviour for standard match data formats and requires no user action.
            matches_df = enrich_match_data(raw_matches)

        if raw_players is not None and not raw_players.empty:
            players_df, missing_cols = validate_and_standardize_players(raw_players)
            if missing_cols:
                warnings.append(f"Players file: Missing fields required for player-level intelligence: {', '.join(missing_cols)}. Inferred or filled with defaults.")

        # Save to file system so state persists and determine paths for training
        if matches_df is not None and not matches_df.empty:
            matches_out.parent.mkdir(parents=True, exist_ok=True)
            matches_df.to_csv(matches_out, index=False, encoding="utf-8")
            matches_arg = matches_df
        else:
            # Fallback to existing or default sample matches
            matches_arg = matches_out if matches_out.exists() else (project_root / "data" / "sample_matches.csv")
            
        if players_df is not None and not players_df.empty:
            players_out.parent.mkdir(parents=True, exist_ok=True)
            players_df.to_csv(players_out, index=False, encoding="utf-8")
            players_arg = players_df
        else:
            # Fallback to existing or default sample players
            players_arg = players_out if players_out.exists() else (project_root / "data" / "sample_players.csv")
            
        # Train model
        artifact = train_model(matches_path=matches_arg, players_path=players_arg, model_path=MODEL_PATH)
        
        # Reload predictor with the new production model
        predictor = FootballPredictor(MODEL_PATH)

        # ── Persist trained model to Vercel Blob so cold starts keep it ──
        blob_synced = False
        try:
            from .blob_storage import upload_model, is_blob_enabled
            if is_blob_enabled():
                result = upload_model(MODEL_PATH)
                blob_synced = result is not None
        except Exception as _blob_err:
            print(f"[Blob] Upload after training failed: {_blob_err}")

        required_labels = ["name", "nationality", "apps", "mins", "goals", "xg", "Goals vs xG", "shots", "sot", "conv %", "xG per Shot", "goals x90", "xg x90", "Goals vs xG x90", "shots x90", "sot x90"]
        player_validation_status = {col: (col not in missing_cols) for col in required_labels} if raw_players is not None else None

        return {
            "status": "success",
            "message": "Model trained and loaded successfully from uploaded files.",
            "uploaded_matches": len(matches_df) if matches_df is not None else 0,
            "uploaded_players": len(players_df) if players_df is not None else 0,
            "metrics": artifact["metrics"],
            "teams": sorted(list(artifact["team_profiles"].keys())),
            "warnings": warnings,
            "player_validation_status": player_validation_status,
            "version": artifact.get("version", "1.0.0"),
            "blob_synced": blob_synced,
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
        
        matches_out = DATA_DIR / "sample_matches.csv"
        players_out = DATA_DIR / "sample_players.csv"
        
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    if "matches" in config and "output" in config["matches"]:
                        matches_out = get_writable_path(DATA_DIR / Path(config["matches"]["output"]).name)
                    if "players" in config and "output" in config["players"]:
                        players_out = get_writable_path(DATA_DIR / Path(config["players"]["output"]).name)
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
            matches_df.to_csv(matches_out, index=False, encoding="utf-8")
            matches_arg = matches_df
        else:
            matches_arg = matches_out if matches_out.exists() else (project_root / "data" / "sample_matches.csv")
            
        if players_df is not None and not players_df.empty:
            players_out.parent.mkdir(parents=True, exist_ok=True)
            players_df.to_csv(players_out, index=False, encoding="utf-8")
            players_arg = players_df
        else:
            players_arg = players_out if players_out.exists() else (project_root / "data" / "sample_players.csv")
            
        # Train model
        artifact = train_model(matches_path=matches_arg, players_path=players_arg, model_path=MODEL_PATH)
        
        # Reload predictor with the new production model
        predictor = FootballPredictor(MODEL_PATH)

        # ── Persist trained model to Vercel Blob so cold starts keep it ──
        blob_synced = False
        try:
            from .blob_storage import upload_model, is_blob_enabled
            if is_blob_enabled():
                result = upload_model(MODEL_PATH)
                blob_synced = result is not None
        except Exception as _blob_err:
            print(f"[Blob] Upload after training failed: {_blob_err}")

        # Check warnings
        warnings = []
        if matches_df is not None and not matches_df.empty:
            # Only warn about ELO if the scraped data did not contain those columns
            # (enrich_match_data always computes them, so check the enriched result)
            elo_cols_present = all(
                col in matches_df.columns
                and pd.to_numeric(matches_df[col], errors="coerce").notna().any()
                for col in ("home_elo", "away_elo", "home_elo_rank", "away_elo_rank")
            )
            if not elo_cols_present:
                warnings.append(
                    "Matches: ELO columns were not available from the source URL; "
                    "they were dynamically computed from the scraped match history."
                )
        missing_cols = []
        if players_df is not None and not players_df.empty:
            _, missing_cols = validate_and_standardize_players(players_df)
            if missing_cols:
                warnings.append(f"Players Webpage: Missing fields required for player-level intelligence: {', '.join(missing_cols)}. Inferred or filled with defaults.")

        required_labels = ["name", "nationality", "apps", "mins", "goals", "xg", "Goals vs xG", "shots", "sot", "conv %", "xG per Shot", "goals x90", "xg x90", "Goals vs xG x90", "shots x90", "sot x90"]
        player_validation_status = {col: (col not in missing_cols) for col in required_labels} if players_df is not None else None

        return {
            "status": "success",
            "message": "Model trained and loaded successfully.",
            "scraped_matches": len(matches_df) if matches_df is not None else 0,
            "scraped_players": len(players_df) if players_df is not None else 0,
            "metrics": artifact["metrics"],
            "teams": sorted(list(artifact["team_profiles"].keys())),
            "warnings": warnings,
            "player_validation_status": player_validation_status,
            "version": artifact.get("version", "1.0.0"),
            "blob_synced": blob_synced,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
def reset_system() -> dict[str, str]:
    global predictor
    try:
        # Determine output paths
        project_root = Path(__file__).resolve().parents[2]
        matches_out = DATA_DIR / "sample_matches.csv"
        players_out = DATA_DIR / "sample_players.csv"
        
        import tempfile
        tmp_dir = Path(tempfile.gettempdir())
        
        # Reset custom files
        for p, name in [(matches_out, "sample_matches.csv"), (players_out, "sample_players.csv")]:
            is_tmp = tmp_dir in p.parents or "soccer_sense" in str(p)
            if is_tmp:
                if p.exists() and p.is_file():
                    try:
                        p.unlink()
                    except Exception as e:
                        import logging
                        logging.warning(f"Could not delete {p}: {e}")
            else:
                # Local environment: copy backup over it
                backup_file = project_root / "data" / f"{name}.bak"
                if backup_file.exists():
                    import shutil
                    shutil.copy2(backup_file, p)
                    
        # Delete custom model from tmp if present
        is_model_tmp = tmp_dir in MODEL_PATH.parents or "soccer_sense" in str(MODEL_PATH)
        if is_model_tmp:
            if MODEL_PATH.exists() and MODEL_PATH.is_file():
                try:
                    MODEL_PATH.unlink()
                except Exception as e:
                    import logging
                    logging.warning(f"Could not delete {MODEL_PATH}: {e}")
                    
            versioned_path = MODEL_PATH.parent / "soccer_sense_v1.0.0.pkl"
            if versioned_path.exists():
                try:
                    versioned_path.unlink()
                except Exception:
                    pass

        # Resolve the default data paths for retraining.
        # On Vercel, DATA_DIR points to /tmp which we just cleared — fall back to the
        # read-only bundled defaults shipped with the deployment package.
        # On local, DATA_DIR is writable and the .bak restore above already refreshed it.
        from .train import READ_ONLY_DATA_DIR

        def _resolve_default(tmp_path: Path, filename: str) -> Path:
            """Return tmp_path if it still exists (local, after .bak restore),
            otherwise return the read-only bundled default file."""
            if tmp_path.exists():
                return tmp_path
            bundled = READ_ONLY_DATA_DIR / filename
            if bundled.exists():
                return bundled
            bak = READ_ONLY_DATA_DIR / f"{filename}.bak"
            if bak.exists():
                return bak
            return tmp_path  # let train_model surface a clear error

        matches_default = _resolve_default(matches_out, "sample_matches.csv")
        players_default = _resolve_default(players_out, "sample_players.csv")

        # Retrain model using default packaged data
        train_model(matches_path=matches_default, players_path=players_default, model_path=MODEL_PATH)
        
        # Reload predictor to pick up the clean default model
        predictor = FootballPredictor(MODEL_PATH)

        # ── Remove custom model from Vercel Blob so cold starts use bundled default ──
        try:
            from .blob_storage import delete_model, is_blob_enabled
            if is_blob_enabled():
                delete_model()
        except Exception as _blob_err:
            print(f"[Blob] Delete after reset failed: {_blob_err}")

        return {
            "status": "success",
            "message": "System data and models successfully reset to clean defaults."
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


@app.get("/model-version")
def get_model_version() -> dict[str, str]:
    try:
        return {"version": predictor.artifact.get("version", "1.0.0")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update-version")
def update_version(payload: UpdateVersionInput):
    global predictor
    try:
        from .train import save_model_artifact
        # Update version in loaded predictor artifact
        predictor.artifact["version"] = payload.version
        
        # Save updated model to MODEL_PATH
        save_model_artifact(predictor, MODEL_PATH)
        
        # Save versioned copy
        versioned_path = MODEL_PATH.parent / f"soccer_sense_v{payload.version}.pkl"
        save_model_artifact(predictor, versioned_path)
        
        # Re-initialize predictor
        predictor = FootballPredictor(MODEL_PATH)
        
        return {
            "status": "success",
            "message": f"Model version successfully updated to {payload.version}.",
            "version": payload.version
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/matches")
def download_matches():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "pipeline_config.json"
    matches_out = DATA_DIR / "sample_matches.csv"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                if "matches" in config and "output" in config["matches"]:
                    matches_out = get_writable_path(project_root / config["matches"]["output"])
        except Exception:
            pass
    if not matches_out.exists():
        matches_out = project_root / "data" / "sample_matches.csv"
    if matches_out.exists():
        return FileResponse(path=matches_out, filename="processed_matches.csv", media_type="text/csv")
    raise HTTPException(status_code=404, detail="Processed matches file not found.")


@app.get("/download/players")
def download_players():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "pipeline_config.json"
    players_out = DATA_DIR / "sample_players.csv"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                if "players" in config and "output" in config["players"]:
                    players_out = get_writable_path(project_root / config["players"]["output"])
        except Exception:
            pass
    if not players_out.exists():
        players_out = project_root / "data" / "sample_players.csv"
    if players_out.exists():
        return FileResponse(path=players_out, filename="processed_players.csv", media_type="text/csv")
    raise HTTPException(status_code=404, detail="Processed players file not found.")


@app.get("/download/dev_model")
def download_dev_model():
    project_root = Path(__file__).resolve().parents[2]
    dev_model = DATA_DIR.parent / "models" / "dev_model.pkl"
    if not dev_model.exists():
        dev_model = project_root / "models" / "dev_model.pkl"
    if dev_model.exists():
        return FileResponse(path=dev_model, filename="dev_model.pkl", media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="Dev model file not found.")


@app.get("/download/production_model")
def download_production_model():
    project_root = Path(__file__).resolve().parents[2]
    prod_model = MODEL_PATH
    if not prod_model.exists():
        prod_model = project_root / "models" / "soccer_sense.pkl"
    if prod_model.exists():
        return FileResponse(path=prod_model, filename="soccer_sense.pkl", media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="Production model file not found.")


@app.get("/download/production_model_versioned")
def download_production_model_versioned():
    try:
        version = predictor.artifact.get("version", "1.0.0")
        project_root = Path(__file__).resolve().parents[2]
        prod_model_v = MODEL_PATH.parent / f"soccer_sense_v{version}.pkl"
        if not prod_model_v.exists():
            prod_model_v = project_root / "models" / f"soccer_sense_v{version}.pkl"
        if prod_model_v.exists():
            return FileResponse(path=prod_model_v, filename=f"soccer_sense_v{version}.pkl", media_type="application/octet-stream")
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="Versioned production model file not found.")

# Trigger reload: model updated with goalkeeper names

