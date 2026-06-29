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
        data = predictor.predict(payload.home_team, payload.away_team)
        return Response(content=json.dumps(data), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/teams")
def get_teams() -> list[str]:
    try:
        teams = sorted(list(predictor.artifact.get("team_profiles", {}).keys()))
        return [t for t in teams if not any(p in t.lower() for p in ["group", "winner", "runner", "loser", "playoff", "to be decided", "tbd", "qualification", "qualifier"])]
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
    """Read multiple uploaded files and combine them into a single DataFrame.

    Each file is processed **independently** to prevent field corruption from
    column-wise merge collisions on shared fields (e.g. home_goals, date).
    After individual processing the results are combined with row-wise
    concatenation followed by deduplication on (date, home_team, away_team).

    File categories:

    1. **Match-level files** — contain a match key (date + home_team +
       away_team).  Each is individually cleaned; all are then row-wise
       concatenated and deduplicated.

    2. **Team-stats files** — have a team column but no match key
       (e.g. attack_rating, form, xg_avg).  Pivoted onto the combined
       match frame as home_<stat> / away_<stat> columns.
    """
    if not upload_files or not isinstance(upload_files, list):
        return None

    import pandas as pd

    # Canonical match-key columns (checked after clean_column_name normalisation)
    MATCH_KEY_CANDIDATES = [
        ("date", "home_team", "away_team"),
        ("date", "home_team_name", "away_team_name"),
        ("date_gmt", "home_team", "away_team"),
        ("date_gmt", "home_team_name", "away_team_name"),
        ("match_date", "home_team", "away_team"),
    ]

    # Team-stats column names that will be pivoted as home_* / away_* on the match frame.
    # We exclude identity/key columns like 'team', 'elo_rating' (already in match ELO cols).
    TEAM_STAT_PIVOT_COLS = [
        "attack_rating", "defense_rating",
        "goals_avg", "conceded_avg",
        "xg_avg", "xga_avg",
        "win_rate_last10", "form_points_last5", "form_points_last10",
        "clean_sheet_rate", "first_goal_rate",
        "fifa_rank", "fifa_points",
    ]

    def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [clean_column_name(c) for c in df.columns]
        return df

    def _find_key(df: pd.DataFrame):
        """Return the first tuple of columns that all exist in df, or None."""
        for key in MATCH_KEY_CANDIDATES:
            if all(k in df.columns for k in key):
                return list(key)
        return None

    def _is_team_stats_file(df: pd.DataFrame) -> bool:
        """True when the file has a team column but no match-level key."""
        has_team_col = any(c in df.columns for c in ("team", "team_name", "club", "club_name"))
        has_match_key = _find_key(df) is not None
        return has_team_col and not has_match_key

    def _pivot_team_stats_onto_matches(
        result: pd.DataFrame, team_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """Join team-level stats onto the match frame as home_* / away_* columns.

        Uses resolve_country() so that variant country names in the team-stats
        file still match the canonical names used in the match frame.
        """
        from .country_aliases import resolve_country as _rc
        team_col = next(
            (c for c in ("team", "team_name", "club", "club_name") if c in team_stats.columns),
            None,
        )
        if team_col is None:
            return result

        # Dynamic pivoting: get all columns except the team identifier
        stat_cols = [c for c in team_stats.columns if c != team_col]
        if not stat_cols:
            return result

        # Build lookup: canonical-lower -> stat dict
        lookup: dict[str, dict[str, Any]] = {}
        for _, row in team_stats.iterrows():
            raw_name = str(row.get(team_col, "")).strip()
            if not raw_name:
                continue
            canonical_key = _rc(raw_name).lower()
            
            row_dict = {}
            for col in stat_cols:
                val = row[col]
                try:
                    if pd.notna(val):
                        row_dict[col] = float(val)
                except (ValueError, TypeError):
                    row_dict[col] = val
            lookup[canonical_key] = row_dict

        if not lookup:
            return result

        home_team_col = next(
            (c for c in ("home_team", "home_team_name") if c in result.columns), None
        )
        away_team_col = next(
            (c for c in ("away_team", "away_team_name") if c in result.columns), None
        )
        if home_team_col is None or away_team_col is None:
            return result

        result = result.copy()
        for stat_col in stat_cols:
            home_out = f"home_{stat_col}"
            away_out = f"away_{stat_col}"

            def _lookup(t: str, _sc=stat_col) -> Any:
                val = lookup.get(_rc(str(t).strip()).lower(), {}).get(_sc)
                return val if val is not None else float("nan")

            if home_out not in result.columns:
                result[home_out] = result[home_team_col].apply(_lookup)
            else:
                mask = result[home_out].isna()
                if mask.any():
                    result.loc[mask, home_out] = result.loc[mask, home_team_col].apply(_lookup)

            if away_out not in result.columns:
                result[away_out] = result[away_team_col].apply(_lookup)
            else:
                mask = result[away_out].isna()
                if mask.any():
                    result.loc[mask, away_out] = result.loc[mask, away_team_col].apply(_lookup)

        return result

    match_dfs: list[pd.DataFrame] = []
    team_stat_dfs: list[pd.DataFrame] = []

    import tempfile
    from .country_aliases import resolve_country as _rc
    debug_log_path = Path(tempfile.gettempdir()) / "soccersense" / "debug_upload.log"
    debug_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(debug_log_path, "a", encoding="utf-8") as log_f:
        log_f.write("\n--- Individual Files (independent processing) ---\n")

        for uf in upload_files:
            if not uf.filename:
                continue
            try:
                uf.file.seek(0)
                df = load_uploaded_file(uf)
                if df is None or df.empty:
                    continue
                df = _clean_cols(df)
                if _is_team_stats_file(df):
                    team_stat_dfs.append(df)
                    log_f.write(
                        f"File: {uf.filename} | Type: TEAM-STATS | Shape: {df.shape} | Columns: {list(df.columns)}\n"
                    )
                else:
                    match_dfs.append(df)
                    log_f.write(
                        f"File: {uf.filename} | Type: MATCH-LEVEL | Shape: {df.shape} | Columns: {list(df.columns)}\n"
                    )
            except Exception as e:
                log_f.write(f"Error reading file {uf.filename}: {e}\n")
                print(f"Error loading uploaded file {uf.filename}: {e}")
                raise ValueError(f"Failed to parse file {uf.filename}: {str(e)}")

    if not match_dfs and not team_stat_dfs:
        return None

    # ------------------------------------------------------------------ #
    # Step 1: Standardise column names and key columns of each dataframe #
    # ------------------------------------------------------------------ #
    processed_match_dfs = []
    for df in match_dfs:
        df = df.copy()
        df.columns = [clean_column_name(c) for c in df.columns]

        # Rename key columns to standard names if aliases are present
        rename_map = {}
        date_c = next((c for c in ("date", "date_gmt", "match_date", "kickoff") if c in df.columns), None)
        home_c = next((c for c in ("home_team", "home_team_name", "home_club_name") if c in df.columns), None)
        away_c = next((c for c in ("away_team", "away_team_name", "away_club_name") if c in df.columns), None)
        goals_h_c = next((c for c in ("home_goals", "home_team_goal_count", "home_score", "fthg") if c in df.columns), None)
        goals_a_c = next((c for c in ("away_goals", "away_team_goal_count", "away_score", "ftag") if c in df.columns), None)

        if date_c: rename_map[date_c] = "date"
        if home_c: rename_map[home_c] = "home_team"
        if away_c: rename_map[away_c] = "away_team"
        if goals_h_c: rename_map[goals_h_c] = "home_goals"
        if goals_a_c: rename_map[goals_a_c] = "away_goals"

        if rename_map:
            df = df.rename(columns=rename_map)

        if "date" in df.columns:
            try:
                # Standardise date to YYYY-MM-DD
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            except Exception:
                pass

        if "home_team" in df.columns and "away_team" in df.columns:
            df["home_team"] = df["home_team"].apply(lambda t: _rc(str(t).strip()))
            df["away_team"] = df["away_team"].apply(lambda t: _rc(str(t).strip()))

        processed_match_dfs.append(df)

    if not processed_match_dfs:
        # Only team-stats files uploaded — load existing/default matches to pivot onto
        project_root = Path(__file__).resolve().parents[2]
        matches_out = DATA_DIR / "sample_matches.csv"
        config_path = project_root / "pipeline_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    if "matches" in config and "output" in config["matches"]:
                        matches_out = DATA_DIR / Path(config["matches"]["output"]).name
            except Exception:
                pass
        
        base_path = matches_out if matches_out.exists() else (project_root / "data" / "sample_matches.csv")
        if base_path.exists():
            try:
                result = pd.read_csv(base_path, encoding="utf-8")
                # Clean columns so key checks and renames work
                result.columns = [clean_column_name(c) for c in result.columns]
            except Exception as e:
                print(f"Error loading base matches: {e}")
                return None
        else:
            return None
    else:
        if len(processed_match_dfs) == 1:
            result = processed_match_dfs[0].copy()
        else:
            result = pd.concat(processed_match_dfs, ignore_index=True)

    # ------------------------------------------------------------------ #
    # Step 1b: Merge duplicate matches to preserve detailed columns      #
    # ------------------------------------------------------------------ #
    if "date" in result.columns and "home_team" in result.columns and "away_team" in result.columns:
        # Sort so rows with the most filled columns (non-nulls) come first
        non_null_counts = result.notna().sum(axis=1)
        result_sorted = result.iloc[non_null_counts.sort_values(ascending=False).index]
        
        # groupby().first() ignores NaN/None by default, merging columns from duplicate rows!
        result = result_sorted.groupby(["date", "home_team", "away_team"], as_index=False, dropna=False).first()

    # ------------------------------------------------------------------ #
    # Step 2: Pivot team-stats files onto match rows (home_* / away_*)   #
    # ------------------------------------------------------------------ #
    for ts_df in team_stat_dfs:
        result = _pivot_team_stats_onto_matches(result, ts_df)

    return result


def load_uploaded_player_files(upload_files: list[UploadFile] | None) -> pd.DataFrame | None:
    """Read multiple uploaded player files and combine them row-wise."""
    if not upload_files or not isinstance(upload_files, list):
        return None
    import pandas as pd
    dfs = []
    for uf in upload_files:
        if not uf.filename:
            continue
        try:
            uf.file.seek(0)
            df = load_uploaded_file(uf)
            if df is not None and not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"Error loading player file {uf.filename}: {e}")
            raise ValueError(f"Failed to parse player file {uf.filename}: {str(e)}")
    if not dfs:
        return None
    if len(dfs) == 1:
        return dfs[0].copy()
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
        raw_players = load_uploaded_player_files(players_files)
        
        matches_df = None
        players_df = None
        
        # Write debug log of uploaded files and columns
        try:
            import tempfile
            debug_log_path = Path(tempfile.gettempdir()) / "soccersense" / "debug_upload.log"
            debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write("\n=== Upload Debug ===\n")
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
            # Column names are already cleaned by load_uploaded_files.
            # Re-applying clean_column_name is idempotent but we leave it
            # here as a safety net in case load_uploaded_files receives a
            # single-file path that bypassed the merge logic.
            raw_matches.columns = [clean_column_name(c) for c in raw_matches.columns]

            # Check if any required fields (or their aliases) are missing
            has_home = any(c in raw_matches.columns for c in ("home_team", "home_team_name", "home_club_name"))
            has_away = any(c in raw_matches.columns for c in ("away_team", "away_team_name", "away_club_name"))
            has_home_goals = any(c in raw_matches.columns for c in ("home_goals", "home_team_goal_count", "home_score", "fthg"))
            has_away_goals = any(c in raw_matches.columns for c in ("away_goals", "away_team_goal_count", "away_score", "ftag"))

            # Only warn and initialize to default if the field is completely missing (no alias exists either)
            if not has_home:
                raw_matches["home_team"] = "Unknown Home Team"
                warnings.append("Matches file: Critical column 'home_team' was missing and initialized to defaults.")
            if not has_away:
                raw_matches["away_team"] = "Unknown Away Team"
                warnings.append("Matches file: Critical column 'away_team' was missing and initialized to defaults.")
            if not has_home_goals:
                raw_matches["home_goals"] = 0
                warnings.append("Matches file: Critical column 'home_goals' was missing and initialized to defaults.")
            if not has_away_goals:
                raw_matches["away_goals"] = 0
                warnings.append("Matches file: Critical column 'away_goals' was missing and initialized to defaults.")

            # enrich_match_data() preserves ELO columns if present (or aliased) in the upload;
            # otherwise it computes them automatically from match history — this is normal
            # expected behaviour for standard match data formats and requires no user action.
            matches_df = enrich_match_data(raw_matches)

        missing_cols: list[str] = []
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
            is_tmp = tmp_dir in p.parents or "soccersense" in str(p)
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
        is_model_tmp = tmp_dir in MODEL_PATH.parents or "soccersense" in str(MODEL_PATH)
        if is_model_tmp:
            if MODEL_PATH.exists() and MODEL_PATH.is_file():
                try:
                    MODEL_PATH.unlink()
                except Exception as e:
                    import logging
                    logging.warning(f"Could not delete {MODEL_PATH}: {e}")
                    
            versioned_path = MODEL_PATH.parent / "soccersense_v1.pkl"
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
        
        # Save versioned copy using get_version_suffix()
        def get_version_suffix(version_str: str) -> str:
            version_str = str(version_str).strip().lower()
            for prefix in ("version_", "version", "v"):
                if version_str.startswith(prefix):
                    version_str = version_str[len(prefix):]
            if "." in version_str:
                version_str = version_str.split(".")[0]
            version_str = "".join(c for c in version_str if c.isdigit())
            if not version_str:
                version_str = "1"
            return f"v{version_str}"

        v_suffix = get_version_suffix(payload.version)
        versioned_path = MODEL_PATH.parent / f"soccersense_{v_suffix}.pkl"
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
        prod_model = project_root / "models" / "soccersense.pkl"
    if prod_model.exists():
        return FileResponse(path=prod_model, filename="soccersense.pkl", media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="Production model file not found.")


@app.get("/download/production_model_versioned")
def download_production_model_versioned():
    try:
        def get_version_suffix(version_str: str) -> str:
            version_str = str(version_str).strip().lower()
            for prefix in ("version_", "version", "v"):
                if version_str.startswith(prefix):
                    version_str = version_str[len(prefix):]
            if "." in version_str:
                version_str = version_str.split(".")[0]
            version_str = "".join(c for c in version_str if c.isdigit())
            if not version_str:
                version_str = "1"
            return f"v{version_str}"

        version = predictor.artifact.get("version", "1.0.0")
        v_suffix = get_version_suffix(version)
        project_root = Path(__file__).resolve().parents[2]
        prod_model_v = MODEL_PATH.parent / f"soccersense_{v_suffix}.pkl"
        if not prod_model_v.exists():
            prod_model_v = project_root / "models" / f"soccersense_{v_suffix}.pkl"
        if prod_model_v.exists():
            return FileResponse(path=prod_model_v, filename=f"soccersense_{v_suffix}.pkl", media_type="application/octet-stream")
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="Versioned production model file not found.")
