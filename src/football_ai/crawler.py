from __future__ import annotations

import io
import re
import logging
from pathlib import Path
import urllib.parse
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard headers to mimic a normal browser request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def fetch_html(url: str) -> str:
    """Fetch HTML content from a URL."""
    logger.info(f"Fetching URL: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        raise ValueError(f"Failed to fetch content from URL: {url}. Error: {e}")


def clean_column_name(col: str) -> str:
    """Standardize column names to snake_case."""
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9_\s]", "", col)
    col = re.sub(r"[\s_]+", "_", col)
    return col


def normalize_player_name(name: str) -> str:
    """Normalize names by stripping accents and converting to NFKD Unicode."""
    if not isinstance(name, str):
        if pd.isna(name):
            return ""
        name = str(name)
    import unicodedata
    nfkd_form = unicodedata.normalize('NFKD', name)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).strip()


def standardize_position(pos: str) -> str:
    pos_clean = str(pos).strip().upper()
    if pos_clean in ["GK", "GOALKEEPER"]:
        return "GK"
    if pos_clean in ["DF", "DEFENDER", "BACK", "DEFENCE"]:
        return "DF"
    if pos_clean in ["MF", "MIDFIELDER", "MIDFIELD"]:
        return "MF"
    if pos_clean in ["FW", "FORWARD", "STRIKER", "ATTACKER", "WINGER"]:
        return "FW"
    return pos_clean


def infer_position_from_stats(row: pd.Series | dict) -> str:
    """Infer player position based on stats and player name keywords."""
    if "position" in row and pd.notna(row["position"]) and str(row["position"]).strip():
        pos = standardize_position(row["position"])
        if pos in ["GK", "DF", "MF", "FW"]:
            return pos

    # Prioritize active outfield stats to avoid classifying outfield players with certain names as GK
    goals = float(row.get("goals", 0.0))
    shots = float(row.get("shots", 0.0))
    xg = float(row.get("xg", 0.0))
    xg_90 = float(row.get("xg_x90", row.get("xg_per_90", 0.0)))
    shots_90 = float(row.get("shots_x90", row.get("shots_per_90", 0.0)))
    goals_90 = float(row.get("goals_x90", row.get("goals_per_90", 0.0)))

    if goals > 0 or shots > 0 or xg > 0.01 or xg_90 > 0.01 or shots_90 > 0.1 or goals_90 > 0.01:
        if xg_90 > 0.25 or shots_90 > 1.5:
            return "FW"
        elif xg_90 > 0.08 or shots_90 > 0.6:
            return "MF"
        else:
            return "DF"

    # Heuristic based on name
    name_val = row.get("name", "")
    name_lower = normalize_player_name(str(name_val)).lower()
    gk_keywords = [
        "turner", "gunok", "cakir", "bayindir", "horvath", "johnson", "alisson", "ederson", 
        "pickford", "ter stegen", "neuer", "courtois", "oblak", "donnarumma", "sommer", 
        "e. martinez", "emiliano martinez", "e martinez", "szczesny", "mignolet", "gk", "goalkeeper"
    ]
    if any(kw in name_lower for kw in gk_keywords):
        return "GK"

    # Heuristic based on stats if not matched as GK
    if xg_90 > 0.25 or shots_90 > 1.5:
        return "FW"
    elif xg_90 > 0.08 or shots_90 > 0.6:
        return "MF"
    else:
        return "DF"


def validate_and_standardize_players(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Standardize a player DataFrame to ensure it conforms to the required 16-column structure.
    Also performs position inference and fills missing fields with defaults.
    
    Required columns to match:
    - name, nationality, apps, mins, goals, xg, Goals vs xG, shots, sot, conv %, xG per Shot, goals x90, xg x90, Goals vs xG x90, shots x90, sot x90
    
    Returns:
    - Standardized DataFrame
    - List of missing fields (original user-facing names)
    """
    import numpy as np
    
    # Required columns with original labels
    required_labels = {
        "name": "name",
        "nationality": "nationality",
        "apps": "apps",
        "mins": "mins",
        "goals": "goals",
        "xg": "xg",
        "goals_vs_xg": "Goals vs xG",
        "shots": "shots",
        "sot": "sot",
        "conv_pct": "conv %",
        "xg_per_shot": "xG per Shot",
        "goals_x90": "goals x90",
        "xg_x90": "xg x90",
        "goals_vs_xg_x90": "Goals vs xG x90",
        "shots_x90": "shots x90",
        "sot_x90": "sot x90",
        "saves_made": "saves_made",
        "goals_conceded": "goals_conceded",
        "save_perc": "save_perc",
        "goals_prevented": "goals_prevented",
        "xgot_conceded": "xgot_conceded"
    }

    # Potential column variants to check (case-insensitive, cleaned check)
    field_variants = {
        "name": ["name", "player", "player_name", "fullname"],
        "nationality": ["nationality", "team", "country", "national_team", "club"],
        "apps": ["apps", "appearances", "caps", "games", "played"],
        "mins": ["mins", "minutes", "min", "time_played"],
        "goals": ["goals", "gls", "goals_scored"],
        "xg": ["xg", "expected_goals"],
        "goals_vs_xg": ["goals_vs_xg", "goals_xg_diff", "goals_vs_xg_diff", "goals_xg_difference"],
        "shots": ["shots", "sh", "total_shots"],
        "sot": ["sot", "shots_on_target", "ontarget"],
        "conv_pct": ["conv_pct", "conv_percent", "conv", "conversion_rate", "conv_"],
        "xg_per_shot": ["xg_per_shot", "xg_shot", "expected_goals_per_shot"],
        "goals_x90": ["goals_x90", "goals_per_90", "gls_90", "goals_90", "goals_x90"],
        "xg_x90": ["xg_x90", "xg_per_90", "xg_90", "xg_x90"],
        "goals_vs_xg_x90": ["goals_vs_xg_x90", "goals_vs_xg_per_90", "goals_vs_xg_90"],
        "shots_x90": ["shots_x90", "shots_per_90", "shots_90"],
        "sot_x90": ["sot_x90", "sot_per_90", "sot_90"],
        "saves_made": ["saves_made", "saves", "saves_made_90"],
        "goals_conceded": ["goals_conceded", "conceded"],
        "save_perc": ["save_perc", "save_percent", "save_", "save_pct"],
        "goals_prevented": ["goals_prevented", "prevented"],
        "xgot_conceded": ["xgot_conceded", "xgot"]
    }

    cleaned_input_cols = {clean_column_name(c): c for c in df.columns}
    missing_fields = []
    standardized = pd.DataFrame()

    for internal_key, variants in field_variants.items():
        matched_col = None
        for var in variants:
            var_cleaned = clean_column_name(var)
            if var_cleaned in cleaned_input_cols:
                matched_col = cleaned_input_cols[var_cleaned]
                break

        # Fallback: look for substring matches
        if matched_col is None:
            for col_clean, col_orig in cleaned_input_cols.items():
                # Avoid cross-matching key absolute metrics with per-90 rates or other suffixes
                if internal_key in ["goals", "xg", "shots", "sot", "goals_vs_xg"]:
                    if any(x in col_clean for x in ["90", "per", "vs", "shot"]) and not (internal_key == "goals_vs_xg" and "vs" in col_clean and "90" not in col_clean):
                        continue
                if internal_key in ["goals_x90", "xg_x90", "shots_x90", "sot_x90", "goals_vs_xg_x90"]:
                    if "90" not in col_clean and "per" not in col_clean:
                        continue
                if any(v in col_clean for v in variants):
                    matched_col = col_orig
                    break

        if matched_col is not None:
            if internal_key in ["name", "nationality"]:
                standardized[internal_key] = df[matched_col].astype(str).str.strip()
            else:
                # Numeric fields: replace percent signs, commas, etc.
                s = df[matched_col].astype(str).str.replace("%", "").str.replace(",", "")
                standardized[internal_key] = pd.to_numeric(s, errors="coerce").fillna(0.0)
        else:
            if internal_key not in ["saves_made", "goals_conceded", "save_perc", "goals_prevented", "xgot_conceded"]:
                missing_fields.append(required_labels[internal_key])
            if internal_key in ["name", "nationality"]:
                standardized[internal_key] = "Unknown"
            else:
                standardized[internal_key] = 0.0

    # Ensure clean team names
    standardized["team"] = standardized["nationality"]

    # Infer position if not explicitly present
    position_col = next((c for c in df.columns if clean_column_name(c) in ["pos", "position"]), None)
    if position_col is not None:
        standardized["position"] = df[position_col].apply(standardize_position)
    else:
        # Infer using our heuristic
        inferred = []
        for _, row in standardized.iterrows():
            inferred.append(infer_position_from_stats(row))
        standardized["position"] = [standardize_position(pos) for pos in inferred]

    # Add backward compatibility fields
    standardized["xg_per_90"] = standardized["xg_x90"]
    standardized["goals_per_90"] = standardized["goals_x90"]
    standardized["xa_per_90"] = df[cleaned_input_cols["xa_per_90"]].astype(float) if "xa_per_90" in cleaned_input_cols else 0.05
    standardized["assists_per_90"] = df[cleaned_input_cols["assists_per_90"]].astype(float) if "assists_per_90" in cleaned_input_cols else 0.05
    
    start_prob_col = next((c for c in df.columns if clean_column_name(c) in ["start_probability", "start_prob", "start"]), None)
    if start_prob_col is not None:
        standardized["start_probability"] = pd.to_numeric(df[start_prob_col], errors="coerce").fillna(0.8)
    else:
        standardized["start_probability"] = [1.0 if pos == "GK" else 0.8 for pos in standardized["position"]]

    status_col = next((c for c in df.columns if clean_column_name(c) == "status"), None)
    if status_col is not None:
        standardized["status"] = df[status_col].astype(str).str.strip()
    else:
        standardized["status"] = "active"

    # Rearrange columns logically
    cols = [
        "team", "name", "nationality", "position", "apps", "mins", "goals", "xg", "goals_vs_xg", 
        "shots", "sot", "conv_pct", "xg_per_shot", "goals_x90", "xg_x90", "goals_vs_xg_x90", 
        "shots_x90", "sot_x90", "xg_per_90", "goals_per_90", "xa_per_90", "assists_per_90", 
        "start_probability", "status",
        "saves_made", "goals_conceded", "save_perc", "goals_prevented", "xgot_conceded"
    ]
    
    return standardized[cols], missing_fields


def parse_score(score_str: str) -> tuple[int, int] | None:
    """Parse score strings like '2-1', '1 - 0', '3 : 0' into home and away goals."""
    if not isinstance(score_str, str):
        return None
    
    # Remove citations, footnotes, extra details like (a.e.t.), (p), etc.
    score_clean = re.sub(r"\(.*?\)", "", score_str)
    score_clean = re.sub(r"\[.*?\]", "", score_clean)
    score_clean = score_clean.strip()
    
    # Match pattern of two numbers separated by a dash, en-dash, em-dash, colon, or "to"
    match = re.search(r"(\d+)\s*[-–—:to]\s*(\d+)", score_clean)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def detect_and_parse_matches(html_text: str) -> pd.DataFrame:
    """Scrape and parse a matches table from HTML text."""
    soup = BeautifulSoup(html_text, "html.parser")
    tables = soup.find_all("table")
    
    best_table = None
    best_score = 0
    parsed_df = None
    
    for i, table in enumerate(tables):
        try:
            # Wrap table HTML in StringIO to avoid pandas warning/error
            df = pd.read_html(io.StringIO(str(table)))[0]
        except Exception:
            continue
            
        if df.empty or len(df) < 2:
            continue
            
        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[-1] if col[-1] else f"col_{j}" for j, col in enumerate(df.columns.values)]
            
        cols = [clean_column_name(c) for c in df.columns]
        
        # Scrape match indicators
        home_hits = sum(1 for c in cols if any(k in c for k in ["home", "host", "team_1", "team1"]))
        away_hits = sum(1 for c in cols if any(k in c for k in ["away", "guest", "team_2", "team2"]))
        score_hits = sum(1 for c in cols if any(k in c for k in ["score", "result", "goals", "ft"]))
        date_hits = sum(1 for c in cols if any(k in c for k in ["date", "day", "time"]))
        
        score = home_hits + away_hits + score_hits + date_hits
        
        if score > best_score:
            best_score = score
            best_table = table
            parsed_df = df
            
    if parsed_df is None:
        raise ValueError("Could not find a valid match table on the page.")
        
    df = parsed_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[-1] if col[-1] else f"col_{j}" for j, col in enumerate(df.columns.values)]
        
    df.columns = [clean_column_name(c) for c in df.columns]
    
    # Map fields
    mapped_df = pd.DataFrame()
    
    # Find home team
    home_col = next((c for c in df.columns if any(k in c for k in ["home_team", "team_1", "team1", "home", "host"])), None)
    # Find away team
    away_col = next((c for c in df.columns if any(k in c for k in ["away_team", "team_2", "team2", "away", "visitor", "guest"])), None)
    
    if not home_col or not away_col:
        # Fallback: look for columns that might just be called "team" or similar, or take first two string columns
        str_cols = [c for c in df.columns if df[c].dtype == object]
        if len(str_cols) >= 2:
            home_col, away_col = str_cols[0], str_cols[1]
        else:
            raise ValueError(f"Could not map home/away team columns. Table columns: {df.columns.tolist()}")
            
    mapped_df["home_team"] = df[home_col].astype(str).str.strip()
    mapped_df["away_team"] = df[away_col].astype(str).str.strip()
    
    # Parse score/goals
    home_goals_col = next((c for c in df.columns if c in ["home_goals", "goals_home", "goals1", "home_score"]), None)
    away_goals_col = next((c for c in df.columns if c in ["away_goals", "goals_away", "goals2", "away_score"]), None)
    
    if home_goals_col and away_goals_col:
        mapped_df["home_goals"] = pd.to_numeric(df[home_goals_col], errors="coerce").fillna(0).astype(int)
        mapped_df["away_goals"] = pd.to_numeric(df[away_goals_col], errors="coerce").fillna(0).astype(int)
    else:
        # Look for unified score column
        score_col = next((c for c in df.columns if any(k in c for k in ["score", "result", "ft", "r"])), None)
        if score_col:
            home_goals = []
            away_goals = []
            for val in df[score_col]:
                parsed = parse_score(str(val))
                if parsed:
                    home_goals.append(parsed[0])
                    away_goals.append(parsed[1])
                else:
                    home_goals.append(0)
                    away_goals.append(0)
            mapped_df["home_goals"] = home_goals
            mapped_df["away_goals"] = away_goals
        else:
            # Fallback to zero goals
            mapped_df["home_goals"] = 0
            mapped_df["away_goals"] = 0
            
    # Find Date
    date_col = next((c for c in df.columns if any(k in c for k in ["date", "match_date", "day", "time"])), None)
    if date_col:
        # Strip reference links from dates (e.g. "1 June 2024[23]")
        clean_dates = df[date_col].astype(str).apply(lambda x: re.sub(r"\[.*?\]", "", x).strip())
        mapped_df["date"] = pd.to_datetime(clean_dates, errors="coerce").dt.strftime("%Y-%m-%d")
        mapped_df["date"] = mapped_df["date"].fillna("2024-06-01")
    else:
        mapped_df["date"] = "2024-06-01"
        
    # Optional competition
    comp_col = next((c for c in df.columns if any(k in c for k in ["competition", "tournament", "league"])), None)
    if comp_col:
        mapped_df["competition"] = df[comp_col].astype(str).str.strip()
    else:
        mapped_df["competition"] = "Scraped Match"
        
    # Drop rows where home_team or away_team is empty/header residue
    mapped_df = mapped_df[
        (mapped_df["home_team"].str.len() > 1) & 
        (mapped_df["away_team"].str.len() > 1) &
        (~mapped_df["home_team"].str.contains("team", case=False))
    ].reset_index(drop=True)
    
    return mapped_df


def detect_and_parse_players(html_text: str) -> pd.DataFrame:
    """Scrape and parse squad/players tables from HTML text. Works with multiple tables per page."""
    soup = BeautifulSoup(html_text, "html.parser")
    
    # In Wikipedia, team squad lists are usually tables with columns like 'No.', 'Pos.', 'Player', 'Goals'
    # And the team name is the preceding h3 or h4 heading
    all_players = []
    
    current_team = "Unknown Team"
    
    # Walk the DOM to associate headings with tables
    for elem in soup.descendants:
        if elem.name in ["h2", "h3", "h4"]:
            heading_text = elem.get_text().strip()
            # Clean wikipedia [edit] links
            heading_text = re.sub(r"\[edit\]", "", heading_text).strip()
            # If it looks like a country or team, save it
            if len(heading_text) > 1 and not any(k in heading_text.lower() for k in ["squad", "staff", "schedule", "group", "referee", "reference"]):
                current_team = heading_text
                
        elif elem.name == "table":
            try:
                df = pd.read_html(io.StringIO(str(elem)))[0]
            except Exception:
                continue
                
            if df.empty or len(df) < 2:
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[-1] if col[-1] else f"col_{j}" for j, col in enumerate(df.columns.values)]
                
            cols = [clean_column_name(c) for c in df.columns]
            
            # Check if this table is a players list
            player_hits = sum(1 for c in cols if any(k in c for k in ["player", "name", "member"]))
            pos_hits = sum(1 for c in cols if any(k in c for k in ["pos", "position", "role"]))
            
            if player_hits > 0 or (pos_hits > 0 and len(df.columns) >= 4):
                # Clean and parse this players table
                df.columns = [clean_column_name(c) for c in df.columns]
                
                # Find columns
                player_col = next((c for c in df.columns if any(k in c for k in ["player", "name", "fullname"])), None)
                pos_col = next((c for c in df.columns if any(k in c for k in ["pos", "position"])), None)
                team_col = next((c for c in df.columns if any(k in c for k in ["team", "country", "national_team", "club"])), None)
                goals_col = next((c for c in df.columns if c in ["goals", "gls", "goals_scored"]), None)
                caps_col = next((c for c in df.columns if c in ["caps", "apps", "caps_apps"]), None)
                
                if not player_col:
                    continue
                    
                for idx, row in df.iterrows():
                    name = str(row[player_col]).strip()
                    # Remove Wikipedia marks like (captain) or page link anchors
                    name = re.sub(r"\(c\)", "", name, flags=re.IGNORECASE)
                    name = re.sub(r"\[.*?\]", "", name)
                    name = name.strip()
                    
                    if len(name) <= 2 or name.lower() in ["player", "name", "total"]:
                        continue
                        
                    raw_pos = str(row[pos_col]).strip() if pos_col else "FW"
                    # Map position
                    pos = "FW"
                    raw_pos_lower = raw_pos.lower()
                    if "gk" in raw_pos_lower or "goalkeeper" in raw_pos_lower:
                        pos = "GK"
                    elif "df" in raw_pos_lower or "defender" in raw_pos_lower or "back" in raw_pos_lower:
                        pos = "DF"
                    elif "mf" in raw_pos_lower or "midfielder" in raw_pos_lower:
                        pos = "MF"
                        
                    team = current_team
                    if team_col and pd.notna(row[team_col]):
                        team = str(row[team_col]).strip()
                        
                    # Standardize stats
                    goals = pd.to_numeric(row[goals_col], errors="coerce") if goals_col else 0
                    caps = pd.to_numeric(row[caps_col], errors="coerce") if caps_col else 0
                    
                    if pd.isna(goals): goals = 0
                    if pd.isna(caps): caps = 0
                    
                    # Estimate per-90 stats from caps/goals
                    goals_per_90 = 0.0
                    xg_per_90 = 0.0
                    xa_per_90 = 0.05
                    assists_per_90 = 0.05
                    
                    if caps > 0:
                        goals_per_90 = min(1.0, float(goals) / float(caps))
                        # xG is usually slightly higher than goals for good attackers, or proportional
                        xg_per_90 = goals_per_90 * 1.2
                    
                    # Apply position defaults if no caps/goals available
                    if goals_per_90 == 0:
                        if pos == "FW":
                            xg_per_90, goals_per_90 = 0.4, 0.3
                            xa_per_90, assists_per_90 = 0.2, 0.1
                        elif pos == "MF":
                            xg_per_90, goals_per_90 = 0.15, 0.1
                            xa_per_90, assists_per_90 = 0.25, 0.15
                        elif pos == "DF":
                            xg_per_90, goals_per_90 = 0.05, 0.02
                            xa_per_90, assists_per_90 = 0.05, 0.03
                        else:
                            xg_per_90, goals_per_90 = 0.0, 0.0
                            xa_per_90, assists_per_90 = 0.0, 0.0
                            
                    start_probability = 1.0 if pos == "GK" else 0.8
                    
                    all_players.append({
                        "team": team,
                        "name": name,
                        "position": pos,
                        "xg_per_90": xg_per_90,
                        "goals_per_90": goals_per_90,
                        "xa_per_90": xa_per_90,
                        "assists_per_90": assists_per_90,
                        "start_probability": start_probability,
                        "status": "active",
                        "saves_made": 0.0,
                        "goals_conceded": 0.0,
                        "save_perc": 0.0,
                        "goals_prevented": 0.0,
                        "xgot_conceded": 0.0
                    })
                    
    if not all_players:
        raise ValueError("Could not find any squad tables or parse player data from URL.")
        
    df_players = pd.DataFrame(all_players)
    standardized_df, _ = validate_and_standardize_players(df_players)
    return standardized_df


def enrich_match_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich raw match records with ELO, Elo Ranks, and first goal team.

    Handles multiple common column naming conventions from different data
    sources (e.g. home_team_name / home_team_goal_count / elo_rating).
    If ELO columns are already present they are preserved as-is; otherwise
    they are computed from the match results using an Elo K=32 algorithm.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # Step 0: Column alias resolution
    # Map alternative/source-specific column names to canonical names so
    # that data from different providers is handled transparently.
    # ------------------------------------------------------------------
    ALIAS_MAP = {
        # Team names
        "home_team_name": "home_team",
        "away_team_name": "away_team",
        "home_club_name": "home_team",
        "away_club_name": "away_team",
        # Goal counts
        "home_team_goal_count": "home_goals",
        "away_team_goal_count": "away_goals",
        "home_score": "home_goals",
        "away_score": "away_goals",
        "fthg": "home_goals",
        "ftag": "away_goals",
        # Date aliases
        "date_gmt": "date",
        "match_date": "date",
        "kickoff": "date",
        # Tournament / competition
        "league": "competition",
        "tournament": "competition",
        "div": "competition",
        # ELO rating aliases — common variants from different data sources
        "home_elo_rating": "home_elo",
        "home_elo_score": "home_elo",
        "home_rating": "home_elo",
        "homeelo": "home_elo",
        "elo_home": "home_elo",
        "h_elo": "home_elo",
        "home_team_elo": "home_elo",
        "away_elo_rating": "away_elo",
        "away_elo_score": "away_elo",
        "away_rating": "away_elo",
        "awayelo": "away_elo",
        "elo_away": "away_elo",
        "a_elo": "away_elo",
        "away_team_elo": "away_elo",
        # ELO rank aliases
        "home_elo_rank": "home_elo_rank",   # already canonical, but keep for completeness
        "home_rank": "home_elo_rank",
        "home_fifa_rank": "home_elo_rank",
        "home_team_rank": "home_elo_rank",
        "h_rank": "home_elo_rank",
        "home_ranking": "home_elo_rank",
        "away_rank": "away_elo_rank",
        "away_fifa_rank": "away_elo_rank",
        "away_team_rank": "away_elo_rank",
        "a_rank": "away_elo_rank",
        "away_ranking": "away_elo_rank",
    }
    rename = {}
    for alias, canonical in ALIAS_MAP.items():
        if alias in df.columns and canonical not in df.columns:
            rename[alias] = canonical
    if rename:
        df = df.rename(columns=rename)

    # ------------------------------------------------------------------
    # Step 0b: ELO / rank alias resolution (per-team columns)
    # Some datasets store a single elo_rating / fifa_rank column in a
    # team-stats CSV that gets merged in.  Detect and split them into the
    # home_ / away_ variants if needed.
    #
    # Pattern: if the merged frame has a 'team' column alongside
    # 'elo_rating' / 'fifa_rank', those rows represent per-team stats.
    # We join them onto the match rows by home_team / away_team.
    # ------------------------------------------------------------------
    elo_cols_canonical = {"home_elo", "away_elo", "home_elo_rank", "away_elo_rank"}
    missing_elo = elo_cols_canonical - set(df.columns)

    if missing_elo and "team" in df.columns:
        # Build a team->rating lookup from the elo_rating / fifa_rank columns
        elo_lookup = {}
        rank_lookup = {}
        for _, row in df[["team", "elo_rating" if "elo_rating" in df.columns else "team",
                            "fifa_rank" if "fifa_rank" in df.columns else "team"]].drop_duplicates().iterrows():
            t = row.get("team")
            if pd.notna(t) and str(t).strip():
                if "elo_rating" in df.columns:
                    val = pd.to_numeric(row.get("elo_rating"), errors="coerce")
                    if pd.notna(val):
                        elo_lookup[str(t).strip()] = val
                if "fifa_rank" in df.columns:
                    val = pd.to_numeric(row.get("fifa_rank"), errors="coerce")
                    if pd.notna(val):
                        rank_lookup[str(t).strip()] = val

        if elo_lookup and "home_elo" not in df.columns and "away_elo" not in df.columns:
            df["home_elo"] = df["home_team"].apply(lambda t: elo_lookup.get(str(t).strip(), float("nan")))
            df["away_elo"] = df["away_team"].apply(lambda t: elo_lookup.get(str(t).strip(), float("nan")))
        if rank_lookup and "home_elo_rank" not in df.columns and "away_elo_rank" not in df.columns:
            df["home_elo_rank"] = df["home_team"].apply(lambda t: rank_lookup.get(str(t).strip(), float("nan")))
            df["away_elo_rank"] = df["away_team"].apply(lambda t: rank_lookup.get(str(t).strip(), float("nan")))

    # Drop rows that are missing both home_team and away_team (came from
    # the team-stats-only rows in a merged frame)
    if "home_team" in df.columns and "away_team" in df.columns:
        df = df[df["home_team"].notna() & df["away_team"].notna()]
        df = df[df["home_team"].astype(str).str.strip() != ""]
        df = df[df["away_team"].astype(str).str.strip() != ""]
        df = df.reset_index(drop=True)

    if 'date' not in df.columns or df['date'].empty:
        df['date'] = pd.date_range(start='2024-01-01', periods=len(df)).strftime('%Y-%m-%d')
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').fillna(pd.Timestamp('2024-01-01'))
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
    df = df.sort_values('date').reset_index(drop=True)
    
    df['home_goals'] = pd.to_numeric(df['home_goals'], errors='coerce').fillna(0).astype(int)
    df['away_goals'] = pd.to_numeric(df['away_goals'], errors='coerce').fillna(0).astype(int)
    
    # Calculate first_goal_team
    if 'first_goal_team' not in df.columns:
        first_goal_team = []
        for h, a, ht, at in zip(df['home_goals'], df['away_goals'], df['home_team'], df['away_team']):
            if h > 0 and a == 0:
                first_goal_team.append(ht)
            elif a > 0 and h == 0:
                first_goal_team.append(at)
            elif h > 0 and a > 0:
                first_goal_team.append(ht)  # Default
            else:
                first_goal_team.append("")
        df['first_goal_team'] = first_goal_team

    # ------------------------------------------------------------------
    # ELO ratings: only compute if not already supplied by the caller.
    # This preserves real-world ELO values from user-uploaded CSV files
    # and prevents goal predictions from being flattened toward the mean.
    #
    # Three cases:
    #   1. All ELO columns present and fully populated → preserve as-is.
    #   2. ELO columns absent entirely → compute from match history.
    #   3. ELO columns present but partially NaN (mixed file upload) →
    #      preserve supplied values; fill missing rows via K=32 algorithm.
    # ------------------------------------------------------------------
    elo_cols_exist = all(col in df.columns for col in ('home_elo', 'away_elo', 'home_elo_rank', 'away_elo_rank'))

    if elo_cols_exist:
        # Coerce to numeric in case they came in as strings
        for col in ('home_elo', 'away_elo', 'home_elo_rank', 'away_elo_rank'):
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Check if any rows are missing ELO data (e.g. mixed-format upload)
        elo_missing_mask = df['home_elo'].isna() | df['away_elo'].isna()

        if elo_missing_mask.any():
            # Compute ELO from match history for ALL rows using K=32,
            # then backfill only the rows that were missing values.
            K = 32
            elo = {}
            computed_home = []
            computed_away = []

            for _, row in df.iterrows():
                h, a = row['home_team'], row['away_team']
                if h not in elo: elo[h] = 1500.0
                if a not in elo: elo[a] = 1500.0

                computed_home.append(elo[h])
                computed_away.append(elo[a])

                r_home, r_away = elo[h], elo[a]
                e_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
                e_away = 1 / (1 + 10 ** ((r_home - r_away) / 400))

                hg, ag = row['home_goals'], row['away_goals']
                if hg > ag:   s_home, s_away = 1.0, 0.0
                elif hg < ag: s_home, s_away = 0.0, 1.0
                else:         s_home, s_away = 0.5, 0.5

                elo[h] += K * (s_home - e_home)
                elo[a] += K * (s_away - e_away)

            computed_home_s = pd.Series(computed_home, index=df.index)
            computed_away_s = pd.Series(computed_away, index=df.index)

            # Fill NaN cells with computed values; keep supplied values intact
            df['home_elo'] = df['home_elo'].fillna(computed_home_s)
            df['away_elo'] = df['away_elo'].fillna(computed_away_s)

            # Compute ranks from final ELO state for missing rank rows
            latest_elos = sorted(elo.items(), key=lambda x: x[1], reverse=True)
            ranks = {team: rank + 1 for rank, (team, _) in enumerate(latest_elos)}
            rank_missing_mask = df['home_elo_rank'].isna() | df['away_elo_rank'].isna()
            df.loc[rank_missing_mask, 'home_elo_rank'] = df.loc[rank_missing_mask, 'home_team'].map(ranks).fillna(50)
            df.loc[rank_missing_mask, 'away_elo_rank'] = df.loc[rank_missing_mask, 'away_team'].map(ranks).fillna(50)
        else:
            # All ELO values are present — nothing to compute
            pass

        # Final type cleanup
        for col in ('home_elo', 'away_elo'):
            df[col] = df[col].fillna(1500.0)
        for col in ('home_elo_rank', 'away_elo_rank'):
            df[col] = df[col].fillna(50)

    else:
        # Dynamically compute ELO ratings from match results
        K = 32
        elo = {}
        home_elos = []
        away_elos = []
        
        for idx, row in df.iterrows():
            h, a = row['home_team'], row['away_team']
            if h not in elo: elo[h] = 1500.0
            if a not in elo: elo[a] = 1500.0
            
            home_elos.append(elo[h])
            away_elos.append(elo[a])
            
            r_home = elo[h]
            r_away = elo[a]
            e_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
            e_away = 1 / (1 + 10 ** ((r_home - r_away) / 400))
            
            hg, ag = row['home_goals'], row['away_goals']
            if hg > ag:
                s_home, s_away = 1.0, 0.0
            elif hg < ag:
                s_home, s_away = 0.0, 1.0
            else:
                s_home, s_away = 0.5, 0.5
                
            elo[h] += K * (s_home - e_home)
            elo[a] += K * (s_away - e_away)
            
        df['home_elo'] = home_elos
        df['away_elo'] = away_elos

        # Elo ranks
        latest_elos = sorted(elo.items(), key=lambda x: x[1], reverse=True)
        ranks = {team: rank + 1 for rank, (team, _) in enumerate(latest_elos)}
        df['home_elo_rank'] = [ranks.get(t, 50) for t in df['home_team']]
        df['away_elo_rank'] = [ranks.get(t, 50) for t in df['away_team']]

    if 'competition' not in df.columns:
        df['competition'] = 'Scraped Match'

    # Build output column list — preserve any extra columns the user included
    required_cols = [
        'date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'first_goal_team',
        'home_elo', 'away_elo', 'home_elo_rank', 'away_elo_rank', 'competition'
    ]
    extra_cols = [c for c in df.columns if c not in required_cols]
    return df[required_cols + extra_cols]



def crawl_and_process(
    matches_url: str | None = None,
    players_url: str | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Scrapes, parses, and enriches data from matches and/or players URLs."""
    matches_df = None
    players_df = None
    
    if matches_url:
        logger.info(f"Crawling matches from: {matches_url}")
        html = fetch_html(matches_url)
        raw_matches = detect_and_parse_matches(html)
        logger.info(f"Successfully scraped {len(raw_matches)} raw match rows. Enriching matches...")
        matches_df = enrich_match_data(raw_matches)
        logger.info(f"Successfully enriched matches dataframe.")
        
    if players_url:
        logger.info(f"Crawling players from: {players_url}")
        html = fetch_html(players_url)
        players_df = detect_and_parse_players(html)
        logger.info(f"Successfully scraped {len(players_df)} players rows.")
        
    return matches_df, players_df
