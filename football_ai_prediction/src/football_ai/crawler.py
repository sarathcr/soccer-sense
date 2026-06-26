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
                        "status": "active"
                    })
                    
    if not all_players:
        raise ValueError("Could not find any squad tables or parse player data from URL.")
        
    return pd.DataFrame(all_players)


def enrich_match_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich scraped raw match records with ELO, Elo Ranks, and first goal team."""
    df = df.copy()
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

    # Calculate Elo ratings
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
        
    cols = [
        'date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'first_goal_team',
        'home_elo', 'away_elo', 'home_elo_rank', 'away_elo_rank', 'competition'
    ]
    return df[cols]


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
