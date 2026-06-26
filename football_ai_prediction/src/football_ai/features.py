from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


RESULT_LABELS = ["away_win", "draw", "home_win"]
FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "home_elo_rank",
    "away_elo_rank",
    "elo_diff",
    "rank_diff",
]


@dataclass(frozen=True)
class TeamProfile:
    team: str
    elo: float
    elo_rank: float


def normalize_team_name(team) -> str:
    if not isinstance(team, str):
        import pandas as pd
        if pd.isna(team):
            return ""
        team = str(team)
    return " ".join(team.strip().split()).title()


def result_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def build_training_frame(matches: pd.DataFrame) -> pd.DataFrame:
    frame = matches.copy()
    frame["result"] = [
        result_label(home, away)
        for home, away in zip(frame["home_goals"], frame["away_goals"], strict=True)
    ]
    frame["total_goals"] = frame["home_goals"] + frame["away_goals"]
    frame["both_teams_to_score"] = (
        (frame["home_goals"] > 0) & (frame["away_goals"] > 0)
    ).astype(int)
    frame["home_clean_sheet"] = (frame["away_goals"] == 0).astype(int)
    frame["away_clean_sheet"] = (frame["home_goals"] == 0).astype(int)
    
    # Handle missing first_goal_team column
    if "first_goal_team" not in frame.columns:
        first_goal_team = []
        for h, a, ht, at in zip(frame["home_goals"], frame["away_goals"], frame["home_team"], frame["away_team"]):
            if h > 0:
                first_goal_team.append(ht)
            elif a > 0:
                first_goal_team.append(at)
            else:
                first_goal_team.append("")
        frame["first_goal_team"] = first_goal_team

    frame["first_goal_home"] = (frame["first_goal_team"] == frame["home_team"]).astype(int)
    return add_difference_features(frame)


def add_difference_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["elo_diff"] = frame["home_elo"] - frame["away_elo"]
    frame["rank_diff"] = frame["away_elo_rank"] - frame["home_elo_rank"]
    return frame


def build_team_profiles(matches: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rows = []
    for side in ("home", "away"):
        rows.append(
            matches[
                [
                    "date",
                    f"{side}_team",
                    f"{side}_elo",
                    f"{side}_elo_rank",
                ]
            ].rename(
                columns={
                    f"{side}_team": "team",
                    f"{side}_elo": "elo",
                    f"{side}_elo_rank": "elo_rank",
                }
            )
        )

    all_teams = pd.concat(rows, ignore_index=True)
    all_teams = all_teams.sort_values("date")
    profiles: dict[str, dict[str, Any]] = {}
    for team, team_rows in all_teams.groupby("team", sort=False):
        latest = team_rows.tail(3).mean(numeric_only=True)
        normalized = normalize_team_name(team)
        profiles[normalized] = {
            "team": normalized,
            "elo": float(latest["elo"]),
            "elo_rank": float(latest["elo_rank"]),
        }
    return profiles


def get_val(profile: Any, attr: str, default: float = 0.0) -> Any:
    if isinstance(profile, dict):
        return profile.get(attr, default)
    return getattr(profile, attr, default)


def build_inference_features(
    home_team: str,
    away_team: str,
    team_profiles: dict[str, Any],
) -> pd.DataFrame:
    home_name = normalize_team_name(home_team)
    away_name = normalize_team_name(away_team)
    
    # Graceful fallback for home team
    if home_name not in team_profiles:
        import logging
        logging.warning(f"Unknown home_team: {home_team}. Using fallback profile.")
        if team_profiles:
            avg_elo = sum(get_val(p, "elo") for p in team_profiles.values()) / len(team_profiles)
            avg_rank = sum(get_val(p, "elo_rank") for p in team_profiles.values()) / len(team_profiles)
        else:
            avg_elo, avg_rank = 1500.0, 50.0
        
        home = {
            "team": home_name,
            "elo": avg_elo,
            "elo_rank": avg_rank,
        }
    else:
        home = team_profiles[home_name]

    # Graceful fallback for away team
    if away_name not in team_profiles:
        import logging
        logging.warning(f"Unknown away_team: {away_team}. Using fallback profile.")
        if team_profiles:
            avg_elo = sum(get_val(p, "elo") for p in team_profiles.values()) / len(team_profiles)
            avg_rank = sum(get_val(p, "elo_rank") for p in team_profiles.values()) / len(team_profiles)
        else:
            avg_elo, avg_rank = 1400.0, 50.0
        
        away = {
            "team": away_name,
            "elo": avg_elo,
            "elo_rank": avg_rank,
        }
    else:
        away = team_profiles[away_name]

    frame = pd.DataFrame(
        [
            {
                "home_elo": get_val(home, "elo"),
                "away_elo": get_val(away, "elo"),
                "home_elo_rank": get_val(home, "elo_rank"),
                "away_elo_rank": get_val(away, "elo_rank"),
            }
        ]
    )
    return add_difference_features(frame)[FEATURE_COLUMNS]
