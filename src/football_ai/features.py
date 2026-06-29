from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .country_aliases import resolve_country


RESULT_LABELS = ["away_win", "draw", "home_win"]

# ---------------------------------------------------------------------------
# Feature column registry
# ---------------------------------------------------------------------------
# Core ELO + goalkeeper features (always present — computed from match results
# and player data even when no extra files are uploaded).
_CORE_FEATURES = [
    "home_elo",
    "away_elo",
    "home_elo_rank",
    "away_elo_rank",
    "elo_diff",
    "rank_diff",
    "home_gk_save_ratio",
    "away_gk_save_ratio",
    "home_gk_prevented_per_90",
    "away_gk_prevented_per_90",
]

# Extended performance & form features.
# These are computed from (a) uploaded team-stats columns (File 3 after pivot),
# (b) uploaded pre-match match-stats columns (File 2), or
# (c) rolling statistics derived from match results in the matches DataFrame.
# The same stats are stored in team_profiles so inference works without extra files.
_EXTENDED_FEATURES = [
    # Attack / defense ratings (0-100, higher = better)
    "home_attack_rating",
    "away_attack_rating",
    "attack_diff",            # home - away
    "home_defense_rating",
    "away_defense_rating",
    "defense_diff",           # home - away
    # Goal-scoring tendencies
    "home_goals_avg",
    "away_goals_avg",
    "goals_avg_diff",
    "home_conceded_avg",
    "away_conceded_avg",
    "conceded_avg_diff",
    # Win / clean-sheet / first-goal rates  (0.0 – 1.0)
    "home_win_rate",
    "away_win_rate",
    "win_rate_diff",
    "home_clean_sheet_rate",
    "away_clean_sheet_rate",
    # Expected goals
    "home_xg_avg",
    "away_xg_avg",
    "xg_avg_diff",
    # Rolling form (points from last 5 matches, 0–15)
    "home_form_pts",
    "away_form_pts",
    "form_diff",
    # First-goal tendency
    "home_first_goal_rate",
    "away_first_goal_rate",
]

FEATURE_COLUMNS: list[str] = _CORE_FEATURES + _EXTENDED_FEATURES

# ---------------------------------------------------------------------------
# Sensible global-average defaults (used when data is absent)
# ---------------------------------------------------------------------------
_STAT_DEFAULTS: dict[str, float] = {
    "attack_rating":    50.0,   # Neutral mid-range (0-100 scale)
    "defense_rating":   50.0,
    "goals_avg":         1.3,   # Approx. global goals-per-game average
    "conceded_avg":      1.3,
    "win_rate":          0.33,  # One outcome in three
    "clean_sheet_rate":  0.25,
    "xg_avg":            1.2,
    "xga_avg":           1.2,
    "form_pts":          5.0,   # ~1.0 pts/game × 5 games
    "first_goal_rate":   0.50,
}

# Names of team-level stat columns that may arrive in the matches DataFrame
# after the home_/away_ pivot done in api.py (from the uploaded team-stats CSV).
_TEAM_STAT_UPLOAD_COLS: list[str] = [
    "attack_rating",
    "defense_rating",
    "goals_avg",
    "conceded_avg",
    "xg_avg",
    "xga_avg",
    "win_rate_last10",   # → mapped to "win_rate" in profile
    "form_points_last5", # → mapped to "form_pts"
    "form_points_last10",
    "clean_sheet_rate",
    "first_goal_rate",
]

# How uploaded column names map to internal profile keys
_UPLOAD_TO_PROFILE: dict[str, str] = {
    "attack_rating":     "attack_rating",
    "defense_rating":    "defense_rating",
    "goals_avg":         "goals_avg",
    "conceded_avg":      "conceded_avg",
    "xg_avg":            "xg_avg",
    "xga_avg":           "xga_avg",
    "win_rate_last10":   "win_rate",
    "form_points_last5": "form_pts",
    "form_points_last10":"form_pts",
    "clean_sheet_rate":  "clean_sheet_rate",
    "first_goal_rate":   "first_goal_rate",
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def normalize_team_name(team: Any) -> str:
    if not isinstance(team, str):
        if pd.isna(team):
            return ""
        team = str(team)
    # First collapse whitespace and resolve common-English canonical name
    return resolve_country(" ".join(team.strip().split()))


def result_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def get_val(profile: Any, attr: str, default: float = 0.0) -> float:
    """Get a numeric attribute from a profile dict or dataclass."""
    if isinstance(profile, dict):
        return float(profile.get(attr, default))
    return float(getattr(profile, attr, default))


def _get_stat(profile: Any, attr: str, default: float = 0.0) -> float:
    return get_val(profile, attr, default)


# ---------------------------------------------------------------------------
# Rolling team statistics (computed from match history)
# ---------------------------------------------------------------------------

def _compute_team_rolling_stats(
    matches: pd.DataFrame,
    team: str,
    n_form: int = 5,
) -> dict[str, float]:
    """Derive per-team performance statistics from historical match results.

    Returns a dict with keys matching _STAT_DEFAULTS.  Falls back to defaults
    for any metric that cannot be computed from the available data.
    """
    hm = matches[matches["home_team"] == team]
    am = matches[matches["away_team"] == team]
    total = len(hm) + len(am)

    if total == 0:
        return dict(_STAT_DEFAULTS)

    # --- Goal averages ---
    goals_scored  = int(hm["home_goals"].sum()) + int(am["away_goals"].sum())
    goals_conceded = int(hm["away_goals"].sum()) + int(am["home_goals"].sum())
    goals_avg    = goals_scored  / total
    conceded_avg = goals_conceded / total

    # --- Win / clean-sheet rates ---
    home_wins = int((hm["home_goals"] > hm["away_goals"]).sum())
    away_wins = int((am["away_goals"] > am["home_goals"]).sum())
    win_rate  = (home_wins + away_wins) / total

    home_cs = int((hm["away_goals"] == 0).sum())
    away_cs = int((am["home_goals"] == 0).sum())
    clean_sheet_rate = (home_cs + away_cs) / total

    # --- First-goal rate ---
    first_goal_rate = 0.50
    if "first_goal_team" in matches.columns:
        fgt_home = int((hm["first_goal_team"] == team).sum())
        fgt_away = int((am["first_goal_team"] == team).sum())
        # Only count matches where a goal was scored
        scored_matches = total - int(((hm["home_goals"] + hm["away_goals"]) == 0).sum()) \
                               - int(((am["home_goals"] + am["away_goals"]) == 0).sum())
        if scored_matches > 0:
            first_goal_rate = (fgt_home + fgt_away) / scored_matches

    # --- Rolling form (last n_form matches by date) ---
    hm_pts = pd.DataFrame({
        "date": hm["date"].values,
        "pts":  np.where(hm["home_goals"].values > hm["away_goals"].values, 3,
                np.where(hm["home_goals"].values == hm["away_goals"].values, 1, 0)),
    })
    am_pts = pd.DataFrame({
        "date": am["date"].values,
        "pts":  np.where(am["away_goals"].values > am["home_goals"].values, 3,
                np.where(am["away_goals"].values == am["home_goals"].values, 1, 0)),
    })
    all_form = pd.concat([hm_pts, am_pts]).sort_values("date").tail(n_form)
    form_pts = float(all_form["pts"].sum()) if not all_form.empty else 5.0

    # --- xG averages (from pre-match xG columns if present) ---
    xg_avg  = _STAT_DEFAULTS["xg_avg"]
    xga_avg = _STAT_DEFAULTS["xga_avg"]

    if "home_team_prematch_xg" in matches.columns and "away_team_prematch_xg" in matches.columns:
        h_xg  = pd.to_numeric(hm["home_team_prematch_xg"], errors="coerce").dropna()
        a_xg  = pd.to_numeric(am["away_team_prematch_xg"], errors="coerce").dropna()
        h_xga = pd.to_numeric(hm["away_team_prematch_xg"], errors="coerce").dropna()
        a_xga = pd.to_numeric(am["home_team_prematch_xg"], errors="coerce").dropna()
        combined_xg  = pd.concat([h_xg,  a_xg])
        combined_xga = pd.concat([h_xga, a_xga])
        if not combined_xg.empty:
            xg_avg  = float(combined_xg.mean())
        if not combined_xga.empty:
            xga_avg = float(combined_xga.mean())

    # --- Derived attack / defense ratings (0-100) ---
    # Based on goals relative to a global mean so they stay in a sensible range.
    GLOBAL_MEAN = 1.3
    attack_rating  = min(100.0, max(0.0, (goals_avg  / GLOBAL_MEAN) * 50.0))
    defense_rating = min(100.0, max(0.0, (1.0 - conceded_avg / (GLOBAL_MEAN * 2.0)) * 100.0))

    return {
        "goals_avg":        round(goals_avg,    4),
        "conceded_avg":     round(conceded_avg, 4),
        "win_rate":         round(win_rate,          4),
        "clean_sheet_rate": round(clean_sheet_rate,  4),
        "first_goal_rate":  round(first_goal_rate,   4),
        "form_pts":         round(form_pts, 2),
        "xg_avg":           round(xg_avg,  4),
        "xga_avg":          round(xga_avg, 4),
        "attack_rating":    round(attack_rating,  2),
        "defense_rating":   round(defense_rating, 2),
    }


# ---------------------------------------------------------------------------
# Team profiles
# ---------------------------------------------------------------------------

def build_team_profiles(
    matches: pd.DataFrame,
    players: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    """Build per-team profiles combining ELO, GK stats, and performance metrics.

    The performance metrics come from three sources (in increasing priority):
      1. Rolling statistics computed from the match results history.
      2. Uploaded match-level stat columns (File 2: prematch_xg, ppg, etc.).
      3. Uploaded team-level stat columns (File 3 after pivot: home_attack_rating,
         away_defense_rating, etc.).
    """
    # --- Gather latest ELO per team ---
    rows = []
    for side in ("home", "away"):
        rows.append(
            matches[["date", f"{side}_team", f"{side}_elo", f"{side}_elo_rank"]]
            .rename(columns={
                f"{side}_team":     "team",
                f"{side}_elo":      "elo",
                f"{side}_elo_rank": "elo_rank",
            })
        )
    all_teams = pd.concat(rows, ignore_index=True).sort_values("date")

    profiles: dict[str, dict[str, Any]] = {}

    for team, team_rows in all_teams.groupby("team", sort=False):
        # Skip tournament placeholder names (like '3rd Group A/B/C/D/F', 'Winner Group A')
        name_lower = str(team).lower()
        if any(p in name_lower for p in ["group", "winner", "runner", "loser", "playoff", "to be decided", "tbd", "qualification", "qualifier"]):
            continue

        latest = team_rows.iloc[-1][["elo", "elo_rank"]].apply(pd.to_numeric, errors="coerce")
        normalized = normalize_team_name(team)

        # --- Goalkeeper features ---
        gk_save_ratio      = 0.70
        gk_prevented_per_90 = 0.0
        if players is not None and not players.empty:
            team_norm   = normalized.lower()
            team_players = players[
                players["team"].apply(lambda t: normalize_team_name(t).lower()) == team_norm
            ]
            gks = team_players[team_players["position"].str.upper() == "GK"]
            if not gks.empty:
                primary_gk  = gks.sort_values("mins", ascending=False).iloc[0]
                save_perc   = float(primary_gk.get("save_perc", 70.0))
                gk_save_ratio = (save_perc / 100.0) if save_perc > 1.0 else (save_perc if save_perc > 0.0 else 0.70)
                goals_prev  = float(primary_gk.get("goals_prevented", 0.0))
                mins        = float(primary_gk.get("mins", 0.0))
                if mins > 90.0:
                    gk_prevented_per_90 = goals_prev / (mins / 90.0)

        # --- Rolling stats from match history (baseline) ---
        rolled = _compute_team_rolling_stats(matches, team)

        # --- Override with uploaded team-stat pivot columns (File 3) ---
        # After api.py's pivot, the matches DataFrame will have columns like
        # 'home_attack_rating', 'away_defense_rating', etc.
        for upload_col, profile_key in _UPLOAD_TO_PROFILE.items():
            for side in ("home", "away"):
                col = f"{side}_{upload_col}"
                if col in matches.columns:
                    side_col   = f"{side}_team"
                    side_vals  = matches.loc[matches[side_col] == team, col]
                    side_vals  = pd.to_numeric(side_vals, errors="coerce").dropna()
                    if not side_vals.empty:
                        rolled[profile_key] = float(side_vals.iloc[-1])

        # --- ppg from prematch_ppg columns ---
        for side, ppg_col in [("home", "prematch_ppg_home"), ("away", "prematch_ppg_away")]:
            if ppg_col in matches.columns:
                side_col  = f"{side}_team"
                side_vals = matches.loc[matches[side_col] == team, ppg_col]
                side_vals = pd.to_numeric(side_vals, errors="coerce").dropna()
                if not side_vals.empty:
                    rolled["ppg"] = float(side_vals.iloc[-1])

        profiles[normalized] = {
            "team":                normalized,
            "elo":                 float(latest["elo"]),
            "elo_rank":            float(latest["elo_rank"]),
            "gk_save_ratio":       gk_save_ratio,
            "gk_prevented_per_90": gk_prevented_per_90,
            **{k: rolled.get(k, _STAT_DEFAULTS.get(k, 0.0)) for k in _STAT_DEFAULTS},
        }

    return profiles


# ---------------------------------------------------------------------------
# Training frame construction
# ---------------------------------------------------------------------------

def _fill_col(
    frame: pd.DataFrame,
    out_col: str,
    team_col: str,
    profile_key: str,
    team_profiles: dict[str, Any],
    default: float,
) -> pd.Series:
    """Return a Series for out_col: use frame column if present & non-NaN,
    otherwise fall back to team profile, then to hard default."""
    if out_col in frame.columns:
        series = pd.to_numeric(frame[out_col], errors="coerce")
        nan_mask = series.isna()
        if nan_mask.any():
            fallback = frame.loc[nan_mask, team_col].apply(
                lambda t: _get_stat(
                    team_profiles.get(normalize_team_name(t), {}),
                    profile_key,
                    default,
                )
            )
            series = series.copy()
            series.loc[nan_mask] = fallback
        return series.fillna(default)
    else:
        return frame[team_col].apply(
            lambda t: _get_stat(
                team_profiles.get(normalize_team_name(t), {}),
                profile_key,
                default,
            )
        )


def build_training_frame(
    matches: pd.DataFrame,
    team_profiles: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if team_profiles is None:
        team_profiles = {}

    frame = matches.copy()

    # --- Basic labels ---
    frame["result"] = [
        result_label(home, away)
        for home, away in zip(frame["home_goals"], frame["away_goals"], strict=True)
    ]
    frame["total_goals"]        = frame["home_goals"] + frame["away_goals"]
    frame["both_teams_to_score"] = (
        (frame["home_goals"] > 0) & (frame["away_goals"] > 0)
    ).astype(int)
    frame["home_clean_sheet"] = (frame["away_goals"] == 0).astype(int)
    frame["away_clean_sheet"] = (frame["home_goals"] == 0).astype(int)

    if "first_goal_team" not in frame.columns:
        first_goal_team = []
        for h, a, ht, at in zip(frame["home_goals"], frame["away_goals"],
                                  frame["home_team"],  frame["away_team"]):
            if h > 0:
                first_goal_team.append(ht)
            elif a > 0:
                first_goal_team.append(at)
            else:
                first_goal_team.append("")
        frame["first_goal_team"] = first_goal_team

    frame["first_goal_home"] = (frame["first_goal_team"] == frame["home_team"]).astype(int)

    # --- GK features ---
    home_gk_save_ratio      = []
    away_gk_save_ratio      = []
    home_gk_prevented_per_90 = []
    away_gk_prevented_per_90 = []
    for h_team, a_team in zip(frame["home_team"], frame["away_team"]):
        h_prof = team_profiles.get(normalize_team_name(h_team), {})
        a_prof = team_profiles.get(normalize_team_name(a_team), {})
        home_gk_save_ratio.append(_get_stat(h_prof, "gk_save_ratio", 0.70))
        away_gk_save_ratio.append(_get_stat(a_prof, "gk_save_ratio", 0.70))
        home_gk_prevented_per_90.append(_get_stat(h_prof, "gk_prevented_per_90", 0.0))
        away_gk_prevented_per_90.append(_get_stat(a_prof, "gk_prevented_per_90", 0.0))

    frame["home_gk_save_ratio"]       = home_gk_save_ratio
    frame["away_gk_save_ratio"]       = away_gk_save_ratio
    frame["home_gk_prevented_per_90"] = home_gk_prevented_per_90
    frame["away_gk_prevented_per_90"] = away_gk_prevented_per_90

    # --- Extended performance features ---
    # Each feature tries the uploaded match column first, then falls back to
    # the team profile (so training and inference share the same code path).

    # Attack / defense ratings
    frame["home_attack_rating"]  = _fill_col(frame, "home_attack_rating",  "home_team", "attack_rating",  team_profiles, _STAT_DEFAULTS["attack_rating"])
    frame["away_attack_rating"]  = _fill_col(frame, "away_attack_rating",  "away_team", "attack_rating",  team_profiles, _STAT_DEFAULTS["attack_rating"])
    frame["home_defense_rating"] = _fill_col(frame, "home_defense_rating", "home_team", "defense_rating", team_profiles, _STAT_DEFAULTS["defense_rating"])
    frame["away_defense_rating"] = _fill_col(frame, "away_defense_rating", "away_team", "defense_rating", team_profiles, _STAT_DEFAULTS["defense_rating"])

    # Goals averages — try match-level columns first (ppg / goal_count roll-ups)
    frame["home_goals_avg"]    = _fill_col(frame, "home_goals_avg",    "home_team", "goals_avg",    team_profiles, _STAT_DEFAULTS["goals_avg"])
    frame["away_goals_avg"]    = _fill_col(frame, "away_goals_avg",    "away_team", "goals_avg",    team_profiles, _STAT_DEFAULTS["goals_avg"])
    frame["home_conceded_avg"] = _fill_col(frame, "home_conceded_avg", "home_team", "conceded_avg", team_profiles, _STAT_DEFAULTS["conceded_avg"])
    frame["away_conceded_avg"] = _fill_col(frame, "away_conceded_avg", "away_team", "conceded_avg", team_profiles, _STAT_DEFAULTS["conceded_avg"])

    # Win / clean-sheet rates
    frame["home_win_rate"]         = _fill_col(frame, "home_win_rate",         "home_team", "win_rate",         team_profiles, _STAT_DEFAULTS["win_rate"])
    frame["away_win_rate"]         = _fill_col(frame, "away_win_rate",         "away_team", "win_rate",         team_profiles, _STAT_DEFAULTS["win_rate"])
    frame["home_clean_sheet_rate"] = _fill_col(frame, "home_clean_sheet_rate", "home_team", "clean_sheet_rate", team_profiles, _STAT_DEFAULTS["clean_sheet_rate"])
    frame["away_clean_sheet_rate"] = _fill_col(frame, "away_clean_sheet_rate", "away_team", "clean_sheet_rate", team_profiles, _STAT_DEFAULTS["clean_sheet_rate"])

    # xG averages — check prematch_xg columns first
    if "home_team_prematch_xg" in frame.columns:
        frame["home_xg_avg"] = _fill_col(frame, "home_team_prematch_xg", "home_team", "xg_avg", team_profiles, _STAT_DEFAULTS["xg_avg"])
    else:
        frame["home_xg_avg"] = _fill_col(frame, "home_xg_avg", "home_team", "xg_avg", team_profiles, _STAT_DEFAULTS["xg_avg"])

    if "away_team_prematch_xg" in frame.columns:
        frame["away_xg_avg"] = _fill_col(frame, "away_team_prematch_xg", "away_team", "xg_avg", team_profiles, _STAT_DEFAULTS["xg_avg"])
    else:
        frame["away_xg_avg"] = _fill_col(frame, "away_xg_avg", "away_team", "xg_avg", team_profiles, _STAT_DEFAULTS["xg_avg"])

    # Form points
    frame["home_form_pts"] = _fill_col(frame, "home_form_pts", "home_team", "form_pts", team_profiles, _STAT_DEFAULTS["form_pts"])
    frame["away_form_pts"] = _fill_col(frame, "away_form_pts", "away_team", "form_pts", team_profiles, _STAT_DEFAULTS["form_pts"])

    # First-goal rate
    frame["home_first_goal_rate"] = _fill_col(frame, "home_first_goal_rate", "home_team", "first_goal_rate", team_profiles, _STAT_DEFAULTS["first_goal_rate"])
    frame["away_first_goal_rate"] = _fill_col(frame, "away_first_goal_rate", "away_team", "first_goal_rate", team_profiles, _STAT_DEFAULTS["first_goal_rate"])

    return add_difference_features(frame)


# ---------------------------------------------------------------------------
# Difference / interaction features
# ---------------------------------------------------------------------------

def add_difference_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["elo_diff"]          = frame["home_elo"]          - frame["away_elo"]
    frame["rank_diff"]         = frame["away_elo_rank"]     - frame["home_elo_rank"]
    frame["attack_diff"]       = frame["home_attack_rating"]  - frame["away_attack_rating"]
    frame["defense_diff"]      = frame["home_defense_rating"] - frame["away_defense_rating"]
    frame["goals_avg_diff"]    = frame["home_goals_avg"]    - frame["away_goals_avg"]
    frame["conceded_avg_diff"] = frame["home_conceded_avg"] - frame["away_conceded_avg"]
    frame["win_rate_diff"]     = frame["home_win_rate"]     - frame["away_win_rate"]
    frame["xg_avg_diff"]       = frame["home_xg_avg"]       - frame["away_xg_avg"]
    frame["form_diff"]         = frame["home_form_pts"]     - frame["away_form_pts"]
    return frame


# ---------------------------------------------------------------------------
# Inference feature construction
# ---------------------------------------------------------------------------

def build_inference_features(
    home_team: str,
    away_team: str,
    team_profiles: dict[str, Any],
) -> pd.DataFrame:
    """Build the full FEATURE_COLUMNS vector for a single match prediction."""
    home_name = normalize_team_name(home_team)
    away_name = normalize_team_name(away_team)

    def _fallback_profile(name: str) -> dict[str, Any]:
        """Compute average profile when a team is unknown."""
        if team_profiles:
            avg_elo  = sum(get_val(p, "elo")      for p in team_profiles.values()) / len(team_profiles)
            avg_rank = sum(get_val(p, "elo_rank")  for p in team_profiles.values()) / len(team_profiles)
        else:
            avg_elo, avg_rank = 1500.0, 50.0
        return {"team": name, "elo": avg_elo, "elo_rank": avg_rank, **_STAT_DEFAULTS}

    import logging
    if home_name not in team_profiles:
        logging.warning(f"Unknown home_team: {home_team!r}. Using fallback profile.")
        home = _fallback_profile(home_name)
    else:
        home = team_profiles[home_name]

    if away_name not in team_profiles:
        logging.warning(f"Unknown away_team: {away_team!r}. Using fallback profile.")
        away = _fallback_profile(away_name)
    else:
        away = team_profiles[away_name]

    row = {
        # Core ELO
        "home_elo":      get_val(home, "elo"),
        "away_elo":      get_val(away, "elo"),
        "home_elo_rank": get_val(home, "elo_rank"),
        "away_elo_rank": get_val(away, "elo_rank"),
        # GK
        "home_gk_save_ratio":       get_val(home, "gk_save_ratio",       0.70),
        "away_gk_save_ratio":       get_val(away, "gk_save_ratio",       0.70),
        "home_gk_prevented_per_90": get_val(home, "gk_prevented_per_90", 0.0),
        "away_gk_prevented_per_90": get_val(away, "gk_prevented_per_90", 0.0),
        # Extended
        "home_attack_rating":  get_val(home, "attack_rating",  _STAT_DEFAULTS["attack_rating"]),
        "away_attack_rating":  get_val(away, "attack_rating",  _STAT_DEFAULTS["attack_rating"]),
        "home_defense_rating": get_val(home, "defense_rating", _STAT_DEFAULTS["defense_rating"]),
        "away_defense_rating": get_val(away, "defense_rating", _STAT_DEFAULTS["defense_rating"]),
        "home_goals_avg":      get_val(home, "goals_avg",      _STAT_DEFAULTS["goals_avg"]),
        "away_goals_avg":      get_val(away, "goals_avg",      _STAT_DEFAULTS["goals_avg"]),
        "home_conceded_avg":   get_val(home, "conceded_avg",   _STAT_DEFAULTS["conceded_avg"]),
        "away_conceded_avg":   get_val(away, "conceded_avg",   _STAT_DEFAULTS["conceded_avg"]),
        "home_win_rate":         get_val(home, "win_rate",         _STAT_DEFAULTS["win_rate"]),
        "away_win_rate":         get_val(away, "win_rate",         _STAT_DEFAULTS["win_rate"]),
        "home_clean_sheet_rate": get_val(home, "clean_sheet_rate", _STAT_DEFAULTS["clean_sheet_rate"]),
        "away_clean_sheet_rate": get_val(away, "clean_sheet_rate", _STAT_DEFAULTS["clean_sheet_rate"]),
        "home_xg_avg":    get_val(home, "xg_avg",    _STAT_DEFAULTS["xg_avg"]),
        "away_xg_avg":    get_val(away, "xg_avg",    _STAT_DEFAULTS["xg_avg"]),
        "home_form_pts":  get_val(home, "form_pts",  _STAT_DEFAULTS["form_pts"]),
        "away_form_pts":  get_val(away, "form_pts",  _STAT_DEFAULTS["form_pts"]),
        "home_first_goal_rate": get_val(home, "first_goal_rate", _STAT_DEFAULTS["first_goal_rate"]),
        "away_first_goal_rate": get_val(away, "first_goal_rate", _STAT_DEFAULTS["first_goal_rate"]),
    }

    frame = pd.DataFrame([row])

    # Compute differentials
    frame["elo_diff"]          = frame["home_elo"]           - frame["away_elo"]
    frame["rank_diff"]         = frame["away_elo_rank"]      - frame["home_elo_rank"]
    frame["attack_diff"]       = frame["home_attack_rating"] - frame["away_attack_rating"]
    frame["defense_diff"]      = frame["home_defense_rating"]- frame["away_defense_rating"]
    frame["goals_avg_diff"]    = frame["home_goals_avg"]     - frame["away_goals_avg"]
    frame["conceded_avg_diff"] = frame["home_conceded_avg"]  - frame["away_conceded_avg"]
    frame["win_rate_diff"]     = frame["home_win_rate"]      - frame["away_win_rate"]
    frame["xg_avg_diff"]       = frame["home_xg_avg"]        - frame["away_xg_avg"]
    frame["form_diff"]         = frame["home_form_pts"]      - frame["away_form_pts"]

    return frame[FEATURE_COLUMNS]
