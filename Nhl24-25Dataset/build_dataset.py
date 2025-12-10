import os
import json
import pandas as pd
import numpy as np
from glob import glob
from typing import Any

DEFAULT_REST_DAYS = 3
MAX_REST_DAYS = 7
OUTLIER_QUANTILE = 0.99
ROLLING_WINDOWS = [3, 10]

NEUTRAL_SEASON_WIN_PCT = 0.45
NEUTRAL_SAVE_PCT = 0.91
NEUTRAL_RATE_METRIC = 0.50

def load_and_flatten_data(directory: str) -> pd.DataFrame:
    """
    Reads JSON files from a directory and flattens them into a pandas DataFrame.

    Args:
        directory (str): The path to the directory containing JSON data files.

    Returns:
        pd.DataFrame: A flattened DataFrame containing game and team statistics.
    """
    files = glob(os.path.join(directory, "*.json"))
    rows = []

    for file in files:
        try:
            with open(file, "r") as f:
                games_list = json.load(f)

            if not isinstance(games_list, list):
                continue

            for game in games_list:
                game_id = game.get("game_id")
                date = game.get("date")
                d_obj = pd.to_datetime(date)

                season = d_obj.year + 1 if d_obj.month > 8 else d_obj.year

                venue = game.get("venue", "Unknown")
                attendance = game.get("attendance", 0)

                spread = game.get("spread")
                over_under = game.get("over_under")
                moneyline = game.get("favorite_moneyline")

                season_series = game.get("season_series_summary", "")

                officials_str = "|".join([o.get("name", "") for o in game.get("officials", [])])

                teams = game.get("teams_stats", [])
                if len(teams) != 2:
                    continue

                s0, s1 = int(teams[0].get("score", 0)), int(teams[1].get("score", 0))
                teams[0]["won"], teams[1]["won"] = int(s0 > s1), int(s1 > s0)

                for team in teams:
                    def f(x: Any) -> float:
                        try: return float(x)
                        except: return 0.0

                    rows.append({
                        "game_id": game_id,
                        "date": d_obj,
                        "season": season,
                        "venue": venue,
                        "attendance": float(attendance),
                        "officials": officials_str,
                        "season_series": season_series,
                        "spread": str(spread),
                        "over_under": f(over_under),
                        "favorite_moneyline": f(moneyline),
                        "team_id": team.get("team_id"),
                        "team_name": team.get("team_name"),
                        "home_away": team.get("home_away"),
                        "team_record": team.get("record_summary",""),
                        "won": team["won"],
                        "score": int(team.get("score",0)),
                        "shots": f(team.get("shots")),
                        "power_play_goals": f(team.get("power_play_goals")),
                        "power_play_opportunities": f(team.get("power_play_opportunities")),
                        "faceoff_win_pct": f(team.get("faceoff_win_pct")),
                        "hits": f(team.get("hits")),
                        "blocked_shots": f(team.get("blocked_shots")),
                        "pim": f(team.get("pim")),
                        "giveaways": f(team.get("giveaways")),
                        "takeaways": f(team.get("takeaways")),
                    })

        except Exception:
            pass

    df = pd.DataFrame(rows).sort_values(["team_id","date"]).reset_index(drop=True)
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates rolling statistics, lag features, and merges opponent data.

    Args:
        df (pd.DataFrame): The raw flattened DataFrame.

    Returns:
        pd.DataFrame: The processed DataFrame suitable for modeling.
    """
    df['date'] = pd.to_datetime(df['date'])
    # do not fill this will default value as 0.0 means equal odds
    df['spread'] = pd.to_numeric(df['spread'], errors="coerce")
    df = df.sort_values(["team_id","season","date"])

    shifted = df.groupby(['team_id','season'])['won'].transform(lambda x: x.shift())
    df['cum_wins'] = shifted.cumsum().fillna(0)

    df['cum_games'] = df.groupby(['team_id','season']).cumcount()
    df['season_win_pct'] = (df['cum_wins'] / df['cum_games']).replace([np.inf,np.nan], NEUTRAL_SEASON_WIN_PCT)

    df['prev_date'] = df.groupby(['team_id','season'])['date'].shift(1)
    df['rest_days'] = (df['date'] - df['prev_date']).dt.days
    df['rest_days'] = df['rest_days'].fillna(DEFAULT_REST_DAYS).clip(upper=MAX_REST_DAYS)
    df.drop(columns=['prev_date'], inplace=True)

    stats = ["score","power_play_goals","power_play_opportunities","faceoff_win_pct",
             "hits","blocked_shots","shots","pim","giveaways","takeaways"]
    
    for col in stats:
        df[col] = df[col].clip(upper=df[col].quantile(OUTLIER_QUANTILE))

    for w in ROLLING_WINDOWS:
        rolled = (
            df.groupby(['team_id','season'])[stats]
              .rolling(w, min_periods=1, closed='left').mean()
              .reset_index(drop=True)
        )
        rolled.columns = [f"rolling_{c}_{w}" for c in stats]
        df = pd.concat([df, rolled], axis=1)

        df[f"rolling_pp_efficiency_{w}"] = \
            df[f"rolling_power_play_goals_{w}"] / (df[f"rolling_power_play_opportunities_{w}"] + 1e-6)

    df = df.replace([np.inf,-np.inf], np.nan)

    opp = df[[
        "game_id","team_id","team_name","won","rest_days","season_win_pct","score","shots",
        *[f"rolling_{s}_{w}" for s in stats for w in ROLLING_WINDOWS],
        *[f"rolling_pp_efficiency_{w}" for w in ROLLING_WINDOWS]
    ]].copy()

    opp.columns = [c if c=="game_id" else f"opp_{c}" for c in opp.columns]

    df = df.merge(opp, on="game_id")
    df = df[df.team_id != df.opp_team_id]

    df['save_pct'] = np.where(
        df['opp_shots'] > 0,
        (df['opp_shots'] - df['opp_score']) / df['opp_shots'],
        NEUTRAL_SAVE_PCT
    )

    df['is_home'] = (df['home_away']=="home").astype(int)
    df['rest_advantage'] = df['rest_days'] - df['opp_rest_days']

    cols = list(df.columns)
    cols.insert(0, cols.pop(cols.index("won")))
    df = df[cols]

    return df

if __name__ == "__main__":
    df_raw = load_and_flatten_data("./data")
    if not df_raw.empty:
        df_final = build_features(df_raw)
        df_final.to_csv("nhl_dataset.csv", index=False)