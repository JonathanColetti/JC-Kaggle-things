# NHL Game Data (2024-2025): Advanced Metrics & Betting

![NHL Analytics](https://images.unsplash.com/photo-1515703407324-5f753afd8be8?q=80&w=1000&auto=format&fit=crop) 

# Disclaimer

This is a public API provided by ESPN, and I am not affiliated with ESPN or responsible for any usage of this API. This information is provided solely for educational and informational purposes. Please use this API responsibly and abide by ESPN's terms of service.

## Context
This dataset provides granular game-level statistics for the 2024-2025 NHL season. Unlike standard boxscore datasets, this dataset is preprocessed for Machine Learning. It includes rolling averages, rest day calculations, and opponent statistics joined onto each row, allowing for immediate time-series forecasting or classification modeling without extensive feature engineering.

The data is sourced from ESPN's public API and includes betting odds (spread, moneyline, over/under) to facilitate handicapping analysis.

## Content
The dataset follows a **Row-Per-Team** structure.
* **Total Rows:** approx 2,600 (Approx. 1,312 games * 2 teams)
* **Granularity:** Each row represents one team's performance/status for a specific game.
* **Leakage Prevention:** Rolling features use `closed='left'`, ensuring Game N only includes history up to Game N-1.

### Key Features
1. **Rolling Averages (3 & 10 games)** — scoring, shots, hits, penalties, and power play efficiency.
2. **Rest Tracking** — days since last game, plus rest advantage vs opponent.
3. **Opponent Features Included** — mirrored metrics for cross-team predictive modeling.
4. **Betting Lines Included** — closing spreads, totals, and moneylines.
5. **Playoff data included** - Playoff games are also included in the dataset

## Data Dictionary

### Raw Game Statistics
| Column | Type | Description |
| :--- | :--- | :--- |
| `shots` | Float | Shots on goal in this game. |
| `power_play_goals` | Float | Power play goals scored. |
| `power_play_opportunities` | Float | Number of power play chances. |
| `faceoff_win_pct` | Float | Faceoff win percentage for the game. |
| `hits` | Float | Total hits recorded. |
| `blocked_shots` | Float | Shots blocked by this team. |
| `pim` | Float | Penalty minutes accumulated. |
| `giveaways` | Float | Giveaways committed. |
| `takeaways` | Float | Takeaways recorded. |

### Team Performance Features
| Column | Type | Description |
| :--- | :--- | :--- |
| `rest_days` | Float | Days since last game (capped at 7). |
| `rest_advantage` | Float | `rest_days - opp_rest_days`. |
| `season_win_pct` | Float | Win% to date (before this game). |
| `cum_wins` | Integer | Cumulative season wins prior to game. |
| `cum_games` | Integer | Games played to date in season. |
| `save_pct` | Float | Goalie save% for this specific game. |
| `is_home` | Binary | `1` = Home, `0` = Away. |
| `team_record` | String | Win-Loss-OT record at time of game. |

---

### Rolling Averages (3-Game Window)
| Column | Type | Description |
| :--- | :--- | :--- |
| `rolling_score_3` | Float | Avg goals scored over previous 3 games. |
| `rolling_power_play_goals_3` | Float | Avg PP goals (3-game window). |
| `rolling_power_play_opportunities_3` | Float | Avg PP chances (3-game window). |
| `rolling_faceoff_win_pct_3` | Float | Avg faceoff win% (3-game window). |
| `rolling_hits_3` | Float | Avg hits (3-game window). |
| `rolling_blocked_shots_3` | Float | Avg blocked shots (3-game window). |
| `rolling_shots_3` | Float | Avg shots on goal (3-game window). |
| `rolling_pim_3` | Float | Avg penalty minutes (3-game window). |
| `rolling_giveaways_3` | Float | Avg giveaways (3-game window). |
| `rolling_takeaways_3` | Float | Avg takeaways (3-game window). |
| `rolling_pp_efficiency_3` | Float | PP Goals / PP Opps (3-game window). |

---

### Rolling Averages (10-Game Window)
| Column | Type | Description |
| :--- | :--- | :--- |
| `rolling_score_10` | Float | Avg goals scored over previous 10 games. |
| `rolling_power_play_goals_10` | Float | Avg PP goals (10-game window). |
| `rolling_power_play_opportunities_10` | Float | Avg PP chances (10-game window). |
| `rolling_faceoff_win_pct_10` | Float | Avg faceoff win% (10-game window). |
| `rolling_hits_10` | Float | Avg hits (10-game window). |
| `rolling_blocked_shots_10` | Float | Avg blocked shots (10-game window). |
| `rolling_shots_10` | Float | Avg shots on goal (10-game window). |
| `rolling_pim_10` | Float | Avg penalty minutes (10-game window). |
| `rolling_giveaways_10` | Float | Avg giveaways (10-game window). |
| `rolling_takeaways_10` | Float | Avg takeaways (10-game window). |
| `rolling_pp_efficiency_10`| Float | PP Goals / PP Opps (10-game window). |

---

### Opponent Features
All opponent metrics mirror the team's features with an `opp_` prefix:

| Column | Type | Description |
| :--- | :--- | :--- |
| `opp_team_id` | Integer | Opponent's team identifier. |
| `opp_team_name` | String | Opponent's team name. |
| `opp_won` | Binary | Whether opponent won (`1`) or lost (`0`). |
| `opp_rest_days` | Float | Opponent's days since last game. |
| `opp_season_win_pct` | Float | Opponent's win% to date. |
| `opp_score` | Integer | Goals scored by opponent. |
| `opp_shots` | Float | Opponent's shots on goal. |
| `opp_rolling_score_3` | Float | Opponent's 3-game avg goals. |
| `opp_rolling_score_10` | Float | Opponent's 10-game avg goals. |
| `opp_rolling_power_play_goals_3` | Float | Opponent's 3-game PP goals avg. |
| `opp_rolling_power_play_goals_10` | Float | Opponent's 10-game PP goals avg. |
| `opp_rolling_power_play_opportunities_3` | Float | Opponent's 3-game PP opps avg. |
| `opp_rolling_power_play_opportunities_10` | Float | Opponent's 10-game PP opps avg. |
| `opp_rolling_faceoff_win_pct_3` | Float | Opponent's 3-game faceoff% avg. |
| `opp_rolling_faceoff_win_pct_10` | Float | Opponent's 10-game faceoff% avg. |
| `opp_rolling_hits_3` | Float | Opponent's 3-game hits avg. |
| `opp_rolling_hits_10` | Float | Opponent's 10-game hits avg. |
| `opp_rolling_blocked_shots_3` | Float | Opponent's 3-game blocked shots avg. |
| `opp_rolling_blocked_shots_10` | Float | Opponent's 10-game blocked shots avg. |
| `opp_rolling_shots_3` | Float | Opponent's 3-game shots avg. |
| `opp_rolling_shots_10` | Float | Opponent's 10-game shots avg. |
| `opp_rolling_pim_3` | Float | Opponent's 3-game PIM avg. |
| `opp_rolling_pim_10` | Float | Opponent's 10-game PIM avg. |
| `opp_rolling_giveaways_3` | Float | Opponent's 3-game giveaways avg. |
| `opp_rolling_giveaways_10` | Float | Opponent's 10-game giveaways avg. |
| `opp_rolling_takeaways_3` | Float | Opponent's 3-game takeaways avg. |
| `opp_rolling_takeaways_10` | Float | Opponent's 10-game takeaways avg. |
| `opp_rolling_pp_efficiency_3` | Float | Opponent's 3-game PP efficiency. |
| `opp_rolling_pp_efficiency_10` | Float | Opponent's 10-game PP efficiency. |

---

## Methodology

1. **Ingested via ESPN API** using `requests`.
2. **Flattened boxscores** into tabular rows.
3. **Derived labels** (`won`, cumulative win tracking).
4. **Feature engineering**:
   * Rolling windows via `groupby().rolling().shift(1)`
   * Opponent joins applied row-wise
   * Betting NaNs preserved to avoid bias

---

## Inspiration — Projects to Explore
- Predict **win probability** vs Vegas moneyline  
- Forecast **total goals** to beat market O/U  
- Analyze **fatigue impact** via rest differential  
- Cluster teams based on physical vs offensive profile
- Add the is playoff game column

---

### Acknowledgements
Publicly available data courtesy of ESPN.  
Dataset intended for educational + analytical research.

