# Data Usage in Premier League ML Model

This document explains what data from the premier-league API is used to train the model.

## Raw Data Extracted from API

The `premier-league` package provides match data from FBRef.com. From each match, we extract:

### Core Match Data (Directly from API)
1. **Date** - Match date and time
2. **HomeTeam** - Name of the home team
3. **AwayTeam** - Name of the away team
4. **FTHG** (Full Time Home Goals) - Goals scored by home team
5. **FTAG** (Full Time Away Goals) - Goals scored by away team
6. **FTR** (Full Time Result) - Match result: 'H' (Home win), 'D' (Draw), 'A' (Away win)

**Note**: The FTR is calculated from the scores: `'H' if home_score > away_score else 'A' if away_score > home_score else 'D'`

## Feature Engineering

The raw match data is transformed into **25 engineered features** for machine learning:

### 1. Home Team Statistics (8 features)
- `home_win_rate` - Overall win rate of home team
- `home_draw_rate` - Overall draw rate of home team
- `home_loss_rate` - Overall loss rate of home team
- `home_goals_avg` - Average goals scored by home team (at home)
- `home_goals_conceded_avg` - Average goals conceded by home team (at home)
- `home_home_win_rate` - Win rate when playing at home
- `home_recent_form` - Average points from last 5 matches (1.0=win, 0.5=draw, 0.0=loss)
- `home_home_form` - Average points from last 5 home matches

### 2. Away Team Statistics (8 features)
- `away_win_rate` - Overall win rate of away team
- `away_draw_rate` - Overall draw rate of away team
- `away_loss_rate` - Overall loss rate of away team
- `away_goals_avg` - Average goals scored by away team (away from home)
- `away_goals_conceded_avg` - Average goals conceded by away team (away from home)
- `away_away_win_rate` - Win rate when playing away
- `away_recent_form` - Average points from last 5 matches
- `away_away_form` - Average points from last 5 away matches

### 3. Head-to-Head Statistics (4 features)
- `h2h_home_wins` - Number of times home team won in previous meetings
- `h2h_draws` - Number of draws in previous meetings
- `h2h_away_wins` - Number of times away team won in previous meetings
- `h2h_home_goal_diff` - Average goal difference from home team's perspective

### 4. Comparative Features (3 features)
- `goal_diff` - Difference in average goals scored (home_avg - away_avg)
- `defensive_diff` - Difference in average goals conceded (away_conceded - home_conceded)
- `points_diff` - Difference in league points (home_points - away_points)

### 5. League Position (2 features)
- `home_points` - Total points accumulated by home team
- `away_points` - Total points accumulated by away team

## How Features Are Calculated

All features are calculated using **only historical data** up to the current match:
- For each match, we look at all previous matches in the dataset
- Team statistics are calculated from matches before the current match
- This ensures no data leakage (we don't use future information to predict past matches)

## Target Variable

The model predicts one of three outcomes:
- **0** = Home Win (H)
- **1** = Draw (D)
- **2** = Away Win (A)

## Data Flow Summary

```
API Data (premier-league package)
    ↓
Raw Match Data: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR
    ↓
Feature Engineering (25 features calculated from historical matches)
    ↓
Scaled Features (StandardScaler normalization)
    ↓
ML Model Training (Random Forest, Gradient Boosting, Logistic Regression, XGBoost)
    ↓
Trained Model (predicts: Home Win / Draw / Away Win)
```

## Key Points

1. **Only 6 raw fields** are extracted from the API per match
2. **25 features** are engineered from historical match data
3. **No external data** is used (no player stats, no betting odds, etc.)
4. **Temporal ordering** is preserved - features only use past matches
5. **Home/Away context** is important - separate statistics for home vs away performance

## Example

For a match between Arsenal (home) vs Chelsea (away):

**Raw Data:**
- Date: 2024-12-22
- HomeTeam: Arsenal
- AwayTeam: Chelsea
- FTHG: 2
- FTAG: 1
- FTR: H

**Engineered Features (sample):**
- home_win_rate: 0.65 (Arsenal wins 65% of their matches)
- away_win_rate: 0.55 (Chelsea wins 55% of their matches)
- home_recent_form: 0.8 (Arsenal averaged 0.8 points in last 5 matches)
- h2h_home_wins: 3 (Arsenal won 3 previous meetings)
- goal_diff: 0.3 (Arsenal scores 0.3 more goals on average)
- ... and 20 more features

**Model Prediction:** Uses all 25 features to predict the outcome

