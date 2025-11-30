"""
Feature engineering module for Premier League match prediction.
Creates features from historical match data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class FeatureEngineer:
    """Engineers features from raw match data for ML models."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.team_stats = {}
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features from match data.
        
        Args:
            df: DataFrame with match data (Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR)
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Initialize team statistics
        self._initialize_team_stats(df)
        
        # Create features for each match
        features_list = []
        for idx, row in df.iterrows():
            features = self._extract_match_features(df, idx, row)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Add target variable
        features_df['target'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
        
        return features_df
    
    def _initialize_team_stats(self, df: pd.DataFrame):
        """Initialize team statistics dictionary."""
        teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
        for team in teams:
            self.team_stats[team] = {
                'matches_played': 0,
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'goals_for': 0,
                'goals_against': 0,
                'home_wins': 0,
                'home_draws': 0,
                'home_losses': 0,
                'away_wins': 0,
                'away_draws': 0,
                'away_losses': 0,
                'recent_form': [],  # Last 5 results (1=win, 0.5=draw, 0=loss)
                'home_form': [],
                'away_form': [],
            }
    
    def _extract_match_features(self, df: pd.DataFrame, idx: int, row: pd.Series) -> Dict:
        """Extract features for a single match."""
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Get historical data up to this match
        historical = df.iloc[:idx]
        
        # Team statistics (up to this point)
        home_stats = self._get_team_stats(historical, home_team, is_home=True)
        away_stats = self._get_team_stats(historical, away_team, is_home=False)
        
        # Head-to-head statistics
        h2h = self._get_head_to_head(historical, home_team, away_team)
        
        # Recent form (multiple time windows for better prediction)
        home_recent_form_3 = self._get_recent_form(historical, home_team, n=3)
        home_recent_form_5 = self._get_recent_form(historical, home_team, n=5)
        home_recent_form_10 = self._get_recent_form(historical, home_team, n=10)
        away_recent_form_3 = self._get_recent_form(historical, away_team, n=3)
        away_recent_form_5 = self._get_recent_form(historical, away_team, n=5)
        away_recent_form_10 = self._get_recent_form(historical, away_team, n=10)
        
        # Home/Away specific form
        home_home_form = self._get_recent_form(historical, home_team, n=5, venue='home')
        away_away_form = self._get_recent_form(historical, away_team, n=5, venue='away')
        
        # Goal statistics
        home_goals_avg = self._get_avg_goals(historical, home_team, venue='home')
        away_goals_avg = self._get_avg_goals(historical, away_team, venue='away')
        home_conceded_avg = self._get_avg_goals_conceded(historical, home_team, venue='home')
        away_conceded_avg = self._get_avg_goals_conceded(historical, away_team, venue='away')
        
        # Goal momentum (recent scoring trends)
        home_goals_momentum = self._get_goals_momentum(historical, home_team, venue='home')
        away_goals_momentum = self._get_goals_momentum(historical, away_team, venue='away')
        home_conceded_momentum = self._get_goals_momentum(historical, home_team, venue='home', conceded=True)
        away_conceded_momentum = self._get_goals_momentum(historical, away_team, venue='away', conceded=True)
        
        # Goal difference trends
        home_gd_trend = self._get_goal_difference_trend(historical, home_team)
        away_gd_trend = self._get_goal_difference_trend(historical, away_team)
        
        # Combine all features
        features = {
            # Home team statistics
            'home_win_rate': home_stats['win_rate'],
            'home_draw_rate': home_stats['draw_rate'],
            'home_loss_rate': home_stats['loss_rate'],
            'home_goals_avg': home_goals_avg,
            'home_goals_conceded_avg': home_conceded_avg,
            'home_home_win_rate': home_stats['home_win_rate'],
            'home_recent_form_3': np.mean(home_recent_form_3) if home_recent_form_3 else 0.5,
            'home_recent_form_5': np.mean(home_recent_form_5) if home_recent_form_5 else 0.5,
            'home_recent_form_10': np.mean(home_recent_form_10) if home_recent_form_10 else 0.5,
            'home_home_form': np.mean(home_home_form) if home_home_form else 0.5,
            'home_goals_momentum': home_goals_momentum,
            'home_conceded_momentum': home_conceded_momentum,
            'home_gd_trend': home_gd_trend,
            
            # Away team statistics
            'away_win_rate': away_stats['win_rate'],
            'away_draw_rate': away_stats['draw_rate'],
            'away_loss_rate': away_stats['loss_rate'],
            'away_goals_avg': away_goals_avg,
            'away_goals_conceded_avg': away_conceded_avg,
            'away_away_win_rate': away_stats['away_win_rate'],
            'away_recent_form_3': np.mean(away_recent_form_3) if away_recent_form_3 else 0.5,
            'away_recent_form_5': np.mean(away_recent_form_5) if away_recent_form_5 else 0.5,
            'away_recent_form_10': np.mean(away_recent_form_10) if away_recent_form_10 else 0.5,
            'away_away_form': np.mean(away_away_form) if away_away_form else 0.5,
            'away_goals_momentum': away_goals_momentum,
            'away_conceded_momentum': away_conceded_momentum,
            'away_gd_trend': away_gd_trend,
            
            # Head-to-head
            'h2h_home_wins': h2h['home_wins'],
            'h2h_draws': h2h['draws'],
            'h2h_away_wins': h2h['away_wins'],
            'h2h_home_goal_diff': h2h['home_goal_diff'],
            
            # Goal difference
            'goal_diff': home_goals_avg - away_goals_avg,
            'defensive_diff': away_conceded_avg - home_conceded_avg,
            'momentum_diff': home_goals_momentum - away_goals_momentum,
            
            # League position (if available)
            'home_points': home_stats['points'],
            'away_points': away_stats['points'],
            'points_diff': home_stats['points'] - away_stats['points'],
        }
        
        return features
    
    def _get_team_stats(self, historical: pd.DataFrame, team: str, is_home: bool) -> Dict:
        """Get team statistics from historical matches."""
        if len(historical) == 0:
            return {
                'win_rate': 0.33, 'draw_rate': 0.33, 'loss_rate': 0.33,
                'home_win_rate': 0.33, 'away_win_rate': 0.33,
                'points': 0
            }
        
        if is_home:
            team_matches = historical[historical['HomeTeam'] == team]
            wins = len(team_matches[team_matches['FTR'] == 'H'])
            draws = len(team_matches[team_matches['FTR'] == 'D'])
            losses = len(team_matches[team_matches['FTR'] == 'A'])
        else:
            team_matches = historical[historical['AwayTeam'] == team]
            wins = len(team_matches[team_matches['FTR'] == 'A'])
            draws = len(team_matches[team_matches['FTR'] == 'D'])
            losses = len(team_matches[team_matches['FTR'] == 'H'])
        
        all_matches = historical[
            (historical['HomeTeam'] == team) | (historical['AwayTeam'] == team)
        ]
        total_matches = len(all_matches)
        
        if total_matches == 0:
            return {
                'win_rate': 0.33, 'draw_rate': 0.33, 'loss_rate': 0.33,
                'home_win_rate': 0.33, 'away_win_rate': 0.33,
                'points': 0
            }
        
        # Overall stats
        all_wins = len(all_matches[
            ((all_matches['HomeTeam'] == team) & (all_matches['FTR'] == 'H')) |
            ((all_matches['AwayTeam'] == team) & (all_matches['FTR'] == 'A'))
        ])
        all_draws = len(all_matches[all_matches['FTR'] == 'D'])
        all_losses = total_matches - all_wins - all_draws
        
        # Home/Away specific
        home_matches = historical[historical['HomeTeam'] == team]
        home_wins = len(home_matches[home_matches['FTR'] == 'H'])
        home_total = len(home_matches)
        
        away_matches = historical[historical['AwayTeam'] == team]
        away_wins = len(away_matches[away_matches['FTR'] == 'A'])
        away_total = len(away_matches)
        
        points = all_wins * 3 + all_draws
        
        return {
            'win_rate': all_wins / total_matches if total_matches > 0 else 0.33,
            'draw_rate': all_draws / total_matches if total_matches > 0 else 0.33,
            'loss_rate': all_losses / total_matches if total_matches > 0 else 0.33,
            'home_win_rate': home_wins / home_total if home_total > 0 else 0.33,
            'away_win_rate': away_wins / away_total if away_total > 0 else 0.33,
            'points': points
        }
    
    def _get_head_to_head(self, historical: pd.DataFrame, home_team: str, away_team: str) -> Dict:
        """Get head-to-head statistics between two teams."""
        h2h_matches = historical[
            ((historical['HomeTeam'] == home_team) & (historical['AwayTeam'] == away_team)) |
            ((historical['HomeTeam'] == away_team) & (historical['AwayTeam'] == home_team))
        ]
        
        if len(h2h_matches) == 0:
            return {'home_wins': 0, 'draws': 0, 'away_wins': 0, 'home_goal_diff': 0}
        
        home_wins = len(h2h_matches[
            ((h2h_matches['HomeTeam'] == home_team) & (h2h_matches['FTR'] == 'H')) |
            ((h2h_matches['AwayTeam'] == home_team) & (h2h_matches['FTR'] == 'A'))
        ])
        draws = len(h2h_matches[h2h_matches['FTR'] == 'D'])
        away_wins = len(h2h_matches) - home_wins - draws
        
        # Goal difference from home team's perspective
        goal_diff = 0
        for _, match in h2h_matches.iterrows():
            if match['HomeTeam'] == home_team:
                goal_diff += match['FTHG'] - match['FTAG']
            else:
                goal_diff += match['FTAG'] - match['FTHG']
        
        return {
            'home_wins': home_wins,
            'draws': draws,
            'away_wins': away_wins,
            'home_goal_diff': goal_diff / len(h2h_matches) if len(h2h_matches) > 0 else 0
        }
    
    def _get_recent_form(self, historical: pd.DataFrame, team: str, n: int = 5, 
                        venue: Optional[str] = None) -> List[float]:
        """Get recent form (last n matches) for a team.
        Returns list of results: 1.0 for win, 0.5 for draw, 0.0 for loss.
        """
        if venue == 'home':
            team_matches = historical[historical['HomeTeam'] == team].tail(n)
            results = [
                1.0 if row['FTR'] == 'H' else 0.5 if row['FTR'] == 'D' else 0.0
                for _, row in team_matches.iterrows()
            ]
        elif venue == 'away':
            team_matches = historical[historical['AwayTeam'] == team].tail(n)
            results = [
                1.0 if row['FTR'] == 'A' else 0.5 if row['FTR'] == 'D' else 0.0
                for _, row in team_matches.iterrows()
            ]
        else:
            team_matches = historical[
                (historical['HomeTeam'] == team) | (historical['AwayTeam'] == team)
            ].tail(n)
            results = []
            for _, row in team_matches.iterrows():
                if row['HomeTeam'] == team:
                    results.append(1.0 if row['FTR'] == 'H' else 0.5 if row['FTR'] == 'D' else 0.0)
                else:
                    results.append(1.0 if row['FTR'] == 'A' else 0.5 if row['FTR'] == 'D' else 0.0)
        
        return results
    
    def _get_avg_goals(self, historical: pd.DataFrame, team: str, venue: str = 'all') -> float:
        """Get average goals scored by team."""
        if venue == 'home':
            matches = historical[historical['HomeTeam'] == team]
            if len(matches) == 0:
                return 1.0
            return matches['FTHG'].mean()
        elif venue == 'away':
            matches = historical[historical['AwayTeam'] == team]
            if len(matches) == 0:
                return 1.0
            return matches['FTAG'].mean()
        else:
            home_goals = historical[historical['HomeTeam'] == team]['FTHG'].sum()
            away_goals = historical[historical['AwayTeam'] == team]['FTAG'].sum()
            total_matches = len(historical[
                (historical['HomeTeam'] == team) | (historical['AwayTeam'] == team)
            ])
            return (home_goals + away_goals) / total_matches if total_matches > 0 else 1.0
    
    def _get_avg_goals_conceded(self, historical: pd.DataFrame, team: str, venue: str = 'all') -> float:
        """Get average goals conceded by team."""
        if venue == 'home':
            matches = historical[historical['HomeTeam'] == team]
            if len(matches) == 0:
                return 1.0
            return matches['FTAG'].mean()
        elif venue == 'away':
            matches = historical[historical['AwayTeam'] == team]
            if len(matches) == 0:
                return 1.0
            return matches['FTHG'].mean()
        else:
            home_conceded = historical[historical['HomeTeam'] == team]['FTAG'].sum()
            away_conceded = historical[historical['AwayTeam'] == team]['FTHG'].sum()
            total_matches = len(historical[
                (historical['HomeTeam'] == team) | (historical['AwayTeam'] == team)
            ])
            return (home_conceded + away_conceded) / total_matches if total_matches > 0 else 1.0
    
    def _get_goals_momentum(self, historical: pd.DataFrame, team: str, venue: str = 'home', 
                            conceded: bool = False, n: int = 5) -> float:
        """Get goal momentum (recent average vs overall average).
        Positive means scoring more recently, negative means scoring less.
        
        Args:
            historical: Historical matches
            team: Team name
            venue: 'home', 'away', or 'all'
            conceded: If True, calculate for goals conceded
            n: Number of recent matches to consider
        """
        if venue == 'home':
            team_matches = historical[historical['HomeTeam'] == team].tail(n)
            if len(team_matches) == 0:
                return 0.0
            if conceded:
                recent_avg = team_matches['FTAG'].mean()
                overall_avg = self._get_avg_goals_conceded(historical, team, venue='home')
            else:
                recent_avg = team_matches['FTHG'].mean()
                overall_avg = self._get_avg_goals(historical, team, venue='home')
        elif venue == 'away':
            team_matches = historical[historical['AwayTeam'] == team].tail(n)
            if len(team_matches) == 0:
                return 0.0
            if conceded:
                recent_avg = team_matches['FTHG'].mean()
                overall_avg = self._get_avg_goals_conceded(historical, team, venue='away')
            else:
                recent_avg = team_matches['FTAG'].mean()
                overall_avg = self._get_avg_goals(historical, team, venue='away')
        else:
            team_matches = historical[
                (historical['HomeTeam'] == team) | (historical['AwayTeam'] == team)
            ].tail(n)
            if len(team_matches) == 0:
                return 0.0
            if conceded:
                goals = []
                for _, match in team_matches.iterrows():
                    if match['HomeTeam'] == team:
                        goals.append(match['FTAG'])
                    else:
                        goals.append(match['FTHG'])
                recent_avg = np.mean(goals) if goals else 1.0
            else:
                goals = []
                for _, match in team_matches.iterrows():
                    if match['HomeTeam'] == team:
                        goals.append(match['FTHG'])
                    else:
                        goals.append(match['FTAG'])
                recent_avg = np.mean(goals) if goals else 1.0
            overall_avg = self._get_avg_goals(historical, team, venue='all') if not conceded else \
                          self._get_avg_goals_conceded(historical, team, venue='all')
        
        return recent_avg - overall_avg
    
    def _get_goal_difference_trend(self, historical: pd.DataFrame, team: str, n: int = 5) -> float:
        """Get goal difference trend (recent goal difference vs overall).
        Positive means improving, negative means declining.
        
        Args:
            historical: Historical matches
            team: Team name
            n: Number of recent matches to consider
        """
        team_matches = historical[
            (historical['HomeTeam'] == team) | (historical['AwayTeam'] == team)
        ].tail(n)
        
        if len(team_matches) == 0:
            return 0.0
        
        recent_gd = []
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team:
                gd = match['FTHG'] - match['FTAG']
            else:
                gd = match['FTAG'] - match['FTHG']
            recent_gd.append(gd)
        
        recent_avg_gd = np.mean(recent_gd) if recent_gd else 0.0
        
        # Overall goal difference
        all_matches = historical[
            (historical['HomeTeam'] == team) | (historical['AwayTeam'] == team)
        ]
        if len(all_matches) == 0:
            return 0.0
        
        overall_gd = []
        for _, match in all_matches.iterrows():
            if match['HomeTeam'] == team:
                gd = match['FTHG'] - match['FTAG']
            else:
                gd = match['FTAG'] - match['FTHG']
            overall_gd.append(gd)
        
        overall_avg_gd = np.mean(overall_gd) if overall_gd else 0.0
        
        return recent_avg_gd - overall_avg_gd

