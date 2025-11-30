"""
Data collection module for Premier League match data.
Uses the premier-league package (https://pypi.org/project/premier-league/) 
for comprehensive Premier League data access.
"""

import pandas as pd
import os
import requests
from typing import Optional, Dict, List
from datetime import datetime
try:
    from premier_league import MatchStatistics  # type: ignore
    PREMIER_LEAGUE_AVAILABLE = True
except ImportError:
    PREMIER_LEAGUE_AVAILABLE = False
    MatchStatistics = None  # type: ignore
    print("Warning: premier-league package not installed. Install with: pip install premier-league")


class PremierLeagueDataCollector:
    """Collects Premier League match data using the premier-league package."""
    
    def __init__(self, data_dir: str = "data", db_filename: str = "premier_league.db"):
        """Initialize the data collector.
        
        Args:
            data_dir: Directory to store collected data and database
            db_filename: Name of the SQLite database file
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.db_filename = db_filename
        self.stats = None
        
        if PREMIER_LEAGUE_AVAILABLE:
            # Initialize MatchStatistics with custom database location
            db_path = os.path.join(data_dir, db_filename)
            self.stats = MatchStatistics(db_filename=db_filename, db_directory=data_dir)
            print(f"Initialized Premier League data collector. Database: {db_path}")
        else:
            print("premier-league package not available. Please install it first.")
    
    def update_data(self):
        """Update the database with the latest Premier League match data.
        
        This method fetches and stores the latest available match data.
        """
        if not PREMIER_LEAGUE_AVAILABLE:
            raise ImportError("premier-league package is required. Install with: pip install premier-league")
        
        print("Updating Premier League data...")
        self.stats.update_data_set()
        print("Data update complete!")
    
    def get_all_matches(self, update_first: bool = False, seasons: Optional[List[str]] = None) -> pd.DataFrame:
        """Get all Premier League matches from the database.
        
        Args:
            update_first: If True, update data before retrieving
            seasons: List of seasons to retrieve (e.g., ["2023-2024", "2022-2023"]). 
                    If None, gets all available seasons.
            
        Returns:
            DataFrame with all match data
        """
        if not PREMIER_LEAGUE_AVAILABLE:
            raise ImportError("premier-league package is required. Install with: pip install premier-league")
        
        if update_first:
            self.update_data()
        
        # Get games - use get_games_before_date as primary method since it doesn't require match_week
        all_games = []
        
        try:
            # Get games before current date (most reliable method)
            all_games = self.stats.get_games_before_date(datetime.now(), limit=5000)
            if all_games:
                print(f"Retrieved {len(all_games)} games from database")
        except Exception as e:
            print(f"Warning: Could not retrieve games: {e}")
            # Fallback: try to get games by season (requires match_week, so we skip for now)
            if seasons is None:
                current_year = datetime.now().year
                seasons = [f"{year}-{year+1}" for year in range(current_year - 3, current_year + 1)]
            
            for season in seasons:
                try:
                    # Note: get_games_by_season requires match_week parameter
                    # We skip this method and use get_games_before_date instead
                    pass
                except Exception as e:
                    continue
        
        # Convert to DataFrame format expected by our feature engineering
        match_data = []
        for game in all_games:
            # Handle both dict and object types
            if not isinstance(game, dict):
                # Convert to dict if it's an object
                try:
                    game = game.__dict__ if hasattr(game, '__dict__') else dict(game)
                except:
                    continue
            
            # Handle nested structures - extract values from nested dicts
            def extract_value(obj, *keys):
                """Extract value from nested dict structure."""
                if isinstance(obj, dict):
                    for key in keys:
                        if key in obj:
                            val = obj[key]
                            # If it's a nested dict, try to get 'name' or the value itself
                            if isinstance(val, dict):
                                return val.get('name', val.get('Name', str(val)))
                            return val
                return None
            
            # Get scores - try multiple field names
            home_score = (extract_value(game, 'home_score', 'FTHG', 'home_goals', 'homeScore') or 
                         game.get('home_score') or game.get('FTHG') or game.get('home_goals') or 
                         game.get('homeScore', 0))
            away_score = (extract_value(game, 'away_score', 'FTAG', 'away_goals', 'awayScore') or 
                         game.get('away_score') or game.get('FTAG') or game.get('away_goals') or 
                         game.get('awayScore', 0))
            
            # Skip if scores are None or invalid
            if home_score is None or away_score is None:
                continue
            
            try:
                home_score = int(home_score)
                away_score = int(away_score)
            except (ValueError, TypeError):
                continue
            
            # Determine result
            if home_score > away_score:
                ftr = 'H'
            elif away_score > home_score:
                ftr = 'A'
            else:
                ftr = 'D'
            
            # Get date and team names - handle nested structures
            date = (extract_value(game, 'date', 'Date', 'match_date', 'game_date') or 
                   game.get('date') or game.get('Date') or game.get('match_date') or 
                   game.get('game_date', ''))
            
            home_team = (extract_value(game, 'home_team', 'HomeTeam', 'home', 'homeTeam') or 
                        game.get('home_team') or game.get('HomeTeam') or game.get('home') or 
                        game.get('homeTeam', ''))
            away_team = (extract_value(game, 'away_team', 'AwayTeam', 'away', 'awayTeam') or 
                        game.get('away_team') or game.get('AwayTeam') or game.get('away') or 
                        game.get('awayTeam', ''))
            
            # Ensure team names are strings, not dicts
            if isinstance(home_team, dict):
                home_team = home_team.get('name', str(home_team))
            if isinstance(away_team, dict):
                away_team = away_team.get('name', str(away_team))
            
            # Skip if essential fields are missing
            if not date or not home_team or not away_team:
                continue
            
            match_data.append({
                'Date': str(date),
                'HomeTeam': str(home_team),
                'AwayTeam': str(away_team),
                'FTHG': home_score,
                'FTAG': away_score,
                'FTR': ftr
            })
        
        df = pd.DataFrame(match_data)
        if len(df) > 0:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])  # Remove rows with invalid dates
            df = df.sort_values('Date').reset_index(drop=True)
            print(f"Retrieved {len(df)} matches")
        else:
            print("Warning: No matches found. Database may be empty. Try running update_data() first.")
        
        return df
    
    def get_team_matches(self, team_name: str) -> pd.DataFrame:
        """Get all matches for a specific team.
        
        Args:
            team_name: Name of the team
            
        Returns:
            DataFrame with team's match data
        """
        if not PREMIER_LEAGUE_AVAILABLE:
            raise ImportError("premier-league package is required. Install with: pip install premier-league")
        
        games = self.stats.get_team_games(team_name)
        
        match_data = []
        for game in games:
            # Handle different possible field names
            home_score = game.get('home_score') or game.get('FTHG') or game.get('home_goals', 0)
            away_score = game.get('away_score') or game.get('FTAG') or game.get('away_goals', 0)
            
            if home_score is None or away_score is None:
                continue
            
            if home_score > away_score:
                ftr = 'H'
            elif away_score > home_score:
                ftr = 'A'
            else:
                ftr = 'D'
            
            date = game.get('date') or game.get('Date') or game.get('match_date', '')
            home_team = game.get('home_team') or game.get('HomeTeam') or game.get('home', '')
            away_team = game.get('away_team') or game.get('AwayTeam') or game.get('away', '')
            
            match_data.append({
                'Date': date,
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'FTHG': int(home_score),
                'FTAG': int(away_score),
                'FTR': ftr
            })
        
        df = pd.DataFrame(match_data)
        if len(df) > 0:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
        
        return df
    
    def create_ml_dataset(self, output_path: str = "data/premier_league_ml_data.csv", 
                         lag: int = 10, update_first: bool = False):
        """Create a machine learning ready dataset using the premier-league package.
        
        This uses the built-in create_dataset method which creates features
        from the last N games (lag parameter).
        
        Args:
            output_path: Path where the dataset will be saved
            lag: Number of previous games to aggregate (default: 10)
            update_first: If True, update data before creating dataset
        """
        if not PREMIER_LEAGUE_AVAILABLE:
            raise ImportError("premier-league package is required. Install with: pip install premier-league")
        
        if update_first:
            self.update_data()
        
        print(f"Creating ML dataset with lag={lag}...")
        self.stats.create_dataset(output_path, lag=lag)
        print(f"ML dataset saved to {output_path}")
        
        # Load and return the dataset
        return pd.read_csv(output_path)
    
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """Load match data from a CSV file.
        
        Expected CSV format:
        Date, HomeTeam, AwayTeam, FTHG (Full Time Home Goals), FTAG (Full Time Away Goals),
        FTR (Full Time Result: H/A/D), HTHG, HTAG, HTR, etc.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with match data
        """
        df = pd.read_csv(filepath)
        return df
    
    def fetch_from_api(self, api_key: Optional[str] = None, 
                      season: str = "2023") -> pd.DataFrame:
        """Fetch data from football-data.org API (requires free API key).
        
        Args:
            api_key: API key for football-data.org (optional)
            season: Season year (e.g., "2023" for 2023-24 season)
            
        Returns:
            DataFrame with match data
        """
        if not api_key:
            print("Note: API key not provided. Using sample data structure.")
            return self._create_sample_structure()
        
        base_url = "https://api.football-data.org/v4"
        headers = {"X-Auth-Token": api_key}
        
        # Fetch Premier League matches
        url = f"{base_url}/competitions/PL/matches"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            matches = data.get('matches', [])
            return self._parse_api_response(matches)
        else:
            print(f"API Error: {response.status_code}")
            return self._create_sample_structure()
    
    def _parse_api_response(self, matches: List[Dict]) -> pd.DataFrame:
        """Parse API response into DataFrame format."""
        data = []
        for match in matches:
            if match['status'] == 'FINISHED':
                data.append({
                    'Date': match['utcDate'][:10],
                    'HomeTeam': match['homeTeam']['name'],
                    'AwayTeam': match['awayTeam']['name'],
                    'FTHG': match['score']['fullTime']['home'],
                    'FTAG': match['score']['fullTime']['away'],
                    'FTR': 'H' if match['score']['fullTime']['home'] > match['score']['fullTime']['away'] 
                           else 'A' if match['score']['fullTime']['away'] > match['score']['fullTime']['home'] 
                           else 'D'
                })
        return pd.DataFrame(data)
    
    def _create_sample_structure(self) -> pd.DataFrame:
        """Create a sample DataFrame structure for reference."""
        return pd.DataFrame(columns=[
            'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
            'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF',
            'HC', 'AC', 'HY', 'AY', 'HR', 'AR'
        ])
    
    def save_data(self, df: pd.DataFrame, filename: str = "premier_league_matches.csv"):
        """Save collected data to CSV."""
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath


if __name__ == "__main__":
    collector = PremierLeagueDataCollector()
    
    # Example: Update data first (downloads latest Premier League data)
    # collector.update_data()
    
    # Example: Get all matches
    # df = collector.get_all_matches(update_first=True)
    
    # Example: Get matches for specific team
    # df = collector.get_team_matches("Arsenal")
    
    # Example: Create ML-ready dataset (uses built-in feature engineering)
    # df = collector.create_ml_dataset("data/ml_data.csv", lag=10, update_first=True)
    
    # Example: Load from CSV (fallback option)
    # df = collector.load_from_csv("path/to/your/data.csv")
    
    print("Data collector ready!")
    print("Primary method: Use get_all_matches() or create_ml_dataset() with premier-league package")
    print("Fallback: Use load_from_csv() for custom CSV files")

