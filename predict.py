"""
Prediction script for Premier League matches.
Uses trained model to predict match outcomes.
"""

import pandas as pd
import numpy as np
import joblib
import os
from feature_engineering import FeatureEngineer
from data_collector import PremierLeagueDataCollector


class MatchPredictor:
    """Class for making predictions on new matches."""
    
    def __init__(self, model_path: str = "models/best_model.joblib"):
        """Initialize the predictor.
        
        Args:
            model_path: Path to saved model file
        """
        self.model_path = model_path
        self.model_data = None
        self.feature_engineer = FeatureEngineer()
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Please train the model first."
            )
        self.model_data = joblib.load(self.model_path)
        print(f"Model loaded from {self.model_path}")
    
    def predict_match(self, home_team: str, away_team: str, historical_data: pd.DataFrame):
        """Predict the outcome of a single match.
        
        Args:
            home_team: Name of home team
            away_team: Name of away team
            historical_data: DataFrame with historical matches (for feature engineering)
            
        Returns:
            Dictionary with predictions
        """
        # Create a temporary row for this match
        temp_match = pd.DataFrame([{
            'Date': pd.Timestamp.now(),
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'FTHG': 0,  # Placeholder
            'FTAG': 0,  # Placeholder
            'FTR': 'D'  # Placeholder
        }])
        
        # Combine with historical data
        combined_data = pd.concat([historical_data, temp_match], ignore_index=True)
        
        # Create features
        features_df = self.feature_engineer.create_features(combined_data)
        
        # Get features for the last match (our prediction match)
        match_features = features_df.iloc[-1:].drop('target', axis=1)
        
        # Ensure feature order matches training
        feature_names = self.model_data['feature_names']
        match_features = match_features[feature_names]
        
        # Scale features
        scaled_features = self.model_data['scaler'].transform(match_features)
        
        # Predict
        model = self.model_data['model']
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        
        # Map prediction to result with team names
        if prediction == 0:  # Home Win
            predicted_result = home_team
        elif prediction == 2:  # Away Win
            predicted_result = away_team
        else:  # Draw
            predicted_result = 'Draw'
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_result': predicted_result,
            'prediction_code': prediction,  # Keep original code (0=Home, 1=Draw, 2=Away)
            'probabilities': {
                'Home Win': probabilities[0],
                'Draw': probabilities[1],
                'Away Win': probabilities[2]
            },
            'team_probabilities': {
                home_team: probabilities[0],
                'Draw': probabilities[1],
                away_team: probabilities[2]
            }
        }
    
    def predict_multiple_matches(self, matches: list, historical_data: pd.DataFrame):
        """Predict outcomes for multiple matches.
        
        Args:
            matches: List of tuples (home_team, away_team)
            historical_data: DataFrame with historical matches
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        for home_team, away_team in matches:
            pred = self.predict_match(home_team, away_team, historical_data)
            predictions.append(pred)
        return predictions
    
    def print_prediction(self, prediction: dict):
        """Print a formatted prediction.
        
        Args:
            prediction: Prediction dictionary
        """
        home_team = prediction['home_team']
        away_team = prediction['away_team']
        predicted_result = prediction['predicted_result']
        
        print(f"\n{'='*50}")
        print(f"Match: {home_team} vs {away_team}")
        
        # Show predicted winner with team name
        if predicted_result == 'Draw':
            print(f"Predicted Result: Draw")
        else:
            print(f"Predicted Winner: {predicted_result}")
        
        print(f"\nProbabilities:")
        # Show probabilities with team names
        if 'team_probabilities' in prediction:
            for team, prob in prediction['team_probabilities'].items():
                print(f"  {team}: {prob:.2%}")
        else:
            # Fallback to original format
            for outcome, prob in prediction['probabilities'].items():
                print(f"  {outcome}: {prob:.2%}")
        print(f"{'='*50}")


def main():
    """Main prediction function."""
    from data_collector import PremierLeagueDataCollector
    
    # Initialize
    collector = PremierLeagueDataCollector()
    predictor = MatchPredictor()
    
    # Load historical data
    print("="*60)
    print("Premier League Match Prediction")
    print("="*60)
    print("\n[Step 1] Loading historical data...")
    try:
        df = collector.get_all_matches(update_first=False)
        
        if df.empty:
            print("No data found. Please ensure the database has been updated.")
            return
        
        print(f"Loaded {len(df)} historical matches")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Example predictions
    print("\n[Step 2] Making predictions...")
    
    # Predict a single match
    print("\nExample: Arsenal vs Chelsea")
    try:
        prediction = predictor.predict_match('Arsenal', 'Chelsea', df)
        predictor.print_prediction(prediction)
    except Exception as e:
        print(f"Error making prediction: {e}")
        print("Make sure team names match exactly (case-sensitive)")
    
    # Predict multiple matches
    print("\n" + "="*60)
    print("Predicting multiple matches...")
    print("="*60)
    
    matches = [
        ('Arsenal', 'Chelsea'),
        ('Manchester United', 'Liverpool'),
        ('Manchester City', 'Tottenham'),
        ('Newcastle', 'Brighton')
    ]
    
    try:
        predictions = predictor.predict_multiple_matches(matches, df)
        for pred in predictions:
            predictor.print_prediction(pred)
    except Exception as e:
        print(f"Error making predictions: {e}")
        print("Make sure team names match exactly (case-sensitive)")
    
    print("\n" + "="*60)
    print("Predictions complete!")
    print("="*60)
    print("\nTip: Team names are case-sensitive. Use exact names from the database.")


if __name__ == "__main__":
    main()

