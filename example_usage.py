"""
Example usage script demonstrating the complete workflow.
This script shows how to use the Premier League prediction system.
"""

from data_collector import PremierLeagueDataCollector
from feature_engineering import FeatureEngineer
from train_model import PremierLeaguePredictor
from predict import MatchPredictor
import os


def example_workflow():
    """Complete example workflow."""
    
    print("="*60)
    print("Premier League ML Prediction - Example Workflow")
    print("="*60)
    
    # Step 1: Initialize and Update Data
    print("\n[Step 1] Initializing data collector...")
    collector = PremierLeagueDataCollector()
    
    print("\n[Step 2] Updating Premier League data...")
    print("This will download the latest match data from the premier-league package.")
    # collector.update_data()  # Uncomment to update data
    
    # Step 2: Load Data
    print("\n[Step 3] Loading match data...")
    # Option 1: Get all matches (recommended)
    # df = collector.get_all_matches(update_first=False)
    
    # Option 2: Create ML-ready dataset (uses built-in feature engineering)
    # df = collector.create_ml_dataset("data/ml_data.csv", lag=10, update_first=False)
    
    # Option 3: Get specific team matches
    # df = collector.get_team_matches("Arsenal")
    
    # Option 4: Load from CSV (fallback)
    # df = collector.load_from_csv("data/premier_league_matches.csv")
    
    print("Note: Uncomment the data loading code above to continue.")
    
    # Uncomment when you have data:
    # # Step 3: Train Model
    # print("\n[Step 4] Training model...")
    # predictor = PremierLeaguePredictor()
    # X_train, X_test, y_train, y_test, feature_names = predictor.prepare_data(df)
    # results = predictor.train_models(X_train, y_train, X_test, y_test)
    # predictor.evaluate_model(predictor.best_model, X_test, y_test, predictor.best_model_name)
    # predictor.save_model(predictor.best_model, predictor.scaler, feature_names)
    # 
    # # Step 4: Make Predictions
    # print("\n[Step 5] Making predictions...")
    # match_predictor = MatchPredictor("models/best_model.joblib")
    # 
    # # Predict a single match
    # print("\nPredicting: Arsenal vs Chelsea")
    # prediction = match_predictor.predict_match("Arsenal", "Chelsea", df)
    # match_predictor.print_prediction(prediction)
    # 
    # # Predict multiple matches
    # print("\nPredicting multiple matches...")
    # matches = [
    #     ("Arsenal", "Chelsea"),
    #     ("Manchester United", "Liverpool"),
    #     ("Manchester City", "Tottenham"),
    #     ("Newcastle", "Brighton")
    # ]
    # predictions = match_predictor.predict_multiple_matches(matches, df)
    # for pred in predictions:
    #     match_predictor.print_prediction(pred)
    
    print("\n" + "="*60)
    print("Example workflow complete!")
    print("="*60)
    print("\nTo use this system:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run: python train_model.py (this will update data and train automatically)")
    print("3. Or uncomment the code sections above and run: python example_usage.py")


def example_feature_engineering():
    """Example of feature engineering."""
    print("\n" + "="*60)
    print("Feature Engineering Example")
    print("="*60)
    
    # This would require actual data
    print("\nFeature engineering creates the following features:")
    print("- Team win/draw/loss rates")
    print("- Average goals scored and conceded")
    print("- Recent form (last 5 matches)")
    print("- Head-to-head statistics")
    print("- Home/away specific performance")
    print("- Goal difference statistics")
    print("- League points and position")
    
    print("\nTo see features in action, load your data and run:")
    print("from feature_engineering import FeatureEngineer")
    print("engineer = FeatureEngineer()")
    print("features_df = engineer.create_features(your_dataframe)")


if __name__ == "__main__":
    example_workflow()
    example_feature_engineering()

