import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_model import PremierLeaguePredictor

def test_prepare_data_returns_scaled_sets(sample_matches_df):
    predictor = PremierLeaguePredictor(models_dir="models")
    X_train, X_test, y_train, y_test, feature_names = predictor.prepare_data(sample_matches_df)

    # Ensure outputs are not empty
    assert len(X_train) > 0 and len(X_test) > 0
    assert len(y_train) > 0 and len(y_test) > 0
    assert len(feature_names) > 0

    # Scaler should produce approximately zero mean on training data
    train_mean = X_train.mean(axis=0)
    assert abs(train_mean).max() < 1e-6