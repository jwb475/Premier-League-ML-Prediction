import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from feature_engineering import FeatureEngineer


def test_feature_engineering_creates_expected_features(sample_matches_df):
    engineer = FeatureEngineer()
    features = engineer.create_features(sample_matches_df)
    features = features.dropna()

    assert not features.empty, "Feature dataframe should not be empty after dropna"
    required_columns = [
        "home_recent_form_3",
        "home_goals_momentum",
        "home_gd_trend",
        "away_recent_form_10",
        "momentum_diff",
    ]
    for column in required_columns:
        assert column in features.columns, f"{column} should be present in features"

    # Ensure no inf/nan values remain
    assert np.isfinite(features.values).all(), "Features should not contain inf/nan values"

