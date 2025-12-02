import pandas as pd
import pytest


@pytest.fixture
def sample_matches_df():
    """Small but diverse set of historical matches for testing."""
    data = [
        {"Date": "2024-01-01", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "FTHG": 2, "FTAG": 1, "FTR": "H"},
        {"Date": "2024-01-05", "HomeTeam": "Liverpool", "AwayTeam": "Manchester City", "FTHG": 1, "FTAG": 1, "FTR": "D"},
        {"Date": "2024-01-10", "HomeTeam": "Chelsea", "AwayTeam": "Arsenal", "FTHG": 0, "FTAG": 3, "FTR": "A"},
        {"Date": "2024-01-15", "HomeTeam": "Manchester City", "AwayTeam": "Liverpool", "FTHG": 2, "FTAG": 2, "FTR": "D"},
        {"Date": "2024-01-20", "HomeTeam": "Arsenal", "AwayTeam": "Liverpool", "FTHG": 1, "FTAG": 0, "FTR": "H"},
        {"Date": "2024-01-25", "HomeTeam": "Chelsea", "AwayTeam": "Manchester City", "FTHG": 1, "FTAG": 2, "FTR": "A"},
        {"Date": "2024-01-30", "HomeTeam": "Liverpool", "AwayTeam": "Arsenal", "FTHG": 0, "FTAG": 1, "FTR": "A"},
        {"Date": "2024-02-04", "HomeTeam": "Manchester City", "AwayTeam": "Chelsea", "FTHG": 3, "FTAG": 1, "FTR": "H"},
        {"Date": "2024-02-09", "HomeTeam": "Arsenal", "AwayTeam": "Manchester City", "FTHG": 2, "FTAG": 2, "FTR": "D"},
        {"Date": "2024-02-14", "HomeTeam": "Chelsea", "AwayTeam": "Liverpool", "FTHG": 0, "FTAG": 1, "FTR": "A"},
        {"Date": "2024-02-19", "HomeTeam": "Liverpool", "AwayTeam": "Chelsea", "FTHG": 2, "FTAG": 2, "FTR": "D"},
        {"Date": "2024-02-24", "HomeTeam": "Manchester City", "AwayTeam": "Arsenal", "FTHG": 1, "FTAG": 0, "FTR": "H"},
    ]
    return pd.DataFrame(data)


