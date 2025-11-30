# Premier League ML Prediction

A machine learning project to predict the outcomes of Premier League football matches using historical data and various ML algorithms.

## Features

- **Data Collection**: Automatic data collection using the [premier-league](https://pypi.org/project/premier-league/) package with SQLite database storage
- **Feature Engineering**: Comprehensive feature extraction including:
  - Team form and statistics
  - Head-to-head records
  - Home/away performance
  - Goal statistics
  - Recent match form
- **Multiple ML Models**: 
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - Logistic Regression
- **Model Evaluation**: Detailed evaluation with confusion matrices and feature importance
- **Prediction Interface**: Easy-to-use prediction API for new matches

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Premier-League-ML-Prediction.git
cd Premier-League-ML-Prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Source

This project uses the **[premier-league](https://pypi.org/project/premier-league/)** Python package for comprehensive Premier League data access. The package provides:

- Automatic data updates
- SQLite database storage
- Match statistics and game data
- Built-in ML dataset creation
- Access to historical and current season data

The package automatically initializes a local SQLite database on first use and can be updated with the latest match data.

### Alternative Data Sources

If you prefer to use your own data, the system also supports:
- CSV files with match data
- Custom data formats (with minor modifications)

The expected CSV format includes:
- `Date`: Match date
- `HomeTeam`: Home team name
- `AwayTeam`: Away team name
- `FTHG`: Full-time home goals
- `FTAG`: Full-time away goals
- `FTR`: Full-time result (H=Home win, D=Draw, A=Away win)

## Usage

### 1. Prepare Your Data

The easiest way is to use the premier-league package (automatically installed):

```python
from data_collector import PremierLeagueDataCollector

# Initialize collector (creates database automatically)
collector = PremierLeagueDataCollector()

# Update with latest data (downloads from premier-league package)
collector.update_data()

# Get all matches
df = collector.get_all_matches()

# Or create ML-ready dataset directly (uses built-in feature engineering)
df = collector.create_ml_dataset("data/ml_data.csv", lag=10, update_first=True)
```

**Alternative**: If you have your own CSV file:

```python
collector = PremierLeagueDataCollector()
df = collector.load_from_csv("path/to/your/data.csv")
```

### 2. Train the Model

**Easy way** - Just run the training script (it handles everything automatically):

```bash
python train_model.py
```

This will:
1. Update Premier League data automatically
2. Load all available matches
3. Engineer features
4. Train multiple models
5. Select and save the best model

**Manual way** - For more control:

```python
from train_model import PremierLeaguePredictor
from data_collector import PremierLeagueDataCollector

# Initialize and update data
collector = PremierLeagueDataCollector()
collector.update_data()  # Download latest data
df = collector.get_all_matches()

# Initialize predictor
predictor = PremierLeaguePredictor()

# Prepare data
X_train, X_test, y_train, y_test, feature_names = predictor.prepare_data(df)

# Train models
results = predictor.train_models(X_train, y_train, X_test, y_test)

# Evaluate best model
predictor.evaluate_model(predictor.best_model, X_test, y_test, predictor.best_model_name)

# Save model
predictor.save_model(predictor.best_model, predictor.scaler, feature_names)
```

### 3. Make Predictions

Predict the outcome of a match:

```python
from predict import MatchPredictor
from data_collector import PremierLeagueDataCollector

# Load historical data
collector = PremierLeagueDataCollector()
df = collector.get_all_matches()  # Uses premier-league package

# Initialize predictor (loads trained model)
predictor = MatchPredictor("models/best_model.joblib")

# Predict a single match
prediction = predictor.predict_match("Arsenal", "Chelsea", df)
predictor.print_prediction(prediction)

# Predict multiple matches
matches = [
    ("Arsenal", "Chelsea"),
    ("Manchester United", "Liverpool"),
    ("Manchester City", "Tottenham")
]
predictions = predictor.predict_multiple_matches(matches, df)
for pred in predictions:
    predictor.print_prediction(pred)
```

### 4. Evaluate the Model

Generate evaluation metrics and visualizations:

```python
python evaluate.py
```

This will create:
- Confusion matrix plot
- Feature importance plot
- Detailed classification report

## Project Structure

```
Premier-League-ML-Prediction/
├── data/                      # Data directory (auto-created)
│   ├── premier_league.db      # SQLite database (auto-created by premier-league package)
│   └── premier_league_matches.csv  # Optional CSV exports
├── models/                    # Saved models (created after training)
│   └── best_model.joblib
├── data_collector.py          # Data collection module (uses premier-league package)
├── feature_engineering.py     # Feature engineering module
├── train_model.py             # Model training script
├── predict.py                 # Prediction script
├── evaluate.py                # Evaluation and visualization
├── example_usage.py           # Example workflow
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

## Features Explained

The model uses the following features:

1. **Team Statistics**:
   - Win/draw/loss rates
   - Average goals scored and conceded
   - Home and away specific statistics

2. **Recent Form**:
   - Last 5 matches performance
   - Home/away specific form

3. **Head-to-Head**:
   - Historical matchups between teams
   - Goal difference in previous meetings

4. **Goal Statistics**:
   - Average goals scored at home/away
   - Average goals conceded at home/away

5. **League Position**:
   - Points accumulated
   - Points difference between teams

## Model Performance

The models are evaluated using:
- **Accuracy**: Overall prediction accuracy
- **Per-class Accuracy**: Accuracy for each outcome (Home Win, Draw, Away Win)
- **Cross-validation**: 5-fold cross-validation for robust evaluation

Typical performance:
- Overall accuracy: ~50-55% (better than random 33%)
- Home win prediction: Usually highest accuracy
- Draw prediction: Most challenging (lowest accuracy)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See LICENSE file for details.

## Acknowledgments

- **[premier-league](https://pypi.org/project/premier-league/)** package by Michael Li for comprehensive Premier League data access
- Premier League for the exciting matches to predict!

## Notes

- The model requires sufficient historical data (at least one full season) for good performance
- Early matches in a season may have limited features due to lack of historical data
- Draw predictions are inherently more difficult due to their lower frequency
- Model performance may vary depending on data quality and feature engineering
