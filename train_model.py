"""
Model training script for Premier League match prediction.
Trains multiple ML models and selects the best one.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os
from feature_engineering import FeatureEngineer
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost (optional - requires OpenMP on macOS)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception as e:
    # Catch any exception (ImportError, OSError, XGBoostError, etc.)
    XGBOOST_AVAILABLE = False
    xgb = None
    if "OpenMP" in str(e) or "libomp" in str(e):
        print("Warning: XGBoost not available - OpenMP runtime missing.")
        print("  Install with: brew install libomp")
    else:
        print(f"Warning: XGBoost not available: {type(e).__name__}")
    print("Continuing without XGBoost model...")


class PremierLeaguePredictor:
    """Main class for training and using Premier League prediction models."""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize the predictor.
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2):
        """Prepare data for training.
        
        Args:
            df: DataFrame with match data
            test_size: Proportion of data to use for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Create features
        features_df = self.feature_engineer.create_features(df)
        
        # Remove rows with NaN values (first few matches won't have enough history)
        features_df = features_df.dropna()
        
        # Separate features and target
        X = features_df.drop('target', axis=1)
        y = features_df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns.tolist()
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models and select the best one.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
        """
        print("Training multiple models...")
        
        # Calculate class weights to handle imbalance (especially draws)
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        print(f"Class weights: {class_weight_dict}")
        
        # Define models with improved hyperparameters and class weights
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight=class_weight_dict,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                min_samples_split=5,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=2000,
                class_weight=class_weight_dict,
                random_state=42,
                multi_class='multinomial',
                solver='lbfgs',
                C=1.0
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                min_child_weight=3,
                scale_pos_weight=1,
                random_state=42,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
        
        # Train and evaluate each model
        results = {}
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'model': model,
                'train_score': train_score,
                'test_score': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  Train Accuracy: {train_score:.4f}")
            print(f"  Test Accuracy: {test_score:.4f}")
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            self.models[name] = model
        
        # Create ensemble voting classifier from top 3 models
        sorted_models = sorted(results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)
        top_3_models = sorted_models[:3]
        
        print(f"\nCreating ensemble from top 3 models...")
        ensemble_estimators = [(name, results[name]['model']) for name, _ in top_3_models]
        ensemble = VotingClassifier(estimators=ensemble_estimators, voting='soft')
        ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        ensemble_train_score = ensemble.score(X_train, y_train)
        ensemble_test_score = ensemble.score(X_test, y_test)
        ensemble_cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
        
        print(f"  Ensemble Train Accuracy: {ensemble_train_score:.4f}")
        print(f"  Ensemble Test Accuracy: {ensemble_test_score:.4f}")
        print(f"  Ensemble CV Accuracy: {ensemble_cv_scores.mean():.4f} (+/- {ensemble_cv_scores.std() * 2:.4f})")
        
        # Select best model (either best single model or ensemble)
        best_single_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
        best_single_cv = results[best_single_name]['cv_mean']
        
        if ensemble_cv_scores.mean() > best_single_cv:
            self.best_model = ensemble
            self.best_model_name = 'Ensemble (Voting)'
            print(f"\n{'='*50}")
            print(f"Best Model: Ensemble (Voting)")
            print(f"CV Accuracy: {ensemble_cv_scores.mean():.4f}")
            print(f"{'='*50}")
        else:
            self.best_model = results[best_single_name]['model']
            self.best_model_name = best_single_name
            print(f"\n{'='*50}")
            print(f"Best Model: {best_single_name}")
            print(f"CV Accuracy: {best_single_cv:.4f}")
            print(f"{'='*50}")
        
        # Store ensemble in results for reference
        results['Ensemble'] = {
            'model': ensemble,
            'train_score': ensemble_train_score,
            'test_score': ensemble_test_score,
            'cv_mean': ensemble_cv_scores.mean(),
            'cv_std': ensemble_cv_scores.std()
        }
        
        return results
    
    def evaluate_model(self, model, X_test, y_test, model_name: str = "Model"):
        """Evaluate a trained model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
        """
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{model_name} Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Home Win', 'Draw', 'Away Win']))
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return accuracy, y_pred
    
    def save_model(self, model, scaler, feature_names, filename: str = "best_model.joblib"):
        """Save the best model and scaler.
        
        Args:
            model: Trained model
            scaler: Fitted scaler
            feature_names: List of feature names
            filename: Name of the file to save
        """
        filepath = os.path.join(self.models_dir, filename)
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filename: str = "best_model.joblib"):
        """Load a saved model.
        
        Args:
            filename: Name of the file to load
        """
        filepath = os.path.join(self.models_dir, filename)
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        print(f"Model loaded from {filepath}")
        return model_data


def main():
    """Main training function."""
    from data_collector import PremierLeagueDataCollector
    
    # Initialize components
    collector = PremierLeagueDataCollector()
    predictor = PremierLeaguePredictor()
    
    print("="*60)
    print("Premier League ML Model Training")
    print("="*60)
    
    # Update and load data using premier-league package
    print("\n[Step 1] Updating Premier League data...")
    try:
        collector.update_data()
    except Exception as e:
        print(f"Warning: Could not update data: {e}")
        print("Continuing with existing database...")
    
    print("\n[Step 2] Loading match data...")
    try:
        df = collector.get_all_matches(update_first=False)
        
        if df.empty:
            print("No data found. Please ensure the database has been updated.")
            print("Run: collector.update_data() first")
            return
        
        print(f"Loaded {len(df)} matches")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nFallback: Try loading from CSV file")
        print("Example: df = collector.load_from_csv('data/premier_league_matches.csv')")
        return
    
    # Prepare data
    print("\n[Step 3] Engineering features...")
    X_train, X_test, y_train, y_test, feature_names = predictor.prepare_data(df)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train models
    print("\n[Step 4] Training models...")
    results = predictor.train_models(X_train, y_train, X_test, y_test)
    
    # Evaluate best model
    print("\n[Step 5] Evaluating best model...")
    predictor.evaluate_model(predictor.best_model, X_test, y_test, predictor.best_model_name)
    
    # Save model
    print("\n[Step 6] Saving model...")
    predictor.save_model(predictor.best_model, predictor.scaler, feature_names)
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()

