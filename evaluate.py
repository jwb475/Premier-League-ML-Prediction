"""
Evaluation and visualization script for Premier League prediction model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os
from train_model import PremierLeaguePredictor
from data_collector import PremierLeagueDataCollector

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model"):
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Home Win', 'Draw', 'Away Win']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to confusion_matrix.png")
    plt.close()


def plot_feature_importance(model, feature_names, top_n: int = 15):
    """Plot feature importance for tree-based models or coefficients for linear models.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        top_n: Number of top features to display
    """
    # Try feature_importances_ (tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(np.abs(importances))[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("Feature importance plot saved to feature_importance.png")
        plt.close()
    # Try coefficients (linear models like Logistic Regression)
    elif hasattr(model, 'coef_'):
        # For multi-class, take mean absolute coefficient across classes
        if len(model.coef_.shape) > 1:
            coef = np.mean(np.abs(model.coef_), axis=0)
        else:
            coef = np.abs(model.coef_[0])
        
        indices = np.argsort(coef)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), coef[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Average Absolute Coefficient')
        plt.title(f'Top {top_n} Most Important Features (Coefficients)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("Feature importance plot saved to feature_importance.png")
        plt.close()
    else:
        print("Model does not support feature importance visualization.")


def evaluate_predictions(y_true, y_pred, y_proba=None):
    """Comprehensive evaluation of predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
    """
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                               target_names=['Home Win', 'Draw', 'Away Win']))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred)
    
    # Calculate per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class Accuracies:")
    for i, label in enumerate(['Home Win', 'Draw', 'Away Win']):
        print(f"  {label}: {class_accuracies[i]:.4f}")


def main():
    """Main evaluation function."""
    from feature_engineering import FeatureEngineer
    
    print("Model Evaluation")
    print("="*50)
    
    # Load data
    collector = PremierLeagueDataCollector()
    # df = collector.load_from_csv("data/premier_league_matches.csv")
    
    # Load model
    model_path = "models/best_model.joblib"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return
    
    model_data = joblib.load(model_path)
    predictor = PremierLeaguePredictor()
    
    # Load data
    print("\n[Step 1] Loading data...")
    try:
        df = collector.get_all_matches(update_first=False)
        
        if df.empty:
            print("No data found. Please ensure the database has been updated.")
            return
        
        print(f"Loaded {len(df)} matches")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Prepare data using the SAME scaler that was used during training
    print("\n[Step 2] Preparing data...")
    
    # Load the scaler and feature names from the saved model
    saved_scaler = model_data['scaler']
    saved_feature_names = model_data['feature_names']
    
    # Create features (same as training)
    features_df = predictor.feature_engineer.create_features(df)
    features_df = features_df.dropna()
    
    # Separate features and target
    X = features_df.drop('target', axis=1)
    y = features_df['target']
    
    # Split data (same random_state as training to get same split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Ensure feature order matches training
    X_test = X_test[saved_feature_names]
    
    # Use the SAVED scaler (not a new one) to transform test data
    X_test_scaled = saved_scaler.transform(X_test)
    
    print(f"Test samples: {len(X_test)}")
    
    # Load model
    print("\n[Step 3] Loading trained model...")
    model = model_data['model']
    
    # Make predictions using correctly scaled features
    print("\n[Step 4] Making predictions on test set...")
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    # Evaluate
    print("\n[Step 5] Evaluating model performance...")
    evaluate_predictions(y_test, y_pred, y_proba)
    
    # Plot feature importance
    print("\n[Step 6] Generating feature importance plot...")
    plot_feature_importance(model, saved_feature_names)
    
    print("\n" + "="*60)
    print("Evaluation complete! Check the generated plots.")
    print("="*60)


if __name__ == "__main__":
    main()

