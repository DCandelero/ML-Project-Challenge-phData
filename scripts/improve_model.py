#!/usr/bin/env python3
"""
Improve the house price prediction model.

Strategy:
1. Feature engineering (add 8-9 derived features)
2. Algorithm upgrade (KNN → Random Forest)
3. Hyperparameter tuning (Randomized Search)
4. Evaluation and comparison with v1
5. Save as model v2
"""

import json
import pathlib
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn import model_selection, metrics, ensemble, preprocessing, pipeline
from sklearn.model_selection import RandomizedSearchCV

# Import our feature engineering
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from ml.feature_engineering import engineer_features, get_engineered_feature_names

# Paths
SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"
SALES_COLUMNS = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode',
    'yr_built', 'yr_renovated', 'grade', 'condition'
]
OUTPUT_DIR = pathlib.Path("model/v2")
V1_METRICS_PATH = "model/evaluation_metrics.json"


def load_data():
    """Load and merge sales + demographics data"""
    print("Loading data...")
    sales = pd.read_csv(SALES_PATH, usecols=SALES_COLUMNS, dtype={'zipcode': str})
    demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={'zipcode': str})

    merged = sales.merge(demographics, on='zipcode', how='left')
    merged = merged.drop(columns='zipcode')

    y = merged.pop('price')
    X = merged

    return X, y


def create_random_forest_model():
    """Create Random Forest model with good defaults"""
    return pipeline.make_pipeline(
        preprocessing.RobustScaler(),
        ensemble.RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    )


def tune_hyperparameters(X_train, y_train):
    """Tune Random Forest hyperparameters using Randomized Search"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning (Randomized Search)")
    print("="*60)

    param_distributions = {
        'randomforestregressor__n_estimators': [50, 100, 150, 200],
        'randomforestregressor__max_depth': [10, 15, 20, 25, None],
        'randomforestregressor__min_samples_split': [5, 10, 15, 20],
        'randomforestregressor__min_samples_leaf': [2, 5, 10],
        'randomforestregressor__max_features': ['sqrt', 'log2']
    }

    random_search = RandomizedSearchCV(
        create_random_forest_model(),
        param_distributions=param_distributions,
        n_iter=20,  # Try 20 random combinations
        cv=3,       # 3-fold CV (faster than 5-fold)
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    print("Fitting RandomizedSearchCV (this may take 5-10 minutes)...")
    random_search.fit(X_train, y_train)

    print(f"\nBest parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")

    best_score_rmse = np.sqrt(-random_search.best_score_)
    print(f"\nBest CV RMSE: ${best_score_rmse:,.2f}")

    return random_search.best_estimator_


def calculate_mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true))


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model and return metrics"""
    # Training set
    y_train_pred = model.predict(X_train)
    train_rmse = float(np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))

    # Test set
    y_pred = model.predict(X_test)

    test_metrics = {
        'rmse': float(np.sqrt(metrics.mean_squared_error(y_test, y_pred))),
        'mae': float(metrics.mean_absolute_error(y_test, y_pred)),
        'r2': float(metrics.r2_score(y_test, y_pred)),
        'mape': float(calculate_mape(y_test, y_pred))
    }

    return train_rmse, test_metrics


def get_feature_importance(model, feature_names):
    """Extract feature importance from Random Forest"""
    # Get the Random Forest model from pipeline
    rf_model = model.named_steps['randomforestregressor']

    # Get feature importances (already scaled 0-1 by sklearn)
    importances = rf_model.feature_importances_

    # Create sorted list
    feature_importance = []
    for name, importance in zip(feature_names, importances):
        feature_importance.append({
            'feature': name,
            'importance': float(importance)
        })

    # Sort by importance (descending)
    feature_importance.sort(key=lambda x: x['importance'], reverse=True)

    return feature_importance


def load_v1_metrics():
    """Load v1 metrics for comparison"""
    try:
        with open(V1_METRICS_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Could not find v1 metrics at {V1_METRICS_PATH}")
        return None


def compare_models(v1_metrics, v2_metrics):
    """Compare v1 and v2 performance"""
    print("\n" + "="*60)
    print("MODEL COMPARISON: v1 vs v2")
    print("="*60)

    if v1_metrics is None:
        print("v1 metrics not available for comparison")
        return

    v1_test = v1_metrics.get('test_set_metrics', {})
    v2_test = v2_metrics['test_set_metrics']

    metrics_to_compare = ['rmse', 'mae', 'r2', 'mape']

    print(f"\n{'Metric':<10} {'v1':<15} {'v2':<15} {'Improvement':<15}")
    print("-" * 60)

    for metric in metrics_to_compare:
        if metric in v1_test and metric in v2_test:
            v1_val = v1_test[metric]
            v2_val = v2_test[metric]

            # Calculate improvement
            if metric == 'r2':
                # For R², higher is better
                improvement = ((v2_val - v1_val) / v1_val) * 100
                improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
            else:
                # For RMSE, MAE, MAPE, lower is better
                improvement = ((v1_val - v2_val) / v1_val) * 100
                improvement_str = f"-{improvement:.2f}%" if improvement > 0 else f"+{abs(improvement):.2f}%"

            # Format values
            if metric in ['rmse', 'mae']:
                v1_str = f"${v1_val:,.2f}"
                v2_str = f"${v2_val:,.2f}"
            elif metric == 'mape':
                v1_str = f"{v1_val:.2%}"
                v2_str = f"{v2_val:.2%}"
            else:  # r2
                v1_str = f"{v1_val:.4f}"
                v2_str = f"{v2_val:.4f}"

            print(f"{metric.upper():<10} {v1_str:<15} {v2_str:<15} {improvement_str:<15}")


def save_model_and_metadata(model, feature_names, train_rmse, test_metrics, feature_importance, X):
    """Save model, features, metrics, and feature importance"""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save model
    import pickle
    model_path = OUTPUT_DIR / "model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Model saved: {model_path}")

    # 2. Save feature names
    features_path = OUTPUT_DIR / "model_features.json"
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"✓ Features saved: {features_path}")

    # 3. Save evaluation metrics
    overfitting_ratio = test_metrics['rmse'] / train_rmse

    metrics_data = {
        'model_version': 'v2',
        'model_type': 'RandomForestRegressor',
        'test_set_metrics': test_metrics,
        'training_metrics': {
            'rmse': train_rmse
        },
        'overfitting_ratio': float(overfitting_ratio),
        'evaluation_date': datetime.now().isoformat(),
        'total_features': len(feature_names),
        'engineered_features_count': len([f for f in feature_names if f in get_engineered_feature_names()])
    }

    metrics_path = OUTPUT_DIR / "evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"✓ Metrics saved: {metrics_path}")

    # 4. Save feature importance
    importance_path = OUTPUT_DIR / "feature_importance.json"
    with open(importance_path, 'w') as f:
        json.dump(feature_importance, f, indent=2)
    print(f"✓ Feature importance saved: {importance_path}")

    print(f"\n✓ All v2 model artifacts saved to: {OUTPUT_DIR}")


def main():
    """Main improvement script"""
    print("\n" + "="*60)
    print("MODEL IMPROVEMENT: v1 → v2")
    print("="*60)
    print("\nStrategy:")
    print("  1. Feature Engineering (8-9 new features)")
    print("  2. Algorithm Upgrade (KNN → Random Forest)")
    print("  3. Hyperparameter Tuning (Randomized Search)")
    print("  4. Evaluation & Comparison")
    print("  5. Save as v2")

    # Load data
    print("\n" + "="*60)
    print("1. Loading Data")
    print("="*60)
    X, y = load_data()
    print(f"✓ Loaded {len(X):,} samples with {X.shape[1]} features")

    # Feature engineering
    print("\n" + "="*60)
    print("2. Feature Engineering")
    print("="*60)
    X_engineered = engineer_features(X)
    new_features = set(X_engineered.columns) - set(X.columns)
    print(f"✓ Added {len(new_features)} engineered features:")
    for feature in sorted(new_features):
        print(f"  - {feature}")

    feature_names = list(X_engineered.columns)
    print(f"\nTotal features: {len(feature_names)} (was {X.shape[1]})")

    # Train-test split
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X_engineered, y, test_size=0.25, random_state=42
    )

    # Hyperparameter tuning
    print("\n" + "="*60)
    print("3. Algorithm Upgrade & Hyperparameter Tuning")
    print("="*60)
    print("Upgrading: KNN → Random Forest")

    best_model = tune_hyperparameters(X_train, y_train)

    # Evaluate
    print("\n" + "="*60)
    print("4. Evaluation")
    print("="*60)
    train_rmse, test_metrics = evaluate_model(best_model, X_train, X_test, y_train, y_test)

    print(f"\nTest Set Metrics:")
    print(f"  RMSE: ${test_metrics['rmse']:,.2f}")
    print(f"  MAE:  ${test_metrics['mae']:,.2f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    print(f"  MAPE: {test_metrics['mape']:.2%}")

    print(f"\nOverfitting Check:")
    print(f"  Training RMSE: ${train_rmse:,.2f}")
    print(f"  Test RMSE:     ${test_metrics['rmse']:,.2f}")
    print(f"  Ratio:         {test_metrics['rmse']/train_rmse:.2f}")

    # Feature importance
    print("\n" + "="*60)
    print("Feature Importance (Top 10)")
    print("="*60)
    feature_importance = get_feature_importance(best_model, feature_names)

    for i, item in enumerate(feature_importance[:10], 1):
        feature = item['feature']
        importance = item['importance']
        is_engineered = "🔧" if feature in get_engineered_feature_names() else "  "
        print(f"{i:2d}. {is_engineered} {feature:<30s} {importance:.4f}")

    # Compare with v1
    v1_metrics = load_v1_metrics()
    compare_models(v1_metrics, {'test_set_metrics': test_metrics})

    # Save everything
    print("\n" + "="*60)
    print("5. Saving v2 Model")
    print("="*60)
    save_model_and_metadata(
        best_model,
        feature_names,
        train_rmse,
        test_metrics,
        feature_importance,
        X_engineered
    )

    # Final summary
    print("\n" + "="*60)
    print("✓ MODEL IMPROVEMENT COMPLETE!")
    print("="*60)
    print(f"\nv2 Model Summary:")
    print(f"  Algorithm:          Random Forest")
    print(f"  Features:           {len(feature_names)} (including {len(new_features)} engineered)")
    print(f"  Test RMSE:          ${test_metrics['rmse']:,.2f}")
    print(f"  Test R²:            {test_metrics['r2']:.4f}")
    print(f"\nTo use v2 model:")
    print(f"  1. Update .env: MODEL_VERSION=v2")
    print(f"  2. Restart API: docker-compose restart api")
    print("="*60)


if __name__ == "__main__":
    main()
