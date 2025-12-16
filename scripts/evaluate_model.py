#!/usr/bin/env python3
"""Evaluate the house price prediction model"""

import json
import pathlib
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn import model_selection, metrics, pipeline, preprocessing, neighbors

# Paths
SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"
SALES_COLUMNS = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]
OUTPUT_PATH = "model/evaluation_metrics.json"


def load_data():
    """Load and merge sales + demographics data"""
    print("Loading sales data...")
    sales = pd.read_csv(SALES_PATH, usecols=SALES_COLUMNS, dtype={'zipcode': str})

    print("Loading demographics data...")
    demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={'zipcode': str})

    print("Merging datasets...")
    merged = sales.merge(demographics, on='zipcode', how='left')
    merged = merged.drop(columns='zipcode')

    y = merged.pop('price')
    X = merged

    return X, y


def create_model():
    """Create the same model as create_model.py"""
    return pipeline.make_pipeline(
        preprocessing.RobustScaler(),
        neighbors.KNeighborsRegressor()
    )


def calculate_mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true))


def evaluate_model(X, y):
    """Evaluate model with train-test split and cross-validation"""

    # Train-test split (same as create_model.py)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Train model
    print("\nTraining model...")
    model = create_model()
    model.fit(X_train, y_train)

    # Training set evaluation (to check overfitting)
    print("Evaluating on training set...")
    y_train_pred = model.predict(X_train)
    train_rmse = float(np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))

    # Test set evaluation
    print("Evaluating on test set...")
    y_pred = model.predict(X_test)

    test_metrics = {
        'rmse': float(np.sqrt(metrics.mean_squared_error(y_test, y_pred))),
        'mae': float(metrics.mean_absolute_error(y_test, y_pred)),
        'r2': float(metrics.r2_score(y_test, y_pred)),
        'mape': float(calculate_mape(y_test, y_pred))
    }

    print("\n" + "="*50)
    print("Test Set Metrics:")
    print("="*50)
    print(f"  RMSE: ${test_metrics['rmse']:,.2f}")
    print(f"  MAE:  ${test_metrics['mae']:,.2f}")
    print(f"  R²:   {test_metrics['r2']:.3f}")
    print(f"  MAPE: {test_metrics['mape']:.1%}")

    # Overfitting check
    print("\n" + "="*50)
    print("Overfitting Analysis:")
    print("="*50)
    print(f"  Training RMSE: ${train_rmse:,.2f}")
    print(f"  Test RMSE:     ${test_metrics['rmse']:,.2f}")
    overfitting_ratio = test_metrics['rmse'] / train_rmse
    print(f"  Ratio:         {overfitting_ratio:.2f}")
    if overfitting_ratio > 1.2:
        print("  ⚠️  Possible overfitting detected (ratio > 1.2)")
    else:
        print("  ✓ No significant overfitting")

    # Cross-validation
    print("\n" + "="*50)
    print("Running 5-fold cross-validation...")
    print("="*50)
    cv_scores = model_selection.cross_val_score(
        create_model(), X, y,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    cv_rmse_scores = np.sqrt(-cv_scores)
    cv_metrics = {
        'cv_folds': 5,
        'rmse_mean': float(cv_rmse_scores.mean()),
        'rmse_std': float(cv_rmse_scores.std()),
        'rmse_scores': cv_rmse_scores.tolist()
    }

    print(f"\nCross-Validation Results:")
    print(f"  RMSE Mean: ${cv_metrics['rmse_mean']:,.0f}")
    print(f"  RMSE Std:  ${cv_metrics['rmse_std']:,.0f}")
    print(f"  Range:     ${cv_rmse_scores.min():,.0f} - ${cv_rmse_scores.max():,.0f}")

    if cv_metrics['rmse_std'] > 30000:
        print("  ⚠️  High variance detected (std > $30k)")
    else:
        print("  ✓ Low variance across folds")

    # Compile results
    results = {
        'model_version': 'v1',
        'model_type': 'KNeighborsRegressor',
        'test_set_metrics': test_metrics,
        'training_metrics': {
            'rmse': train_rmse
        },
        'cross_validation': cv_metrics,
        'overfitting_ratio': float(overfitting_ratio),
        'evaluation_date': datetime.now().isoformat(),
        'test_set_size': len(X_test),
        'training_set_size': len(X_train),
        'total_features': X.shape[1]
    }

    return results


def main():
    """Main evaluation script"""
    print("\n" + "="*50)
    print("Housing ML Model Evaluation")
    print("="*50)

    # Load data
    print("\nLoading data...")
    X, y = load_data()
    print(f"✓ Loaded {len(X):,} samples with {X.shape[1]} features")

    # Evaluate
    results = evaluate_model(X, y)

    # Save results
    output_path = pathlib.Path(OUTPUT_PATH)
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*50)
    print(f"✓ Results saved to {output_path}")
    print("="*50)

    # Summary
    print("\n" + "="*50)
    print("Summary:")
    print("="*50)
    print(f"Model Type:      {results['model_type']}")
    print(f"Test RMSE:       ${results['test_set_metrics']['rmse']:,.2f}")
    print(f"Test R²:         {results['test_set_metrics']['r2']:.3f}")
    print(f"CV RMSE:         ${results['cross_validation']['rmse_mean']:,.0f} ± ${results['cross_validation']['rmse_std']:,.0f}")
    print("="*50)


if __name__ == "__main__":
    main()
