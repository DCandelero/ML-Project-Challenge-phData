#!/usr/bin/env python3
"""Evaluate the house price prediction model"""

import argparse
import json
import pathlib
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn import model_selection, metrics, pipeline, preprocessing, neighbors

# Paths
SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"
SALES_COLUMNS = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode',
    # Additional columns needed for feature engineering (v2)
    'yr_built', 'yr_renovated', 'grade', 'condition'
]


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


def load_existing_model(model_path, features_path):
    """Load an existing model and its features"""
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"Loading features from {features_path}...")
    with open(features_path, 'r') as f:
        features = json.load(f)

    return model, features


def evaluate_model(X, y, model_version='v1', use_existing_model=False):
    """Evaluate model with train-test split and cross-validation"""

    # Train-test split (same as create_model.py)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Load or train model
    if use_existing_model:
        model_path = f"model/{model_version}/model.pkl"
        features_path = f"model/{model_version}/model_features.json"

        model, required_features = load_existing_model(model_path, features_path)

        # Apply feature engineering if v2
        if model_version == 'v2':
            from ml.feature_engineering import engineer_features
            X_train = engineer_features(X_train)
            X_test = engineer_features(X_test)

        # Align features
        X_train = X_train[required_features]
        X_test = X_test[required_features]
        X_aligned = X.copy()
        if model_version == 'v2':
            X_aligned = engineer_features(X_aligned)
        X_aligned = X_aligned[required_features]
    else:
        # Train model from scratch
        print("\nTraining model...")
        model = create_model()
        model.fit(X_train, y_train)
        X_aligned = X

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

    # Cross-validation (only if training from scratch)
    if use_existing_model:
        print("\n" + "="*50)
        print("Skipping cross-validation (using existing model)")
        print("="*50)
        cv_metrics = None
    else:
        print("\n" + "="*50)
        print("Running 5-fold cross-validation...")
        print("="*50)
        cv_scores = model_selection.cross_val_score(
            create_model(), X_aligned, y,
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
    # Determine model type
    if use_existing_model:
        model_type = type(model.named_steps['randomforestregressor'] if hasattr(model, 'named_steps') else model).__name__
    else:
        model_type = 'KNeighborsRegressor'

    results = {
        'model_version': model_version,
        'model_type': model_type,
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'test_r2': test_metrics['r2'],
        'test_mape': test_metrics['mape'],
        'train_rmse': train_rmse,
        'overfitting_ratio': float(overfitting_ratio),
        'evaluation_date': datetime.now().isoformat(),
        'test_set_size': len(X_test),
        'training_set_size': len(X_train),
        'total_features': X_train.shape[1]
    }

    # Add cross-validation if available
    if cv_metrics:
        results['cv_rmse_mean'] = cv_metrics['rmse_mean']
        results['cv_rmse_std'] = cv_metrics['rmse_std']

    return results


def main():
    """Main evaluation script"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate house price prediction model')
    parser.add_argument(
        '--model-version',
        type=str,
        default='v1',
        choices=['v1', 'v2'],
        help='Model version to evaluate (default: v1)'
    )
    parser.add_argument(
        '--use-existing',
        action='store_true',
        help='Evaluate existing model from model/{version}/ instead of training new one'
    )
    args = parser.parse_args()

    print("\n" + "="*50)
    print("Housing ML Model Evaluation")
    print("="*50)
    print(f"Model Version: {args.model_version}")
    print(f"Mode: {'Evaluate existing model' if args.use_existing else 'Train and evaluate'}")

    # Load data
    print("\nLoading data...")
    X, y = load_data()
    print(f"✓ Loaded {len(X):,} samples with {X.shape[1]} features")

    # Evaluate
    results = evaluate_model(X, y, model_version=args.model_version, use_existing_model=args.use_existing)

    # Save results
    output_dir = pathlib.Path(f"model/{args.model_version}")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "evaluation_metrics.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*50)
    print(f"✓ Results saved to {output_path}")
    print("="*50)

    # Summary
    print("\n" + "="*50)
    print("Summary:")
    print("="*50)
    print(f"Model Version:   {results['model_version']}")
    print(f"Model Type:      {results['model_type']}")
    print(f"Test RMSE:       ${results['test_rmse']:,.2f}")
    print(f"Test R²:         {results['test_r2']:.3f}")
    if 'cv_rmse_mean' in results:
        print(f"CV RMSE:         ${results['cv_rmse_mean']:,.0f} ± ${results['cv_rmse_std']:,.0f}")
    print("="*50)


if __name__ == "__main__":
    main()
