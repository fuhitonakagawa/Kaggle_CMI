#!/usr/bin/env python3
# ====================================================================================================
# CMI BFRB Detection - Quick Test Script
# Test the comprehensive solution with a small subset of data
# ====================================================================================================

import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Import solution components
from comprehensive_solution import (
    Config, DataProcessor, HierarchicalClassifier,
    EnsembleModel, calculate_competition_metric,
    FeatureEngineering
)

def quick_test():
    """Quick test of the comprehensive solution."""
    print("="*70)
    print("CMI BFRB Detection - Quick Test")
    print("="*70)
    
    # Check data availability
    data_path = Config.DATA_PATH
    if not data_path.exists():
        print(f"⚠️ Data not found at {data_path}")
        print("Please ensure the data is downloaded and placed in the correct directory.")
        return
    
    # Load a small subset of data
    print("\nLoading sample data...")
    train_df = pd.read_csv(data_path / "train.csv")
    demo_df = pd.read_csv(data_path / "train_demographics.csv")
    
    # Use only first 10 sequences for quick test
    unique_sequences = train_df['sequence_id'].unique()[:10]
    train_df_subset = train_df[train_df['sequence_id'].isin(unique_sequences)]
    
    print(f"Using {len(unique_sequences)} sequences for quick test")
    
    # Test 1: Feature Engineering
    print("\n" + "-"*60)
    print("Test 1: Feature Engineering")
    print("-"*60)
    
    fe = FeatureEngineering()
    
    # Test gravity removal
    sample_acc = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    sample_rot = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]])
    linear_accel = fe.remove_gravity(sample_acc, sample_rot)
    print(f"✓ Gravity removal: Input shape {sample_acc.shape} -> Output shape {linear_accel.shape}")
    
    # Test angular velocity
    angular_vel = fe.calculate_angular_velocity(sample_rot)
    print(f"✓ Angular velocity: Output shape {angular_vel.shape}")
    
    # Test FFT features
    sample_signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100))
    fft_features = fe.extract_fft_features(sample_signal)
    print(f"✓ FFT features: {len(fft_features)} features extracted")
    print(f"  Dominant frequency: {fft_features['dominant_freq']:.2f} Hz")
    
    # Test 2: Data Processing
    print("\n" + "-"*60)
    print("Test 2: Data Processing")
    print("-"*60)
    
    processor = DataProcessor(Config)
    X, y, groups = processor.process_all_sequences(train_df_subset, demo_df)
    
    print(f"✓ Feature matrix shape: {X.shape}")
    print(f"✓ Number of features: {X.shape[1]}")
    print(f"✓ Labels shape: {y.shape}")
    print(f"✓ Unique classes: {len(np.unique(y))}")
    print(f"✓ Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        gesture = Config.REVERSE_GESTURE_MAPPER[cls]
        print(f"  {cls:2d} ({gesture[:30]:30s}): {count:3d} samples")
    
    # Test 3: Hierarchical Classification
    print("\n" + "-"*60)
    print("Test 3: Hierarchical Classification")
    print("-"*60)
    
    hier_clf = HierarchicalClassifier()
    
    # Train on subset
    print("Training hierarchical classifier on subset...")
    hier_clf.fit(X, y, groups)
    
    # Predict
    y_pred = hier_clf.predict(X)
    
    # Calculate metrics
    metrics = calculate_competition_metric(y, y_pred)
    print(f"✓ Training set performance:")
    print(f"  Binary F1: {metrics['binary_f1']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Combined Score: {metrics['combined_score']:.4f}")
    
    # Test 4: Ensemble Model
    print("\n" + "-"*60)
    print("Test 4: Ensemble Model")
    print("-"*60)
    
    ensemble = EnsembleModel()
    
    # Add models
    ensemble.add_model(hier_clf, 'hierarchical', weight=2.0)
    ensemble.add_model(None, 'lgbm_test', weight=1.0)
    
    print("Training ensemble on subset...")
    ensemble.fit(X, y, groups)
    
    # Predict
    y_ensemble = ensemble.predict(X)
    
    # Calculate metrics
    ensemble_metrics = calculate_competition_metric(y, y_ensemble)
    print(f"✓ Ensemble performance:")
    print(f"  Binary F1: {ensemble_metrics['binary_f1']:.4f}")
    print(f"  Macro F1: {ensemble_metrics['macro_f1']:.4f}")
    print(f"  Combined Score: {ensemble_metrics['combined_score']:.4f}")
    
    # Test 5: Probability Predictions
    print("\n" + "-"*60)
    print("Test 5: Probability Predictions")
    print("-"*60)
    
    # Test probability predictions
    proba = ensemble.predict_proba(X[:5])
    print(f"✓ Probability shape: {proba.shape}")
    print(f"✓ Sample probabilities for first sequence:")
    for i in range(min(5, len(proba[0]))):
        print(f"  Class {i}: {proba[0][i]:.4f}")
    print(f"  Sum of probabilities: {proba[0].sum():.4f}")
    
    print("\n" + "="*70)
    print("✓ All tests passed successfully!")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n💡 Next steps:")
        print("1. Run full training: python comprehensive_solution.py --mode train")
        print("2. For Kaggle submission: python kaggle_submission.py")
        print("3. Check outputs/ folder for detailed results")