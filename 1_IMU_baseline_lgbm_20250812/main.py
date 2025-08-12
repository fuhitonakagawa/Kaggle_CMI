#!/usr/bin/env python
"""
CMI BFRB Detection - IMU-only LightGBM Baseline
Main execution script for training and evaluation
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import DataLoader, REVERSE_GESTURE_MAPPER
from src.feature_engineering import FeatureEngineer
from src.train import Trainer
from src.evaluate import competition_metric

warnings.filterwarnings('ignore')


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main execution pipeline."""
    print("="*60)
    print("CMI BFRB Detection - IMU-only LightGBM Baseline")
    print("="*60)
    print("🚀 Key Features:")
    print("  - World Coordinate Transformation")
    print("  - Statistical Feature Engineering")
    print("  - StratifiedGroupKFold Cross-Validation")
    print("="*60)
    
    # Load configuration
    config = load_config()
    print(f"\n✓ Configuration loaded")
    
    # Set random seed
    np.random.seed(config['training']['seed'])
    
    # Initialize components
    data_loader = DataLoader(config)
    feature_engineer = FeatureEngineer(config)
    trainer = Trainer(config)
    
    # Load data
    print("\n" + "="*60)
    print("STEP 1: Loading Data")
    print("="*60)
    train_df, train_demographics, test_df, test_demographics = data_loader.load_data()
    
    # Get IMU columns
    imu_cols = data_loader.get_imu_columns(train_df, test_df)
    
    # Prepare sequences
    print("\n" + "="*60)
    print("STEP 2: Preparing Sequences")
    print("="*60)
    
    train_sequences, y_train, subjects, train_seq_ids = data_loader.prepare_sequences(
        train_df, train_demographics, imu_cols, is_train=True
    )
    
    test_sequences, _, _, test_seq_ids = data_loader.prepare_sequences(
        test_df, test_demographics, imu_cols, is_train=False
    )
    
    print(f"✓ Prepared {len(train_sequences)} training sequences")
    print(f"✓ Prepared {len(test_sequences)} test sequences")
    
    # Extract features
    print("\n" + "="*60)
    print("STEP 3: Feature Engineering")
    print("="*60)
    
    print("Extracting training features...")
    X_train = feature_engineer.extract_all_features(train_sequences)
    
    print("Extracting test features...")
    X_test = feature_engineer.extract_all_features(test_sequences)
    
    # Train models
    print("\n" + "="*60)
    print("STEP 4: Model Training")
    print("="*60)
    
    oof_predictions, cv_scores = trainer.train(X_train, y_train, subjects)
    
    # Save results
    print("\n" + "="*60)
    print("STEP 5: Saving Results")
    print("="*60)
    
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config['output']['model_dir']).parent / f"run_{timestamp}"
    
    trainer.save_results(results_dir)
    
    # Generate test predictions
    print("\n" + "="*60)
    print("STEP 6: Generating Test Predictions")
    print("="*60)
    
    test_predictions = trainer.predict(X_test)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'sequence_id': test_seq_ids,
        'gesture': [REVERSE_GESTURE_MAPPER[pred] for pred in test_predictions]
    })
    
    submission_path = results_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"✓ Submission saved to {submission_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"✓ Final CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"✓ Results saved to: {results_dir}")
    print(f"✓ Models trained: {config['training']['n_folds']}")
    print(f"✓ Test predictions generated: {len(test_predictions)}")
    print("="*60)
    
    return trainer, results_dir


if __name__ == "__main__":
    trainer, results_dir = main()