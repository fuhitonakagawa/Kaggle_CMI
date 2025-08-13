#!/usr/bin/env python3
# ====================================================================================================
# CMI BFRB Detection - Kaggle Submission Script
# This script is optimized for Kaggle notebook environment
# ====================================================================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
import polars as pl
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

# Import the comprehensive solution
from comprehensive_solution import (
    Config, DataProcessor, HierarchicalClassifier, 
    EnsembleModel, predict_for_kaggle
)

print("="*70)
print("CMI BFRB Detection - Kaggle Submission")
print("="*70)

# ====================================================================================================
# LIGHTWEIGHT INFERENCE VERSION
# ====================================================================================================

def load_pretrained_models():
    """Load pre-trained models for inference."""
    print("Loading pre-trained models...")
    
    # Check if models exist
    model_path = Path("5_CMI_comprehensive/models/")
    if not model_path.exists():
        print("⚠️ No pre-trained models found. Training new models...")
        # Import training function
        from comprehensive_solution import train_full_pipeline
        ensemble, processor = train_full_pipeline()
        return ensemble, processor
    
    # Load saved models
    processor = joblib.load(model_path / 'processor.pkl')
    ensemble = joblib.load(model_path / 'final_ensemble.pkl')
    
    print("✓ Models loaded successfully")
    return ensemble, processor

# Global variables for models (loaded once)
ENSEMBLE = None
PROCESSOR = None

def initialize_models():
    """Initialize models globally."""
    global ENSEMBLE, PROCESSOR
    if ENSEMBLE is None or PROCESSOR is None:
        ENSEMBLE, PROCESSOR = load_pretrained_models()

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Predict function for Kaggle evaluation API.
    
    Args:
        sequence: Polars DataFrame with sensor data for one sequence
        demographics: Polars DataFrame with demographic data
    
    Returns:
        str: Predicted gesture name
    """
    # Initialize models if not already loaded
    initialize_models()
    
    # Convert to pandas
    df_seq = sequence.to_pandas()
    
    # Process sequence
    features = PROCESSOR.process_sequence(df_seq)
    features = features.reshape(1, -1)
    
    # Handle NaN/inf values
    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
    
    # Predict
    prediction = ENSEMBLE.predict(features)[0]
    
    # Convert to gesture name
    gesture_name = Config.REVERSE_GESTURE_MAPPER[prediction]
    
    return gesture_name

# ====================================================================================================
# KAGGLE EVALUATION SERVER
# ====================================================================================================

if __name__ == "__main__":
    import kaggle_evaluation.cmi_inference_server
    
    print("\nInitializing Kaggle Inference Server...")
    
    # Create inference server with our predict function
    inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
    
    # Check if running in competition environment
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        print("Running in Kaggle competition environment...")
        inference_server.serve()
    else:
        print("Running in local/test environment...")
        # Local testing with sample data
        test_data_path = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
        
        if not test_data_path.exists():
            # Try alternative path
            test_data_path = Path("cmi-detect-behavior-with-sensor-data")
        
        if test_data_path.exists():
            inference_server.run_local_gateway(
                data_paths=(
                    str(test_data_path / "test.csv"),
                    str(test_data_path / "test_demographics.csv"),
                )
            )
            print("\n✓ Submission parquet file generated successfully!")
        else:
            print("⚠️ Test data not found. Please ensure data is in the correct path.")
            print("Expected path:", test_data_path)
    
    print("\n" + "="*70)
    print("Inference complete!")
    print("="*70)