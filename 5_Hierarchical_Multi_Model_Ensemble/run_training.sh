#!/bin/bash
# ====================================================================================================
# CMI BFRB Detection - Training Script
# ====================================================================================================

echo "======================================================================"
echo "CMI BFRB Detection - Comprehensive Solution v5.0"
echo "Starting Training Pipeline..."
echo "======================================================================"

# Create necessary directories
mkdir -p 5_CMI_comprehensive/outputs
mkdir -p 5_CMI_comprehensive/models

# Install required packages if not already installed
echo "Checking dependencies..."
uv add tensorflow lightgbm xgboost catboost scipy scikit-learn pandas polars joblib

# Run training
echo ""
echo "Starting model training..."
python 5_CMI_comprehensive/comprehensive_solution.py --mode train

echo ""
echo "Training complete! Check outputs/ folder for results."
echo "======================================================================"