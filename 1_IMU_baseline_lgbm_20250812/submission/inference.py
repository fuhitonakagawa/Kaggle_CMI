#!/usr/bin/env python
"""
CMI BFRB Detection - IMU-only LightGBM Inference Script
This script performs inference using the trained model and generates submission.parquet
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial.transform import Rotation as R

# Add kaggle evaluation path
sys.path.append('/kaggle/input/cmi-detect-behavior-with-sensor-data')
import kaggle_evaluation.cmi_inference_server

warnings.filterwarnings('ignore')

# Configuration
ACC_COLS = ['acc_x', 'acc_y', 'acc_z']
ROT_COLS = ['rot_w', 'rot_x', 'rot_y', 'rot_z']

# Load model (update path as needed)
MODEL_PATH = '/kaggle/input/imu-lgbm-model/imu_lgbm_model.pkl'
print('Loading model...')
model_data = joblib.load(MODEL_PATH)

models = model_data['models']
feature_names = model_data['feature_names']
reverse_gesture_mapper = model_data['reverse_gesture_mapper']

print(f'✓ Loaded {len(models)} models')
print(f'✓ CV Score: {model_data["mean_cv_score"]:.4f}')


def handle_quaternion_missing_values(rot_data: np.ndarray) -> np.ndarray:
    """Handle missing values in quaternion data."""
    rot_cleaned = rot_data.copy()
    
    for i in range(len(rot_data)):
        row = rot_data[i]
        missing_count = np.isnan(row).sum()
        
        if missing_count == 0:
            norm = np.linalg.norm(row)
            if norm > 1e-8:
                rot_cleaned[i] = row / norm
            else:
                rot_cleaned[i] = [1.0, 0.0, 0.0, 0.0]
        elif missing_count == 1:
            missing_idx = np.where(np.isnan(row))[0][0]
            valid_values = row[~np.isnan(row)]
            sum_squares = np.sum(valid_values**2)
            if sum_squares <= 1.0:
                missing_value = np.sqrt(max(0, 1.0 - sum_squares))
                if i > 0 and not np.isnan(rot_cleaned[i-1, missing_idx]):
                    if rot_cleaned[i-1, missing_idx] < 0:
                        missing_value = -missing_value
                rot_cleaned[i, missing_idx] = missing_value
                rot_cleaned[i, ~np.isnan(row)] = valid_values
            else:
                rot_cleaned[i] = [1.0, 0.0, 0.0, 0.0]
        else:
            rot_cleaned[i] = [1.0, 0.0, 0.0, 0.0]
    
    return rot_cleaned


def compute_world_acceleration(acc: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Convert acceleration from device to world coordinates."""
    try:
        rot_scipy = rot[:, [1, 2, 3, 0]]  # Convert to scipy format
        norms = np.linalg.norm(rot_scipy, axis=1)
        if np.any(norms < 1e-8):
            mask = norms < 1e-8
            rot_scipy[mask] = [0.0, 0.0, 0.0, 1.0]
        r = R.from_quat(rot_scipy)
        acc_world = r.apply(acc)
    except Exception:
        acc_world = acc.copy()
    return acc_world


def extract_statistical_features(data: np.ndarray, prefix: str) -> dict:
    """Extract statistical features from 1D time series."""
    features = {}
    
    # Basic statistics
    features[f'{prefix}_mean'] = np.mean(data)
    features[f'{prefix}_std'] = np.std(data)
    features[f'{prefix}_var'] = np.var(data)
    features[f'{prefix}_min'] = np.min(data)
    features[f'{prefix}_max'] = np.max(data)
    features[f'{prefix}_median'] = np.median(data)
    features[f'{prefix}_q25'] = np.percentile(data, 25)
    features[f'{prefix}_q75'] = np.percentile(data, 75)
    features[f'{prefix}_iqr'] = features[f'{prefix}_q75'] - features[f'{prefix}_q25']
    features[f'{prefix}_range'] = features[f'{prefix}_max'] - features[f'{prefix}_min']
    
    # Boundary features
    features[f'{prefix}_first'] = data[0] if len(data) > 0 else 0
    features[f'{prefix}_last'] = data[-1] if len(data) > 0 else 0
    features[f'{prefix}_delta'] = features[f'{prefix}_last'] - features[f'{prefix}_first']
    
    # Higher order moments
    if len(data) > 1 and np.std(data) > 1e-8:
        features[f'{prefix}_skew'] = pd.Series(data).skew()
        features[f'{prefix}_kurt'] = pd.Series(data).kurtosis()
    else:
        features[f'{prefix}_skew'] = 0
        features[f'{prefix}_kurt'] = 0
    
    # Differential features
    if len(data) > 1:
        diff_data = np.diff(data)
        features[f'{prefix}_diff_mean'] = np.mean(diff_data)
        features[f'{prefix}_diff_std'] = np.std(diff_data)
        features[f'{prefix}_n_changes'] = np.sum(np.abs(diff_data) > np.std(data) * 0.1)
    else:
        features[f'{prefix}_diff_mean'] = 0
        features[f'{prefix}_diff_std'] = 0
        features[f'{prefix}_n_changes'] = 0
    
    # Segment features (3 segments)
    seq_len = len(data)
    if seq_len >= 9:
        seg_size = seq_len // 3
        for i in range(3):
            start_idx = i * seg_size
            end_idx = (i + 1) * seg_size if i < 2 else seq_len
            segment = data[start_idx:end_idx]
            features[f'{prefix}_seg{i+1}_mean'] = np.mean(segment)
            features[f'{prefix}_seg{i+1}_std'] = np.std(segment)
        # Segment transitions
        features[f'{prefix}_seg1_to_seg2'] = features[f'{prefix}_seg2_mean'] - features[f'{prefix}_seg1_mean']
        features[f'{prefix}_seg2_to_seg3'] = features[f'{prefix}_seg3_mean'] - features[f'{prefix}_seg2_mean']
    else:
        for i in range(3):
            features[f'{prefix}_seg{i+1}_mean'] = features[f'{prefix}_mean']
            features[f'{prefix}_seg{i+1}_std'] = features[f'{prefix}_std']
        features[f'{prefix}_seg1_to_seg2'] = 0
        features[f'{prefix}_seg2_to_seg3'] = 0
    
    return features


def extract_features(sequence: pl.DataFrame, demographics: pl.DataFrame) -> pd.DataFrame:
    """Extract features from IMU sequence."""
    # Convert to pandas
    seq_df = sequence.to_pandas()
    demo_df = demographics.to_pandas()
    
    # Get available columns
    available_acc_cols = [col for col in ACC_COLS if col in seq_df.columns]
    available_rot_cols = [col for col in ROT_COLS if col in seq_df.columns]
    
    # Handle missing values
    acc_data = seq_df[available_acc_cols].copy()
    acc_data = acc_data.ffill().bfill().fillna(0)
    
    rot_data = seq_df[available_rot_cols].copy()
    rot_data = rot_data.ffill().bfill()
    
    # Handle quaternion missing values
    rot_data_clean = handle_quaternion_missing_values(rot_data.values)
    
    # Compute world acceleration
    world_acc_data = compute_world_acceleration(acc_data.values, rot_data_clean)
    
    # Initialize features
    features = {}
    
    # Sequence metadata
    features['sequence_length'] = len(seq_df)
    
    # Demographics features
    if len(demo_df) > 0:
        demo_row = demo_df.iloc[0]
        features['age'] = demo_row.get('age', 0)
        features['adult_child'] = demo_row.get('adult_child', 0)
        features['sex'] = demo_row.get('sex', 0)
        features['handedness'] = demo_row.get('handedness', 0)
        features['height_cm'] = demo_row.get('height_cm', 0)
        features['shoulder_to_wrist_cm'] = demo_row.get('shoulder_to_wrist_cm', 0)
        features['elbow_to_wrist_cm'] = demo_row.get('elbow_to_wrist_cm', 0)
    else:
        # Default values
        features.update({
            'age': 0, 'adult_child': 0, 'sex': 0, 'handedness': 0,
            'height_cm': 0, 'shoulder_to_wrist_cm': 0, 'elbow_to_wrist_cm': 0
        })
    
    # Extract statistical features for each axis
    for i, axis in enumerate(['x', 'y', 'z']):
        if i < acc_data.shape[1]:
            # Device acceleration
            features.update(extract_statistical_features(acc_data.values[:, i], f'acc_{axis}'))
            # World acceleration
            features.update(extract_statistical_features(world_acc_data[:, i], f'world_acc_{axis}'))
    
    # Rotation features
    for i, comp in enumerate(['w', 'x', 'y', 'z']):
        if i < rot_data_clean.shape[1]:
            features.update(extract_statistical_features(rot_data_clean[:, i], f'rot_{comp}'))
    
    # Magnitude features
    acc_magnitude = np.linalg.norm(acc_data.values, axis=1)
    world_acc_magnitude = np.linalg.norm(world_acc_data, axis=1)
    
    features.update(extract_statistical_features(acc_magnitude, 'acc_magnitude'))
    features.update(extract_statistical_features(world_acc_magnitude, 'world_acc_magnitude'))
    
    # Difference between device and world acceleration
    acc_world_diff = acc_magnitude - world_acc_magnitude
    features.update(extract_statistical_features(acc_world_diff, 'acc_world_diff'))
    
    # Convert to DataFrame
    result_df = pd.DataFrame([features])
    
    # Ensure all expected features are present
    for col in feature_names:
        if col not in result_df.columns:
            result_df[col] = 0
    
    # Select only the features used in training
    result_df = result_df[feature_names]
    result_df = result_df.fillna(0)
    
    return result_df


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Prediction function for CMI inference server.
    Takes a single sequence and returns the predicted gesture name.
    """
    try:
        # Extract features
        features = extract_features(sequence, demographics)
        
        # Get predictions from all models
        probabilities = []
        
        for model in models:
            # Get prediction probabilities
            pred_proba = model.predict_proba(features)
            probabilities.append(pred_proba[0])
        
        # Ensemble: average probabilities
        avg_proba = np.mean(probabilities, axis=0)
        final_prediction = np.argmax(avg_proba)
        
        # Convert to gesture name
        gesture_name = reverse_gesture_mapper[final_prediction]
        
        return gesture_name
        
    except Exception as e:
        print(f"Prediction error: {e}")
        # Return default prediction in case of error
        return 'Text on phone'


# Initialize and run inference server
if __name__ == '__main__':
    print('Initializing inference server...')
    inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
    
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        print('Running in competition environment...')
        inference_server.serve()
    else:
        print('Running in local testing mode...')
        inference_server.run_local_gateway(
            data_paths=(
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
            )
        )
        print('✓ Inference complete!')
        print('✓ submission.parquet has been generated')