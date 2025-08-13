#!/usr/bin/env python
"""
CMI BFRB Detection - IMU Improved Model
Complete Training and Inference Pipeline
Version: 2.1.0 - Fixed for Kaggle Submission
Date: 2025-01-12
"""

import os
import sys
import json
import pickle
import joblib
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from scipy.signal import find_peaks, welch
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

# ======================== Configuration ========================

# Paths
IS_KAGGLE = '/kaggle' in os.getcwd() or os.path.exists('/kaggle/input')
if IS_KAGGLE:
    DATA_PATH = '/kaggle/input/cmi-detect-behavior-with-sensor-data/'
    MODEL_PATH = '/kaggle/working/'
    OUTPUT_PATH = '/kaggle/working/'
else:
    DATA_PATH = '../cmi-detect-behavior-with-sensor-data/'
    MODEL_PATH = './models/'
    OUTPUT_PATH = './'

# Create directories
os.makedirs(MODEL_PATH, exist_ok=True)

# General settings
SEED = 42
N_FOLDS = 5
DEBUG = False  # Set to True for quick testing

# Gesture mapping
GESTURE_MAPPER = {
    'Above ear - pull hair': 0, 
    'Cheek - pinch skin': 1, 
    'Eyebrow - pull hair': 2,
    'Eyelash - pull hair': 3, 
    'Forehead - pull hairline': 4, 
    'Forehead - scratch': 5,
    'Neck - pinch skin': 6, 
    'Neck - scratch': 7, 
    'Drink from bottle/cup': 8,
    'Feel around in tray and pull out an object': 9, 
    'Glasses on/off': 10,
    'Pinch knee/leg skin': 11, 
    'Pull air toward your face': 12,
    'Scratch knee/leg skin': 13, 
    'Text on phone': 14, 
    'Wave hello': 15,
    'Write name in air': 16, 
    'Write name on leg': 17,
}
REVERSE_GESTURE_MAPPER = {v: k for k, v in GESTURE_MAPPER.items()}

# BFRB behaviors (indices 0-7)
BFRB_INDICES = list(range(8))

# Model parameters
LGBM_PARAMS = {
    'objective': 'multiclass',
    'num_class': 18,
    'n_estimators': 500 if not DEBUG else 50,
    'max_depth': 8,
    'learning_rate': 0.03,
    'num_leaves': 31,
    'colsample_bytree': 0.6,
    'subsample': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': SEED,
    'n_jobs': -1,
    'verbosity': -1,
    'min_child_samples': 20,
    'min_split_gain': 0.001
}

XGB_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 18,
    'n_estimators': 500 if not DEBUG else 50,
    'max_depth': 8,
    'learning_rate': 0.03,
    'colsample_bytree': 0.6,
    'subsample': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': SEED,
    'n_jobs': -1,
    'verbosity': 0,
    'min_child_weight': 5,
    'gamma': 0.1,
    'tree_method': 'hist'
}

# ======================== Feature Engineering ========================

def handle_quaternion_missing_values(rot_w, rot_x, rot_y, rot_z):
    """Handle missing values in quaternion data"""
    mask = ~(np.isnan(rot_w) | np.isnan(rot_x) | np.isnan(rot_y) | np.isnan(rot_z))
    if np.sum(mask) == 0:
        return np.ones(len(rot_w)), np.zeros(len(rot_w)), np.zeros(len(rot_w)), np.zeros(len(rot_w))
    
    # Forward fill then backward fill
    df_temp = pd.DataFrame({'w': rot_w, 'x': rot_x, 'y': rot_y, 'z': rot_z})
    df_temp = df_temp.ffill().bfill()
    
    # If still NaN, use identity quaternion
    df_temp = df_temp.fillna({'w': 1.0, 'x': 0.0, 'y': 0.0, 'z': 0.0})
    
    return df_temp['w'].values, df_temp['x'].values, df_temp['y'].values, df_temp['z'].values

def compute_world_acceleration(acc_x, acc_y, acc_z, rot_w, rot_x, rot_y, rot_z):
    """Convert local acceleration to world coordinates using quaternion rotation"""
    # Handle missing values
    rot_w, rot_x, rot_y, rot_z = handle_quaternion_missing_values(rot_w, rot_x, rot_y, rot_z)
    
    # Normalize quaternions
    norm = np.sqrt(rot_w**2 + rot_x**2 + rot_y**2 + rot_z**2) + 1e-10
    rot_w, rot_x, rot_y, rot_z = rot_w/norm, rot_x/norm, rot_y/norm, rot_z/norm
    
    # Rotation matrix components
    wx, wy, wz = rot_w*rot_x, rot_w*rot_y, rot_w*rot_z
    xx, xy, xz = rot_x*rot_x, rot_x*rot_y, rot_x*rot_z
    yy, yz, zz = rot_y*rot_y, rot_y*rot_z, rot_z*rot_z
    
    # Transform to world coordinates
    world_x = (1 - 2*(yy + zz)) * acc_x + 2*(xy - wz) * acc_y + 2*(xz + wy) * acc_z
    world_y = 2*(xy + wz) * acc_x + (1 - 2*(xx + zz)) * acc_y + 2*(yz - wx) * acc_z
    world_z = 2*(xz - wy) * acc_x + 2*(yz + wx) * acc_y + (1 - 2*(xx + yy)) * acc_z
    
    return world_x, world_y, world_z

def extract_statistical_features(data: np.ndarray, prefix: str) -> Dict[str, float]:
    """Extract statistical features from time series data"""
    features = {}
    
    # Basic statistics
    features[f'{prefix}_mean'] = np.mean(data)
    features[f'{prefix}_std'] = np.std(data)
    features[f'{prefix}_min'] = np.min(data)
    features[f'{prefix}_max'] = np.max(data)
    features[f'{prefix}_median'] = np.median(data)
    features[f'{prefix}_q25'] = np.percentile(data, 25)
    features[f'{prefix}_q75'] = np.percentile(data, 75)
    features[f'{prefix}_iqr'] = features[f'{prefix}_q75'] - features[f'{prefix}_q25']
    features[f'{prefix}_range'] = features[f'{prefix}_max'] - features[f'{prefix}_min']
    features[f'{prefix}_cv'] = np.std(data) / (np.mean(data) + 1e-10)
    
    # Higher moments
    if len(data) > 1:
        features[f'{prefix}_skew'] = stats.skew(data)
        features[f'{prefix}_kurt'] = stats.kurtosis(data)
    else:
        features[f'{prefix}_skew'] = 0
        features[f'{prefix}_kurt'] = 0
    
    # Time series features
    if len(data) > 1:
        # Convert to numpy array if it's a pandas Series
        if hasattr(data, 'values'):
            data_array = data.values
        else:
            data_array = np.array(data)
        features[f'{prefix}_first'] = data_array[0]
        features[f'{prefix}_last'] = data_array[-1]
        features[f'{prefix}_delta'] = data_array[-1] - data_array[0]
        
        # Difference statistics
        diff_data = np.diff(data)
        features[f'{prefix}_diff_mean'] = np.mean(diff_data)
        features[f'{prefix}_diff_std'] = np.std(diff_data)
        features[f'{prefix}_diff_max'] = np.max(np.abs(diff_data))
        
        # Trend
        time_indices = np.arange(len(data))
        corr_coef = np.corrcoef(time_indices, data)[0, 1]
        features[f'{prefix}_trend'] = corr_coef if not np.isnan(corr_coef) else 0
    
    return features

def extract_frequency_features(data: np.ndarray, prefix: str, sampling_rate: float = 20) -> Dict[str, float]:
    """Extract frequency domain features"""
    features = {}
    
    if len(data) < 4:
        # Return default values for short sequences
        for feat in ['fft_max_freq', 'fft_max_power', 'spectral_centroid', 
                    'spectral_rolloff', 'spectral_entropy', 'zero_crossing_rate']:
            features[f'{prefix}_{feat}'] = 0
        return features
    
    # FFT features
    fft = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), 1/sampling_rate)
    positive_freqs = freqs[:len(freqs)//2]
    power_spectrum = np.abs(fft[:len(fft)//2])**2
    
    if len(power_spectrum) > 0:
        max_power_idx = np.argmax(power_spectrum)
        features[f'{prefix}_fft_max_freq'] = positive_freqs[max_power_idx]
        features[f'{prefix}_fft_max_power'] = power_spectrum[max_power_idx]
        
        # Spectral features
        total_power = np.sum(power_spectrum) + 1e-10
        normalized_spectrum = power_spectrum / total_power
        
        # Spectral centroid
        spectral_centroid = np.sum(positive_freqs * normalized_spectrum)
        features[f'{prefix}_spectral_centroid'] = spectral_centroid
        
        # Spectral rolloff
        cumsum = np.cumsum(normalized_spectrum)
        rolloff_idx = np.where(cumsum >= 0.85)[0]
        if len(rolloff_idx) > 0:
            features[f'{prefix}_spectral_rolloff'] = positive_freqs[rolloff_idx[0]]
        else:
            features[f'{prefix}_spectral_rolloff'] = positive_freqs[-1]
        
        # Spectral entropy
        spectral_entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum + 1e-10))
        features[f'{prefix}_spectral_entropy'] = spectral_entropy
    
    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
    features[f'{prefix}_zero_crossing_rate'] = zero_crossings / len(data)
    
    return features

def extract_segment_features(data: np.ndarray, prefix: str, n_segments: int = 3) -> Dict[str, float]:
    """Extract features from segments of the time series"""
    features = {}
    
    if len(data) < n_segments * 3:
        # Too short for segmentation
        for i in range(n_segments):
            features[f'{prefix}_seg{i+1}_mean'] = np.mean(data)
            features[f'{prefix}_seg{i+1}_std'] = np.std(data)
        return features
    
    segment_size = len(data) // n_segments
    for i in range(n_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < n_segments - 1 else len(data)
        segment = data[start:end]
        
        features[f'{prefix}_seg{i+1}_mean'] = np.mean(segment)
        features[f'{prefix}_seg{i+1}_std'] = np.std(segment)
        features[f'{prefix}_seg{i+1}_max'] = np.max(segment)
        features[f'{prefix}_seg{i+1}_min'] = np.min(segment)
    
    return features

def extract_comprehensive_features(seq_df: pd.DataFrame, demo_df: pd.DataFrame = None) -> pd.DataFrame:
    """Extract all features from a sequence"""
    features = {}
    
    # Sequence length
    features['sequence_length'] = len(seq_df)
    
    # Demographics with defaults
    if demo_df is not None and len(demo_df) > 0:
        demo = demo_df.iloc[0]
        features['age'] = demo.get('age', 30)
        features['adult_child'] = demo.get('adult_child', 1)
        features['sex'] = demo.get('sex', 0)
        features['handedness'] = demo.get('handedness', 1)
        features['height_cm'] = demo.get('height_cm', 170)
        features['shoulder_to_wrist_cm'] = demo.get('shoulder_to_wrist_cm', 50)
        features['elbow_to_wrist_cm'] = demo.get('elbow_to_wrist_cm', 30)
    else:
        features.update({
            'age': 30, 'adult_child': 1, 'sex': 0, 'handedness': 1,
            'height_cm': 170, 'shoulder_to_wrist_cm': 50, 'elbow_to_wrist_cm': 30
        })
    
    # Extract IMU features
    imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
    
    for col in imu_cols:
        if col in seq_df.columns:
            data = seq_df[col].fillna(0).values
            
            # Statistical features
            features.update(extract_statistical_features(data, col))
            
            # Frequency features
            features.update(extract_frequency_features(data, col))
            
            # Segment features
            features.update(extract_segment_features(data, col))
    
    # World acceleration features
    if all(col in seq_df.columns for col in ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']):
        world_x, world_y, world_z = compute_world_acceleration(
            seq_df['acc_x'].fillna(0).values,
            seq_df['acc_y'].fillna(0).values,
            seq_df['acc_z'].fillna(0).values,
            seq_df['rot_w'].fillna(1).values,
            seq_df['rot_x'].fillna(0).values,
            seq_df['rot_y'].fillna(0).values,
            seq_df['rot_z'].fillna(0).values
        )
        
        # Extract features for world coordinates
        for data, prefix in [(world_x, 'world_acc_x'), (world_y, 'world_acc_y'), (world_z, 'world_acc_z')]:
            features.update(extract_statistical_features(data, prefix))
        
        # World acceleration magnitude
        world_mag = np.sqrt(world_x**2 + world_y**2 + world_z**2)
        features.update(extract_statistical_features(world_mag, 'world_acc_mag'))
    
    # Acceleration magnitude
    if all(col in seq_df.columns for col in ['acc_x', 'acc_y', 'acc_z']):
        acc_mag = np.sqrt(
            seq_df['acc_x'].fillna(0).values**2 + 
            seq_df['acc_y'].fillna(0).values**2 + 
            seq_df['acc_z'].fillna(0).values**2
        )
        features.update(extract_statistical_features(acc_mag, 'acc_magnitude'))
    
    # Rotation energy
    if all(col in seq_df.columns for col in ['rot_x', 'rot_y', 'rot_z']):
        rot_energy = seq_df['rot_x'].fillna(0).values**2 + seq_df['rot_y'].fillna(0).values**2 + seq_df['rot_z'].fillna(0).values**2
        features.update(extract_statistical_features(rot_energy, 'rotation_energy'))
    
    # Jerk (acceleration derivative)
    if all(col in seq_df.columns for col in ['acc_x', 'acc_y', 'acc_z']) and len(seq_df) > 1:
        for col in ['acc_x', 'acc_y', 'acc_z']:
            jerk = np.diff(seq_df[col].fillna(0).values)
            features[f'{col}_jerk_mean'] = np.mean(jerk)
            features[f'{col}_jerk_std'] = np.std(jerk)
            features[f'{col}_jerk_max'] = np.max(np.abs(jerk))
    
    return pd.DataFrame([features])

# ======================== Model Training ========================

def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """Calculate competition metric: (Binary F1 + Macro F1) / 2"""
    # Binary F1: BFRB vs non-BFRB
    binary_f1 = f1_score(
        np.where(y_true <= 7, 1, 0),
        np.where(y_pred <= 7, 1, 0),
        zero_division=0.0,
    )
    
    # Macro F1: within BFRB behaviors
    macro_f1 = f1_score(
        np.where(y_true <= 7, y_true, 99),
        np.where(y_pred <= 7, y_pred, 99),
        average='macro',
        zero_division=0.0,
    )
    
    final_score = 0.5 * (binary_f1 + macro_f1)
    return final_score, binary_f1, macro_f1

def train_models(X_train: pd.DataFrame, y_train: np.ndarray, subjects: np.ndarray) -> Dict:
    """Train LightGBM and XGBoost models with cross-validation"""
    print('\n========== Model Training ==========')
    
    # Initialize models storage
    lgb_models = []
    xgb_models = []
    oof_predictions_lgb = np.zeros((len(X_train), 18))
    oof_predictions_xgb = np.zeros((len(X_train), 18))
    feature_importance = pd.DataFrame()
    
    # Cross-validation
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y_train, subjects)):
        print(f'\n--- Fold {fold + 1}/{N_FOLDS} ---')
        
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]
        
        # Train LightGBM
        print('Training LightGBM...')
        lgb_model = lgb.LGBMClassifier(**LGBM_PARAMS)
        lgb_model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
        )
        lgb_models.append(lgb_model)
        
        # Train XGBoost
        print('Training XGBoost...')
        xgb_model = xgb.XGBClassifier(**XGB_PARAMS, enable_categorical=False)
        xgb_model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False
        )
        xgb_models.append(xgb_model)
        
        # OOF predictions
        oof_predictions_lgb[val_idx] = lgb_model.predict_proba(X_fold_val)
        oof_predictions_xgb[val_idx] = xgb_model.predict_proba(X_fold_val)
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': lgb_model.feature_importances_,
            'fold': fold
        })
        feature_importance = pd.concat([feature_importance, importance])
        
        # Evaluate fold
        val_pred_lgb = np.argmax(oof_predictions_lgb[val_idx], axis=1)
        val_pred_xgb = np.argmax(oof_predictions_xgb[val_idx], axis=1)
        val_pred_ensemble = np.argmax(0.6 * oof_predictions_lgb[val_idx] + 0.4 * oof_predictions_xgb[val_idx], axis=1)
        
        score, binary_f1, macro_f1 = competition_metric(y_fold_val, val_pred_ensemble)
        print(f'Fold {fold + 1} - Score: {score:.4f}, Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f}')
    
    # Final ensemble predictions
    oof_predictions_ensemble = 0.6 * oof_predictions_lgb + 0.4 * oof_predictions_xgb
    final_predictions = np.argmax(oof_predictions_ensemble, axis=1)
    
    # Overall metrics
    final_score, binary_f1, macro_f1 = competition_metric(y_train, final_predictions)
    print(f'\n========== Overall Results ==========')
    print(f'Competition Score: {final_score:.4f}')
    print(f'Binary F1: {binary_f1:.4f}')
    print(f'Macro F1: {macro_f1:.4f}')
    print(f'Accuracy: {accuracy_score(y_train, final_predictions):.4f}')
    
    # Top features
    feature_importance_mean = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
    print(f'\nTop 10 Features:')
    for i, (feat, imp) in enumerate(feature_importance_mean.head(10).items(), 1):
        print(f'{i:2d}. {feat}: {imp:.2f}')
    
    return {
        'lgb_models': lgb_models,
        'xgb_models': xgb_models,
        'feature_columns': list(X_train.columns),
        'feature_importance': feature_importance_mean.to_dict(),
        'oof_predictions': oof_predictions_ensemble,
        'metrics': {
            'competition_score': final_score,
            'binary_f1': binary_f1,
            'macro_f1': macro_f1,
            'accuracy': accuracy_score(y_train, final_predictions)
        }
    }

# ======================== Inference ========================

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """Prediction function for Kaggle inference server"""
    try:
        # Convert to pandas
        seq_df = sequence.to_pandas() if isinstance(sequence, pl.DataFrame) else sequence
        demo_df = demographics.to_pandas() if isinstance(demographics, pl.DataFrame) else demographics
        
        # Extract features
        features = extract_comprehensive_features(seq_df, demo_df)
        
        # Ensure all features present
        for col in feature_cols:
            if col not in features.columns:
                features[col] = 0
        
        X_pred = features[feature_cols]
        
        # Get predictions from all models
        lgb_preds = np.zeros((1, 18))
        xgb_preds = np.zeros((1, 18))
        
        for lgb_model in lgb_models:
            lgb_preds += lgb_model.predict_proba(X_pred) / len(lgb_models)
        
        for xgb_model in xgb_models:
            xgb_preds += xgb_model.predict_proba(X_pred) / len(xgb_models)
        
        # Ensemble
        ensemble_pred = 0.6 * lgb_preds + 0.4 * xgb_preds
        
        # Apply post-processing (boost BFRB behaviors)
        if np.max(ensemble_pred[0, :8]) > 0.35:  # If any BFRB behavior has high confidence
            ensemble_pred[0, :8] *= 1.25  # Boost BFRB probabilities
        
        final_pred = np.argmax(ensemble_pred[0])
        
        return REVERSE_GESTURE_MAPPER.get(final_pred, 'Text on phone')
        
    except Exception as e:
        print(f'Prediction error: {e}')
        return 'Text on phone'  # Default fallback

# ======================== Main Execution ========================

def main():
    """Main execution function"""
    global lgb_models, xgb_models, feature_cols
    
    print('CMI BFRB Detection - IMU Improved Model')
    print('=' * 50)
    
    # Check if we're in training or inference mode
    # IMPORTANT: In Kaggle submission environment, KAGGLE_IS_COMPETITION_RERUN is set
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        # ========== Inference Mode ==========
        print('Running in INFERENCE mode (Competition Environment)')
        
        # Load pre-trained models
        print('Loading models...')
        model_file = os.path.join(MODEL_PATH, 'model_data.pkl')
        
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            lgb_models = model_data['lgb_models']
            xgb_models = model_data['xgb_models']
            feature_cols = model_data['feature_columns']
            print(f'Loaded {len(lgb_models)} LightGBM and {len(xgb_models)} XGBoost models')
        else:
            print('ERROR: Model file not found! Looking for alternative paths...')
            # Try alternative paths
            alternative_paths = [
                '/kaggle/input/cmi-imu-models/model_data.pkl',
                '/kaggle/working/model_data.pkl',
                './model_data.pkl'
            ]
            
            model_loaded = False
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    print(f'Found model at: {alt_path}')
                    with open(alt_path, 'rb') as f:
                        model_data = pickle.load(f)
                    lgb_models = model_data['lgb_models']
                    xgb_models = model_data['xgb_models']
                    feature_cols = model_data['feature_columns']
                    model_loaded = True
                    break
            
            if not model_loaded:
                print('CRITICAL ERROR: No model file found!')
                lgb_models = []
                xgb_models = []
                feature_cols = []
        
        # Initialize inference server
        print('Initializing inference server...')
        sys.path.append('/kaggle/input/cmi-detect-behavior-with-sensor-data')
        import kaggle_evaluation.cmi_inference_server
        
        inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
        print('Serving predictions...')
        inference_server.serve()
        
        print('\n========== Inference Complete ==========')
        print('Note: submission.parquet is generated automatically by Kaggle')
        
    else:
        # ========== Training/Local Mode ==========
        print('Running in TRAINING/LOCAL mode')
        
        # Load training data
        print('\nLoading data...')
        train_df = pl.read_csv(os.path.join(DATA_PATH, 'train.csv'))
        train_demographics = pl.read_csv(os.path.join(DATA_PATH, 'train_demographics.csv'))
        
        # Filter for 'Performs gesture' phase only
        train_filtered = train_df.filter(pl.col('behavior') == 'Performs gesture')
        print(f'Total sequences: {train_filtered["sequence_id"].n_unique()}')
        
        # Extract features for all sequences
        print('\nExtracting features...')
        features_list = []
        labels = []
        subjects = []
        
        sequences = list(train_filtered.group_by('sequence_id', maintain_order=True))
        
        # Use subset for debugging
        if DEBUG:
            sequences = sequences[:500]
            print(f'DEBUG MODE: Using only {len(sequences)} sequences')
        
        for idx, (seq_id, seq_data) in enumerate(sequences):
            if idx % 500 == 0:
                print(f'Processing: {idx}/{len(sequences)} sequences')
            
            # Get sequence info
            sequence_id = seq_id[0] if isinstance(seq_id, tuple) else seq_id
            subject_id = seq_data['subject'][0]
            gesture = seq_data['gesture'][0]
            
            # Get demographics
            subject_demographics = train_demographics.filter(pl.col('subject') == subject_id)
            
            # Convert to pandas and extract features
            seq_df = seq_data.to_pandas()
            demo_df = subject_demographics.to_pandas() if not subject_demographics.is_empty() else pd.DataFrame()
            
            features = extract_comprehensive_features(seq_df, demo_df)
            features_list.append(features)
            labels.append(GESTURE_MAPPER[gesture])
            subjects.append(subject_id)
        
        # Combine features
        X_train = pd.concat(features_list, ignore_index=True)
        y_train = np.array(labels)
        subjects = np.array(subjects)
        
        print(f'\nFeature matrix shape: {X_train.shape}')
        print(f'Labels shape: {y_train.shape}')
        print(f'Unique subjects: {len(np.unique(subjects))}')
        
        # Train models
        model_results = train_models(X_train, y_train, subjects)
        
        # Save models and metadata
        print('\nSaving models...')
        model_data = {
            'lgb_models': model_results['lgb_models'],
            'xgb_models': model_results['xgb_models'],
            'feature_columns': model_results['feature_columns'],
            'feature_importance': model_results['feature_importance'],
            'metrics': model_results['metrics']
        }
        
        with open(os.path.join(MODEL_PATH, 'model_data.pkl'), 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save training results
        with open(os.path.join(MODEL_PATH, 'training_results.json'), 'w') as f:
            json.dump(model_results['metrics'], f, indent=2)
        
        print(f'\nModels saved to {MODEL_PATH}')
        print('\n========== Training Complete ==========')
        
        # Test inference locally if not in Kaggle
        if not IS_KAGGLE:
            print('\nTesting inference function...')
            lgb_models = model_results['lgb_models']
            xgb_models = model_results['xgb_models']
            feature_cols = model_results['feature_columns']
            
            test_seq = pl.DataFrame({
                'acc_x': np.random.randn(100),
                'acc_y': np.random.randn(100),
                'acc_z': np.random.randn(100),
                'rot_w': np.random.randn(100),
                'rot_x': np.random.randn(100),
                'rot_y': np.random.randn(100),
                'rot_z': np.random.randn(100)
            })
            test_demo = pl.DataFrame({
                'age': [25],
                'adult_child': [1],
                'sex': [0],
                'handedness': [1]
            })
            
            result = predict(test_seq, test_demo)
            print(f'Test prediction: {result}')
            assert isinstance(result, str) and result in GESTURE_MAPPER, 'Invalid prediction!'
            print('Inference test passed!')
        
        # If in Kaggle but not in competition environment, run local gateway
        elif IS_KAGGLE and not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            print('\nRunning local gateway for testing...')
            lgb_models = model_results['lgb_models']
            xgb_models = model_results['xgb_models']
            feature_cols = model_results['feature_columns']
            
            sys.path.append('/kaggle/input/cmi-detect-behavior-with-sensor-data')
            import kaggle_evaluation.cmi_inference_server
            
            inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
            inference_server.run_local_gateway(
                data_paths=(
                    os.path.join(DATA_PATH, 'test.csv'),
                    os.path.join(DATA_PATH, 'test_demographics.csv'),
                )
            )
            print('\n✓ Local test complete!')
            print('✓ submission.parquet has been generated for local testing')
    
    print('\n========== Process Complete ==========')

if __name__ == '__main__':
    main()