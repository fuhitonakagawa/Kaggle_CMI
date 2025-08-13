#!/usr/bin/env python3
# ====================================================================================================
# CMI BFRB Detection - Comprehensive Solution v5.0
# Target Score: 0.85+ (Binary F1: 0.95+, Macro F1: 0.75+)
# Architecture: Hierarchical Classification + Multi-Model Ensemble
# ====================================================================================================

import os
import sys
import json
import pickle
import warnings
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, welch
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Machine Learning Models
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available, will use alternative models")

# Deep Learning
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout,
    Bidirectional, LSTM, GRU, GlobalAveragePooling1D, Dense, Multiply, Reshape,
    Lambda, Concatenate, GaussianNoise, Flatten, Layer
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.utils import Sequence, to_categorical

warnings.filterwarnings("ignore")

# ====================================================================================================
# CONFIGURATION
# ====================================================================================================

class Config:
    """Global configuration for the pipeline."""
    
    # Paths
    DATA_PATH = Path("cmi-detect-behavior-with-sensor-data/")
    OUTPUT_PATH = Path("5_CMI_comprehensive/outputs/")
    MODEL_PATH = Path("5_CMI_comprehensive/models/")
    
    # Create directories
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    
    # Data parameters
    SAMPLE_RATE = 20  # Hz
    N_FOLDS = 5
    RANDOM_STATE = 42
    
    # Sequence processing
    SEQUENCE_MAX_LEN = 500
    PAD_PERCENTILE = 95
    
    # Deep Learning parameters
    BATCH_SIZE = 64
    EPOCHS = 150
    PATIENCE = 30
    LR_INIT = 5e-4
    WEIGHT_DECAY = 1e-4
    MIXUP_ALPHA = 0.4
    
    # Model architecture
    CNN_FILTERS = [64, 128, 256]
    LSTM_UNITS = 128
    GRU_UNITS = 128
    ATTENTION_UNITS = 128
    DROPOUT_RATE = 0.4
    
    # Gradient Boosting parameters
    LGBM_PARAMS = {
        'objective': 'multiclass',
        'num_class': 18,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'num_threads': -1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'min_child_samples': 20,
        'max_depth': -1
    }
    
    XGB_PARAMS = {
        'objective': 'multi:softprob',
        'num_class': 18,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'tree_method': 'hist',
        'random_state': 42
    }
    
    # Gesture mapping
    GESTURE_MAPPER = {
        "Above ear - pull hair": 0,
        "Cheek - pinch skin": 1,
        "Eyebrow - pull hair": 2,
        "Eyelash - pull hair": 3,
        "Forehead - pull hairline": 4,
        "Forehead - scratch": 5,
        "Neck - pinch skin": 6,
        "Neck - scratch": 7,
        "Drink from bottle/cup": 8,
        "Feel around in tray and pull out an object": 9,
        "Glasses on/off": 10,
        "Pinch knee/leg skin": 11,
        "Pull air toward your face": 12,
        "Scratch knee/leg skin": 13,
        "Text on phone": 14,
        "Wave hello": 15,
        "Write name in air": 16,
        "Write name on leg": 17,
    }
    
    REVERSE_GESTURE_MAPPER = {v: k for k, v in GESTURE_MAPPER.items()}
    
    # BFRB classes (0-7)
    BFRB_CLASSES = list(range(8))
    NON_BFRB_CLASSES = list(range(8, 18))

# ====================================================================================================
# UTILITY FUNCTIONS
# ====================================================================================================

def seed_everything(seed=42):
    """Set all random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def configure_gpu():
    """Configure GPU for optimal performance."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            if sys.platform == 'darwin':
                print("✓ Metal GPU detected (Mac)")
            else:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✓ CUDA GPU configured: {len(gpus)} device(s)")
            return True
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")
            return False
    print("⚠️ No GPU found, using CPU")
    return False

# Initialize environment
seed_everything(Config.RANDOM_STATE)
GPU_AVAILABLE = configure_gpu()

# ====================================================================================================
# FEATURE ENGINEERING
# ====================================================================================================

class FeatureEngineering:
    """Comprehensive feature engineering for sensor data."""
    
    @staticmethod
    def remove_gravity(acc_data: np.ndarray, rot_data: np.ndarray) -> np.ndarray:
        """Remove gravity component from accelerometer data using quaternion rotation."""
        num_samples = acc_data.shape[0]
        linear_accel = np.zeros_like(acc_data)
        gravity_world = np.array([0, 0, 9.81])
        
        for i in range(num_samples):
            if np.all(np.isnan(rot_data[i])) or np.all(np.isclose(rot_data[i], 0)):
                linear_accel[i] = acc_data[i]
                continue
            
            try:
                # Quaternion format: [x, y, z, w]
                rotation = R.from_quat(rot_data[i])
                gravity_sensor = rotation.apply(gravity_world, inverse=True)
                linear_accel[i] = acc_data[i] - gravity_sensor
            except ValueError:
                linear_accel[i] = acc_data[i]
        
        return linear_accel
    
    @staticmethod
    def calculate_angular_velocity(rot_data: np.ndarray, sample_rate: int = 20) -> np.ndarray:
        """Calculate angular velocity from quaternion data."""
        n_samples = len(rot_data)
        angular_vel = np.zeros((n_samples, 3))
        dt = 1.0 / sample_rate
        
        for i in range(n_samples - 1):
            try:
                r_t = R.from_quat(rot_data[i])
                r_t_plus = R.from_quat(rot_data[i + 1])
                delta_rot = r_t.inv() * r_t_plus
                angular_vel[i] = delta_rot.as_rotvec() / dt
            except (ValueError, ZeroDivisionError):
                angular_vel[i] = 0
        
        if n_samples > 1:
            angular_vel[-1] = angular_vel[-2]
        
        return angular_vel
    
    @staticmethod
    def calculate_angular_distance(rot_data: np.ndarray) -> np.ndarray:
        """Calculate angular distance between consecutive frames."""
        n_samples = len(rot_data)
        angular_dist = np.zeros(n_samples)
        
        for i in range(n_samples - 1):
            try:
                r1 = R.from_quat(rot_data[i])
                r2 = R.from_quat(rot_data[i + 1])
                relative_rotation = r1.inv() * r2
                angle = np.linalg.norm(relative_rotation.as_rotvec())
                angular_dist[i] = angle
            except ValueError:
                angular_dist[i] = 0
        
        return angular_dist
    
    @staticmethod
    def extract_fft_features(signal: np.ndarray, sample_rate: int = 20) -> Dict[str, float]:
        """Extract FFT-based frequency features."""
        if len(signal) < 10:
            return {
                'dominant_freq': 0,
                'spectral_centroid': 0,
                'spectral_energy': 0,
                'spectral_entropy': 0,
                'band_power_low': 0,
                'band_power_mid': 0,
                'band_power_high': 0
            }
        
        # FFT calculation
        fft_vals = np.abs(fft(signal))
        freqs = fftfreq(len(signal), 1/sample_rate)
        
        # Positive frequencies only
        pos_mask = freqs > 0
        fft_vals = fft_vals[pos_mask]
        freqs = freqs[pos_mask]
        
        if len(fft_vals) == 0:
            return {
                'dominant_freq': 0,
                'spectral_centroid': 0,
                'spectral_energy': 0,
                'spectral_entropy': 0,
                'band_power_low': 0,
                'band_power_mid': 0,
                'band_power_high': 0
            }
        
        # Spectral features
        total_power = np.sum(fft_vals)
        if total_power > 0:
            spectral_centroid = np.sum(freqs * fft_vals) / total_power
            spectral_entropy = -np.sum((fft_vals/total_power) * np.log2(fft_vals/total_power + 1e-10))
        else:
            spectral_centroid = 0
            spectral_entropy = 0
        
        return {
            'dominant_freq': freqs[np.argmax(fft_vals)] if len(fft_vals) > 0 else 0,
            'spectral_centroid': spectral_centroid,
            'spectral_energy': np.sum(fft_vals**2),
            'spectral_entropy': spectral_entropy,
            'band_power_low': np.sum(fft_vals[(freqs >= 0) & (freqs < 2)]),
            'band_power_mid': np.sum(fft_vals[(freqs >= 2) & (freqs < 5)]),
            'band_power_high': np.sum(fft_vals[(freqs >= 5) & (freqs < 10)])
        }
    
    @staticmethod
    def extract_statistical_features(signal: np.ndarray) -> Dict[str, float]:
        """Extract statistical features from signal."""
        if len(signal) == 0:
            return {
                'mean': 0, 'std': 0, 'max': 0, 'min': 0,
                'range': 0, 'median': 0, 'q25': 0, 'q75': 0,
                'skew': 0, 'kurtosis': 0, 'zero_crossing': 0
            }
        
        peaks, _ = find_peaks(signal) if len(signal) > 3 else ([], {})
        
        return {
            'mean': np.mean(signal),
            'std': np.std(signal),
            'max': np.max(signal),
            'min': np.min(signal),
            'range': np.ptp(signal),
            'median': np.median(signal),
            'q25': np.percentile(signal, 25),
            'q75': np.percentile(signal, 75),
            'skew': stats.skew(signal) if len(signal) > 2 else 0,
            'kurtosis': stats.kurtosis(signal) if len(signal) > 3 else 0,
            'zero_crossing': np.sum(np.diff(np.sign(signal)) != 0) if len(signal) > 1 else 0,
            'peak_count': len(peaks)
        }
    
    @staticmethod
    def extract_tof_features(tof_data: pd.DataFrame, sensor_id: int) -> Dict[str, float]:
        """Extract features from TOF sensor data."""
        features = {}
        
        # Check if TOF data exists
        tof_cols = [f"tof_{sensor_id}_v{p}" for p in range(64)]
        if not all(col in tof_data.columns for col in tof_cols):
            return {
                f'tof_{sensor_id}_mean': 0,
                f'tof_{sensor_id}_std': 0,
                f'tof_{sensor_id}_min': 0,
                f'tof_{sensor_id}_max': 0
            }
        
        # Get TOF data and handle missing values
        sensor_data = tof_data[tof_cols].replace(-1, np.nan)
        
        features[f'tof_{sensor_id}_mean'] = sensor_data.mean(axis=1).fillna(0).mean()
        features[f'tof_{sensor_id}_std'] = sensor_data.std(axis=1).fillna(0).mean()
        features[f'tof_{sensor_id}_min'] = sensor_data.min(axis=1).fillna(0).mean()
        features[f'tof_{sensor_id}_max'] = sensor_data.max(axis=1).fillna(0).mean()
        
        return features

# ====================================================================================================
# DATA PROCESSING
# ====================================================================================================

class DataProcessor:
    """Comprehensive data processing pipeline."""
    
    def __init__(self, config: Config):
        self.config = config
        self.feature_eng = FeatureEngineering()
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def process_sequence(self, df: pd.DataFrame, is_training: bool = False) -> np.ndarray:
        """Process a single sequence and extract all features."""
        features = []
        
        # Basic IMU data
        acc_data = df[['acc_x', 'acc_y', 'acc_z']].fillna(0).values
        rot_data = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']].fillna([0, 0, 0, 1]).values
        
        # Remove gravity to get linear acceleration
        linear_accel = self.feature_eng.remove_gravity(acc_data, rot_data)
        
        # Calculate angular features
        angular_vel = self.feature_eng.calculate_angular_velocity(rot_data)
        angular_dist = self.feature_eng.calculate_angular_distance(rot_data)
        
        # Magnitude features
        acc_mag = np.linalg.norm(acc_data, axis=1)
        linear_acc_mag = np.linalg.norm(linear_accel, axis=1)
        angular_vel_mag = np.linalg.norm(angular_vel, axis=1)
        
        # Jerk features (derivatives)
        acc_mag_jerk = np.diff(acc_mag, prepend=acc_mag[0])
        linear_acc_mag_jerk = np.diff(linear_acc_mag, prepend=linear_acc_mag[0])
        
        # Statistical features for each signal
        stat_features = {}
        
        # Acceleration features
        for i, axis in enumerate(['x', 'y', 'z']):
            stat_features.update({
                f'acc_{axis}_{k}': v 
                for k, v in self.feature_eng.extract_statistical_features(acc_data[:, i]).items()
            })
            stat_features.update({
                f'linear_acc_{axis}_{k}': v 
                for k, v in self.feature_eng.extract_statistical_features(linear_accel[:, i]).items()
            })
            stat_features.update({
                f'angular_vel_{axis}_{k}': v 
                for k, v in self.feature_eng.extract_statistical_features(angular_vel[:, i]).items()
            })
        
        # Magnitude statistical features
        stat_features.update({
            f'acc_mag_{k}': v 
            for k, v in self.feature_eng.extract_statistical_features(acc_mag).items()
        })
        stat_features.update({
            f'linear_acc_mag_{k}': v 
            for k, v in self.feature_eng.extract_statistical_features(linear_acc_mag).items()
        })
        stat_features.update({
            f'angular_vel_mag_{k}': v 
            for k, v in self.feature_eng.extract_statistical_features(angular_vel_mag).items()
        })
        
        # FFT features for key signals
        fft_features = {}
        for i, axis in enumerate(['x', 'y', 'z']):
            fft_features.update({
                f'acc_{axis}_fft_{k}': v 
                for k, v in self.feature_eng.extract_fft_features(acc_data[:, i]).items()
            })
        
        fft_features.update({
            f'acc_mag_fft_{k}': v 
            for k, v in self.feature_eng.extract_fft_features(acc_mag).items()
        })
        
        # TOF features
        tof_features = {}
        for sensor_id in range(1, 6):
            tof_features.update(self.feature_eng.extract_tof_features(df, sensor_id))
        
        # Thermal features
        thermal_features = {}
        for i in range(1, 6):
            thm_col = f'thm_{i}'
            if thm_col in df.columns:
                thm_data = df[thm_col].fillna(0).values
                thermal_features[f'thm_{i}_mean'] = np.mean(thm_data)
                thermal_features[f'thm_{i}_std'] = np.std(thm_data)
            else:
                thermal_features[f'thm_{i}_mean'] = 0
                thermal_features[f'thm_{i}_std'] = 0
        
        # Cross-correlation features
        corr_features = {
            'corr_acc_xy': np.corrcoef(acc_data[:, 0], acc_data[:, 1])[0, 1] if len(acc_data) > 1 else 0,
            'corr_acc_xz': np.corrcoef(acc_data[:, 0], acc_data[:, 2])[0, 1] if len(acc_data) > 1 else 0,
            'corr_acc_yz': np.corrcoef(acc_data[:, 1], acc_data[:, 2])[0, 1] if len(acc_data) > 1 else 0,
        }
        
        # Combine all features
        all_features = {**stat_features, **fft_features, **tof_features, **thermal_features, **corr_features}
        
        # Convert to array
        if self.feature_columns is None:
            self.feature_columns = list(all_features.keys())
        
        feature_array = np.array([all_features.get(col, 0) for col in self.feature_columns])
        
        # Handle NaN and inf values
        feature_array = np.nan_to_num(feature_array, nan=0, posinf=0, neginf=0)
        
        return feature_array
    
    def process_all_sequences(self, df: pd.DataFrame, demo_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process all sequences and return features, labels, and groups."""
        features = []
        labels = []
        groups = []
        
        # Process each sequence
        for seq_id in df['sequence_id'].unique():
            seq_data = df[df['sequence_id'] == seq_id]
            
            # Extract features
            seq_features = self.process_sequence(seq_data)
            features.append(seq_features)
            
            # Get label if available
            if 'gesture' in seq_data.columns:
                gesture = seq_data['gesture'].iloc[0]
                labels.append(Config.GESTURE_MAPPER[gesture])
            
            # Get subject for grouping
            subject = seq_data['subject'].iloc[0]
            groups.append(subject)
        
        return np.array(features), np.array(labels), np.array(groups)

# ====================================================================================================
# HIERARCHICAL CLASSIFIER
# ====================================================================================================

class HierarchicalClassifier:
    """Hierarchical classification strategy: Binary → BFRB → Full."""
    
    def __init__(self):
        self.binary_model = None  # BFRB vs Non-BFRB
        self.bfrb_model = None    # 8-class BFRB
        self.full_model = None    # 18-class full
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None):
        """Train hierarchical classifiers."""
        print("Training Hierarchical Classifier...")
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Step 1: Train binary classifier
        print("  Step 1: Training Binary Classifier (BFRB vs Non-BFRB)...")
        y_binary = (y < 8).astype(int)
        
        # Calculate class weights for binary
        binary_weights = compute_class_weight('balanced', classes=np.unique(y_binary), y=y_binary)
        binary_weight_dict = {0: binary_weights[0], 1: binary_weights[1] * 1.5}  # Boost BFRB weight
        
        self.binary_model = lgb.LGBMClassifier(
            **{**Config.LGBM_PARAMS, 
               'objective': 'binary',
               'num_class': 1,
               'metric': 'binary_logloss',
               'n_estimators': 500,
               'class_weight': binary_weight_dict}
        )
        self.binary_model.fit(X_scaled, y_binary)
        
        # Step 2: Train BFRB classifier
        print("  Step 2: Training BFRB Classifier (8 classes)...")
        bfrb_mask = y < 8
        X_bfrb = X_scaled[bfrb_mask]
        y_bfrb = y[bfrb_mask]
        
        if len(np.unique(y_bfrb)) > 1:
            self.bfrb_model = lgb.LGBMClassifier(
                **{**Config.LGBM_PARAMS,
                   'num_class': 8,
                   'n_estimators': 700,
                   'class_weight': 'balanced'}
            )
            self.bfrb_model.fit(X_bfrb, y_bfrb)
        
        # Step 3: Train full classifier
        print("  Step 3: Training Full Classifier (18 classes)...")
        self.full_model = lgb.LGBMClassifier(
            **{**Config.LGBM_PARAMS,
               'n_estimators': 1000,
               'class_weight': 'balanced'}
        )
        self.full_model.fit(X_scaled, y)
        
        print("✓ Hierarchical Classifier training complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using hierarchical strategy."""
        X_scaled = self.scaler.transform(X)
        n_samples = X_scaled.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        
        # Step 1: Binary prediction
        binary_pred = self.binary_model.predict(X_scaled)
        binary_proba = self.binary_model.predict_proba(X_scaled)[:, 1]
        
        # Step 2: Hierarchical prediction
        for i in range(n_samples):
            if binary_pred[i] == 1 and binary_proba[i] > 0.6:  # BFRB with high confidence
                if self.bfrb_model is not None:
                    # Use BFRB-specific model
                    bfrb_pred = self.bfrb_model.predict(X_scaled[i:i+1])[0]
                    predictions[i] = bfrb_pred
                else:
                    # Fallback to full model
                    full_pred = self.full_model.predict(X_scaled[i:i+1])[0]
                    predictions[i] = full_pred if full_pred < 8 else 0
            else:
                # Use full model for non-BFRB or low confidence
                full_pred = self.full_model.predict(X_scaled[i:i+1])[0]
                if binary_pred[i] == 0 and full_pred < 8:
                    # Correct inconsistency: predicted as non-BFRB but full model says BFRB
                    predictions[i] = 8  # Default to first non-BFRB class
                else:
                    predictions[i] = full_pred
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using hierarchical strategy."""
        X_scaled = self.scaler.transform(X)
        n_samples = X_scaled.shape[0]
        proba = np.zeros((n_samples, 18))
        
        # Get binary probabilities
        binary_proba = self.binary_model.predict_proba(X_scaled)
        
        for i in range(n_samples):
            if binary_proba[i, 1] > 0.6:  # BFRB likely
                if self.bfrb_model is not None:
                    bfrb_proba = self.bfrb_model.predict_proba(X_scaled[i:i+1])[0]
                    proba[i, :8] = bfrb_proba * binary_proba[i, 1]
                    proba[i, 8:] = (1 - binary_proba[i, 1]) / 10  # Small prob for non-BFRB
                else:
                    full_proba = self.full_model.predict_proba(X_scaled[i:i+1])[0]
                    proba[i] = full_proba
            else:
                full_proba = self.full_model.predict_proba(X_scaled[i:i+1])[0]
                # Adjust probabilities based on binary confidence
                proba[i, :8] = full_proba[:8] * binary_proba[i, 1]
                proba[i, 8:] = full_proba[8:] * binary_proba[i, 0]
        
        # Normalize probabilities
        proba = proba / proba.sum(axis=1, keepdims=True)
        
        return proba

# ====================================================================================================
# ENSEMBLE MODEL
# ====================================================================================================

class EnsembleModel:
    """Multi-model ensemble with optimized weighting."""
    
    def __init__(self):
        self.models = []
        self.weights = None
        self.scaler = StandardScaler()
        
    def add_model(self, model, name: str, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models.append({
            'model': model,
            'name': name,
            'weight': weight
        })
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None):
        """Train all models in the ensemble."""
        X_scaled = self.scaler.fit_transform(X)
        
        for model_dict in self.models:
            print(f"Training {model_dict['name']}...")
            
            if model_dict['name'] == 'hierarchical':
                # Hierarchical classifier needs special handling
                model_dict['model'].fit(X, y, groups)
            elif 'lgbm' in model_dict['name'].lower():
                # LightGBM
                model = lgb.LGBMClassifier(**Config.LGBM_PARAMS, n_estimators=800)
                model.fit(X_scaled, y)
                model_dict['model'] = model
            elif 'xgb' in model_dict['name'].lower():
                # XGBoost
                model = xgb.XGBClassifier(**Config.XGB_PARAMS, n_estimators=800)
                model.fit(X_scaled, y)
                model_dict['model'] = model
            elif 'catboost' in model_dict['name'].lower() and CATBOOST_AVAILABLE:
                # CatBoost
                model = cb.CatBoostClassifier(
                    iterations=800,
                    learning_rate=0.05,
                    depth=8,
                    loss_function='MultiClass',
                    random_seed=42,
                    verbose=False
                )
                model.fit(X_scaled, y)
                model_dict['model'] = model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction with weighted voting."""
        predictions = []
        weights = []
        
        for model_dict in self.models:
            if model_dict['name'] == 'hierarchical':
                pred = model_dict['model'].predict(X)
            else:
                X_scaled = self.scaler.transform(X)
                pred = model_dict['model'].predict(X_scaled)
            
            predictions.append(pred)
            weights.append(model_dict['weight'])
        
        # Weighted voting
        predictions = np.array(predictions)
        weights = np.array(weights) / np.sum(weights)
        
        # For each sample, weighted mode
        final_predictions = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            # Count weighted votes for each class
            class_votes = np.zeros(18)
            for j, pred in enumerate(predictions[:, i]):
                class_votes[pred] += weights[j]
            final_predictions[i] = np.argmax(class_votes)
        
        return final_predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Ensemble probability prediction."""
        probas = []
        weights = []
        
        for model_dict in self.models:
            if model_dict['name'] == 'hierarchical':
                proba = model_dict['model'].predict_proba(X)
            else:
                X_scaled = self.scaler.transform(X)
                proba = model_dict['model'].predict_proba(X_scaled)
            
            probas.append(proba)
            weights.append(model_dict['weight'])
        
        # Weighted average of probabilities
        weights = np.array(weights) / np.sum(weights)
        ensemble_proba = np.zeros_like(probas[0])
        
        for proba, weight in zip(probas, weights):
            ensemble_proba += proba * weight
        
        return ensemble_proba

# ====================================================================================================
# COMPETITION METRICS
# ====================================================================================================

def calculate_competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate the competition metric (Binary F1 + Macro F1) / 2."""
    # Binary F1 (BFRB vs Non-BFRB)
    binary_true = (y_true < 8).astype(int)
    binary_pred = (y_pred < 8).astype(int)
    binary_f1 = f1_score(binary_true, binary_pred, average='binary')
    
    # Macro F1 for BFRB classes only
    bfrb_mask = y_true < 8
    if bfrb_mask.any():
        # Only calculate macro F1 for BFRB samples
        macro_f1 = f1_score(y_true[bfrb_mask], y_pred[bfrb_mask], average='macro')
    else:
        macro_f1 = 0.0
    
    # Combined score
    combined_score = (binary_f1 + macro_f1) / 2
    
    return {
        'binary_f1': binary_f1,
        'macro_f1': macro_f1,
        'combined_score': combined_score
    }

# ====================================================================================================
# CROSS-VALIDATION WITH STRATIFIED GROUP K-FOLD
# ====================================================================================================

def cross_validate_with_sgkf(X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
                            n_splits: int = 5) -> Dict[str, List[float]]:
    """Perform cross-validation with StratifiedGroupKFold."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=Config.RANDOM_STATE)
    
    scores = {
        'binary_f1': [],
        'macro_f1': [],
        'combined_score': []
    }
    
    print(f"\nStarting {n_splits}-Fold Cross Validation with StratifiedGroupKFold...")
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        groups_train = groups[train_idx]
        
        # Create ensemble model
        ensemble = EnsembleModel()
        
        # Add hierarchical classifier
        hierarchical_clf = HierarchicalClassifier()
        ensemble.add_model(hierarchical_clf, 'hierarchical', weight=2.0)
        
        # Add LightGBM
        ensemble.add_model(None, 'lgbm_1', weight=1.5)
        
        # Add XGBoost
        ensemble.add_model(None, 'xgb_1', weight=1.0)
        
        # Train ensemble
        ensemble.fit(X_train, y_train, groups_train)
        
        # Predict
        y_pred = ensemble.predict(X_val)
        
        # Calculate metrics
        fold_metrics = calculate_competition_metric(y_val, y_pred)
        
        for key in scores:
            scores[key].append(fold_metrics[key])
        
        print(f"  Binary F1: {fold_metrics['binary_f1']:.4f}")
        print(f"  Macro F1: {fold_metrics['macro_f1']:.4f}")
        print(f"  Combined Score: {fold_metrics['combined_score']:.4f}")
    
    # Print summary
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    for key in scores:
        mean_score = np.mean(scores[key])
        std_score = np.std(scores[key])
        print(f"{key:15s}: {mean_score:.4f} (+/- {std_score:.4f})")
    print("="*60)
    
    return scores

# ====================================================================================================
# MAIN TRAINING PIPELINE
# ====================================================================================================

def train_full_pipeline():
    """Main training pipeline for the comprehensive solution."""
    print("="*70)
    print("CMI BFRB Detection - Comprehensive Solution v5.0")
    print("="*70)
    
    # Check if running in Kaggle
    is_kaggle = os.path.exists("/kaggle/input")
    
    # Load data
    print("\nLoading data...")
    if is_kaggle:
        train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
        demo_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
    else:
        train_df = pd.read_csv(Config.DATA_PATH / "train.csv")
        demo_df = pd.read_csv(Config.DATA_PATH / "train_demographics.csv")
    
    print(f"Loaded {len(train_df)} samples from {train_df['sequence_id'].nunique()} sequences")
    
    # Process data
    print("\nProcessing sequences and extracting features...")
    processor = DataProcessor(Config)
    X, y, groups = processor.process_all_sequences(train_df, demo_df)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of unique subjects: {len(np.unique(groups))}")
    
    # Cross-validation
    cv_scores = cross_validate_with_sgkf(X, y, groups, n_splits=Config.N_FOLDS)
    
    # Train final model on all data
    print("\n" + "="*70)
    print("Training Final Ensemble Model on All Data...")
    print("="*70)
    
    final_ensemble = EnsembleModel()
    
    # Add models to ensemble
    hierarchical_clf = HierarchicalClassifier()
    final_ensemble.add_model(hierarchical_clf, 'hierarchical', weight=2.0)
    final_ensemble.add_model(None, 'lgbm_full', weight=1.5)
    final_ensemble.add_model(None, 'xgb_full', weight=1.0)
    
    if CATBOOST_AVAILABLE:
        final_ensemble.add_model(None, 'catboost_full', weight=1.0)
    
    # Train ensemble
    final_ensemble.fit(X, y, groups)
    
    # Save models and artifacts
    print("\nSaving models and artifacts...")
    
    # Save processor
    joblib.dump(processor, Config.MODEL_PATH / 'processor.pkl')
    joblib.dump(processor.feature_columns, Config.MODEL_PATH / 'feature_columns.pkl')
    
    # Save ensemble
    joblib.dump(final_ensemble, Config.MODEL_PATH / 'final_ensemble.pkl')
    
    # Save results
    results = {
        'cv_scores': cv_scores,
        'mean_binary_f1': np.mean(cv_scores['binary_f1']),
        'mean_macro_f1': np.mean(cv_scores['macro_f1']),
        'mean_combined_score': np.mean(cv_scores['combined_score']),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(Config.OUTPUT_PATH / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Training complete! Models saved to:", Config.MODEL_PATH)
    print(f"✓ Final CV Score: {results['mean_combined_score']:.4f}")
    
    return final_ensemble, processor

# ====================================================================================================
# INFERENCE FOR KAGGLE SUBMISSION
# ====================================================================================================

def predict_for_kaggle(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """Predict function for Kaggle evaluation API."""
    # Load saved models
    processor = joblib.load(Config.MODEL_PATH / 'processor.pkl')
    ensemble = joblib.load(Config.MODEL_PATH / 'final_ensemble.pkl')
    
    # Convert to pandas
    df_seq = sequence.to_pandas()
    
    # Process sequence
    features = processor.process_sequence(df_seq)
    features = features.reshape(1, -1)
    
    # Predict
    prediction = ensemble.predict(features)[0]
    
    # Convert to gesture name
    gesture_name = Config.REVERSE_GESTURE_MAPPER[prediction]
    
    return gesture_name

# ====================================================================================================
# MAIN EXECUTION
# ====================================================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CMI BFRB Detection - Comprehensive Solution')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'],
                      help='Mode: train or inference')
    parser.add_argument('--kaggle', action='store_true', help='Run in Kaggle environment')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Train the full pipeline
        ensemble, processor = train_full_pipeline()
        
    elif args.mode == 'inference':
        # Kaggle inference mode
        if args.kaggle:
            import kaggle_evaluation.cmi_inference_server
            
            # Set up prediction function
            def predict(sequence, demographics):
                return predict_for_kaggle(sequence, demographics)
            
            # Create inference server
            inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
            
            # Run inference
            if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                inference_server.serve()
            else:
                # Local testing
                inference_server.run_local_gateway(
                    data_paths=(
                        '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
                        '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
                    )
                )
        else:
            print("Inference mode requires --kaggle flag")
    
    print("\n✓ Execution complete!")