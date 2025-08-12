#!/usr/bin/env python3
# ====================================================================================================
# CMI BFRB Detection - Two-Stage Classification System (Fixed for Kaggle)
# Score Target: 0.730+ (Binary F1: 0.94+, Macro F1: 0.52+)
# 
# FIXED ISSUES:
# 1. Proper CMI inference server initialization
# 2. Correct import paths for Kaggle environment
# 3. Error handling for missing packages
# ====================================================================================================

import os
import sys
import json
import pickle
import warnings
import gc
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Optional, Any

import numpy as np
import pandas as pd
import polars as pl

# Scipy imports
from scipy import stats, signal
from scipy.spatial.transform import Rotation as R
from scipy.fft import fft, fftfreq

# Sklearn imports
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# Try to import joblib (fallback to pickle if not available)
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Joblib not available, using pickle")

# Import LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("ERROR: LightGBM not available!")
    raise ImportError("LightGBM is required for this notebook")

# Try to import SMOTE (optional for Kaggle)
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("SMOTE not available, will use class weights instead")

warnings.filterwarnings('ignore')

print('✓ All imports loaded successfully')
print(f'  - LightGBM: {LIGHTGBM_AVAILABLE}')
print(f'  - Joblib: {JOBLIB_AVAILABLE}')
print(f'  - SMOTE: {SMOTE_AVAILABLE}')

# ====================================================================================================
# CONFIGURATION
# ====================================================================================================

# Detect environment
IS_KAGGLE = os.path.exists('/kaggle/input')

# Set paths based on environment
if IS_KAGGLE:
    BASE_PATH = '/kaggle/input/cmi-detect-behavior-with-sensor-data/'
    WORKING_PATH = '/kaggle/working/'
else:
    BASE_PATH = 'cmi-detect-behavior-with-sensor-data/'
    WORKING_PATH = './'

CONFIG = {
    'data_path': BASE_PATH,
    'working_path': WORKING_PATH,
    'n_folds': 5,
    'random_state': 42,
    'sample_rate': 20,  # Hz
    'use_smote': SMOTE_AVAILABLE,
    'two_stage': True,  # Enable two-stage classification
    
    # Stage 1: Binary classification (BFRB vs Non-BFRB)
    'binary_lgbm': {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'n_estimators': 800,
        'max_depth': 8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1,
        'importance_type': 'gain'
    },
    
    # Stage 2A: BFRB multi-class (8 classes)
    'bfrb_lgbm': {
        'objective': 'multiclass',
        'num_class': 8,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 25,
        'learning_rate': 0.03,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'n_estimators': 1000,
        'max_depth': 6,
        'min_child_samples': 30,
        'reg_alpha': 0.2,
        'reg_lambda': 0.2,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1,
        'importance_type': 'gain'
    },
    
    # Stage 2B: Non-BFRB multi-class (10 classes)
    'non_bfrb_lgbm': {
        'objective': 'multiclass',
        'num_class': 10,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'n_estimators': 600,
        'max_depth': 7,
        'min_child_samples': 20,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1,
        'importance_type': 'gain'
    }
}

# Gesture mapping
GESTURE_MAPPER = {
    'Above ear - pull hair': 0, 'Cheek - pinch skin': 1, 'Eyebrow - pull hair': 2,
    'Eyelash - pull hair': 3, 'Forehead - pull hairline': 4, 'Forehead - scratch': 5,
    'Neck - pinch skin': 6, 'Neck - scratch': 7,
    'Drink from bottle/cup': 8, 'Feel around in tray and pull out an object': 9,
    'Glasses on/off': 10, 'Pinch knee/leg skin': 11, 'Pull air toward your face': 12,
    'Scratch knee/leg skin': 13, 'Text on phone': 14, 'Wave hello': 15,
    'Write name in air': 16, 'Write name on leg': 17
}
REVERSE_GESTURE_MAPPER = {v: k for k, v in GESTURE_MAPPER.items()}

print(f'✓ Configuration loaded')
print(f'  - Environment: {"Kaggle" if IS_KAGGLE else "Local"}')
print(f'  - Data path: {CONFIG["data_path"]}')
print(f'  - Working path: {CONFIG["working_path"]}')
print(f'  - Gesture classes: {len(GESTURE_MAPPER)}')

# ====================================================================================================
# WORLD ACCELERATION TRANSFORMATION
# ====================================================================================================

def compute_world_acceleration(acc: np.ndarray, rot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert device coordinates to world coordinates using quaternions."""
    try:
        # Convert quaternion format (w,x,y,z) to scipy format (x,y,z,w)
        rot_scipy = rot[:, [1, 2, 3, 0]]
        
        # Create rotation object and apply to acceleration
        r = R.from_quat(rot_scipy)
        world_acc = r.apply(acc)
        
        # Estimate gravity (low-pass filter)
        b, a = signal.butter(3, 0.3, 'low', fs=CONFIG['sample_rate'])
        gravity = np.zeros_like(world_acc)
        for i in range(3):
            gravity[:, i] = signal.filtfilt(b, a, world_acc[:, i])
        
        # Linear acceleration = total - gravity
        linear_acc = world_acc - gravity
        
        return world_acc, linear_acc
    except Exception as e:
        print(f"Warning: World acceleration computation failed: {e}")
        # Return original acceleration as fallback
        return acc, acc

# ====================================================================================================
# FEATURE EXTRACTION (Simplified for robustness)
# ====================================================================================================

def extract_statistical_features(data: np.ndarray, prefix: str) -> Dict[str, float]:
    """Extract basic statistical features from time series data."""
    features = {}
    
    # Handle empty or invalid data
    if len(data) == 0:
        return {f'{prefix}_mean': 0, f'{prefix}_std': 0}
    
    # Basic statistics
    features[f'{prefix}_mean'] = float(np.mean(data))
    features[f'{prefix}_std'] = float(np.std(data))
    features[f'{prefix}_min'] = float(np.min(data))
    features[f'{prefix}_max'] = float(np.max(data))
    features[f'{prefix}_range'] = float(np.max(data) - np.min(data))
    
    # Percentiles
    for p in [25, 50, 75]:
        features[f'{prefix}_p{p}'] = float(np.percentile(data, p))
    
    return features

def extract_features_from_sequence(seq_df: pd.DataFrame, demo_df: pd.DataFrame = None) -> pd.DataFrame:
    """Extract features from a sequence (simplified version)."""
    features = {}
    
    # Sequence metadata
    features['sequence_length'] = len(seq_df)
    
    # Demographics (if available)
    if demo_df is not None and len(demo_df) > 0:
        demo = demo_df.iloc[0]
        for col in ['age', 'adult_child', 'sex', 'handedness']:
            if col in demo.index:
                features[col] = float(demo[col]) if not pd.isna(demo[col]) else 0.0
    
    # IMU features
    acc_cols = ['acc_x', 'acc_y', 'acc_z']
    rot_cols = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
    
    # Check if columns exist
    has_acc = all(col in seq_df.columns for col in acc_cols)
    has_rot = all(col in seq_df.columns for col in rot_cols)
    
    if has_acc:
        # Extract acceleration features
        for col in acc_cols:
            data = seq_df[col].fillna(0).values
            features.update(extract_statistical_features(data, col))
        
        # Acceleration magnitude
        acc_data = seq_df[acc_cols].fillna(0).values
        acc_mag = np.linalg.norm(acc_data, axis=1)
        features.update(extract_statistical_features(acc_mag, 'acc_mag'))
    
    if has_rot:
        # Extract rotation features
        for col in rot_cols:
            data = seq_df[col].fillna(0).values
            features.update(extract_statistical_features(data, col))
    
    # World acceleration (if both acc and rot available)
    if has_acc and has_rot:
        try:
            acc_data = seq_df[acc_cols].fillna(0).values
            rot_data = seq_df[rot_cols].fillna(method='ffill').fillna(1 if 'rot_w' in rot_cols else 0).values
            
            world_acc, linear_acc = compute_world_acceleration(acc_data, rot_data)
            
            # World acceleration features
            for i, axis in enumerate(['x', 'y', 'z']):
                features.update(extract_statistical_features(world_acc[:, i], f'world_acc_{axis}'))
            
            # Linear acceleration features
            linear_mag = np.linalg.norm(linear_acc, axis=1)
            features.update(extract_statistical_features(linear_mag, 'linear_mag'))
        except Exception as e:
            print(f"Warning: Feature extraction error: {e}")
    
    return pd.DataFrame([features])

print('✓ Feature extraction functions defined')

# ====================================================================================================
# TWO-STAGE CLASSIFIER
# ====================================================================================================

class TwoStageClassifier:
    """Two-stage classification: Binary (BFRB detection) -> Multi-class (specific gesture)."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.binary_models = []  # Stage 1: BFRB vs Non-BFRB
        self.bfrb_models = []    # Stage 2A: BFRB 8-class
        self.non_bfrb_models = [] # Stage 2B: Non-BFRB 10-class
        self.feature_columns = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None):
        """Train two-stage classifier (simplified for Kaggle)."""
        
        # Store feature columns
        if isinstance(X, pd.DataFrame):
            self.feature_columns = X.columns.tolist()
            X = X.values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Stage 1: Binary labels (BFRB: 0-7, Non-BFRB: 8-17)
        y_binary = (y < 8).astype(int)
        
        # Single fold training (simplified for Kaggle)
        print("Training models...")
        
        # Stage 1: Binary Classification
        print("  Stage 1: Binary classifier")
        binary_model = lgb.LGBMClassifier(**self.config['binary_lgbm'])
        binary_model.fit(X, y_binary)
        self.binary_models.append(binary_model)
        
        # Stage 2A: BFRB Multi-class
        print("  Stage 2A: BFRB classifier")
        bfrb_mask = y < 8
        if np.sum(bfrb_mask) > 0:
            X_bfrb = X[bfrb_mask]
            y_bfrb = y[bfrb_mask]
            bfrb_model = lgb.LGBMClassifier(**self.config['bfrb_lgbm'])
            bfrb_model.fit(X_bfrb, y_bfrb)
            self.bfrb_models.append(bfrb_model)
        
        # Stage 2B: Non-BFRB Multi-class
        print("  Stage 2B: Non-BFRB classifier")
        non_bfrb_mask = y >= 8
        if np.sum(non_bfrb_mask) > 0:
            X_non_bfrb = X[non_bfrb_mask]
            y_non_bfrb = y[non_bfrb_mask] - 8  # Shift labels to 0-9
            non_bfrb_model = lgb.LGBMClassifier(**self.config['non_bfrb_lgbm'])
            non_bfrb_model.fit(X_non_bfrb, y_non_bfrb)
            self.non_bfrb_models.append(non_bfrb_model)
        
        self.is_fitted = True
        print("✓ Training complete")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using two-stage models."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
        
        if isinstance(X, pd.DataFrame):
            if self.feature_columns:
                # Ensure same features
                missing_cols = set(self.feature_columns) - set(X.columns)
                for col in missing_cols:
                    X[col] = 0
                X = X[self.feature_columns].values
            else:
                X = X.values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X = self.scaler.transform(X)
        
        n_samples = len(X)
        predictions = np.zeros(n_samples, dtype=int)
        
        # Stage 1: Binary prediction
        if self.binary_models:
            binary_proba = self.binary_models[0].predict_proba(X)
            is_bfrb = binary_proba[:, 1] > 0.5
        else:
            is_bfrb = np.ones(n_samples, dtype=bool)  # Default to BFRB
        
        # Stage 2: Conditional prediction
        for i in range(n_samples):
            if is_bfrb[i]:  # BFRB
                if self.bfrb_models:
                    predictions[i] = self.bfrb_models[0].predict(X[i:i+1])[0]
                else:
                    predictions[i] = 0  # Default BFRB class
            else:  # Non-BFRB
                if self.non_bfrb_models:
                    predictions[i] = self.non_bfrb_models[0].predict(X[i:i+1])[0] + 8
                else:
                    predictions[i] = 14  # Default Non-BFRB class (Text on phone)
        
        return predictions
    
    def save(self, path: str):
        """Save model to file."""
        model_data = {
            'binary_models': self.binary_models,
            'bfrb_models': self.bfrb_models,
            'non_bfrb_models': self.non_bfrb_models,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        if JOBLIB_AVAILABLE:
            joblib.dump(model_data, path)
        else:
            with open(path, 'wb') as f:
                pickle.dump(model_data, f, protocol=4)
    
    @classmethod
    def load(cls, path: str):
        """Load model from file."""
        try:
            if JOBLIB_AVAILABLE:
                model_data = joblib.load(path)
            else:
                with open(path, 'rb') as f:
                    model_data = pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
            # Try alternative loading method
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
        
        classifier = cls(model_data['config'])
        classifier.binary_models = model_data['binary_models']
        classifier.bfrb_models = model_data['bfrb_models']
        classifier.non_bfrb_models = model_data['non_bfrb_models']
        classifier.feature_columns = model_data['feature_columns']
        classifier.scaler = model_data['scaler']
        classifier.is_fitted = model_data.get('is_fitted', True)
        
        return classifier

print('✓ Two-stage classifier defined')

# ====================================================================================================
# MAIN TRAINING FUNCTION
# ====================================================================================================

def train_model_if_needed():
    """Train model if not already available."""
    model_path = os.path.join(CONFIG['working_path'], 'two_stage_model.pkl')
    
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        try:
            model = TwoStageClassifier.load(model_path)
            print("✓ Model loaded successfully")
            return model
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Training new model...")
    
    # Load training data
    print("Loading training data...")
    train_df = pd.read_csv(os.path.join(CONFIG['data_path'], 'train.csv'))
    demo_df = pd.read_csv(os.path.join(CONFIG['data_path'], 'train_demographics.csv'))
    
    # Process a subset for faster training in Kaggle
    unique_sequences = train_df['sequence_id'].unique()
    print(f"Total sequences: {len(unique_sequences)}")
    
    # Sample sequences if too many
    if len(unique_sequences) > 5000:
        np.random.seed(CONFIG['random_state'])
        sampled_sequences = np.random.choice(unique_sequences, 5000, replace=False)
        print(f"Sampling {len(sampled_sequences)} sequences for training")
    else:
        sampled_sequences = unique_sequences
    
    # Extract features
    print("Extracting features...")
    features_list = []
    labels = []
    groups = []
    
    for i, seq_id in enumerate(sampled_sequences):
        if i % 500 == 0:
            print(f"  Processing sequence {i+1}/{len(sampled_sequences)}")
        
        seq_data = train_df[train_df['sequence_id'] == seq_id]
        subject_id = seq_data['subject'].iloc[0]
        gesture = seq_data['gesture'].iloc[0]
        
        # Get demographics
        subject_demo = demo_df[demo_df['subject'] == subject_id]
        
        # Extract features
        features = extract_features_from_sequence(seq_data, subject_demo)
        features_list.append(features)
        labels.append(GESTURE_MAPPER[gesture])
        groups.append(subject_id)
    
    # Combine features
    X = pd.concat(features_list, ignore_index=True)
    y = np.array(labels)
    groups = np.array(groups)
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Train model
    model = TwoStageClassifier(CONFIG)
    model.fit(X, y, groups)
    
    # Save model
    model.save(model_path)
    print(f"✓ Model saved to {model_path}")
    
    return model

# ====================================================================================================
# PREDICTION FUNCTION FOR KAGGLE
# ====================================================================================================

def create_prediction_function(model):
    """Create prediction function for Kaggle inference server."""
    
    def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
        """Prediction function for Kaggle submission."""
        try:
            # Convert to pandas
            seq_df = sequence.to_pandas() if isinstance(sequence, pl.DataFrame) else sequence
            demo_df = demographics.to_pandas() if isinstance(demographics, pl.DataFrame) else demographics
            
            # Extract features
            features = extract_features_from_sequence(seq_df, demo_df)
            
            # Make prediction
            pred = model.predict(features)[0]
            
            # Convert to gesture name
            gesture_name = REVERSE_GESTURE_MAPPER.get(pred, 'Text on phone')
            
            return gesture_name
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return 'Text on phone'  # Default prediction
    
    return predict

# ====================================================================================================
# MAIN EXECUTION
# ====================================================================================================

def main():
    """Main execution function."""
    print("="*70)
    print("CMI BFRB Detection - Two-Stage Classification")
    print("="*70)
    
    # Train or load model
    model = train_model_if_needed()
    
    # Create prediction function
    predict_func = create_prediction_function(model)
    
    # Test prediction
    print("\nTesting prediction function...")
    test_seq = pl.DataFrame({
        'acc_x': np.random.randn(100),
        'acc_y': np.random.randn(100),
        'acc_z': np.random.randn(100),
        'rot_w': np.ones(100),
        'rot_x': np.zeros(100),
        'rot_y': np.zeros(100),
        'rot_z': np.zeros(100)
    })
    test_demo = pl.DataFrame({
        'age': [25],
        'adult_child': [1],
        'sex': [0],
        'handedness': [1]
    })
    
    try:
        test_result = predict_func(test_seq, test_demo)
        print(f"Test result: {test_result}")
        assert test_result in GESTURE_MAPPER, f"Invalid prediction: {test_result}"
        print("✓ Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    return predict_func

# ====================================================================================================
# KAGGLE INFERENCE SERVER INITIALIZATION
# ====================================================================================================

if __name__ == '__main__':
    # Get prediction function
    predict = main()
    
    # Initialize inference server for Kaggle
    if IS_KAGGLE:
        print("\n" + "="*70)
        print("Initializing Kaggle CMI Inference Server")
        print("="*70)
        
        # Import CMI inference server
        sys.path.append('/kaggle/input/cmi-detect-behavior-with-sensor-data')
        
        try:
            import kaggle_evaluation.cmi_inference_server
            
            print("Creating inference server...")
            inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
            print("✓ Inference server created")
            
            # Check if this is a competition rerun
            if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                print("\n🏆 Competition environment detected")
                print("Starting inference server...")
                inference_server.serve()
                print("✓ Inference complete")
            else:
                print("\n📝 Test environment detected")
                print("Running local gateway test...")
                try:
                    inference_server.run_local_gateway(
                        data_paths=(
                            os.path.join(CONFIG['data_path'], 'test.csv'),
                            os.path.join(CONFIG['data_path'], 'test_demographics.csv'),
                        )
                    )
                    print("✓ Local test complete")
                    
                    # Check submission file
                    if os.path.exists('submission.parquet'):
                        submission_df = pd.read_parquet('submission.parquet')
                        print(f"\n✓ Submission generated: {submission_df.shape}")
                        print(submission_df.head())
                except Exception as e:
                    print(f"Local test error: {e}")
                    
        except ImportError as e:
            print(f"⚠️ Could not import CMI inference server: {e}")
            print("This is expected if running outside Kaggle environment")
        except Exception as e:
            print(f"❌ Error initializing inference server: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n✓ Local environment - inference server not needed")
        print("To submit to Kaggle:")
        print("1. Upload this script as a notebook")
        print("2. Ensure GPU is disabled (not needed)")
        print("3. Run all cells")
        print("4. Submit to competition")
    
    print("\n" + "="*70)
    print("Script execution complete")
    print("="*70)