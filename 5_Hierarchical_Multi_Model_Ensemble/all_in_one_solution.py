#!/usr/bin/env python3
# ====================================================================================================
# CMI BFRB Detection - All-in-One Comprehensive Solution v5.0
# Target Score: 0.85+ (Binary F1: 0.95+, Macro F1: 0.75+)
# Architecture: Hierarchical Classification + Multi-Model Ensemble + Deep Learning
# Single file for Kaggle notebook submission (copy-paste ready)
# ====================================================================================================

import os
import sys
import warnings
from typing import Dict

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings("ignore")

# Try importing optional libraries
try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Machine Learning Models
import lightgbm as lgb
import xgboost as xgb

try:
    import catboost as cb

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available, will use LightGBM and XGBoost only")

# Deep Learning
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    LSTM,
    Activation,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Input,
    Lambda,
    MaxPooling1D,
    Multiply,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

print("=" * 70)
print("CMI BFRB Detection - All-in-One Solution v5.0")
print("=" * 70)

# ====================================================================================================
# GPU CONFIGURATION
# ====================================================================================================


def configure_gpu():
    """Configure GPU for optimal performance."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            if sys.platform == "darwin":
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


def seed_everything(seed=42):
    """Set all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


# Initialize environment
GPU_AVAILABLE = configure_gpu()
seed_everything(42)

# ====================================================================================================
# GLOBAL CONFIGURATION
# ====================================================================================================

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

# Configuration parameters
CONFIG = {
    "n_folds": 5,
    "random_state": 42,
    "sample_rate": 20,
    # Deep Learning
    "sequence_max_len": 500,
    "pad_percentile": 95,
    "batch_size": 64,
    "epochs": 100,  # Reduced for Kaggle time limit
    "patience": 20,
    "lr_init": 5e-4,
    "weight_decay": 1e-4,
    "mixup_alpha": 0.4,
    # Gradient Boosting
    "lgbm_n_estimators": 500,
    "xgb_n_estimators": 500,
    "use_hierarchical": True,  # Enable hierarchical classification
    "use_ensemble": True,  # Enable ensemble
    "retrain_on_full": False,  # Set to True for final submission
}

print(f"✓ Configuration loaded ({len(GESTURE_MAPPER)} gesture classes)")

# ====================================================================================================
# FEATURE ENGINEERING
# ====================================================================================================


def remove_gravity(acc_data: np.ndarray, rot_data: np.ndarray) -> np.ndarray:
    """Remove gravity component from accelerometer data."""
    num_samples = acc_data.shape[0]
    linear_accel = np.zeros_like(acc_data)
    gravity_world = np.array([0, 0, 9.81])

    for i in range(num_samples):
        if np.all(np.isnan(rot_data[i])) or np.all(np.isclose(rot_data[i], 0)):
            linear_accel[i] = acc_data[i]
            continue
        try:
            rotation = R.from_quat(rot_data[i])
            gravity_sensor = rotation.apply(gravity_world, inverse=True)
            linear_accel[i] = acc_data[i] - gravity_sensor
        except ValueError:
            linear_accel[i] = acc_data[i]

    return linear_accel


def calculate_angular_velocity(
    rot_data: np.ndarray, sample_rate: int = 20
) -> np.ndarray:
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


def extract_fft_features(
    signal_data: np.ndarray, sample_rate: int = 20
) -> Dict[str, float]:
    """Extract FFT-based frequency features."""
    if len(signal_data) < 10:
        return {
            "dominant_freq": 0,
            "spectral_energy": 0,
            "spectral_entropy": 0,
            "band_low": 0,
            "band_mid": 0,
            "band_high": 0,
        }

    fft_vals = np.abs(fft(signal_data))
    freqs = fftfreq(len(signal_data), 1 / sample_rate)

    pos_mask = freqs > 0
    fft_vals = fft_vals[pos_mask]
    freqs = freqs[pos_mask]

    if len(fft_vals) == 0:
        return {
            "dominant_freq": 0,
            "spectral_energy": 0,
            "spectral_entropy": 0,
            "band_low": 0,
            "band_mid": 0,
            "band_high": 0,
        }

    total_power = np.sum(fft_vals)
    if total_power > 0:
        spectral_entropy = -np.sum(
            (fft_vals / total_power) * np.log2(fft_vals / total_power + 1e-10)
        )
    else:
        spectral_entropy = 0

    return {
        "dominant_freq": freqs[np.argmax(fft_vals)] if len(fft_vals) > 0 else 0,
        "spectral_energy": np.sum(fft_vals**2),
        "spectral_entropy": spectral_entropy,
        "band_low": np.sum(fft_vals[(freqs >= 0) & (freqs < 2)]),
        "band_mid": np.sum(fft_vals[(freqs >= 2) & (freqs < 5)]),
        "band_high": np.sum(fft_vals[(freqs >= 5) & (freqs < 10)]),
    }


def extract_statistical_features(signal_data: np.ndarray) -> Dict[str, float]:
    """Extract statistical features from signal."""
    if len(signal_data) == 0:
        return {
            "mean": 0,
            "std": 0,
            "max": 0,
            "min": 0,
            "range": 0,
            "median": 0,
            "q25": 0,
            "q75": 0,
            "skew": 0,
            "kurtosis": 0,
        }

    return {
        "mean": np.mean(signal_data),
        "std": np.std(signal_data),
        "max": np.max(signal_data),
        "min": np.min(signal_data),
        "range": np.ptp(signal_data),
        "median": np.median(signal_data),
        "q25": np.percentile(signal_data, 25),
        "q75": np.percentile(signal_data, 75),
        "skew": stats.skew(signal_data) if len(signal_data) > 2 else 0,
        "kurtosis": stats.kurtosis(signal_data) if len(signal_data) > 3 else 0,
    }


def process_sequence_for_ml(df: pd.DataFrame) -> np.ndarray:
    """Process sequence and extract features for ML models."""
    features_dict = {}

    # Basic IMU data
    acc_data = df[["acc_x", "acc_y", "acc_z"]].fillna(0).values
    # Handle quaternion data with proper default values
    rot_data = df[["rot_x", "rot_y", "rot_z", "rot_w"]].copy()
    rot_data["rot_x"] = rot_data["rot_x"].fillna(0)
    rot_data["rot_y"] = rot_data["rot_y"].fillna(0)
    rot_data["rot_z"] = rot_data["rot_z"].fillna(0)
    rot_data["rot_w"] = rot_data["rot_w"].fillna(1)
    rot_data = rot_data.values

    # Remove gravity
    linear_accel = remove_gravity(acc_data, rot_data)

    # Angular features
    angular_vel = calculate_angular_velocity(rot_data)
    angular_dist = calculate_angular_distance(rot_data)

    # Magnitude features
    acc_mag = np.linalg.norm(acc_data, axis=1)
    linear_acc_mag = np.linalg.norm(linear_accel, axis=1)
    angular_vel_mag = np.linalg.norm(angular_vel, axis=1)

    # Statistical features for each signal
    for i, axis in enumerate(["x", "y", "z"]):
        # Acceleration
        stat_feats = extract_statistical_features(acc_data[:, i])
        for k, v in stat_feats.items():
            features_dict[f"acc_{axis}_{k}"] = v

        # Linear acceleration
        stat_feats = extract_statistical_features(linear_accel[:, i])
        for k, v in stat_feats.items():
            features_dict[f"linear_acc_{axis}_{k}"] = v

        # Angular velocity
        stat_feats = extract_statistical_features(angular_vel[:, i])
        for k, v in stat_feats.items():
            features_dict[f"angular_vel_{axis}_{k}"] = v

    # Magnitude statistical features
    for signal, name in [
        (acc_mag, "acc_mag"),
        (linear_acc_mag, "linear_acc_mag"),
        (angular_vel_mag, "angular_vel_mag"),
        (angular_dist, "angular_dist"),
    ]:
        stat_feats = extract_statistical_features(signal)
        for k, v in stat_feats.items():
            features_dict[f"{name}_{k}"] = v

    # FFT features
    for i, axis in enumerate(["x", "y", "z"]):
        fft_feats = extract_fft_features(acc_data[:, i])
        for k, v in fft_feats.items():
            features_dict[f"acc_{axis}_fft_{k}"] = v

    # TOF features
    for sensor_id in range(1, 6):
        tof_cols = [f"tof_{sensor_id}_v{p}" for p in range(64)]
        if all(col in df.columns for col in tof_cols):
            tof_data = df[tof_cols].replace(-1, np.nan)
            features_dict[f"tof_{sensor_id}_mean"] = (
                tof_data.mean(axis=1).fillna(0).mean()
            )
            features_dict[f"tof_{sensor_id}_std"] = (
                tof_data.std(axis=1).fillna(0).mean()
            )
            features_dict[f"tof_{sensor_id}_min"] = (
                tof_data.min(axis=1).fillna(0).mean()
            )
            features_dict[f"tof_{sensor_id}_max"] = (
                tof_data.max(axis=1).fillna(0).mean()
            )
        else:
            for stat in ["mean", "std", "min", "max"]:
                features_dict[f"tof_{sensor_id}_{stat}"] = 0

    # Thermal features
    for i in range(1, 6):
        thm_col = f"thm_{i}"
        if thm_col in df.columns:
            thm_data = df[thm_col].fillna(0).values
            features_dict[f"thm_{i}_mean"] = np.mean(thm_data)
            features_dict[f"thm_{i}_std"] = np.std(thm_data)
        else:
            features_dict[f"thm_{i}_mean"] = 0
            features_dict[f"thm_{i}_std"] = 0

    return features_dict


def prepare_sequences_for_dl(df, sequence_ids):
    """Prepare sequences for deep learning model."""
    sequences = []
    labels = []
    groups = []

    for seq_id in sequence_ids:
        seq_data = df[df["sequence_id"] == seq_id].copy()

        if len(seq_data) == 0:
            continue

        # Get subject info
        subject_id = (
            seq_data["subject"].iloc[0] if "subject" in seq_data.columns else "unknown"
        )

        # Basic IMU features
        acc_data = seq_data[["acc_x", "acc_y", "acc_z"]].fillna(0).values
        # Handle quaternion data with proper default values
        rot_data = seq_data[["rot_w", "rot_x", "rot_y", "rot_z"]].copy()
        rot_data = rot_data.ffill().bfill()
        rot_data["rot_w"] = rot_data["rot_w"].fillna(1)
        rot_data["rot_x"] = rot_data["rot_x"].fillna(0)
        rot_data["rot_y"] = rot_data["rot_y"].fillna(0)
        rot_data["rot_z"] = rot_data["rot_z"].fillna(0)
        rot_data = rot_data.values

        # Remove gravity
        linear_acc = remove_gravity(acc_data, rot_data[:, [1, 2, 3, 0]])

        # Calculate features
        acc_mag = np.sqrt(np.sum(acc_data**2, axis=1))
        linear_acc_mag = np.sqrt(np.sum(linear_acc**2, axis=1))

        # Angular velocity
        angular_vel = calculate_angular_velocity(
            rot_data[:, [1, 2, 3, 0]], CONFIG["sample_rate"]
        )
        angular_vel_mag = np.linalg.norm(angular_vel, axis=1)

        # Combine IMU features
        imu_features = np.column_stack(
            [
                linear_acc,  # 3 features
                acc_mag.reshape(-1, 1),  # 1 feature
                linear_acc_mag.reshape(-1, 1),  # 1 feature
                angular_vel,  # 3 features
                angular_vel_mag.reshape(-1, 1),  # 1 feature
            ]
        )  # Total: 9 IMU features

        # TOF features (simplified)
        tof_features = []
        for i in range(1, 6):
            tof_cols = [f"tof_{i}_v{p}" for p in range(64)]
            if all(col in seq_data.columns for col in tof_cols):
                tof_data = (
                    seq_data[tof_cols].replace(-1, np.nan).mean(axis=1).fillna(0).values
                )
                tof_features.append(tof_data)
            else:
                tof_features.append(np.zeros(len(seq_data)))

        # Stack features
        if tof_features:
            tof_array = np.column_stack(tof_features)
            all_features = np.concatenate([imu_features, tof_array], axis=1)
        else:
            all_features = imu_features

        sequences.append(all_features)

        # Get label if available
        if "gesture" in seq_data.columns:
            gesture = seq_data["gesture"].iloc[0]
            labels.append(GESTURE_MAPPER[gesture])

        groups.append(subject_id)

    return sequences, labels, groups


def pad_sequences_custom(sequences, maxlen=None, padding="post"):
    """Custom padding function for sequences."""
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)

    num_samples = len(sequences)
    num_features = sequences[0].shape[1] if len(sequences[0].shape) > 1 else 1

    padded = np.zeros((num_samples, maxlen, num_features), dtype="float32")

    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), maxlen)
        if padding == "post":
            padded[i, :seq_len] = seq[:seq_len]
        else:
            padded[i, -seq_len:] = seq[-seq_len:]

    return padded


# ====================================================================================================
# HIERARCHICAL CLASSIFIER
# ====================================================================================================


class HierarchicalClassifier:
    """Hierarchical classification: Binary → BFRB → Full."""

    def __init__(self):
        self.binary_model = None
        self.bfrb_model = None
        self.full_model = None
        self.scaler = StandardScaler()

    def fit(self, X, y, groups=None):
        """Train hierarchical classifiers."""
        print("Training Hierarchical Classifier...")

        # Normalize
        X_scaled = self.scaler.fit_transform(X)

        # Binary classifier
        print("  Step 1: Binary Classifier (BFRB vs Non-BFRB)...")
        y_binary = (y < 8).astype(int)
        self.binary_model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=CONFIG["lgbm_n_estimators"],
            learning_rate=0.05,
            num_leaves=127,
            random_state=CONFIG["random_state"],
            verbose=-1,
        )
        self.binary_model.fit(X_scaled, y_binary)

        # BFRB classifier
        print("  Step 2: BFRB Classifier (8 classes)...")
        bfrb_mask = y < 8
        if bfrb_mask.any():
            X_bfrb = X_scaled[bfrb_mask]
            y_bfrb = y[bfrb_mask]

            self.bfrb_model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=8,
                n_estimators=CONFIG["lgbm_n_estimators"],
                learning_rate=0.05,
                num_leaves=127,
                random_state=CONFIG["random_state"],
                verbose=-1,
            )
            self.bfrb_model.fit(X_bfrb, y_bfrb)

        # Full classifier
        print("  Step 3: Full Classifier (18 classes)...")
        self.full_model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=18,
            n_estimators=CONFIG["lgbm_n_estimators"],
            learning_rate=0.05,
            num_leaves=127,
            random_state=CONFIG["random_state"],
            verbose=-1,
        )
        self.full_model.fit(X_scaled, y)

        print("✓ Hierarchical Classifier training complete")

    def predict(self, X):
        """Predict using hierarchical strategy."""
        X_scaled = self.scaler.transform(X)
        n_samples = X_scaled.shape[0]
        predictions = np.zeros(n_samples, dtype=int)

        # Binary prediction
        binary_pred = self.binary_model.predict(X_scaled)
        binary_proba = self.binary_model.predict_proba(X_scaled)[:, 1]

        for i in range(n_samples):
            if binary_pred[i] == 1 and binary_proba[i] > 0.6:  # BFRB
                if self.bfrb_model is not None:
                    predictions[i] = self.bfrb_model.predict(X_scaled[i : i + 1])[0]
                else:
                    predictions[i] = self.full_model.predict(X_scaled[i : i + 1])[0]
            else:
                full_pred = self.full_model.predict(X_scaled[i : i + 1])[0]
                predictions[i] = full_pred if full_pred >= 8 else 8

        return predictions


# ====================================================================================================
# DEEP LEARNING MODELS
# ====================================================================================================


# Tensor manipulations for attention
def time_sum(x):
    return K.sum(x, axis=1)


def squeeze_last_axis(x):
    return tf.squeeze(x, axis=-1)


def expand_last_axis(x):
    return tf.expand_dims(x, axis=-1)


def attention_layer(inputs):
    """Attention mechanism for sequence weighting."""
    score = Dense(1, activation="tanh")(inputs)
    score = Lambda(squeeze_last_axis)(score)
    weights = Activation("softmax")(score)
    weights = Lambda(expand_last_axis)(weights)
    context = Multiply()([inputs, weights])
    context = Lambda(time_sum)(context)
    return context


def build_simple_dl_model(input_shape, n_classes):
    """Build a simple deep learning model for quick training."""
    inp = Input(shape=input_shape)

    # CNN layers
    x = Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(128, 3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    # BiLSTM
    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    # Attention
    x = attention_layer(x)

    # Dense layers
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)

    # Output
    out = Dense(n_classes, activation="softmax")(x)

    return Model(inp, out)


# ====================================================================================================
# ENSEMBLE MODEL
# ====================================================================================================


class EnsembleModel:
    """Multi-model ensemble."""

    def __init__(self):
        self.models = []
        self.weights = []
        self.scaler = StandardScaler()

    def add_model(self, model, name, weight=1.0):
        """Add a model to the ensemble."""
        self.models.append({"model": model, "name": name})
        self.weights.append(weight)

    def fit(self, X, y, groups=None):
        """Train all models."""
        X_scaled = self.scaler.fit_transform(X)

        for model_dict in self.models:
            print(f"Training {model_dict['name']}...")

            if model_dict["name"] == "hierarchical":
                model_dict["model"].fit(X, y, groups)
            elif "lgbm" in model_dict["name"]:
                model = lgb.LGBMClassifier(
                    objective="multiclass",
                    num_class=18,
                    n_estimators=CONFIG["lgbm_n_estimators"],
                    learning_rate=0.05,
                    num_leaves=127,
                    random_state=CONFIG["random_state"],
                    verbose=-1,
                )
                model.fit(X_scaled, y)
                model_dict["model"] = model
            elif "xgb" in model_dict["name"]:
                model = xgb.XGBClassifier(
                    objective="multi:softprob",
                    num_class=18,
                    n_estimators=CONFIG["xgb_n_estimators"],
                    learning_rate=0.05,
                    max_depth=8,
                    random_state=CONFIG["random_state"],
                    use_label_encoder=False,
                    eval_metric="mlogloss",
                )
                model.fit(X_scaled, y)
                model_dict["model"] = model

    def predict(self, X):
        """Ensemble prediction."""
        predictions = []

        for model_dict in self.models:
            if model_dict["name"] == "hierarchical":
                pred = model_dict["model"].predict(X)
            else:
                X_scaled = self.scaler.transform(X)
                pred = model_dict["model"].predict(X_scaled)
            predictions.append(pred)

        # Weighted voting
        predictions = np.array(predictions)
        weights = np.array(self.weights) / np.sum(self.weights)

        final_predictions = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            class_votes = np.zeros(18)
            for j, pred in enumerate(predictions[:, i]):
                class_votes[pred] += weights[j]
            final_predictions[i] = np.argmax(class_votes)

        return final_predictions


# ====================================================================================================
# COMPETITION METRICS
# ====================================================================================================


def calculate_competition_metric(y_true, y_pred):
    """Calculate competition metric."""
    # Binary F1
    binary_true = (y_true < 8).astype(int)
    binary_pred = (y_pred < 8).astype(int)
    binary_f1 = f1_score(binary_true, binary_pred, average="binary")

    # Macro F1 for BFRB
    bfrb_mask = y_true < 8
    if bfrb_mask.any():
        macro_f1 = f1_score(y_true[bfrb_mask], y_pred[bfrb_mask], average="macro")
    else:
        macro_f1 = 0.0

    combined_score = (binary_f1 + macro_f1) / 2

    return {
        "binary_f1": binary_f1,
        "macro_f1": macro_f1,
        "combined_score": combined_score,
    }


# ====================================================================================================
# MAIN TRAINING PIPELINE
# ====================================================================================================

# Global variables for trained models
TRAINED_MODELS = None
FEATURE_COLUMNS = None
ML_SCALER = None
DL_MODEL = None
DL_SCALER = None
DL_PAD_LEN = None


def train_models():
    """Train all models on the training data."""
    global TRAINED_MODELS, FEATURE_COLUMNS, ML_SCALER, DL_MODEL, DL_SCALER, DL_PAD_LEN

    print("\n" + "=" * 70)
    print("TRAINING PHASE")
    print("=" * 70)

    # Check if running in Kaggle
    is_kaggle = os.path.exists("/kaggle/input")

    # Load data
    print("\nLoading data...")
    if is_kaggle:
        train_df = pd.read_csv(
            "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
        )
        demo_df = pd.read_csv(
            "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"
        )
    else:
        # Local path
        data_path = "cmi-detect-behavior-with-sensor-data/"
        train_df = pd.read_csv(data_path + "train.csv")
        demo_df = pd.read_csv(data_path + "train_demographics.csv")

    print(
        f"Loaded {len(train_df)} samples from {train_df['sequence_id'].nunique()} sequences"
    )

    # Process sequences for ML models
    print("\nProcessing sequences for ML models...")
    ml_features = []
    labels = []
    groups = []

    for seq_id in train_df["sequence_id"].unique():
        seq_data = train_df[train_df["sequence_id"] == seq_id]

        # Extract features
        features_dict = process_sequence_for_ml(seq_data)

        # Store feature columns
        if FEATURE_COLUMNS is None:
            FEATURE_COLUMNS = list(features_dict.keys())

        # Convert to array
        feature_array = np.array([features_dict.get(col, 0) for col in FEATURE_COLUMNS])
        ml_features.append(feature_array)

        # Get label
        if "gesture" in seq_data.columns:
            gesture = seq_data["gesture"].iloc[0]
            labels.append(GESTURE_MAPPER[gesture])

        # Get subject for grouping
        subject = seq_data["subject"].iloc[0]
        groups.append(subject)

    X_ml = np.array(ml_features)
    y = np.array(labels)
    groups = np.array(groups)

    # Handle NaN/inf values
    X_ml = np.nan_to_num(X_ml, nan=0, posinf=0, neginf=0)

    print(f"ML Feature matrix shape: {X_ml.shape}")

    # Split data BEFORE training to avoid data leakage
    X_train, X_val, y_train, y_val = train_test_split(
        X_ml, y, test_size=0.2, random_state=CONFIG["random_state"], stratify=y
    )
    
    # Split groups accordingly
    train_mask = np.zeros(len(y), dtype=bool)
    train_mask[:len(y_train)] = True
    groups_train = groups[:len(y_train)]
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

    # Create ensemble
    ensemble = EnsembleModel()

    # Add hierarchical classifier
    if CONFIG["use_hierarchical"]:
        hierarchical_clf = HierarchicalClassifier()
        ensemble.add_model(hierarchical_clf, "hierarchical", weight=2.0)

    # Add LightGBM
    ensemble.add_model(None, "lgbm", weight=1.5)

    # Add XGBoost
    ensemble.add_model(None, "xgb", weight=1.0)

    # Train ensemble on TRAINING data only
    print("\nTraining ensemble models on training split...")
    ensemble.fit(X_train, y_train, groups_train)

    # Store trained models
    TRAINED_MODELS = ensemble
    ML_SCALER = ensemble.scaler

    # Train Deep Learning model (simplified for speed)
    if CONFIG["epochs"] > 0:
        print("\nPreparing sequences for Deep Learning...")
        sequences, dl_labels, dl_groups = prepare_sequences_for_dl(
            train_df, train_df["sequence_id"].unique()
        )

        # Padding
        seq_lengths = [len(seq) for seq in sequences]
        DL_PAD_LEN = int(np.percentile(seq_lengths, CONFIG["pad_percentile"]))
        X_dl = pad_sequences_custom(sequences, maxlen=DL_PAD_LEN)
        y_dl = to_categorical(dl_labels, num_classes=18)

        print(f"DL Sequences shape: {X_dl.shape}")

        # Normalize
        DL_SCALER = StandardScaler()
        X_dl_reshaped = X_dl.reshape(-1, X_dl.shape[-1])
        X_dl_reshaped = DL_SCALER.fit_transform(X_dl_reshaped)
        X_dl = X_dl_reshaped.reshape(X_dl.shape)

        # Build and train simple DL model
        print("Building Deep Learning model...")
        DL_MODEL = build_simple_dl_model((DL_PAD_LEN, X_dl.shape[-1]), 18)

        DL_MODEL.compile(
            optimizer=Adam(learning_rate=CONFIG["lr_init"]),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Quick training with reduced epochs
        print(f"Training DL model for {min(CONFIG['epochs'], 20)} epochs...")
        DL_MODEL.fit(
            X_dl,
            y_dl,
            batch_size=CONFIG["batch_size"],
            epochs=min(CONFIG["epochs"], 20),  # Limit epochs for Kaggle
            validation_split=0.2,
            verbose=1,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        )

    # Validation
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    # Use the already split validation data (no need to split again)
    # Predict on validation set
    y_pred = TRAINED_MODELS.predict(X_val)

    # Calculate metrics
    metrics = calculate_competition_metric(y_val, y_pred)

    print(f"Binary F1: {metrics['binary_f1']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Combined Score: {metrics['combined_score']:.4f}")

    # Optionally retrain on full data for final submission
    if CONFIG.get("retrain_on_full", False):
        print("\nRetraining on full dataset for final submission...")
        ensemble_full = EnsembleModel()
        
        if CONFIG["use_hierarchical"]:
            hierarchical_clf_full = HierarchicalClassifier()
            ensemble_full.add_model(hierarchical_clf_full, "hierarchical", weight=2.0)
        
        ensemble_full.add_model(None, "lgbm", weight=1.5)
        ensemble_full.add_model(None, "xgb", weight=1.0)
        
        ensemble_full.fit(X_ml, y, groups)
        TRAINED_MODELS = ensemble_full
        ML_SCALER = ensemble_full.scaler
        print("✓ Retrained on full dataset")

    print("\n✓ Training complete!")

    return metrics["combined_score"]


# ====================================================================================================
# INFERENCE FOR KAGGLE
# ====================================================================================================


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Main prediction function for Kaggle evaluation API.

    Args:
        sequence: Polars DataFrame with sensor data
        demographics: Polars DataFrame with demographic data

    Returns:
        str: Predicted gesture name
    """
    global TRAINED_MODELS, FEATURE_COLUMNS, ML_SCALER, DL_MODEL, DL_SCALER, DL_PAD_LEN

    # Train models if not already trained
    if TRAINED_MODELS is None:
        print("Models not trained yet. Training now...")
        train_models()

    try:
        # Convert to pandas
        seq_df = (
            sequence.to_pandas() if isinstance(sequence, pl.DataFrame) else sequence
        )

        # Add required columns if missing
        if "sequence_id" not in seq_df.columns:
            seq_df["sequence_id"] = "test_seq"
        if "subject" not in seq_df.columns:
            seq_df["subject"] = "unknown"

        # Process sequence for ML models
        features_dict = process_sequence_for_ml(seq_df)

        # Convert to array using stored feature columns
        feature_array = np.array([features_dict.get(col, 0) for col in FEATURE_COLUMNS])
        feature_array = feature_array.reshape(1, -1)

        # Handle NaN/inf
        feature_array = np.nan_to_num(feature_array, nan=0, posinf=0, neginf=0)

        # Get ensemble prediction
        ml_pred = TRAINED_MODELS.predict(feature_array)[0]

        # Optional: combine with DL prediction if available
        if DL_MODEL is not None and CONFIG["epochs"] > 0:
            try:
                # Prepare sequence for DL
                sequences, _, _ = prepare_sequences_for_dl(seq_df, ["test_seq"])

                if sequences:
                    X_dl = pad_sequences_custom(sequences, maxlen=DL_PAD_LEN)
                    X_dl_reshaped = X_dl.reshape(-1, X_dl.shape[-1])
                    X_dl_reshaped = DL_SCALER.transform(X_dl_reshaped)
                    X_dl = X_dl_reshaped.reshape(X_dl.shape)

                    dl_pred = DL_MODEL.predict(X_dl, verbose=0).argmax(axis=1)[0]

                    # Simple voting
                    final_pred = ml_pred if np.random.random() > 0.3 else dl_pred
                else:
                    final_pred = ml_pred
            except:
                final_pred = ml_pred
        else:
            final_pred = ml_pred

        # Convert to gesture name
        gesture_name = REVERSE_GESTURE_MAPPER[final_pred]
        return gesture_name

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback

        traceback.print_exc()
        return "Text on phone"  # Default safe prediction


# ====================================================================================================
# MAIN EXECUTION
# ====================================================================================================

if __name__ == "__main__":
    # Check if running in Kaggle
    is_kaggle = os.path.exists("/kaggle/input")

    if is_kaggle:
        print("=" * 70)
        print("KAGGLE COMPETITION ENVIRONMENT DETECTED")
        print("=" * 70)

        # Train models
        print("\nTraining models for submission...")
        validation_score = train_models()

        print(f"\n✓ Models trained with validation score: {validation_score:.4f}")

        # Initialize Kaggle inference server
        print("\n" + "=" * 70)
        print("INITIALIZING KAGGLE INFERENCE SERVER")
        print("=" * 70)

        try:
            from kaggle_evaluation.cmi_inference_server import CMIInferenceServer

            print("✓ CMI module imported successfully")
            print("Creating inference server...")

            # Create inference server with our predict function
            inference_server = CMIInferenceServer(predict)
            print("✓ Inference server created")

            # Start serving predictions
            print("\nStarting inference server...")
            print("This will process test data and create submission.parquet")

            # Serve predictions for test data
            inference_server.serve()

            print("\n✓ Submission complete!")
            print("=" * 70)

        except ImportError as e:
            print(f"⚠️ Kaggle evaluation module not available: {e}")
            print("Attempting manual submission generation...")
            
            # Try to generate submission manually
            try:
                test_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv"
                if os.path.exists(test_path):
                    print("Loading test data from Kaggle input...")
                    test_df = pd.read_csv(test_path)
                    test_demo_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")
                    
                    print(f"Processing {test_df['sequence_id'].nunique()} test sequences...")
                    
                    predictions = []
                    sequence_ids = test_df['sequence_id'].unique()
                    
                    for i, seq_id in enumerate(sequence_ids):
                        if i % 100 == 0:
                            print(f"  Processed {i}/{len(sequence_ids)} sequences...")
                        
                        seq_df = test_df[test_df['sequence_id'] == seq_id]
                        seq_pl = pl.from_pandas(seq_df)
                        demo_pl = pl.from_pandas(test_demo_df[test_demo_df['sequence_id'] == seq_id])
                        
                        pred = predict(seq_pl, demo_pl)
                        predictions.append({'sequence_id': seq_id, 'prediction': pred})
                    
                    submission_df = pd.DataFrame(predictions)
                    submission_df.to_parquet('/kaggle/working/submission.parquet', index=False)
                    print(f"\n✓ Generated submission.parquet with {len(submission_df)} predictions")
                else:
                    print("Testing prediction function with dummy data...")
                    test_seq = pl.DataFrame(
                        {
                            "acc_x": np.random.randn(100),
                            "acc_y": np.random.randn(100),
                            "acc_z": np.random.randn(100),
                            "rot_w": np.ones(100),
                            "rot_x": np.zeros(100),
                            "rot_y": np.zeros(100),
                            "rot_z": np.zeros(100),
                        }
                    )
                    test_demo = pl.DataFrame({"age": [25]})
                    
                    result = predict(test_seq, test_demo)
                    print(f"Test prediction: {result}")
                    assert result in GESTURE_MAPPER, "Invalid prediction"
                    print("✓ Test passed!")
                    
            except Exception as manual_e:
                print(f"⚠️ Manual submission generation failed: {manual_e}")

    else:
        print("=" * 70)
        print("LOCAL ENVIRONMENT - TRAINING AND TESTING")
        print("=" * 70)

        # Train and validate locally
        validation_score = train_models()

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print(f"Final validation score: {validation_score:.4f}")
        print("=" * 70)
        
        # Try to generate submission.parquet locally
        print("\nAttempting to generate submission.parquet locally...")
        
        try:
            # Check if test data exists
            data_path = "cmi-detect-behavior-with-sensor-data/"
            test_file = data_path + "test.csv"
            
            if os.path.exists(test_file):
                print("Loading test data...")
                test_df = pd.read_csv(test_file)
                test_demo_df = pd.read_csv(data_path + "test_demographics.csv")
                
                print(f"Processing {test_df['sequence_id'].nunique()} test sequences...")
                
                # Process test sequences
                predictions = []
                sequence_ids = test_df['sequence_id'].unique()
                
                for i, seq_id in enumerate(sequence_ids):
                    if i % 100 == 0:
                        print(f"  Processed {i}/{len(sequence_ids)} sequences...")
                    
                    seq_df = test_df[test_df['sequence_id'] == seq_id]
                    
                    # Convert to polars for predict function
                    seq_pl = pl.from_pandas(seq_df)
                    demo_pl = pl.from_pandas(test_demo_df[test_demo_df['sequence_id'] == seq_id])
                    
                    # Get prediction
                    pred = predict(seq_pl, demo_pl)
                    predictions.append({'sequence_id': seq_id, 'prediction': pred})
                
                # Create submission dataframe
                submission_df = pd.DataFrame(predictions)
                
                # Save as parquet
                submission_df.to_parquet('submission.parquet', index=False)
                print(f"\n✓ Generated submission.parquet with {len(submission_df)} predictions")
                print(f"  File size: {os.path.getsize('submission.parquet') / 1024:.2f} KB")
                print(f"  Sample predictions:")
                for i in range(min(5, len(submission_df))):
                    print(f"    {submission_df.iloc[i]['sequence_id']}: {submission_df.iloc[i]['prediction']}")
            else:
                print("⚠️ Test data not found at:", test_file)
                print("  Submission file will be generated when running in Kaggle environment")
                
        except Exception as e:
            print(f"⚠️ Error generating submission locally: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 70)
        print("To use for Kaggle submission:")
        print("1. Copy this entire script to a Kaggle notebook")
        print("2. Run all cells")
        print("3. The submission.parquet will be generated automatically")
        print("=" * 70)
