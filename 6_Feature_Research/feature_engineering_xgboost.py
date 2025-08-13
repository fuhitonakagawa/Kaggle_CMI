#!/usr/bin/env python3
# ====================================================================================================
# CMI BFRB Detection - Advanced Feature Engineering with XGBoost
# Based on plan-pro.md specifications for robust feature engineering
# Single file for Kaggle notebook submission (copy-paste ready)
# ====================================================================================================
#
# USAGE:
# ------
# For KAGGLE:
#   1. Set FORCE_LOCAL = False (line 45)
#   2. Copy entire script to Kaggle notebook
#   3. Run all cells
#
# ====================================================================================================

import json
import os
import pickle
import time
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from scipy import stats
from scipy.signal import find_peaks, welch
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Only print header in main process (not in multiprocessing workers)
if __name__ == "__main__":
    print("=" * 70)
    print("CMI BFRB Detection - Advanced Feature Engineering v1.1")
    print("XGBoost with comprehensive sensor fusion features")
    print("With Export/Import capability for faster iteration")
    print("=" * 70)

# ====================================================================================================
# ENVIRONMENT CONFIGURATION
# ====================================================================================================

# 🔧 MAIN ENVIRONMENT SWITCH - CHANGE THIS TO SWITCH BETWEEN KAGGLE AND LOCAL
IS_KAGGLE_ENV = False  # Set to True for Kaggle, False for local MacBook

# ⚙️ FEATURE EXTRACTION SETTINGS
# Change these variables to control behavior:
USE_EXPORTED_FEATURES = True  # Set to True to skip feature extraction
EXPORT_FEATURES = False  # Set to False when using exported features
EXPORT_NAME = None  # Custom name for export (None = auto-generate timestamp)

# Auto-set exported features path based on environment
if IS_KAGGLE_ENV:
    # Kaggle dataset path
    EXPORTED_FEATURES_PATH = "/kaggle/input/cmi-bfrb-detection-exported-feature-data/features_v1.1.0_20250813_161440"
else:
    # Local path
    EXPORTED_FEATURES_PATH = "exported_features/features_v1.1.0_20250813_161440"

# Example usage:
# For Kaggle: Just set IS_KAGGLE_ENV = True
# For Local: Just set IS_KAGGLE_ENV = False
# The path will be automatically configured!

# Set paths based on environment
if IS_KAGGLE_ENV:
    # Kaggle paths
    EXPORT_DIR = Path("./exported_features")
    DATA_BASE_PATH = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
else:
    # Local MacBook paths
    EXPORT_DIR = Path("exported_features")
    DATA_BASE_PATH = Path("cmi-detect-behavior-with-sensor-data")

EXPORT_DIR.mkdir(exist_ok=True, parents=True)
FEATURE_VERSION = "v1.1.0"

# Only print configuration in main process
if __name__ == "__main__":
    print(f"🌍 Environment: {'KAGGLE' if IS_KAGGLE_ENV else 'LOCAL (MacBook)'}")
    print(f"📁 Export directory: {EXPORT_DIR}")
    print(f"📊 Data directory: {DATA_BASE_PATH}")
    print(
        f"⚡ Parallel processing: {'DISABLED (Kaggle)' if IS_KAGGLE_ENV else 'ENABLED (Local)'}"
    )
    print(
        f"🎮 XGBoost GPU: {'ENABLED (CUDA/T4)' if IS_KAGGLE_ENV else 'DISABLED (CPU only)'}"
    )

# ====================================================================================================
# CONFIGURATION
# ====================================================================================================

# Set data paths based on environment
if IS_KAGGLE_ENV:
    # Kaggle paths
    DATA_PATHS = {
        "train_path": str(DATA_BASE_PATH / "train.csv"),
        "train_demographics_path": str(DATA_BASE_PATH / "train_demographics.csv"),
        "test_path": str(DATA_BASE_PATH / "test.csv"),
        "test_demographics_path": str(DATA_BASE_PATH / "test_demographics.csv"),
    }
else:
    # Local MacBook paths
    DATA_PATHS = {
        "train_path": str(DATA_BASE_PATH / "train.csv"),
        "train_demographics_path": str(DATA_BASE_PATH / "train_demographics.csv"),
        "test_path": str(DATA_BASE_PATH / "test.csv"),
        "test_demographics_path": str(DATA_BASE_PATH / "test_demographics.csv"),
    }

    # Check if local data exists
    if not DATA_BASE_PATH.exists():
        print(f"⚠️ Warning: Local data directory not found: {DATA_BASE_PATH}")
        print("Please ensure the data files are in the correct location.")

CONFIG = {
    # Data paths
    **DATA_PATHS,
    # Feature engineering
    "sampling_rate": 20,  # Hz
    "gravity": 9.81,  # m/s^2
    "use_world_acc": True,
    "use_linear_acc": True,
    "use_angular_velocity": True,
    "use_frequency_features": True,
    "use_tof_spatial": True,
    "use_thermal_trends": True,
    "use_cross_modal": True,
    # Multi-resolution windows (S/M/L)
    "use_multi_resolution": True,
    "window_sizes": {
        "S": (20, 30),  # 1.0-1.5 seconds
        "M": (60, 80),  # 3-4 seconds
        "L": (200, 256),  # 10-12.8 seconds
    },
    "use_tail_emphasis": True,  # Emphasize tail windows for TTA
    # ToF processing
    "tof_pca_components": 8,
    "tof_valid_threshold": 0.2,
    "tof_outlier_percentile": (1, 99),
    "tof_use_pca": True,
    "tof_use_handedness_mirror": True,  # Mirror ToF based on handedness
    "tof_region_analysis": True,  # Analyze different spatial regions
    # Frequency analysis
    "welch_nperseg": 128,
    "welch_noverlap": 64,
    "freq_bands": [(0.3, 3), (3, 8), (8, 12)],  # Hz
    # Normalization
    "sequence_normalize": True,
    "robust_scaler": True,
    # Model
    "n_folds": 5,
    "random_state": 42,
    "xgb_params": {
        "objective": "multi:softprob",
        "num_class": 18,
        "n_estimators": 1000,
        "max_depth": 10,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 3,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    },
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

# Only print configuration in main process
if __name__ == "__main__":
    print(f"✓ Configuration loaded ({len(GESTURE_MAPPER)} gesture classes)")
    print(
        f"📍 Data paths configured for {'Kaggle' if IS_KAGGLE_ENV else 'Local'} environment"
    )

# ====================================================================================================
# QUATERNION AND IMU PROCESSING
# ====================================================================================================


def handle_quaternion_missing(rot_data: np.ndarray) -> np.ndarray:
    """Handle missing values in quaternion data with proper normalization."""
    rot_cleaned = rot_data.copy()

    # Fill NaN values
    for col in range(4):
        mask = np.isnan(rot_cleaned[:, col])
        if mask.any():
            # Forward fill, then backward fill, then use default
            rot_cleaned[:, col] = (
                pd.Series(rot_cleaned[:, col])
                .fillna(method="ffill")
                .fillna(method="bfill")
                .fillna(1.0 if col == 0 else 0.0)
                .values
            )

    # Normalize quaternions
    norms = np.linalg.norm(rot_cleaned, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    rot_cleaned = rot_cleaned / norms

    return rot_cleaned


def compute_world_acceleration(acc: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Convert acceleration from device to world coordinates."""
    try:
        # Convert quaternion format (w,x,y,z) to scipy format (x,y,z,w)
        rot_scipy = rot[:, [1, 2, 3, 0]]
        r = R.from_quat(rot_scipy)
        acc_world = r.apply(acc)
    except Exception:
        acc_world = acc.copy()
    return acc_world


def compute_linear_acceleration(
    acc: np.ndarray, rot: np.ndarray, method: str = "subtract"
) -> np.ndarray:
    """Remove gravity from acceleration to get linear acceleration."""
    if method == "subtract":
        # Method A: Subtract gravity in world coordinates
        acc_world = compute_world_acceleration(acc, rot)
        gravity_world = np.array([0, 0, CONFIG["gravity"]])
        linear_acc = acc_world - gravity_world
    else:
        # Method B: High-pass filter
        from scipy.signal import butter, filtfilt

        b, a = butter(4, 2.0, btype="high", fs=CONFIG["sampling_rate"])
        linear_acc = np.zeros_like(acc)
        for i in range(3):
            linear_acc[:, i] = filtfilt(b, a, acc[:, i])

    return linear_acc


def compute_angular_velocity(rot: np.ndarray, dt: float = 1.0 / 20) -> np.ndarray:
    """Compute angular velocity from quaternion sequence."""
    omega = np.zeros((len(rot) - 1, 3))

    for i in range(len(rot) - 1):
        q1 = rot[i, [1, 2, 3, 0]]  # Convert to scipy format
        q2 = rot[i + 1, [1, 2, 3, 0]]

        try:
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)
            r_diff = r2 * r1.inv()
            omega[i] = r_diff.as_rotvec() / dt
        except:
            omega[i] = 0

    # Pad to match original length
    omega = np.vstack([omega, omega[-1:]])
    return omega


def quaternion_to_euler(rot: np.ndarray) -> np.ndarray:
    """Convert quaternion to Euler angles (roll, pitch, yaw)."""
    rot_scipy = rot[:, [1, 2, 3, 0]]
    try:
        r = R.from_quat(rot_scipy)
        euler = r.as_euler("xyz")
    except:
        euler = np.zeros((len(rot), 3))
    return euler


# ====================================================================================================
# STATISTICAL FEATURES
# ====================================================================================================


def extract_statistical_features(data: np.ndarray, prefix: str) -> dict:
    """Extract comprehensive statistical features from 1D time series."""
    features = {}

    if len(data) == 0 or np.all(np.isnan(data)):
        # Return zeros for empty data
        return {
            f"{prefix}_{k}": 0
            for k in [
                "mean",
                "std",
                "min",
                "max",
                "median",
                "q25",
                "q75",
                "iqr",
                "range",
                "cv",
                "skew",
                "kurt",
                "first",
                "last",
                "delta",
                "diff_mean",
                "diff_std",
                "n_changes",
            ]
        }

    # Clean data
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return {
            f"{prefix}_{k}": 0
            for k in [
                "mean",
                "std",
                "min",
                "max",
                "median",
                "q25",
                "q75",
                "iqr",
                "range",
                "cv",
                "skew",
                "kurt",
                "first",
                "last",
                "delta",
                "diff_mean",
                "diff_std",
                "n_changes",
            ]
        }

    # Basic statistics
    features[f"{prefix}_mean"] = np.mean(data)
    features[f"{prefix}_std"] = np.std(data)
    features[f"{prefix}_min"] = np.min(data)
    features[f"{prefix}_max"] = np.max(data)
    features[f"{prefix}_median"] = np.median(data)
    features[f"{prefix}_q25"] = np.percentile(data, 25)
    features[f"{prefix}_q75"] = np.percentile(data, 75)
    features[f"{prefix}_iqr"] = features[f"{prefix}_q75"] - features[f"{prefix}_q25"]
    features[f"{prefix}_range"] = features[f"{prefix}_max"] - features[f"{prefix}_min"]
    features[f"{prefix}_cv"] = features[f"{prefix}_std"] / (
        abs(features[f"{prefix}_mean"]) + 1e-8
    )

    # Shape metrics
    if len(data) > 1:
        features[f"{prefix}_skew"] = stats.skew(data)
        features[f"{prefix}_kurt"] = stats.kurtosis(data)
    else:
        features[f"{prefix}_skew"] = 0
        features[f"{prefix}_kurt"] = 0

    # Boundary features
    features[f"{prefix}_first"] = data[0]
    features[f"{prefix}_last"] = data[-1]
    features[f"{prefix}_delta"] = data[-1] - data[0]

    # Differential features
    if len(data) > 1:
        diff_data = np.diff(data)
        features[f"{prefix}_diff_mean"] = np.mean(diff_data)
        features[f"{prefix}_diff_std"] = np.std(diff_data)
        features[f"{prefix}_n_changes"] = np.sum(np.abs(diff_data) > np.std(data) * 0.1)
    else:
        features[f"{prefix}_diff_mean"] = 0
        features[f"{prefix}_diff_std"] = 0
        features[f"{prefix}_n_changes"] = 0

    # Segment features (3 segments)
    seq_len = len(data)
    if seq_len >= 9:
        seg_size = seq_len // 3
        for i in range(3):
            start_idx = i * seg_size
            end_idx = (i + 1) * seg_size if i < 2 else seq_len
            segment = data[start_idx:end_idx]
            features[f"{prefix}_seg{i + 1}_mean"] = np.mean(segment)
            features[f"{prefix}_seg{i + 1}_std"] = np.std(segment)

        # Segment transitions
        features[f"{prefix}_seg1_to_seg2"] = (
            features[f"{prefix}_seg2_mean"] - features[f"{prefix}_seg1_mean"]
        )
        features[f"{prefix}_seg2_to_seg3"] = (
            features[f"{prefix}_seg3_mean"] - features[f"{prefix}_seg2_mean"]
        )
    else:
        for i in range(3):
            features[f"{prefix}_seg{i + 1}_mean"] = features[f"{prefix}_mean"]
            features[f"{prefix}_seg{i + 1}_std"] = features[f"{prefix}_std"]
        features[f"{prefix}_seg1_to_seg2"] = 0
        features[f"{prefix}_seg2_to_seg3"] = 0

    return features


def extract_hjorth_parameters(data: np.ndarray, prefix: str) -> dict:
    """Extract Hjorth parameters (activity, mobility, complexity)."""
    features = {}

    if len(data) < 2:
        features[f"{prefix}_hjorth_activity"] = 0
        features[f"{prefix}_hjorth_mobility"] = 0
        features[f"{prefix}_hjorth_complexity"] = 0
        return features

    # Activity: variance of signal
    activity = np.var(data)
    features[f"{prefix}_hjorth_activity"] = activity

    # Mobility: sqrt(var(diff) / var(signal))
    diff1 = np.diff(data)
    if activity > 0:
        mobility = np.sqrt(np.var(diff1) / activity)
    else:
        mobility = 0
    features[f"{prefix}_hjorth_mobility"] = mobility

    # Complexity: mobility(diff) / mobility(signal)
    if len(diff1) > 1 and mobility > 0:
        diff2 = np.diff(diff1)
        mobility2 = np.sqrt(np.var(diff2) / np.var(diff1)) if np.var(diff1) > 0 else 0
        complexity = mobility2 / mobility
    else:
        complexity = 0
    features[f"{prefix}_hjorth_complexity"] = complexity

    return features


def extract_peak_features(data: np.ndarray, prefix: str) -> dict:
    """Extract peak-related features."""
    features = {}

    if len(data) < 3:
        features[f"{prefix}_n_peaks"] = 0
        features[f"{prefix}_peak_mean_height"] = 0
        features[f"{prefix}_peak_mean_distance"] = 0
        return features

    # Find peaks
    peaks, properties = find_peaks(data, height=np.std(data) * 0.5)

    features[f"{prefix}_n_peaks"] = len(peaks)

    if len(peaks) > 0:
        features[f"{prefix}_peak_mean_height"] = np.mean(properties["peak_heights"])
        if len(peaks) > 1:
            features[f"{prefix}_peak_mean_distance"] = np.mean(np.diff(peaks))
        else:
            features[f"{prefix}_peak_mean_distance"] = 0
    else:
        features[f"{prefix}_peak_mean_height"] = 0
        features[f"{prefix}_peak_mean_distance"] = 0

    return features


def extract_line_length(data: np.ndarray, prefix: str) -> dict:
    """Extract line length (sum of absolute differences)."""
    features = {}

    if len(data) < 2:
        features[f"{prefix}_line_length"] = 0
        return features

    features[f"{prefix}_line_length"] = np.sum(np.abs(np.diff(data)))

    return features


def extract_autocorrelation(
    data: np.ndarray, prefix: str, lags: list = [1, 2, 4, 8]
) -> dict:
    """Extract autocorrelation features at different lags."""
    features = {}

    if len(data) < max(lags) + 1:
        for lag in lags:
            features[f"{prefix}_autocorr_lag{lag}"] = 0
        return features

    # Normalize data
    data_norm = (data - np.mean(data)) / (np.std(data) + 1e-8)

    for lag in lags:
        if lag < len(data):
            features[f"{prefix}_autocorr_lag{lag}"] = np.corrcoef(
                data_norm[:-lag], data_norm[lag:]
            )[0, 1]
        else:
            features[f"{prefix}_autocorr_lag{lag}"] = 0

    return features


def extract_gradient_histogram(data: np.ndarray, prefix: str, n_bins: int = 10) -> dict:
    """Extract gradient histogram features."""
    features = {}

    if len(data) < 2:
        for i in range(n_bins):
            features[f"{prefix}_grad_hist_bin{i}"] = 0
        return features

    # Compute gradients
    gradients = np.diff(data)

    # Create histogram
    hist, _ = np.histogram(gradients, bins=n_bins)
    hist = hist / (len(gradients) + 1e-8)  # Normalize

    for i, val in enumerate(hist):
        features[f"{prefix}_grad_hist_bin{i}"] = val

    return features


def extract_jerk_features(
    acc_data: np.ndarray, prefix: str, dt: float = 1.0 / 20
) -> dict:
    """Extract jerk features (first derivative of acceleration)."""
    features = {}

    if len(acc_data) < 2:
        features[f"{prefix}_jerk_mean"] = 0
        features[f"{prefix}_jerk_std"] = 0
        features[f"{prefix}_jerk_max"] = 0
        features[f"{prefix}_jerk_p90"] = 0
        features[f"{prefix}_jerk_L2"] = 0
        return features

    # Compute jerk (derivative of acceleration)
    jerk = np.diff(acc_data) / dt

    features[f"{prefix}_jerk_mean"] = np.mean(np.abs(jerk))
    features[f"{prefix}_jerk_std"] = np.std(jerk)
    features[f"{prefix}_jerk_max"] = np.max(np.abs(jerk))
    features[f"{prefix}_jerk_p90"] = np.percentile(np.abs(jerk), 90)
    features[f"{prefix}_jerk_L2"] = np.sqrt(np.mean(jerk**2))  # L2 norm

    return features


# ====================================================================================================
# FREQUENCY DOMAIN FEATURES
# ====================================================================================================


def extract_frequency_features(data: np.ndarray, prefix: str, fs: float = 20.0) -> dict:
    """Extract frequency domain features using Welch's method."""
    features = {}

    if len(data) < CONFIG["welch_nperseg"]:
        # Not enough data for frequency analysis
        for band_idx, _ in enumerate(CONFIG["freq_bands"]):
            features[f"{prefix}_band{band_idx}_power"] = 0
        features[f"{prefix}_spectral_centroid"] = 0
        features[f"{prefix}_spectral_rolloff"] = 0
        features[f"{prefix}_spectral_entropy"] = 0
        features[f"{prefix}_dominant_freq"] = 0
        features[f"{prefix}_dominant_power"] = 0
        features[f"{prefix}_zcr"] = 0
        return features

    # Compute PSD using Welch's method
    freqs, psd = welch(
        data, fs=fs, nperseg=CONFIG["welch_nperseg"], noverlap=CONFIG["welch_noverlap"]
    )

    # Band power features
    for band_idx, (low, high) in enumerate(CONFIG["freq_bands"]):
        band_mask = (freqs >= low) & (freqs <= high)
        if np.any(band_mask):
            features[f"{prefix}_band{band_idx}_power"] = np.sum(psd[band_mask])
        else:
            features[f"{prefix}_band{band_idx}_power"] = 0

    # Spectral centroid
    if np.sum(psd) > 0:
        features[f"{prefix}_spectral_centroid"] = np.sum(freqs * psd) / np.sum(psd)
    else:
        features[f"{prefix}_spectral_centroid"] = 0

    # Spectral rolloff (85%)
    cumsum = np.cumsum(psd)
    if cumsum[-1] > 0:
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        if len(rolloff_idx) > 0:
            features[f"{prefix}_spectral_rolloff"] = freqs[rolloff_idx[0]]
        else:
            features[f"{prefix}_spectral_rolloff"] = freqs[-1]
    else:
        features[f"{prefix}_spectral_rolloff"] = 0

    # Spectral entropy
    psd_norm = psd / (np.sum(psd) + 1e-8)
    psd_norm = psd_norm[psd_norm > 0]
    if len(psd_norm) > 0:
        features[f"{prefix}_spectral_entropy"] = -np.sum(
            psd_norm * np.log(psd_norm + 1e-8)
        )
    else:
        features[f"{prefix}_spectral_entropy"] = 0

    # Dominant frequency
    if len(psd) > 0:
        dominant_idx = np.argmax(psd)
        features[f"{prefix}_dominant_freq"] = freqs[dominant_idx]
        features[f"{prefix}_dominant_power"] = psd[dominant_idx]
    else:
        features[f"{prefix}_dominant_freq"] = 0
        features[f"{prefix}_dominant_power"] = 0

    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
    features[f"{prefix}_zcr"] = zero_crossings / len(data)

    return features


# ====================================================================================================
# TOF SPATIAL FEATURES
# ====================================================================================================


def mirror_tof_by_handedness(tof_frame: np.ndarray, handedness: int) -> np.ndarray:
    """Mirror ToF frame based on handedness (0=left, 1=right)."""
    if handedness == 1:  # Right-handed, mirror horizontally
        if tof_frame.shape == (8, 8):
            return np.fliplr(tof_frame)
        else:
            # For flattened array, reshape then flip
            return np.fliplr(tof_frame.reshape(8, 8)).flatten()
    return tof_frame


def extract_tof_region_features(tof_frame: np.ndarray, prefix: str) -> dict:
    """Extract features from different spatial regions of ToF frame."""
    features = {}

    if tof_frame.shape != (8, 8):
        return features

    # Handle invalid values
    valid_mask = (tof_frame >= 0) & ~np.isnan(tof_frame)
    tof_clean = np.where(valid_mask, tof_frame, np.inf)

    # Center 3x3 region
    center_region = tof_clean[3:6, 3:6]
    valid_center = center_region[center_region < np.inf]
    if len(valid_center) > 0:
        features[f"{prefix}_center_mean"] = np.mean(valid_center)
        features[f"{prefix}_center_min"] = np.min(valid_center)
    else:
        features[f"{prefix}_center_mean"] = 0
        features[f"{prefix}_center_min"] = 0

    # Inner ring (5x5 excluding center 3x3)
    inner_mask = np.ones((8, 8), dtype=bool)
    inner_mask[2:6, 2:6] = True
    inner_mask[3:5, 3:5] = False
    inner_vals = tof_clean[inner_mask]
    valid_inner = inner_vals[inner_vals < np.inf]
    if len(valid_inner) > 0:
        features[f"{prefix}_inner_mean"] = np.mean(valid_inner)
    else:
        features[f"{prefix}_inner_mean"] = 0

    # Outer ring
    outer_mask = ~inner_mask
    outer_mask[2:6, 2:6] = False
    outer_vals = tof_clean[outer_mask]
    valid_outer = outer_vals[outer_vals < np.inf]
    if len(valid_outer) > 0:
        features[f"{prefix}_outer_mean"] = np.mean(valid_outer)
    else:
        features[f"{prefix}_outer_mean"] = 0

    # Four quadrants
    quadrants = [
        tof_clean[:4, :4],  # Top-left
        tof_clean[:4, 4:],  # Top-right
        tof_clean[4:, :4],  # Bottom-left
        tof_clean[4:, 4:],  # Bottom-right
    ]

    for i, quad in enumerate(quadrants):
        valid_quad = quad[quad < np.inf]
        if len(valid_quad) > 0:
            features[f"{prefix}_quad{i}_mean"] = np.mean(valid_quad)
            features[f"{prefix}_quad{i}_min"] = np.min(valid_quad)
        else:
            features[f"{prefix}_quad{i}_mean"] = 0
            features[f"{prefix}_quad{i}_min"] = 0

    # Left vs Right half
    left_half = tof_clean[:, :4]
    right_half = tof_clean[:, 4:]
    valid_left = left_half[left_half < np.inf]
    valid_right = right_half[right_half < np.inf]

    if len(valid_left) > 0 and len(valid_right) > 0:
        features[f"{prefix}_lr_asymmetry"] = np.mean(valid_left) - np.mean(valid_right)
    else:
        features[f"{prefix}_lr_asymmetry"] = 0

    # Top vs Bottom half
    top_half = tof_clean[:4, :]
    bottom_half = tof_clean[4:, :]
    valid_top = top_half[top_half < np.inf]
    valid_bottom = bottom_half[bottom_half < np.inf]

    if len(valid_top) > 0 and len(valid_bottom) > 0:
        features[f"{prefix}_tb_asymmetry"] = np.mean(valid_top) - np.mean(valid_bottom)
    else:
        features[f"{prefix}_tb_asymmetry"] = 0

    return features


def extract_tof_near_frac(
    tof_frame: np.ndarray, prefix: str, quantiles: list = [10, 20]
) -> dict:
    """Extract fraction of pixels below certain distance quantiles."""
    features = {}

    # Handle invalid values
    valid_mask = (tof_frame >= 0) & ~np.isnan(tof_frame)
    valid_data = tof_frame[valid_mask]

    if len(valid_data) == 0:
        for q in quantiles:
            features[f"{prefix}_near_frac_q{q}"] = 0
        return features

    for q in quantiles:
        threshold = np.percentile(valid_data, q)
        features[f"{prefix}_near_frac_q{q}"] = np.sum(valid_data < threshold) / len(
            valid_data
        )

    return features


def extract_tof_anisotropy(tof_frame: np.ndarray, prefix: str) -> dict:
    """Extract anisotropy features using PCA eigenvalues."""
    features = {}

    if tof_frame.shape != (8, 8):
        features[f"{prefix}_anisotropy"] = 0
        features[f"{prefix}_principal_angle"] = 0
        return features

    # Handle invalid values
    valid_mask = (tof_frame >= 0) & ~np.isnan(tof_frame)

    if np.sum(valid_mask) < 3:
        features[f"{prefix}_anisotropy"] = 0
        features[f"{prefix}_principal_angle"] = 0
        return features

    # Get coordinates of valid pixels weighted by inverse distance
    x, y = np.meshgrid(range(8), range(8))
    weights = np.where(valid_mask, 1.0 / (tof_frame + 1), 0)

    # Create point cloud
    valid_points = []
    for i in range(8):
        for j in range(8):
            if valid_mask[i, j]:
                weight = weights[i, j]
                valid_points.append([x[i, j] * weight, y[i, j] * weight])

    if len(valid_points) < 3:
        features[f"{prefix}_anisotropy"] = 0
        features[f"{prefix}_principal_angle"] = 0
        return features

    points = np.array(valid_points)

    # Compute covariance matrix
    cov = np.cov(points.T)

    # Get eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Anisotropy (ratio of eigenvalues)
    if eigenvalues[0] > 0:
        features[f"{prefix}_anisotropy"] = 1 - eigenvalues[1] / eigenvalues[0]
    else:
        features[f"{prefix}_anisotropy"] = 0

    # Principal direction angle
    principal_vec = eigenvectors[:, 0]
    features[f"{prefix}_principal_angle"] = np.arctan2(
        principal_vec[1], principal_vec[0]
    )

    return features


def extract_tof_sensor_sync_features(
    all_min_dists: dict, prefix: str = "tof_sync"
) -> dict:
    """Extract synchronization features across multiple ToF sensors."""
    features = {}

    if len(all_min_dists) < 2:
        features[f"{prefix}_simultaneous_drop_rate"] = 0
        features[f"{prefix}_avg_time_lag"] = 0
        features[f"{prefix}_coherence"] = 0
        return features

    # Convert to list of arrays
    sensor_data = []
    max_len = 0
    for sensor_id in sorted(all_min_dists.keys()):
        data = all_min_dists[sensor_id]
        sensor_data.append(data)
        max_len = max(max_len, len(data))

    # Pad arrays to same length
    padded_data = []
    for data in sensor_data:
        if len(data) < max_len:
            padded = np.pad(data, (0, max_len - len(data)), "edge")
        else:
            padded = data[:max_len]
        padded_data.append(padded)

    sensor_array = np.array(padded_data)

    # Simultaneous drop detection
    threshold = np.percentile(sensor_array, 20, axis=1, keepdims=True)
    proximity_masks = sensor_array < threshold

    # Count frames where multiple sensors detect proximity
    simultaneous_counts = np.sum(proximity_masks, axis=0)
    features[f"{prefix}_simultaneous_drop_rate"] = np.mean(
        simultaneous_counts >= 3
    )  # At least 3 sensors

    # Time lag analysis using cross-correlation
    lags = []
    for i in range(len(sensor_data)):
        for j in range(i + 1, len(sensor_data)):
            # Compute cross-correlation
            corr = np.correlate(
                sensor_data[i] - np.mean(sensor_data[i]),
                sensor_data[j] - np.mean(sensor_data[j]),
                mode="same",
            )
            # Find lag of maximum correlation
            lag = np.argmax(np.abs(corr)) - len(corr) // 2
            lags.append(abs(lag))

    if lags:
        features[f"{prefix}_avg_time_lag"] = np.mean(lags) / 20.0  # Convert to seconds
    else:
        features[f"{prefix}_avg_time_lag"] = 0

    # Overall coherence (average correlation between sensors)
    correlations = []
    for i in range(len(sensor_data)):
        for j in range(i + 1, len(sensor_data)):
            if len(sensor_data[i]) == len(sensor_data[j]):
                corr = np.corrcoef(sensor_data[i], sensor_data[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

    if correlations:
        features[f"{prefix}_coherence"] = np.mean(correlations)
    else:
        features[f"{prefix}_coherence"] = 0

    return features


def extract_tof_arrival_event_features(
    min_dists: np.ndarray, prefix: str, threshold_percentile: int = 20
) -> dict:
    """Extract arrival event features from ToF min distance time series."""
    features = {}

    if len(min_dists) < 2:
        features[f"{prefix}_arrival_rate"] = 0
        features[f"{prefix}_max_arrival_duration"] = 0
        features[f"{prefix}_arrival_frequency"] = 0
        return features

    # Compute threshold for "arrival" (close distance)
    threshold = np.percentile(min_dists, threshold_percentile)

    # Binary mask for arrival events
    arrival_mask = min_dists < threshold

    # Arrival rate (percentage of time in arrival state)
    features[f"{prefix}_arrival_rate"] = np.mean(arrival_mask)

    # Find continuous arrival segments
    changes = np.diff(np.concatenate(([0], arrival_mask.astype(int), [0])))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    if len(starts) > 0:
        durations = ends - starts
        features[f"{prefix}_max_arrival_duration"] = (
            np.max(durations) / 20.0
        )  # Convert to seconds
        features[f"{prefix}_arrival_frequency"] = len(starts) / (
            len(min_dists) / 20.0
        )  # Events per second
    else:
        features[f"{prefix}_max_arrival_duration"] = 0
        features[f"{prefix}_arrival_frequency"] = 0

    return features


def extract_tof_clustering_features(
    tof_frame: np.ndarray, prefix: str, threshold_percentile: int = 20
) -> dict:
    """Extract clustering features from binarized ToF frame."""
    features = {}

    if tof_frame.shape != (8, 8):
        for key in ["max_cluster_size", "n_clusters", "cluster_circularity"]:
            features[f"{prefix}_{key}"] = 0
        return features

    # Handle invalid values
    valid_mask = (tof_frame >= 0) & ~np.isnan(tof_frame)
    valid_data = tof_frame[valid_mask]

    if len(valid_data) == 0:
        for key in ["max_cluster_size", "n_clusters", "cluster_circularity"]:
            features[f"{prefix}_{key}"] = 0
        return features

    # Binarize based on threshold
    threshold = np.percentile(valid_data, threshold_percentile)
    binary = (tof_frame < threshold) & valid_mask

    # Connected components analysis
    from scipy import ndimage

    labeled, n_clusters = ndimage.label(binary)

    features[f"{prefix}_n_clusters"] = n_clusters

    if n_clusters > 0:
        # Find largest cluster
        cluster_sizes = [np.sum(labeled == i) for i in range(1, n_clusters + 1)]
        max_cluster_size = max(cluster_sizes)
        features[f"{prefix}_max_cluster_size"] = max_cluster_size

        # Compute circularity of largest cluster
        max_cluster_label = cluster_sizes.index(max_cluster_size) + 1
        cluster_mask = labeled == max_cluster_label

        # Perimeter approximation
        perimeter = np.sum(np.abs(np.diff(cluster_mask.astype(int), axis=0))) + np.sum(
            np.abs(np.diff(cluster_mask.astype(int), axis=1))
        )

        if perimeter > 0:
            features[f"{prefix}_cluster_circularity"] = (
                4 * np.pi * max_cluster_size / (perimeter**2)
            )
        else:
            features[f"{prefix}_cluster_circularity"] = 0
    else:
        features[f"{prefix}_max_cluster_size"] = 0
        features[f"{prefix}_cluster_circularity"] = 0

    return features


def extract_tof_spatial_features(tof_frame: np.ndarray, prefix: str) -> dict:
    """Extract spatial features from 8x8 ToF frame."""
    features = {}

    # Handle invalid values
    valid_mask = (tof_frame >= 0) & ~np.isnan(tof_frame)
    valid_data = tof_frame[valid_mask]

    features[f"{prefix}_valid_ratio"] = np.sum(valid_mask) / tof_frame.size

    if len(valid_data) == 0:
        # No valid data
        for key in [
            "mean",
            "std",
            "min",
            "max",
            "p10",
            "p50",
            "p90",
            "centroid_x",
            "centroid_y",
            "moment_xx",
            "moment_yy",
            "moment_xy",
            "eccentricity",
            "edge_sum",
            "gradient_sum",
        ]:
            features[f"{prefix}_{key}"] = 0
        return features

    # Basic statistics
    features[f"{prefix}_mean"] = np.mean(valid_data)
    features[f"{prefix}_std"] = np.std(valid_data)
    features[f"{prefix}_min"] = np.min(valid_data)
    features[f"{prefix}_max"] = np.max(valid_data)
    features[f"{prefix}_p10"] = np.percentile(valid_data, 10)
    features[f"{prefix}_p50"] = np.percentile(valid_data, 50)
    features[f"{prefix}_p90"] = np.percentile(valid_data, 90)

    # Spatial moments and centroid
    if tof_frame.shape == (8, 8):
        # Create coordinate grids
        x, y = np.meshgrid(range(8), range(8))

        # Use valid data for weights
        weights = np.where(
            valid_mask, 1.0 / (tof_frame + 1), 0
        )  # Inverse distance weighting
        total_weight = np.sum(weights)

        if total_weight > 0:
            # Centroid
            cx = np.sum(x * weights) / total_weight
            cy = np.sum(y * weights) / total_weight
            features[f"{prefix}_centroid_x"] = cx
            features[f"{prefix}_centroid_y"] = cy

            # Second moments
            features[f"{prefix}_moment_xx"] = (
                np.sum((x - cx) ** 2 * weights) / total_weight
            )
            features[f"{prefix}_moment_yy"] = (
                np.sum((y - cy) ** 2 * weights) / total_weight
            )
            features[f"{prefix}_moment_xy"] = (
                np.sum((x - cx) * (y - cy) * weights) / total_weight
            )

            # Eccentricity
            if features[f"{prefix}_moment_xx"] + features[f"{prefix}_moment_yy"] > 0:
                lambda1 = 0.5 * (
                    features[f"{prefix}_moment_xx"]
                    + features[f"{prefix}_moment_yy"]
                    + np.sqrt(
                        (
                            features[f"{prefix}_moment_xx"]
                            - features[f"{prefix}_moment_yy"]
                        )
                        ** 2
                        + 4 * features[f"{prefix}_moment_xy"] ** 2
                    )
                )
                lambda2 = 0.5 * (
                    features[f"{prefix}_moment_xx"]
                    + features[f"{prefix}_moment_yy"]
                    - np.sqrt(
                        (
                            features[f"{prefix}_moment_xx"]
                            - features[f"{prefix}_moment_yy"]
                        )
                        ** 2
                        + 4 * features[f"{prefix}_moment_xy"] ** 2
                    )
                )
                if lambda1 > 0:
                    features[f"{prefix}_eccentricity"] = np.sqrt(1 - lambda2 / lambda1)
                else:
                    features[f"{prefix}_eccentricity"] = 0
            else:
                features[f"{prefix}_eccentricity"] = 0
        else:
            features[f"{prefix}_centroid_x"] = 4
            features[f"{prefix}_centroid_y"] = 4
            features[f"{prefix}_moment_xx"] = 0
            features[f"{prefix}_moment_yy"] = 0
            features[f"{prefix}_moment_xy"] = 0
            features[f"{prefix}_eccentricity"] = 0

        # Edge detection (simplified)
        valid_frame = np.where(valid_mask, tof_frame, 0)
        dx = np.abs(np.diff(valid_frame, axis=1))
        dy = np.abs(np.diff(valid_frame, axis=0))
        features[f"{prefix}_edge_sum"] = np.sum(dx) + np.sum(dy)

        # Gradient sum (Sobel-like)
        features[f"{prefix}_gradient_sum"] = np.sqrt(np.sum(dx**2) + np.sum(dy**2))
    else:
        # Flat data
        for key in [
            "centroid_x",
            "centroid_y",
            "moment_xx",
            "moment_yy",
            "moment_xy",
            "eccentricity",
            "edge_sum",
            "gradient_sum",
        ]:
            features[f"{prefix}_{key}"] = 0

    return features


# ====================================================================================================
# THERMAL FEATURES
# ====================================================================================================


def extract_thermal_advanced_features(
    thm_data: np.ndarray, prefix: str, threshold_percentile: float = 75
) -> dict:
    """Extract advanced thermal features including second derivatives and event rates."""
    features = {}

    if len(thm_data) < 3:
        features[f"{prefix}_diff2_mean"] = 0
        features[f"{prefix}_diff2_std"] = 0
        features[f"{prefix}_diff2_max"] = 0
        features[f"{prefix}_change_event_rate"] = 0
        features[f"{prefix}_p90_p10_spread"] = 0
        return features

    # Second derivative (acceleration of temperature change)
    diff1 = np.diff(thm_data)
    diff2 = np.diff(diff1)

    features[f"{prefix}_diff2_mean"] = np.mean(diff2)
    features[f"{prefix}_diff2_std"] = np.std(diff2)
    features[f"{prefix}_diff2_max"] = np.max(np.abs(diff2))

    # Change event rate (percentage of significant changes)
    threshold = np.percentile(np.abs(diff1), threshold_percentile)
    features[f"{prefix}_change_event_rate"] = np.mean(np.abs(diff1) > threshold)

    # Quantile spread (p90 - p10)
    features[f"{prefix}_p90_p10_spread"] = np.percentile(thm_data, 90) - np.percentile(
        thm_data, 10
    )

    return features


# ====================================================================================================
# CROSS-MODAL FEATURES
# ====================================================================================================


def extract_cross_modal_sync_features(
    linear_acc_mag: np.ndarray,
    tof_min_dists: dict,
    thermal_data: dict,
    omega_mag: np.ndarray = None,
) -> dict:
    """Extract sophisticated cross-modal synchronization features."""
    features = {}

    # Find linear acceleration peaks
    if len(linear_acc_mag) < 10:
        return features

    peaks, _ = find_peaks(linear_acc_mag, height=np.std(linear_acc_mag) * 0.5)

    if len(peaks) == 0:
        features["cross_modal_sync_score"] = 0
        features["cross_modal_triplet_consistency"] = 0
        return features

    # Micro-synchronization: Check ToF min_dist around acceleration peaks
    window_size = 10  # ±0.5 seconds at 20Hz
    min_dist_drops = []

    for sensor_id, min_dists in tof_min_dists.items():
        if len(min_dists) != len(linear_acc_mag):
            continue

        for peak_idx in peaks:
            window_start = max(0, peak_idx - window_size)
            window_end = min(len(min_dists), peak_idx + window_size)

            if window_end > window_start:
                window_data = min_dists[window_start:window_end]
                if len(window_data) > 0:
                    # Calculate drop in min_dist around peak
                    baseline = np.mean(min_dists)
                    window_min = np.min(window_data)
                    drop = baseline - window_min
                    min_dist_drops.append(drop)

    if min_dist_drops:
        features["cross_modal_acc_tof_sync_mean"] = np.mean(min_dist_drops)
        features["cross_modal_acc_tof_sync_max"] = np.max(min_dist_drops)
    else:
        features["cross_modal_acc_tof_sync_mean"] = 0
        features["cross_modal_acc_tof_sync_max"] = 0

    # Triplet consistency: min_dist drop → thermal rise → acceleration peak
    triplet_scores = []

    for sensor_id in range(1, 6):
        if (
            f"tof_{sensor_id}" not in tof_min_dists
            or f"thm_{sensor_id}" not in thermal_data
        ):
            continue

        tof_data = tof_min_dists[f"tof_{sensor_id}"]
        thm_data = thermal_data[f"thm_{sensor_id}"]

        if len(tof_data) < 20 or len(thm_data) < 20:
            continue

        # Find ToF proximity events
        tof_threshold = np.percentile(tof_data, 20)
        tof_events = np.where(tof_data < tof_threshold)[0]

        for event_idx in tof_events[:10]:  # Check first 10 events
            if event_idx + 20 < len(thm_data) and event_idx + 20 < len(linear_acc_mag):
                # Check if thermal increases after ToF proximity
                thm_before = np.mean(thm_data[max(0, event_idx - 5) : event_idx])
                thm_after = np.mean(
                    thm_data[event_idx : min(event_idx + 10, len(thm_data))]
                )
                thm_increase = thm_after > thm_before

                # Check if acceleration peak follows
                acc_window = linear_acc_mag[
                    event_idx : min(event_idx + 20, len(linear_acc_mag))
                ]
                acc_peak = np.any(
                    acc_window > np.mean(linear_acc_mag) + np.std(linear_acc_mag)
                )

                if thm_increase and acc_peak:
                    triplet_scores.append(1.0)
                else:
                    triplet_scores.append(0.0)

    if triplet_scores:
        features["cross_modal_triplet_consistency"] = np.mean(triplet_scores)
    else:
        features["cross_modal_triplet_consistency"] = 0

    # Angular velocity correlation with ToF
    if omega_mag is not None and len(omega_mag) > 0:
        for sensor_id, min_dists in tof_min_dists.items():
            if len(min_dists) == len(omega_mag):
                features[f"cross_modal_omega_{sensor_id}_corr"] = np.corrcoef(
                    omega_mag, min_dists
                )[0, 1]

    return features


# ====================================================================================================
# MULTI-RESOLUTION WINDOW PROCESSING
# ====================================================================================================


def extract_multi_resolution_features(sequence_df: pd.DataFrame, config: dict) -> dict:
    """Extract features from multiple time windows (S/M/L) with Temporal Pyramid."""
    features = {}

    if not config.get("use_multi_resolution", False):
        return features

    seq_len = len(sequence_df)
    window_sizes = config.get(
        "window_sizes", {"S": (20, 30), "M": (60, 80), "L": (200, 256)}
    )

    # For each window size
    for window_name, (min_size, max_size) in window_sizes.items():
        # Determine actual window size based on sequence length
        if seq_len < min_size:
            continue

        window_size = min(max_size, seq_len)

        # Extract tail window (emphasized for prediction)
        if config.get("use_tail_emphasis", True):
            start_idx = max(0, seq_len - window_size)
            window_df = sequence_df.iloc[start_idx:]
        else:
            # Use middle window
            start_idx = max(0, (seq_len - window_size) // 2)
            window_df = sequence_df.iloc[start_idx : start_idx + window_size]

        # Extract basic statistics for this window
        for col in ["acc_x", "acc_y", "acc_z"]:
            if col in window_df.columns:
                data = window_df[col].values
                features[f"{window_name}_{col}_mean"] = np.mean(data)
                features[f"{window_name}_{col}_std"] = np.std(data)
                features[f"{window_name}_{col}_max"] = np.max(data)
                features[f"{window_name}_{col}_min"] = np.min(data)

    # Temporal Pyramid: Additional multi-scale aggregation (0.5s, 1s, 2s windows)
    pyramid_windows = {
        "micro": 10,  # 0.5 seconds at 20Hz
        "short": 20,  # 1.0 seconds at 20Hz
        "medium": 40,  # 2.0 seconds at 20Hz
    }

    for col in ["acc_x", "acc_y", "acc_z"]:
        if col in sequence_df.columns:
            data = sequence_df[col].values

            for pyramid_name, pyramid_size in pyramid_windows.items():
                if len(data) >= pyramid_size:
                    # Apply moving average with this window
                    smoothed = (
                        pd.Series(data)
                        .rolling(pyramid_size, center=True, min_periods=1)
                        .mean()
                        .values
                    )

                    # Extract statistics from smoothed signal
                    features[f"pyramid_{pyramid_name}_{col}_mean"] = np.mean(smoothed)
                    features[f"pyramid_{pyramid_name}_{col}_std"] = np.std(smoothed)
                    features[f"pyramid_{pyramid_name}_{col}_p10"] = np.percentile(
                        smoothed, 10
                    )
                    features[f"pyramid_{pyramid_name}_{col}_p90"] = np.percentile(
                        smoothed, 90
                    )

    return features


# ====================================================================================================
# MAIN FEATURE EXTRACTION
# ====================================================================================================


def extract_features_parallel(args):
    """Global function for parallel feature extraction (used only in local environment)."""
    extractor, seq_df, demo_df = args
    return extractor.extract_features(seq_df, demo_df)


class FeatureExtractor:
    """Main feature extraction class with fitted transformers."""

    def __init__(self, config: dict):
        self.config = config
        self.scaler = None
        self.tof_pcas = {}  # Will store PCA transformers for each ToF sensor
        self.feature_names = None
        self.is_fitted = False

    def extract_features(
        self, sequence_df: pd.DataFrame, demographics_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract all features from a sequence."""
        features = {}

        # Sequence metadata
        features["sequence_length"] = len(sequence_df)

        # Demographics features
        if len(demographics_df) > 0:
            demo_row = demographics_df.iloc[0]
            features["age"] = demo_row.get("age", 0)
            features["adult_child"] = demo_row.get("adult_child", 0)
            features["sex"] = demo_row.get("sex", 0)
            features["handedness"] = demo_row.get("handedness", 0)
            features["height_cm"] = demo_row.get("height_cm", 0)
            features["shoulder_to_wrist_cm"] = demo_row.get("shoulder_to_wrist_cm", 0)
            features["elbow_to_wrist_cm"] = demo_row.get("elbow_to_wrist_cm", 0)

        # ========== IMU Features ==========

        # Get IMU data
        acc_cols = ["acc_x", "acc_y", "acc_z"]
        rot_cols = ["rot_w", "rot_x", "rot_y", "rot_z"]

        if all(col in sequence_df.columns for col in acc_cols):
            acc_data = sequence_df[acc_cols].values
            # Handle NaN
            for i in range(3):
                acc_data[:, i] = (
                    pd.Series(acc_data[:, i])
                    .fillna(method="ffill")
                    .fillna(method="bfill")
                    .fillna(0)
                    .values
                )
        else:
            acc_data = np.zeros((len(sequence_df), 3))

        if all(col in sequence_df.columns for col in rot_cols):
            rot_data = sequence_df[rot_cols].values
            rot_data = handle_quaternion_missing(rot_data)
        else:
            rot_data = np.tile([1, 0, 0, 0], (len(sequence_df), 1))

        # World acceleration
        if self.config["use_world_acc"]:
            world_acc = compute_world_acceleration(acc_data, rot_data)
            for i, axis in enumerate(["x", "y", "z"]):
                features.update(
                    extract_statistical_features(world_acc[:, i], f"world_acc_{axis}")
                )
                features.update(
                    extract_hjorth_parameters(world_acc[:, i], f"world_acc_{axis}")
                )
                features.update(
                    extract_peak_features(world_acc[:, i], f"world_acc_{axis}")
                )
                features.update(
                    extract_line_length(world_acc[:, i], f"world_acc_{axis}")
                )
                features.update(
                    extract_autocorrelation(world_acc[:, i], f"world_acc_{axis}")
                )
                if self.config["use_frequency_features"]:
                    features.update(
                        extract_frequency_features(world_acc[:, i], f"world_acc_{axis}")
                    )

            # World acceleration magnitude
            world_acc_mag = np.linalg.norm(world_acc, axis=1)
            features.update(
                extract_statistical_features(world_acc_mag, "world_acc_mag")
            )
            features.update(extract_hjorth_parameters(world_acc_mag, "world_acc_mag"))
            features.update(extract_peak_features(world_acc_mag, "world_acc_mag"))
            features.update(extract_line_length(world_acc_mag, "world_acc_mag"))
            features.update(extract_jerk_features(world_acc_mag, "world_acc_mag"))
            if self.config["use_frequency_features"]:
                features.update(
                    extract_frequency_features(world_acc_mag, "world_acc_mag")
                )

            # Horizontal vs Vertical decomposition
            world_acc_horizontal = np.sqrt(world_acc[:, 0] ** 2 + world_acc[:, 1] ** 2)
            world_acc_vertical = np.abs(world_acc[:, 2])
            features.update(
                extract_statistical_features(
                    world_acc_horizontal, "world_acc_horizontal"
                )
            )
            features.update(
                extract_statistical_features(world_acc_vertical, "world_acc_vertical")
            )
            features["world_acc_h_v_ratio"] = np.mean(world_acc_horizontal) / (
                np.mean(world_acc_vertical) + 1e-8
            )

        # Linear acceleration (gravity removed)
        linear_acc = None
        linear_acc_mag = None
        if self.config["use_linear_acc"]:
            linear_acc = compute_linear_acceleration(acc_data, rot_data)
            for i, axis in enumerate(["x", "y", "z"]):
                features.update(
                    extract_statistical_features(linear_acc[:, i], f"linear_acc_{axis}")
                )
                features.update(
                    extract_hjorth_parameters(linear_acc[:, i], f"linear_acc_{axis}")
                )
                features.update(
                    extract_line_length(linear_acc[:, i], f"linear_acc_{axis}")
                )
                features.update(
                    extract_autocorrelation(linear_acc[:, i], f"linear_acc_{axis}")
                )
                features.update(
                    extract_gradient_histogram(linear_acc[:, i], f"linear_acc_{axis}")
                )
                if self.config["use_frequency_features"]:
                    features.update(
                        extract_frequency_features(
                            linear_acc[:, i], f"linear_acc_{axis}"
                        )
                    )

            # Linear acceleration magnitude
            linear_acc_mag = np.linalg.norm(linear_acc, axis=1)
            features.update(
                extract_statistical_features(linear_acc_mag, "linear_acc_mag")
            )
            features.update(extract_peak_features(linear_acc_mag, "linear_acc_mag"))
            features.update(extract_line_length(linear_acc_mag, "linear_acc_mag"))
            features.update(extract_jerk_features(linear_acc_mag, "linear_acc_mag"))
            if self.config["use_frequency_features"]:
                features.update(
                    extract_frequency_features(linear_acc_mag, "linear_acc_mag")
                )

        # Angular velocity
        omega_mag = None
        if self.config["use_angular_velocity"]:
            omega = compute_angular_velocity(rot_data)
            for i, axis in enumerate(["x", "y", "z"]):
                features.update(
                    extract_statistical_features(omega[:, i], f"omega_{axis}")
                )
                features.update(extract_line_length(omega[:, i], f"omega_{axis}"))
                if self.config["use_frequency_features"]:
                    features.update(
                        extract_frequency_features(omega[:, i], f"omega_{axis}")
                    )

            omega_mag = np.linalg.norm(omega, axis=1)
            features.update(extract_statistical_features(omega_mag, "omega_mag"))
            features.update(extract_line_length(omega_mag, "omega_mag"))

            # Correlation between linear_acc_mag and omega_mag
            if linear_acc_mag is not None and len(linear_acc_mag) == len(omega_mag):
                features["linear_omega_corr"] = np.corrcoef(linear_acc_mag, omega_mag)[
                    0, 1
                ]

        # Euler angles
        euler = quaternion_to_euler(rot_data)
        for i, angle in enumerate(["roll", "pitch", "yaw"]):
            features.update(extract_statistical_features(euler[:, i], angle))

        # Raw acceleration (for comparison)
        for i, axis in enumerate(["x", "y", "z"]):
            features.update(extract_statistical_features(acc_data[:, i], f"acc_{axis}"))
            features.update(extract_hjorth_parameters(acc_data[:, i], f"acc_{axis}"))
            if self.config["use_frequency_features"]:
                features.update(
                    extract_frequency_features(acc_data[:, i], f"acc_{axis}")
                )

        acc_mag = np.linalg.norm(acc_data, axis=1)
        features.update(extract_statistical_features(acc_mag, "acc_mag"))

        # Quaternion features
        for i, comp in enumerate(["w", "x", "y", "z"]):
            features.update(extract_statistical_features(rot_data[:, i], f"rot_{comp}"))

        # Axis correlations
        if self.config["use_world_acc"]:
            features["world_acc_corr_xy"] = np.corrcoef(
                world_acc[:, 0], world_acc[:, 1]
            )[0, 1]
            features["world_acc_corr_yz"] = np.corrcoef(
                world_acc[:, 1], world_acc[:, 2]
            )[0, 1]
            features["world_acc_corr_xz"] = np.corrcoef(
                world_acc[:, 0], world_acc[:, 2]
            )[0, 1]

        if self.config["use_linear_acc"]:
            features["linear_acc_corr_xy"] = np.corrcoef(
                linear_acc[:, 0], linear_acc[:, 1]
            )[0, 1]
            features["linear_acc_corr_yz"] = np.corrcoef(
                linear_acc[:, 1], linear_acc[:, 2]
            )[0, 1]
            features["linear_acc_corr_xz"] = np.corrcoef(
                linear_acc[:, 0], linear_acc[:, 2]
            )[0, 1]

        # ========== ToF Features ==========

        tof_min_dists_all = {}  # Store for cross-modal features
        thermal_data_all = {}  # Store for cross-modal features

        if self.config["use_tof_spatial"]:
            tof_sensors = 5
            handedness = (
                demographics_df.iloc[0].get("handedness", 0)
                if len(demographics_df) > 0
                else 0
            )

            for sensor_id in range(1, tof_sensors + 1):
                tof_cols = [f"tof_{sensor_id}_v{i}" for i in range(64)]

                if all(col in sequence_df.columns for col in tof_cols):
                    tof_data = sequence_df[tof_cols].values

                    # Apply handedness mirroring if enabled
                    if self.config.get("tof_use_handedness_mirror", False):
                        for idx in range(len(tof_data)):
                            tof_data[idx] = mirror_tof_by_handedness(
                                tof_data[idx], handedness
                            )

                    # PCA transformation if enabled
                    if self.config.get("tof_use_pca", False):
                        # Handle invalid values for PCA
                        valid_mask = (tof_data >= 0) & ~np.isnan(tof_data)
                        tof_clean = np.where(valid_mask, tof_data, 0)

                        # Check if we have enough valid samples and features
                        n_samples, n_features = tof_clean.shape
                        # PCA components must be min of (n_samples-1, n_features, requested_components)
                        max_components = min(
                            n_samples - 1, n_features, self.config["tof_pca_components"]
                        )

                        if max_components < 2:
                            # Skip PCA if not enough data
                            tof_pca_features = None
                        else:
                            # Fit PCA if not already fitted
                            if self.is_fitted and f"tof_{sensor_id}" in self.tof_pcas:
                                # Use existing PCA
                                pca = self.tof_pcas[f"tof_{sensor_id}"]
                                try:
                                    tof_pca_features = pca.transform(tof_clean)
                                except:
                                    # If transform fails, skip PCA for this sensor
                                    tof_pca_features = None
                            else:
                                # Fit new PCA (during training)
                                pca = PCA(n_components=max_components)
                                try:
                                    tof_pca_features = pca.fit_transform(tof_clean)
                                    self.tof_pcas[f"tof_{sensor_id}"] = pca
                                except:
                                    # If PCA fails, skip it
                                    tof_pca_features = None

                        # Extract PCA features
                        for comp_idx in range(tof_pca_features.shape[1]):
                            pca_series = tof_pca_features[:, comp_idx]
                            features.update(
                                extract_statistical_features(
                                    pca_series, f"tof_{sensor_id}_pca{comp_idx}"
                                )
                            )

                        # Reconstruction error
                        reconstructed = pca.inverse_transform(tof_pca_features)
                        recon_error = np.mean(np.abs(tof_clean - reconstructed), axis=1)
                        features.update(
                            extract_statistical_features(
                                recon_error, f"tof_{sensor_id}_recon_error"
                            )
                        )

                    # Process each frame for spatial features
                    frame_features = []
                    for frame_idx in range(len(tof_data)):
                        frame_8x8 = tof_data[frame_idx].reshape(8, 8)

                        # Basic spatial features
                        frame_feat = extract_tof_spatial_features(
                            frame_8x8, f"tof_{sensor_id}_frame"
                        )

                        # Additional spatial features if enabled
                        if self.config.get("tof_region_analysis", False):
                            frame_feat.update(
                                extract_tof_region_features(
                                    frame_8x8, f"tof_{sensor_id}_frame"
                                )
                            )
                            frame_feat.update(
                                extract_tof_near_frac(
                                    frame_8x8, f"tof_{sensor_id}_frame"
                                )
                            )
                            frame_feat.update(
                                extract_tof_anisotropy(
                                    frame_8x8, f"tof_{sensor_id}_frame"
                                )
                            )
                            frame_feat.update(
                                extract_tof_clustering_features(
                                    frame_8x8, f"tof_{sensor_id}_frame"
                                )
                            )

                        frame_features.append(frame_feat)

                    # Aggregate over time
                    frame_df = pd.DataFrame(frame_features)
                    for col in frame_df.columns:
                        time_series = frame_df[col].values
                        # Time statistics
                        features.update(extract_statistical_features(time_series, col))

                        # Velocity (first difference)
                        if len(time_series) > 1:
                            velocity = np.diff(time_series)
                            features[f"{col}_velocity_mean"] = np.mean(velocity)
                            features[f"{col}_velocity_std"] = np.std(velocity)
                            features[f"{col}_velocity_max"] = np.max(np.abs(velocity))

                            # Acceleration (second difference)
                            if len(velocity) > 1:
                                acceleration = np.diff(velocity)
                                features[f"{col}_accel_mean"] = np.mean(acceleration)
                                features[f"{col}_accel_std"] = np.std(acceleration)
                                features[f"{col}_accel_max"] = np.max(
                                    np.abs(acceleration)
                                )

                    # Min distance tracking
                    min_dists = []
                    for frame_idx in range(len(tof_data)):
                        frame_8x8 = tof_data[frame_idx].reshape(8, 8)
                        valid_data = frame_8x8[(frame_8x8 >= 0) & ~np.isnan(frame_8x8)]
                        if len(valid_data) > 0:
                            min_dists.append(np.min(valid_data))
                        else:
                            min_dists.append(np.inf)

                    min_dists = np.array(min_dists)
                    min_dists[np.isinf(min_dists)] = np.nanmedian(
                        min_dists[~np.isinf(min_dists)]
                    )
                    if np.all(np.isnan(min_dists)):
                        min_dists[:] = 1000  # Default far distance

                    # Store for cross-modal features
                    tof_min_dists_all[f"tof_{sensor_id}"] = min_dists

                    features.update(
                        extract_statistical_features(
                            min_dists, f"tof_{sensor_id}_min_dist"
                        )
                    )

                    # Arrival event features
                    features.update(
                        extract_tof_arrival_event_features(
                            min_dists, f"tof_{sensor_id}"
                        )
                    )

                    # Proximity events
                    proximity_threshold = np.percentile(
                        min_dists[~np.isnan(min_dists)], 20
                    )
                    proximity_mask = min_dists < proximity_threshold
                    features[f"tof_{sensor_id}_proximity_ratio"] = np.mean(
                        proximity_mask
                    )

                    # Longest proximity duration
                    if np.any(proximity_mask):
                        changes = np.diff(
                            np.concatenate(([0], proximity_mask.astype(int), [0]))
                        )
                        starts = np.where(changes == 1)[0]
                        ends = np.where(changes == -1)[0]
                        durations = ends - starts
                        features[f"tof_{sensor_id}_max_proximity_duration"] = (
                            np.max(durations) if len(durations) > 0 else 0
                        )
                    else:
                        features[f"tof_{sensor_id}_max_proximity_duration"] = 0

            # Cross-sensor features
            all_min_dists = []
            for sensor_id in range(1, tof_sensors + 1):
                key = f"tof_{sensor_id}_min_dist_mean"
                if key in features:
                    all_min_dists.append(features[key])

            if all_min_dists:
                features["tof_global_min_dist"] = np.min(all_min_dists)
                features["tof_global_mean_dist"] = np.mean(all_min_dists)
                features["tof_sensor_variance"] = np.var(all_min_dists)

            # Sensor synchronization features
            if len(tof_min_dists_all) >= 2:
                sync_features = extract_tof_sensor_sync_features(tof_min_dists_all)
                features.update(sync_features)

        # ========== Thermal Features ==========

        if self.config["use_thermal_trends"]:
            for thm_id in range(1, 6):
                thm_col = f"thm_{thm_id}"
                if thm_col in sequence_df.columns:
                    thm_data = sequence_df[thm_col].values
                    thm_data = (
                        pd.Series(thm_data)
                        .fillna(method="ffill")
                        .fillna(method="bfill")
                        .fillna(0)
                        .values
                    )

                    # Store for cross-modal features
                    thermal_data_all[f"thm_{thm_id}"] = thm_data

                    # Smooth with moving average
                    if len(thm_data) > 5:
                        thm_smooth = (
                            pd.Series(thm_data)
                            .rolling(5, center=True, min_periods=1)
                            .mean()
                            .values
                        )
                    else:
                        thm_smooth = thm_data

                    features.update(
                        extract_statistical_features(thm_smooth, f"thm_{thm_id}")
                    )

                    # Advanced thermal features
                    features.update(
                        extract_thermal_advanced_features(thm_smooth, f"thm_{thm_id}")
                    )

                    # Trend
                    if len(thm_smooth) > 1:
                        x = np.arange(len(thm_smooth))
                        features[f"thm_{thm_id}_trend"] = np.corrcoef(x, thm_smooth)[
                            0, 1
                        ]

                        # First derivative (temperature change rate)
                        thm_diff = np.diff(thm_smooth)
                        features[f"thm_{thm_id}_diff_mean"] = np.mean(thm_diff)
                        features[f"thm_{thm_id}_diff_std"] = np.std(thm_diff)
                        features[f"thm_{thm_id}_diff_max"] = np.max(np.abs(thm_diff))
                    else:
                        features[f"thm_{thm_id}_trend"] = 0
                        features[f"thm_{thm_id}_diff_mean"] = 0
                        features[f"thm_{thm_id}_diff_std"] = 0
                        features[f"thm_{thm_id}_diff_max"] = 0

        # ========== Cross-modal Features ==========

        if (
            self.config["use_cross_modal"]
            and self.config["use_linear_acc"]
            and self.config["use_tof_spatial"]
        ):
            # Linear acceleration peaks vs ToF proximity
            if (
                "linear_acc_mag_n_peaks" in features
                and "tof_global_min_dist" in features
            ):
                features["cross_modal_acc_tof_ratio"] = features[
                    "linear_acc_mag_n_peaks"
                ] / (features["tof_global_min_dist"] + 1)

            # Temperature change vs proximity
            if self.config["use_thermal_trends"]:
                for thm_id in range(1, 6):
                    thm_key = f"thm_{thm_id}_diff_max"
                    if thm_key in features and "tof_global_min_dist" in features:
                        features[f"cross_modal_thm{thm_id}_tof"] = features[thm_key] * (
                            1000 / (features["tof_global_min_dist"] + 1)
                        )

            # Advanced cross-modal synchronization features
            if linear_acc_mag is not None and tof_min_dists_all and thermal_data_all:
                sync_features = extract_cross_modal_sync_features(
                    linear_acc_mag, tof_min_dists_all, thermal_data_all, omega_mag
                )
                features.update(sync_features)

        # ========== Multi-resolution Window Features ==========

        if self.config.get("use_multi_resolution", False):
            multi_res_features = extract_multi_resolution_features(
                sequence_df, self.config
            )
            features.update(multi_res_features)

        # Replace NaN/inf with 0
        for key in features:
            if isinstance(features[key], (int, float)):
                if np.isnan(features[key]) or np.isinf(features[key]):
                    features[key] = 0

        return pd.DataFrame([features])

    def fit_transform(
        self,
        sequences: List[pd.DataFrame],
        demographics: List[pd.DataFrame],
        labels: np.ndarray = None,
    ) -> pd.DataFrame:
        """Fit transformers and extract features from training data."""
        print("Extracting features from sequences...")

        start_time = time.time()
        feature_dfs = []

        # Use parallel processing for local environment, sequential for Kaggle
        if not IS_KAGGLE_ENV:
            # Local environment: Use parallel processing
            n_processes = min(cpu_count() - 1, 8)  # Leave one CPU free, max 8 processes
            print(
                f"  Processing {len(sequences)} sequences in PARALLEL (Local environment)"
            )
            print(f"  Using {n_processes} processes")

            # Prepare data for parallel processing
            data_with_extractor = [
                (self, seq, demo) for seq, demo in zip(sequences, demographics)
            ]

            # Extract features in parallel with progress bar
            with Pool(n_processes) as pool:
                # Use imap for better memory efficiency and progress tracking
                with tqdm(
                    total=len(sequences), desc="Processing sequences (parallel)"
                ) as pbar:
                    # Optimize chunksize based on number of sequences
                    chunksize = max(1, len(sequences) // (n_processes * 10))
                    for features in pool.imap(
                        extract_features_parallel,
                        data_with_extractor,
                        chunksize=chunksize,
                    ):
                        feature_dfs.append(features)
                        pbar.update(1)
        else:
            # Kaggle environment: Use sequential processing
            print(
                f"  Processing {len(sequences)} sequences SEQUENTIALLY (Kaggle environment)"
            )

            # Sequential processing with progress bar
            for seq_df, demo_df in tqdm(
                zip(sequences, demographics),
                total=len(sequences),
                desc="Processing sequences (sequential)",
            ):
                features = self.extract_features(seq_df, demo_df)
                feature_dfs.append(features)

        elapsed_time = time.time() - start_time
        print(f"  Extraction completed in {elapsed_time:.1f} seconds")
        print(
            f"  Average time per sequence: {elapsed_time / len(sequences):.3f} seconds"
        )

        print("  Concatenating feature DataFrames...")
        X = pd.concat(feature_dfs, ignore_index=True)
        print(f"  Feature matrix created: {X.shape}")

        # Store feature names
        self.feature_names = list(X.columns)

        # Mark as fitted
        self.is_fitted = True

        # Fit scaler
        print("  Scaling features...")
        if self.config["robust_scaler"]:
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=self.feature_names)

        print(f"✓ Extracted and scaled {len(self.feature_names)} features")

        return X

    def transform(
        self, sequences: List[pd.DataFrame], demographics: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """Extract features from test data using fitted transformers."""
        print("Transforming test sequences...")

        # Extract features sequentially with progress bar
        feature_dfs = []
        for seq_df, demo_df in tqdm(
            zip(sequences, demographics),
            total=len(sequences),
            desc="Processing test sequences",
        ):
            features = self.extract_features(seq_df, demo_df)
            feature_dfs.append(features)

        X = pd.concat(feature_dfs, ignore_index=True)

        # Ensure all expected features are present
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        # Apply scaler
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=self.feature_names)

        return X


# ====================================================================================================
# FEATURE EXPORT/IMPORT UTILITIES
# ====================================================================================================


class FeatureExporter:
    """Handle export and import of features for cross-environment usage."""

    @staticmethod
    def export_features(
        features_df: pd.DataFrame,
        extractor,
        labels: np.ndarray,
        subjects: np.ndarray,
        export_name: str = None,
        compress: bool = True,
    ) -> Path:
        """Export features to portable format."""
        if export_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_name = f"features_{FEATURE_VERSION}_{timestamp}"

        export_path = EXPORT_DIR / export_name
        export_path.mkdir(exist_ok=True, parents=True)

        print(f"  Saving to: {export_path}")

        # Save features as Parquet
        features_file = export_path / "features.parquet"
        features_df.to_parquet(
            features_file, compression="snappy" if compress else None, index=False
        )
        print(
            f"  ✓ Features saved ({len(features_df)} samples, {len(features_df.columns)} features)"
        )

        # Save metadata
        metadata = {
            "labels": labels.tolist(),
            "subjects": subjects.tolist(),
            "n_samples": len(labels),
            "n_features": len(features_df.columns),
            "feature_names": list(features_df.columns),
            "feature_version": FEATURE_VERSION,
            "export_date": datetime.now().isoformat(),
        }

        with open(export_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save extractor state
        extractor_state = {
            "scaler": extractor.scaler,
            "tof_pcas": extractor.tof_pcas,
            "feature_names": extractor.feature_names,
            "config": extractor.config,
            "is_fitted": extractor.is_fitted,
        }
        with open(export_path / "extractor.pkl", "wb") as f:
            pickle.dump(extractor_state, f)

        print(f"  ✓ Export complete: {export_path}")
        return export_path

    @staticmethod
    def import_features(
        import_path: str,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
        """Import features from exported files."""
        import_path = Path(import_path)

        print(f"\nImporting features from: {import_path}")

        # Load features
        features_df = pd.read_parquet(import_path / "features.parquet")

        # Load metadata
        with open(import_path / "metadata.json", "r") as f:
            metadata = json.load(f)

        labels = np.array(metadata["labels"])
        subjects = np.array(metadata["subjects"])

        # Load extractor state
        with open(import_path / "extractor.pkl", "rb") as f:
            extractor_state = pickle.load(f)

        print(
            f"  ✓ Imported {features_df.shape[0]} samples, {features_df.shape[1]} features"
        )
        return features_df, labels, subjects, extractor_state


# ====================================================================================================
# MODEL TRAINING
# ====================================================================================================


def train_models():
    """Train XGBoost models with cross-validation, with feature import/export."""
    # Access global variables
    global USE_EXPORTED_FEATURES, EXPORTED_FEATURES_PATH, EXPORT_FEATURES, EXPORT_NAME

    print("\n" + "=" * 70)
    print("TRAINING PHASE")
    print("=" * 70)

    # Display current settings
    if USE_EXPORTED_FEATURES:
        print("\n📥 Mode: IMPORT (using exported features)")
        print(f"   Path: {EXPORTED_FEATURES_PATH}")
    else:
        print("\n🔄 Mode: EXTRACT (computing features from raw data)")
        if EXPORT_FEATURES:
            print(f"   Export: Enabled (name: {EXPORT_NAME or 'auto-generated'})")
        else:
            print("   Export: Disabled")

    # Load data using CONFIG paths
    print("\nLoading data...")
    print(f"  Train data: {CONFIG['train_path']}")
    print(f"  Demographics: {CONFIG['train_demographics_path']}")

    try:
        train_df = pd.read_csv(CONFIG["train_path"])
        demo_df = pd.read_csv(CONFIG["train_demographics_path"])
    except FileNotFoundError as e:
        print(f"\n⚠️ Error: {e}")
        print("\nPlease check your data paths in CONFIG:")
        print("  - For Kaggle: Use /kaggle/input/... paths")
        print("  - For Local: Update paths to your data directory")
        raise

    # Filter for gesture sequences only
    train_df = train_df[train_df["behavior"] == "Performs gesture"].copy()

    print(
        f"Loaded {len(train_df)} samples from {train_df['sequence_id'].nunique()} sequences"
    )

    # Group sequences
    sequences = []
    demographics = []
    labels = []
    subjects = []

    for seq_id in train_df["sequence_id"].unique():
        seq_data = train_df[train_df["sequence_id"] == seq_id]
        subject_id = seq_data["subject"].iloc[0]

        sequences.append(seq_data)
        demographics.append(demo_df[demo_df["subject"] == subject_id])
        labels.append(GESTURE_MAPPER[seq_data["gesture"].iloc[0]])
        subjects.append(subject_id)

    labels = np.array(labels)
    subjects = np.array(subjects)

    # Check if we should import existing features
    # Handle path adjustment for local environment
    import_path = EXPORTED_FEATURES_PATH
    if USE_EXPORTED_FEATURES and EXPORTED_FEATURES_PATH:
        # If local and path doesn't exist, try adjusting the path
        if not IS_KAGGLE_ENV and not Path(import_path).exists():
            # Try to find the export in the local export directory
            export_name = Path(EXPORTED_FEATURES_PATH).name
            local_path = EXPORT_DIR / export_name
            if local_path.exists():
                import_path = str(local_path)
                print(f"📂 Adjusted import path for local environment: {import_path}")
            else:
                print(f"⚠️ Warning: Export not found at {EXPORTED_FEATURES_PATH}")
                print(f"   Also checked: {local_path}")
                USE_EXPORTED_FEATURES = False

    if USE_EXPORTED_FEATURES and import_path and Path(import_path).exists():
        print("\n📥 Loading exported features for fast training...")
        X, labels, subjects, extractor_state = FeatureExporter.import_features(
            import_path
        )

        # Reconstruct extractor from saved state
        extractor = FeatureExtractor(CONFIG)
        extractor.scaler = extractor_state["scaler"]
        extractor.tof_pcas = extractor_state["tof_pcas"]
        extractor.feature_names = extractor_state["feature_names"]
        extractor.is_fitted = extractor_state["is_fitted"]
        extractor.config = extractor_state["config"]

        y = labels
        print(f"  Features loaded in seconds! Shape: {X.shape}")
    else:
        # Extract features from scratch
        print("\n🔄 Extracting features from raw data...")
        print("  (This will take ~400s on first run)")

        # Initialize feature extractor
        extractor = FeatureExtractor(CONFIG)

        # Extract features
        X = extractor.fit_transform(sequences, demographics, labels)
        y = labels

        # Export features if requested
        if EXPORT_FEATURES:
            print("\n💾 Exporting features for future use...")
            export_path = FeatureExporter.export_features(
                X, extractor, labels, subjects, EXPORT_NAME
            )
            print(f"✓ Features exported to: {export_path}")
            print("\n📝 To use these features in the future, set:")
            print("   USE_EXPORTED_FEATURES = True")
            if IS_KAGGLE_ENV:
                print(f'   EXPORTED_FEATURES_PATH = "./{export_path.name}"')
            else:
                print(f'   EXPORTED_FEATURES_PATH = "{export_path}"')

    print(f"Feature matrix shape: {X.shape}")

    # Cross-validation
    print("\nStarting cross-validation...")
    cv = StratifiedGroupKFold(
        n_splits=CONFIG["n_folds"], shuffle=True, random_state=CONFIG["random_state"]
    )

    models = []
    oof_predictions = np.zeros(len(y))
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, subjects)):
        print(f"\n--- Fold {fold + 1}/{CONFIG['n_folds']} ---")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"Train: {len(X_train)}, Val: {len(X_val)}")

        # Configure XGBoost parameters based on environment
        xgb_params = CONFIG["xgb_params"].copy()

        # GPU acceleration settings
        if IS_KAGGLE_ENV:
            # Kaggle with T4 GPUs
            xgb_params["tree_method"] = "gpu_hist"
            xgb_params["device"] = "cuda"
            xgb_params["gpu_id"] = 0  # Use first GPU
            print("  Using GPU acceleration (CUDA/T4)")
        else:
            # Local Mac - MPS is not yet supported by XGBoost, use CPU
            xgb_params["tree_method"] = "hist"
            xgb_params["device"] = "cpu"
            print("  Using CPU (MPS not supported by XGBoost)")

        # Train XGBoost
        model = xgb.XGBClassifier(**xgb_params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

        models.append(model)

        # Predictions
        val_preds = model.predict(X_val)
        oof_predictions[val_idx] = val_preds

        # Calculate metrics
        binary_f1 = f1_score(
            np.where(y_val <= 7, 1, 0),
            np.where(val_preds <= 7, 1, 0),
            zero_division=0.0,
        )

        macro_f1 = f1_score(
            np.where(y_val <= 7, y_val, 99),
            np.where(val_preds <= 7, val_preds, 99),
            average="macro",
            zero_division=0.0,
        )

        score = 0.5 * (binary_f1 + macro_f1)
        cv_scores.append(score)

        print(
            f"Fold {fold + 1} - Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f}, Score: {score:.4f}"
        )

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    print(f"CV Score: {mean_score:.4f} ± {std_score:.4f}")
    print(f"Fold scores: {cv_scores}")

    # Feature importance
    feature_importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": np.mean([m.feature_importances_ for m in models], axis=0),
        }
    ).sort_values("importance", ascending=False)

    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20))

    # Save models with export if features were exported
    if EXPORT_NAME and EXPORT_FEATURES:
        export_path = EXPORT_DIR / EXPORT_NAME
        if export_path.exists():
            model_file = export_path / f"models_{CONFIG['n_folds']}fold.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(models, f)
            print(f"\n✓ Models saved to: {model_file}")

    return models, extractor, mean_score


# ====================================================================================================
# INFERENCE
# ====================================================================================================

# Global variables for models
MODELS = None
EXTRACTOR = None


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """Prediction function for Kaggle inference server."""
    global MODELS, EXTRACTOR

    # Initialize models if needed
    if MODELS is None or EXTRACTOR is None:
        print("Loading/training models...")
        MODELS, EXTRACTOR, _ = train_models()

    # Convert to pandas
    seq_df = sequence.to_pandas()
    demo_df = demographics.to_pandas()

    # Extract features
    features = EXTRACTOR.transform([seq_df], [demo_df])

    # Get predictions from all models
    predictions = []
    for model in MODELS:
        pred = model.predict_proba(features)[0]
        predictions.append(pred)

    # Average predictions
    avg_pred = np.mean(predictions, axis=0)
    final_class = np.argmax(avg_pred)

    # Convert to gesture name
    gesture_name = REVERSE_GESTURE_MAPPER[final_class]

    return gesture_name


# ====================================================================================================
# MAIN EXECUTION
# ====================================================================================================

if __name__ == "__main__":
    # Check and display available exports
    print("\n" + "=" * 70)
    print("AVAILABLE FEATURE EXPORTS")
    print("=" * 70)

    if EXPORT_DIR.exists():
        exports = sorted(EXPORT_DIR.glob("features_*"))
        if exports:
            print("\nFound exported features:")
            for exp in exports[-3:]:  # Show last 3 exports
                if exp.is_dir():
                    metadata_file = exp / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, "r") as f:
                            meta = json.load(f)
                        print(f"  📁 {exp.name}")
                        print(
                            f"     Samples: {meta.get('n_samples', '?')}, Features: {meta.get('n_features', '?')}"
                        )
            print("\n💡 To use exported features, set:")
            print("   USE_EXPORTED_FEATURES = True")
            if IS_KAGGLE_ENV:
                print(f'   EXPORTED_FEATURES_PATH = "./{exports[-1].name}"')
            else:
                print(f'   EXPORTED_FEATURES_PATH = "{exports[-1]}"')
        else:
            print("No exported features found. First run will extract and export.")

    # Train models
    MODELS, EXTRACTOR, score = train_models()
    print(f"\n✓ Models trained with CV score: {score:.4f}")

    # Show performance summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    if USE_EXPORTED_FEATURES:
        print("✅ Used exported features - execution time: ~30 seconds")
    else:
        print("✅ Extracted features from raw data - execution time: ~400 seconds")
        if EXPORT_FEATURES:
            print("   Features have been exported for future use.")
            print("   Next run will be 10x faster with exported features!")

    # Environment-specific completion message
    print("\n" + "=" * 70)
    if IS_KAGGLE_ENV:
        print("KAGGLE SUBMISSION READY")
    else:
        print("LOCAL EXECUTION COMPLETE")
        print(
            "To use in Kaggle: Copy exported features to Kaggle and set IS_KAGGLE_ENV = True"
        )
    print("=" * 70)

    # Initialize Kaggle inference server
    print("\nInitializing Kaggle inference server...")

    try:
        from kaggle_evaluation.cmi_inference_server import CMIInferenceServer

        inference_server = CMIInferenceServer(predict)
        print("✓ Inference server created")

        print("\nStarting inference...")
        inference_server.serve()

        print("\n✓ Submission complete!")

    except ImportError as e:
        print(f"⚠️ Kaggle evaluation module not available: {e}")
        print("Generating submission manually...")

        # Manual submission generation
        test_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv"
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            test_demo_df = pd.read_csv(
                "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv"
            )

            predictions = []
            for seq_id in test_df["sequence_id"].unique():
                seq_data = test_df[test_df["sequence_id"] == seq_id]
                seq_pl = pl.from_pandas(seq_data)

                subject_id = (
                    seq_data["subject"].iloc[0] if "subject" in seq_data.columns else 0
                )
                demo_pl = pl.from_pandas(
                    test_demo_df[test_demo_df["subject"] == subject_id]
                )

                pred = predict(seq_pl, demo_pl)
                predictions.append({"sequence_id": seq_id, "prediction": pred})

            submission_df = pd.DataFrame(predictions)
            submission_df.to_parquet("/kaggle/working/submission.parquet", index=False)
            print(
                f"✓ Generated submission.parquet with {len(submission_df)} predictions"
            )

        print("\nTo use for Kaggle submission:")
        print("1. Copy this entire script to a Kaggle notebook")
        print("2. Run all cells")
        print("3. submission.parquet will be generated automatically")
