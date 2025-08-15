# CMI BFRB Detection - Training

# Import required libraries
import json
import os
import re
import sys
import threading
import time
import warnings

# Suppress TensorFlow verbose warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    import absl.logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy import signal
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

# PyTorch imports (conditional)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available, DL features disabled")

# === Keras/TensorFlow imports (conditional) ===
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    KERAS_AVAILABLE = True
    print("✓ TensorFlow/Keras available")
except Exception as e:
    KERAS_AVAILABLE = False
    print(f"⚠️ TensorFlow/Keras not available: {e} — Keras pipeline disabled")

# Try to import CMI inference server with fallback
try:
    import kaggle_evaluation.cmi_inference_server as cmi
except ModuleNotFoundError:
    sys.path.append("/kaggle/input/cmi-detect-behavior-with-sensor-data")
    import kaggle_evaluation.cmi_inference_server as cmi

warnings.filterwarnings("ignore")
print("✓ All imports loaded successfully")


# Configuration
class Config:
    # Data paths for Kaggle environment
    TRAIN_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
    TRAIN_DEMOGRAPHICS_PATH = (
        "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"
    )

    # Training parameters
    SEED = 42
    N_FOLDS = 5

    # Feature columns
    ACC_COLS = ["acc_x", "acc_y", "acc_z"]
    ROT_COLS = ["rot_w", "rot_x", "rot_y", "rot_z"]

    # LightGBM parameters
    LGBM_PARAMS = {
        "objective": "multiclass",
        "n_estimators": 1024,
        "max_depth": 8,
        "learning_rate": 0.025,
        "colsample_bytree": 0.5,
        "n_jobs": -1,
        "num_leaves": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "subsample": 0.5,
        "verbosity": -1,
        "random_state": 42,
    }

    # Output path
    OUTPUT_PATH = "/kaggle/working/"

    # === Modality Dropout for Training ===
    MODALITY_DROPOUT_PROB = (
        0.3  # Probability of dropping ToF/THM during training (reduced from 0.5)
    )
    USE_MODALITY_DROPOUT = True  # Enable/disable modality dropout

    # === Model selection ===
    # If None -> train and save; If str(path) -> skip training and use this model bundle
    MODEL_PATH = os.getenv(
        "MODEL_PATH", None
    )  # e.g. "/kaggle/input/my-model/imu_lgbm_model.pkl"
    MODEL_FILENAME = "imu_lgbm_model.pkl"  # filename when we save after training
    USE_TORCH = True
    USE_KERAS = True


# === Torch training/inference config ===
class DLConfig:
    TORCH_OUT_DIR = os.path.join(Config.OUTPUT_PATH, "torch_models")
    N_FOLDS = Config.N_FOLDS
    SEED = Config.SEED

    # frame-level features
    PAD_LEN_PERCENTILE = 95  # P95 を既定
    FIXED_PAD_LEN = 90  # 固定長に設定
    FRAME_FEATURE_DIR = os.path.join(Config.OUTPUT_PATH, "frame_features")  # 1seq 1file

    # training
    MAX_EPOCHS = 30
    BATCH_SIZE = 512  # OOM時は段階的にフォールバック
    ACCUM_STEPS = 1  # 勾配蓄積で実効バッチ拡張
    LR = 1e-3
    WEIGHT_DECAY = 1e-2
    DROPOUT = 0.2
    LABEL_SMOOTHING = 0.05

    # runtime
    AMP = True
    NUM_WORKERS = int(os.getenv("DL_NUM_WORKERS", "2"))  # 0→2 に変更

    # file names
    BUNDLE_NAME = "torch_bundle.pkl"  # メタ（列順、スケーラ、pad_len等）
    WEIGHT_TMPL = "fold{:02d}.pt"  # 各foldの重み


# === Keras training/inference config ===
class KerasConfig:
    OUT_DIR = os.path.join(Config.OUTPUT_PATH, "keras_models")
    N_FOLDS = Config.N_FOLDS
    SEED = Config.SEED

    # 前処理（Torch と共通関数を使用）
    PAD_LEN_PERCENTILE = 95
    FIXED_PAD_LEN = 90  # 固定長に設定

    # 学習設定
    MAX_EPOCHS = int(os.getenv("KERAS_MAX_EPOCHS", "40"))
    BATCH_SIZE = int(os.getenv("KERAS_BATCH_SIZE", "512"))
    LR = float(os.getenv("KERAS_LR", "1e-3"))
    DROPOUT = float(os.getenv("KERAS_DROPOUT", "0.2"))
    LABEL_SMOOTHING = float(os.getenv("KERAS_LABEL_SMOOTHING", "0.1"))
    EARLY_STOPPING_PATIENCE = int(os.getenv("KERAS_ES_PATIENCE", "3"))  # 早めに
    REDUCE_LR_PATIENCE = int(os.getenv("KERAS_RLR_PATIENCE", "2"))

    # 保存ファイル名
    BUNDLE_NAME = "keras_bundle.pkl"
    WEIGHT_TMPL = "fold{:02d}.keras"  # KerasModel 保存
    STATE_JSON = os.path.join(OUT_DIR, "keras_state.json")


# === Ensemble configuration ===
class EnsembleConfig:
    # weights for final soft-voting
    W_LGBM = float(os.getenv("ENSEMBLE_W_LGBM", "0.20"))
    W_TORCH = float(os.getenv("ENSEMBLE_W_TORCH", "0.45"))
    W_KERAS = float(os.getenv("ENSEMBLE_W_KERAS", "0.35"))
    # torch bundle path (for inference-only)
    TORCH_BUNDLE_PATH = os.getenv(
        "TORCH_BUNDLE_PATH", os.path.join(DLConfig.TORCH_OUT_DIR, DLConfig.BUNDLE_NAME)
    )
    LOAD_TORCH_FOLDS_IN_MEMORY = bool(int(os.getenv("LOAD_TORCH_FOLDS_IN_MEMORY", "1")))
    FAIL_IF_TORCH_MISSING = bool(int(os.getenv("FAIL_IF_TORCH_MISSING", "0")))

    # Keras バンドル
    KERAS_BUNDLE_PATH = os.getenv(
        "KERAS_BUNDLE_PATH", os.path.join(KerasConfig.OUT_DIR, KerasConfig.BUNDLE_NAME)
    )
    LOAD_KERAS_FOLDS_IN_MEMORY = bool(int(os.getenv("LOAD_KERAS_FOLDS_IN_MEMORY", "1")))


# === Single source of truth for all paths ====================================
from dataclasses import dataclass


def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


@dataclass
class Paths:
    # ---- bases ----
    OUTPUT: str = Config.OUTPUT_PATH
    CKPT_BASE: str = os.getenv(
        "CKPT_BASE_DIR", os.path.join(Config.OUTPUT_PATH, "checkpoints")
    )
    LOG_BASE: str = os.path.join(Config.OUTPUT_PATH, "logs")

    # ---- bundles (pretrained for inference) ----
    # LGBM は MODEL_PATH と LGBM_BUNDLE_PATH の両方を受け付け、前者を後方互換として採用
    LGBM_BUNDLE: str | None = os.getenv(
        "LGBM_BUNDLE_PATH",
        os.getenv(
            "MODEL_PATH",
            "/kaggle/input/cmi-bfrb-v9-lightgbm/other/default/1/imu_lgbm_model-4.pkl",
        ),
    )
    TORCH_BUNDLE: str = os.getenv(
        "TORCH_BUNDLE_PATH", os.path.join(DLConfig.TORCH_OUT_DIR, DLConfig.BUNDLE_NAME)
    )
    KERAS_BUNDLE: str = os.getenv(
        "KERAS_BUNDLE_PATH", os.path.join(KerasConfig.OUT_DIR, KerasConfig.BUNDLE_NAME)
    )

    # ---- features cache (legacy nameも拾う) ----
    FEATURES_CACHE_MAIN: str = os.path.join(
        os.getenv("CKPT_BASE_DIR", os.path.join(Config.OUTPUT_PATH, "checkpoints")),
        "train_features.joblib",
    )
    FEATURES_CACHE_LEGACY: str = os.path.join(
        os.getenv("CKPT_BASE_DIR", os.path.join(Config.OUTPUT_PATH, "checkpoints")),
        "train_feature.joblib",
    )  # 旧名

    # ---- templates / states ----
    LGBM_FOLD_TMPL: str = os.path.join(
        os.getenv("CKPT_BASE_DIR", os.path.join(Config.OUTPUT_PATH, "checkpoints")),
        "lgbm_fold{:02d}.pkl",
    )
    LGBM_STATE_JSON: str = os.path.join(
        os.getenv("CKPT_BASE_DIR", os.path.join(Config.OUTPUT_PATH, "checkpoints")),
        "lgbm_state.json",
    )

    TORCH_EPOCH_TMPL: str = os.path.join(DLConfig.TORCH_OUT_DIR, "fold{:02d}_last.pth")
    TORCH_STATE_JSON: str = os.path.join(DLConfig.TORCH_OUT_DIR, "torch_state.json")

    KERAS_STATE_JSON: str = os.path.join(KerasConfig.OUT_DIR, "keras_state.json")

    # ---- resolved (後で埋める) ----
    FEATURES_CACHE: str | None = None

    def resolve(self):
        # ディレクトリの確保
        for d in [
            self.CKPT_BASE,
            DLConfig.TORCH_OUT_DIR,
            KerasConfig.OUT_DIR,
            self.LOG_BASE,
        ]:
            os.makedirs(d, exist_ok=True)

        # features cache はメイン or レガシーのうち "最初に存在するもの" を優先
        self.FEATURES_CACHE = (
            _first_existing(self.FEATURES_CACHE_MAIN, self.FEATURES_CACHE_LEGACY)
            or self.FEATURES_CACHE_MAIN
        )

        # 既存の設定へ**一括反映**（以降は従来参照でも同じものが見える）
        Config.MODEL_PATH = self.LGBM_BUNDLE
        EnsembleConfig.TORCH_BUNDLE_PATH = self.TORCH_BUNDLE
        EnsembleConfig.KERAS_BUNDLE_PATH = self.KERAS_BUNDLE

        # CheckpointConfig へも集約結果を反映
        CheckpointConfig.CKPT_DIR = self.CKPT_BASE
        CheckpointConfig.LGBM_STATE_JSON = self.LGBM_STATE_JSON
        CheckpointConfig.LGBM_FOLD_TMPL = (
            os.path.basename(self.LGBM_FOLD_TMPL)
            if os.path.isabs(self.LGBM_FOLD_TMPL)
            else self.LGBM_FOLD_TMPL
        )
        CheckpointConfig.TORCH_EPOCH_CKPT_TMPL = self.TORCH_EPOCH_TMPL
        CheckpointConfig.TORCH_STATE_JSON = self.TORCH_STATE_JSON
        CheckpointConfig.KERAS_STATE_JSON = self.KERAS_STATE_JSON
        CheckpointConfig.FEATURES_CACHE = self.FEATURES_CACHE

    def print_summary(self):
        def _flag(p):  # 存在表示
            return f"{p}  [{'✓' if p and os.path.exists(p) else '×'}]"

        print("========== PATHS ==========")
        print(" LGBM  :", _flag(self.LGBM_BUNDLE or "(train)"))
        print(" Torch :", _flag(self.TORCH_BUNDLE))
        print(" Keras :", _flag(self.KERAS_BUNDLE))
        print(" CKPT  :", self.CKPT_BASE)
        print(" Cache :", _flag(self.FEATURES_CACHE))
        print(" Logs  :", self.LOG_BASE)
        print("===========================")


# ============================================================================


# === Checkpoint/Resume configuration & helpers ===
class CheckpointConfig:
    CKPT_DIR = os.path.join(Config.OUTPUT_PATH, os.getenv("CKPT_DIR", "checkpoints"))
    RESUME = bool(int(os.getenv("RESUME", "1")))  # 1: 再開する
    FEATURES_CACHE = os.path.join(CKPT_DIR, "train_features.joblib")
    LGBM_FOLD_TMPL = "lgbm_fold{:02d}.pkl"
    LGBM_STATE_JSON = os.path.join(CKPT_DIR, "lgbm_state.json")
    TORCH_EPOCH_CKPT_TMPL = os.path.join(DLConfig.TORCH_OUT_DIR, "fold{:02d}_last.pth")
    TORCH_STATE_JSON = os.path.join(DLConfig.TORCH_OUT_DIR, "torch_state.json")
    SKIP_TORCH_FOLD_IF_BEST_EXISTS = True  # 既に最良重みがあれば fold をスキップ

    # Keras用の設定
    KERAS_STATE_JSON = os.path.join(KerasConfig.OUT_DIR, "keras_state.json")
    KERAS_EARLY_EXIT_IF_BEST_EXISTS = True


# === Logging & Cache config (NEW) ===
class LogConfig:
    LOG_EVERY_STEPS = int(
        os.getenv("LOG_EVERY_STEPS", "50")
    )  # Torch batch log interval
    PRINT_EVERY_STEPS = int(os.getenv("PRINT_EVERY_STEPS", "100"))  # NEW: 標準出力
    FEATURE_LOG_EVERY = int(os.getenv("FEATURE_LOG_EVERY", "200"))  # NEW: 特徴量抽出
    SAVE_JSONL = bool(int(os.getenv("SAVE_JSONL", "1")))  # 進捗を JSON Lines でも保存
    OUT_DIR = os.path.join(Config.OUTPUT_PATH, "logs")
    os.makedirs(OUT_DIR, exist_ok=True)


class FrameCacheConfig:
    ENABLE = bool(int(os.getenv("FRAME_CACHE", "1")))
    MAX_ITEMS = int(os.getenv("FRAME_CACHE_MAX_ITEMS", "1600"))  # 4x 拡張
    MAX_BYTES = int(float(os.getenv("FRAME_CACHE_MAX_MB", "2048")) * 1024 * 1024)  # 2GB
    STATS_KEY_MODE = os.getenv("FRAME_CACHE_STATS_KEY", "id")  # "id" or "hash"


# Initialize PATHS after CheckpointConfig is defined
PATHS = Paths()
PATHS.resolve()
PATHS.print_summary()


class KerasSpeedConfig:
    MIXED_PRECISION = bool(int(os.getenv("KERAS_MP", "1")))  # 既定ON


def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def _load_json(path, default=None):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {} if default is None else default


os.makedirs(CheckpointConfig.CKPT_DIR, exist_ok=True)

# === Frame Feature LRU Cache (NEW) ===
from collections import OrderedDict

try:
    from joblib import hash as joblib_hash
except Exception:
    joblib_hash = None


class _FrameFeatureCache:
    def __init__(self, max_items=400, max_bytes=800 * 1024 * 1024):
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.size_bytes = 0
        self._cache = OrderedDict()  # key -> (df, bytes)
        self._lock = threading.Lock()

    def _key(self, seq_pl):
        try:
            sid = seq_pl["sequence_id"][0]
            return f"seq:{sid}"
        except Exception:
            # fallback to object id
            return f"obj:{id(seq_pl)}"

    def get(self, seq_pl):
        if not FrameCacheConfig.ENABLE:
            # Bypass cache
            df = build_frame_features(seq_pl)
            df.replace([np.inf, -np.inf], 0, inplace=True)
            df.fillna(0, inplace=True)
            return df

        k = self._key(seq_pl)
        with self._lock:
            if k in self._cache:
                df, b = self._cache.pop(k)
                self._cache[k] = (df, b)  # move to tail (MRU)
                return df

        # miss → まずディスクを探す
        try:
            sid = int(seq_pl["sequence_id"][0])
            path = os.path.join(DLConfig.FRAME_FEATURE_DIR, f"{sid}.parquet")
            if os.path.exists(path):
                df = pd.read_parquet(path)
                df.replace([np.inf, -np.inf], 0, inplace=True)
                df.fillna(0, inplace=True)
                b = int(df.memory_usage(index=True, deep=True).sum())
                # LRU に put
                with self._lock:
                    # evict if needed
                    while (len(self._cache) >= self.max_items) or (
                        self.size_bytes + b > self.max_bytes and len(self._cache) > 0
                    ):
                        _, (ev_df, ev_b) = self._cache.popitem(last=False)
                        self.size_bytes -= ev_b
                    self._cache[k] = (df, b)
                    self.size_bytes += b
                return df
        except Exception:
            pass

        # それでも無ければ計算
        df = build_frame_features(seq_pl)
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)
        b = int(df.memory_usage(index=True, deep=True).sum())

        with self._lock:
            # evict if needed
            while (len(self._cache) >= self.max_items) or (
                self.size_bytes + b > self.max_bytes and len(self._cache) > 0
            ):
                _, (ev_df, ev_b) = self._cache.popitem(last=False)
                self.size_bytes -= ev_b
            self._cache[k] = (df, b)
            self.size_bytes += b
        return df


FRAME_CACHE = _FrameFeatureCache(
    max_items=FrameCacheConfig.MAX_ITEMS, max_bytes=FrameCacheConfig.MAX_BYTES
)

# Check ensemble weights are valid
assert (EnsembleConfig.W_LGBM + EnsembleConfig.W_TORCH + EnsembleConfig.W_KERAS) > 0, (
    "Invalid ensemble weights (sum must be > 0)"
)

np.random.seed(Config.SEED)

# PyTorch seed setting for reproducibility
if TORCH_AVAILABLE:
    torch.manual_seed(Config.SEED)
    torch.cuda.manual_seed_all(Config.SEED)
    torch.backends.cudnn.benchmark = (
        True  # Speed priority (set to False for deterministic)
    )
    print(f"✓ PyTorch seed set to {Config.SEED}")

print("✓ Configuration loaded")
print(
    f"✓ Ensemble weights: LGBM={EnsembleConfig.W_LGBM:.2f}, Torch={EnsembleConfig.W_TORCH:.2f}"
)

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
print(f"✓ Gesture mapping loaded ({len(GESTURE_MAPPER)} classes)")


# Feature engineering functions
ROT_IDXS = {"rot_w": 0, "rot_x": 1, "rot_y": 2, "rot_z": 3}


def build_full_quaternion(seq_df: pd.DataFrame, available_rot_cols: list) -> np.ndarray:
    """Build a full Nx4 quaternion array from available rotation columns."""
    n = len(seq_df)
    # Initialize with NaN, then set w to 1.0 as default
    q = np.full((n, 4), np.nan, dtype=float)
    q[:, 0] = 1.0  # rot_w defaults to 1 (identity quaternion)

    for c in available_rot_cols:
        if c in ROT_IDXS:
            # Keep NaN values as NaN for proper handling downstream
            vals = seq_df[c].to_numpy(dtype=float)
            q[:, ROT_IDXS[c]] = vals
    return q


def infer_dt_and_fs(seq_df: pd.DataFrame, default_fs: float = 20.0) -> tuple:
    """Infer sampling interval (dt) and frequency (fs) from timestamp data if available."""
    # Try to find timestamp column
    time_cols = ["timestamp", "time", "elapsed_time", "seconds_elapsed"]
    time_col = next((c for c in time_cols if c in seq_df.columns), None)

    if time_col is None:
        # No timestamp column found, use default
        return 1.0 / default_fs, default_fs

    # Calculate dt from timestamps
    t = np.asarray(seq_df[time_col], dtype=float)
    if t.size < 2:
        return 1.0 / default_fs, default_fs

    dt_raw = np.median(np.diff(t))
    if not np.isfinite(dt_raw) or dt_raw <= 0:
        return 1.0 / default_fs, default_fs

    # Try different unit conversions to find one that gives reasonable fs (5-200 Hz)
    candidates = [
        dt_raw,  # Already in seconds
        dt_raw / 1e3,  # Milliseconds to seconds
        dt_raw / 1e6,  # Microseconds to seconds
        dt_raw / 1e9,  # Nanoseconds to seconds
    ]

    for dt in candidates:
        if dt > 0:
            fs = 1.0 / dt
            if 5.0 <= fs <= 200.0:
                return dt, fs

    # If no reasonable fs found, use default
    return 1.0 / default_fs, default_fs


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
                if i > 0 and not np.isnan(rot_cleaned[i - 1, missing_idx]):
                    if rot_cleaned[i - 1, missing_idx] < 0:
                        missing_value = -missing_value
                rot_cleaned[i, missing_idx] = missing_value
                rot_cleaned[i, ~np.isnan(row)] = valid_values
            else:
                rot_cleaned[i] = [1.0, 0.0, 0.0, 0.0]
        else:
            rot_cleaned[i] = [1.0, 0.0, 0.0, 0.0]

    return rot_cleaned


def fix_quaternion_sign(rot: np.ndarray) -> np.ndarray:
    """Fix quaternion sign continuity for smooth rotation representation."""
    rot_fixed = rot.copy()

    for i in range(1, len(rot_fixed)):
        # Calculate dot product between consecutive quaternions
        dot_product = np.dot(rot_fixed[i - 1], rot_fixed[i])

        # If dot product is negative, flip the quaternion sign
        if dot_product < 0:
            rot_fixed[i] = -rot_fixed[i]

    # Normalize quaternions
    norms = np.linalg.norm(rot_fixed, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0  # Avoid division by zero
    rot_fixed = rot_fixed / norms

    return rot_fixed


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


def compute_linear_acceleration(
    world_acc: np.ndarray, fs: float = 20.0, method: str = "lpf"
) -> np.ndarray:
    """Remove gravity from world acceleration to get linear acceleration."""
    if method == "lpf":
        # Low-pass filter to estimate gravity component
        # Use 0.75 Hz cutoff frequency
        wc = 0.75 / (fs / 2.0)
        if wc >= 1.0 or world_acc.shape[0] < 9:  # Cutoff too high or sequence too short
            # Fall back to median
            gravity = np.median(world_acc, axis=0, keepdims=True)
        else:
            b, a = signal.butter(2, wc, btype="low")
            try:
                gravity = signal.filtfilt(b, a, world_acc, axis=0)
            except Exception:
                # If filtfilt fails (e.g., short sequence), use median
                gravity = np.median(world_acc, axis=0, keepdims=True)
        linear_acc = world_acc - gravity
    elif method == "median":
        # Use median as gravity estimate
        gravity = np.median(world_acc, axis=0, keepdims=True)
        linear_acc = world_acc - gravity
    elif method == "subtract":
        # Simple gravity subtraction (assuming gravity = 9.81 m/s^2 in -z direction)
        gravity = np.array([0.0, 0.0, 9.81])
        linear_acc = world_acc - gravity
    else:
        # Default: no filtering
        linear_acc = world_acc.copy()

    return linear_acc


def compute_angular_velocity(rot: np.ndarray, dt: float = 1 / 20) -> np.ndarray:
    """Compute angular velocity from quaternion sequence."""
    # Check input shape
    if rot.ndim != 2 or rot.shape[0] < 2 or rot.shape[1] != 4:
        return np.zeros((max(1, rot.shape[0] if rot.ndim == 2 else 1), 3))

    angular_velocity = []

    # Convert quaternions to scipy format (x, y, z, w)
    rot_scipy = rot[:, [1, 2, 3, 0]]

    for i in range(len(rot_scipy) - 1):
        try:
            # Get rotation objects
            r1 = R.from_quat(rot_scipy[i])
            r2 = R.from_quat(rot_scipy[i + 1])

            # Calculate relative rotation
            r_rel = r2 * r1.inv()

            # Convert to rotation vector (axis-angle representation)
            rotvec = r_rel.as_rotvec()

            # Angular velocity = rotation vector / dt
            omega = rotvec / dt
            angular_velocity.append(omega)
        except Exception:
            angular_velocity.append([0.0, 0.0, 0.0])

    # Pad the last sample
    if angular_velocity:
        angular_velocity.append(angular_velocity[-1])
    else:
        angular_velocity.append([0.0, 0.0, 0.0])

    return np.array(angular_velocity)


def extract_jerk_features(
    acc: np.ndarray, dt: float = 1 / 20, prefix: str = "jerk"
) -> dict:
    """Extract jerk features (rate of change of acceleration)."""
    features = {}

    if len(acc) < 2:
        features[f"{prefix}_mean"] = 0
        features[f"{prefix}_std"] = 0
        features[f"{prefix}_max"] = 0
        features[f"{prefix}_p90"] = 0
        features[f"{prefix}_l2"] = 0
        return features

    # Calculate jerk
    jerk = np.diff(acc, axis=0) / dt
    jerk_magnitude = np.linalg.norm(jerk, axis=1)

    # Extract features
    features[f"{prefix}_mean"] = np.mean(jerk_magnitude)
    features[f"{prefix}_std"] = np.std(jerk_magnitude)
    features[f"{prefix}_max"] = np.max(jerk_magnitude)
    features[f"{prefix}_p90"] = np.percentile(jerk_magnitude, 90)
    features[f"{prefix}_l2"] = np.linalg.norm(jerk_magnitude)

    return features


def extract_correlation_features(data_dict: dict, prefix: str = "corr") -> dict:
    """Extract correlation features between different axes/modalities."""
    features = {}

    # World acceleration XY correlation
    if "world_acc_x" in data_dict and "world_acc_y" in data_dict:
        if len(data_dict["world_acc_x"]) > 1:
            features[f"{prefix}_world_acc_xy"] = np.corrcoef(
                data_dict["world_acc_x"], data_dict["world_acc_y"]
            )[0, 1]
        else:
            features[f"{prefix}_world_acc_xy"] = 0

    # World acceleration XZ correlation
    if "world_acc_x" in data_dict and "world_acc_z" in data_dict:
        if len(data_dict["world_acc_x"]) > 1:
            features[f"{prefix}_world_acc_xz"] = np.corrcoef(
                data_dict["world_acc_x"], data_dict["world_acc_z"]
            )[0, 1]
        else:
            features[f"{prefix}_world_acc_xz"] = 0

    # World acceleration YZ correlation
    if "world_acc_y" in data_dict and "world_acc_z" in data_dict:
        if len(data_dict["world_acc_y"]) > 1:
            features[f"{prefix}_world_acc_yz"] = np.corrcoef(
                data_dict["world_acc_y"], data_dict["world_acc_z"]
            )[0, 1]
        else:
            features[f"{prefix}_world_acc_yz"] = 0

    # Linear acceleration magnitude vs angular velocity magnitude correlation
    if "linear_acc_mag" in data_dict and "angular_vel_mag" in data_dict:
        if len(data_dict["linear_acc_mag"]) > 1:
            features[f"{prefix}_linear_angular"] = np.corrcoef(
                data_dict["linear_acc_mag"], data_dict["angular_vel_mag"]
            )[0, 1]
        else:
            features[f"{prefix}_linear_angular"] = 0

    # Replace NaN values with 0
    for key in features:
        if np.isnan(features[key]):
            features[key] = 0

    return features


def extract_peak_features(
    data: np.ndarray, prefix: str = "peak", fs: float = 20.0
) -> dict:
    """Extract peak-related features from time series."""
    features = {}

    if len(data) < 3:
        features[f"{prefix}_count"] = 0
        features[f"{prefix}_mean_height"] = 0
        features[f"{prefix}_mean_distance"] = 0
        features[f"{prefix}_mean_distance_sec"] = 0
        return features

    # Find peaks using scipy
    try:
        # Find peaks with prominence threshold (0.5 * std) and minimum distance
        threshold = 0.5 * np.std(data) if np.std(data) > 0 else 0.1
        min_distance = max(int(round(0.15 * fs)), 1)  # 150ms minimum between peaks
        peaks, _ = signal.find_peaks(data, prominence=threshold, distance=min_distance)

        features[f"{prefix}_count"] = len(peaks)

        if len(peaks) > 0:
            features[f"{prefix}_mean_height"] = np.mean(data[peaks])
            if len(peaks) > 1:
                mean_distance = np.mean(np.diff(peaks))
                features[f"{prefix}_mean_distance"] = mean_distance
                features[f"{prefix}_mean_distance_sec"] = mean_distance / max(fs, 1e-9)
            else:
                features[f"{prefix}_mean_distance"] = 0
                features[f"{prefix}_mean_distance_sec"] = 0
        else:
            features[f"{prefix}_mean_height"] = 0
            features[f"{prefix}_mean_distance"] = 0
            features[f"{prefix}_mean_distance_sec"] = 0
    except Exception:
        features[f"{prefix}_count"] = 0
        features[f"{prefix}_mean_height"] = 0
        features[f"{prefix}_mean_distance"] = 0
        features[f"{prefix}_mean_distance_sec"] = 0

    return features


def extract_autocorrelation_features(
    data: np.ndarray, lags: list = [1, 2, 4, 8], prefix: str = "autocorr"
) -> dict:
    """Extract autocorrelation features at specified lags."""
    features = {}

    if len(data) < 2:
        for lag in lags:
            features[f"{prefix}_lag{lag}"] = 0
        return features

    for lag in lags:
        if lag < len(data):
            # Calculate autocorrelation at specified lag
            if np.std(data) > 1e-8:
                features[f"{prefix}_lag{lag}"] = np.corrcoef(data[:-lag], data[lag:])[
                    0, 1
                ]
            else:
                features[f"{prefix}_lag{lag}"] = 0

            # Handle NaN values
            if np.isnan(features[f"{prefix}_lag{lag}"]):
                features[f"{prefix}_lag{lag}"] = 0
        else:
            features[f"{prefix}_lag{lag}"] = 0

    return features


def extract_gradient_histogram(
    data: np.ndarray, n_bins: int = 10, prefix: str = "grad_hist"
) -> dict:
    """Extract gradient histogram features."""
    features = {}

    if len(data) < 2:
        for i in range(n_bins):
            features[f"{prefix}_bin{i}"] = 0
        return features

    # Calculate gradients (first difference)
    gradients = np.diff(data)

    # Handle empty gradients
    if len(gradients) == 0:
        for i in range(n_bins):
            features[f"{prefix}_bin{i}"] = 0
        return features

    # Create histogram
    try:
        # Use percentile-based bins for better distribution
        bin_edges = np.percentile(gradients, np.linspace(0, 100, n_bins + 1))
        # Make sure bins are unique (in case of constant values)
        if len(np.unique(bin_edges)) < 2:
            # Fall back to uniform bins
            min_val = np.min(gradients)
            max_val = np.max(gradients)
            if max_val - min_val < 1e-8:
                # All gradients are the same
                for i in range(n_bins):
                    features[f"{prefix}_bin{i}"] = 1.0 if i == n_bins // 2 else 0.0
                return features
            bin_edges = np.linspace(min_val, max_val, n_bins + 1)

        hist, _ = np.histogram(gradients, bins=bin_edges)
        # Normalize histogram to sum to 1
        hist = hist / (np.sum(hist) + 1e-8)

        for i in range(n_bins):
            features[f"{prefix}_bin{i}"] = hist[i]

    except Exception:
        # Fallback to zero features
        for i in range(n_bins):
            features[f"{prefix}_bin{i}"] = 0

    return features


def extract_frequency_features(
    data: np.ndarray, fs: float = 20.0, prefix: str = "freq", compute_zcr: bool = True
) -> dict:
    """Extract frequency domain features using Welch's method."""
    features = {}

    if len(data) < 32:  # Too short for frequency analysis
        # Return zero features with complete key set
        for feat in [
            "band_0.3_3",
            "band_3_8",
            "band_8_12",
            "band_0.3_3_rel",
            "band_3_8_rel",
            "band_8_12_rel",
            "band_0.3_3_log",
            "band_3_8_log",
            "band_8_12_log",
            "total_power",
            "spectral_centroid",
            "spectral_rolloff",
            "spectral_entropy",
            "dominant_freq",
            "zcr",
        ]:
            features[f"{prefix}_{feat}"] = 0
        return features

    try:
        # Calculate power spectral density using Welch's method
        # Reduced nperseg from 128 to 64 for faster computation without significant accuracy loss
        nperseg = max(32, min(64, len(data) // 4, len(data)))
        noverlap = nperseg // 2
        f, psd = signal.welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap)

        # Define frequency bands
        bands = {
            "band_0.3_3": (0.3, 3.0),
            "band_3_8": (3.0, 8.0),
            "band_8_12": (8.0, 12.0),
        }

        # Total power
        total_power = np.sum(psd)
        features[f"{prefix}_total_power"] = total_power

        # Band powers
        for band_name, (low, high) in bands.items():
            # Clamp high frequency to Nyquist frequency
            high_eff = min(high, fs / 2.0)
            band_mask = (f >= low) & (f <= high_eff)
            band_power = np.sum(psd[band_mask])
            features[f"{prefix}_{band_name}"] = band_power
            # Relative power
            if total_power > 0:
                features[f"{prefix}_{band_name}_rel"] = band_power / total_power
            else:
                features[f"{prefix}_{band_name}_rel"] = 0
            # Log power (add small epsilon to avoid log(0))
            features[f"{prefix}_{band_name}_log"] = np.log(band_power + 1e-10)

        # Spectral centroid
        if total_power > 0:
            features[f"{prefix}_spectral_centroid"] = np.sum(f * psd) / total_power
        else:
            features[f"{prefix}_spectral_centroid"] = 0

        # Spectral rolloff (85%)
        cumsum = np.cumsum(psd)
        rolloff_threshold = 0.85 * total_power
        rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
        if len(rolloff_idx) > 0:
            features[f"{prefix}_spectral_rolloff"] = f[rolloff_idx[0]]
        else:
            features[f"{prefix}_spectral_rolloff"] = 0

        # Spectral entropy
        if total_power > 0:
            psd_norm = psd / total_power
            psd_norm[psd_norm == 0] = 1e-10  # Avoid log(0)
            features[f"{prefix}_spectral_entropy"] = -np.sum(
                psd_norm * np.log(psd_norm)
            )
        else:
            features[f"{prefix}_spectral_entropy"] = 0

        # Dominant frequency
        if len(psd) > 0:
            features[f"{prefix}_dominant_freq"] = f[np.argmax(psd)]
        else:
            features[f"{prefix}_dominant_freq"] = 0

        # Zero crossing rate (only for signals that can cross zero)
        if compute_zcr:
            zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
            features[f"{prefix}_zcr"] = zero_crossings / len(data)
        else:
            features[f"{prefix}_zcr"] = 0

    except Exception:
        # If frequency analysis fails, return zero features
        for feat in [
            "band_0.3_3",
            "band_3_8",
            "band_8_12",
            "band_0.3_3_rel",
            "band_3_8_rel",
            "band_8_12_rel",
            "band_0.3_3_log",
            "band_3_8_log",
            "band_8_12_log",
            "total_power",
            "spectral_centroid",
            "spectral_rolloff",
            "spectral_entropy",
            "dominant_freq",
            "zcr",
        ]:
            features[f"{prefix}_{feat}"] = 0

    return features


def extract_pose_invariant_features(
    world_acc: np.ndarray, prefix: str = "pose"
) -> dict:
    """Extract pose-invariant features from world acceleration."""
    features = {}

    if len(world_acc) == 0 or world_acc.shape[1] < 3:
        features[f"{prefix}_vertical_horizontal_ratio"] = 0
        features[f"{prefix}_tilt_angle_mean"] = 0
        features[f"{prefix}_tilt_angle_std"] = 0
        return features

    try:
        # Vertical (Z) vs Horizontal (XY) energy ratio
        vertical_energy = np.var(world_acc[:, 2])
        horizontal_energy = np.var(world_acc[:, 0]) + np.var(world_acc[:, 1])

        if horizontal_energy > 1e-8:
            features[f"{prefix}_vertical_horizontal_ratio"] = (
                vertical_energy / horizontal_energy
            )
        else:
            features[f"{prefix}_vertical_horizontal_ratio"] = 0

        # Tilt angle (angle from vertical)
        acc_magnitude = np.linalg.norm(world_acc, axis=1)
        acc_magnitude[acc_magnitude < 1e-8] = 1e-8  # Avoid division by zero

        # Calculate tilt angle in radians
        tilt_angles = np.arccos(np.clip(world_acc[:, 2] / acc_magnitude, -1, 1))
        features[f"{prefix}_tilt_angle_mean"] = np.mean(tilt_angles)
        features[f"{prefix}_tilt_angle_std"] = np.std(tilt_angles)

    except Exception:
        features[f"{prefix}_vertical_horizontal_ratio"] = 0
        features[f"{prefix}_tilt_angle_mean"] = 0
        features[f"{prefix}_tilt_angle_std"] = 0

    return features


def extract_temporal_pyramid_features(
    data: np.ndarray, prefix: str = "pyramid"
) -> dict:
    """Extract multi-scale temporal features."""
    features = {}

    windows = {"micro": 10, "short": 20, "medium": 40}

    for window_name, window_size in windows.items():
        if len(data) >= window_size:
            # Rolling statistics with specified window
            rolling_mean = (
                pd.Series(data).rolling(window=window_size, min_periods=1).mean()
            )
            rolling_std = (
                pd.Series(data).rolling(window=window_size, min_periods=1).std()
            )

            features[f"{prefix}_{window_name}_mean_mean"] = rolling_mean.mean()
            features[f"{prefix}_{window_name}_mean_std"] = rolling_mean.std()
            features[f"{prefix}_{window_name}_std_mean"] = rolling_std.mean()
            features[f"{prefix}_{window_name}_std_std"] = rolling_std.std()
        else:
            features[f"{prefix}_{window_name}_mean_mean"] = np.mean(data)
            features[f"{prefix}_{window_name}_mean_std"] = 0
            features[f"{prefix}_{window_name}_std_mean"] = np.std(data)
            features[f"{prefix}_{window_name}_std_std"] = 0

    return features


def extract_tail_window_features(
    data: np.ndarray, tail_fraction: float = 0.2, prefix: str = "tail"
) -> dict:
    """Extract features from the tail (last portion) of the signal."""
    features = {}

    if len(data) < 10:
        features[f"{prefix}_mean"] = np.mean(data)
        features[f"{prefix}_std"] = np.std(data)
        features[f"{prefix}_max"] = np.max(data)
        features[f"{prefix}_min"] = np.min(data)
        return features

    # Get tail window size (20% of sequence, at least 1 sample)
    tail_size = max(int(round(len(data) * tail_fraction)), 1)
    tail_size = min(tail_size, len(data))  # Ensure we don't exceed data length
    tail_data = data[-tail_size:]

    features[f"{prefix}_mean"] = np.mean(tail_data)
    features[f"{prefix}_std"] = np.std(tail_data)
    features[f"{prefix}_max"] = np.max(tail_data)
    features[f"{prefix}_min"] = np.min(tail_data)

    return features


print("✓ Feature engineering functions defined")

# Modality detection utilities
TOF_PREFIXES = ("tof", "time_of_flight", "tof_px", "tof_dist")
THM_PREFIXES = ("thermal", "thm", "temp", "ir")


def _cols_startswith(df_cols, prefixes):
    """Find columns that start with any of the given prefixes (case-insensitive)."""
    prefixes = tuple(p.lower() for p in prefixes)
    return [c for c in df_cols if c.lower().startswith(prefixes)]


def detect_modalities(seq_df: pd.DataFrame):
    """Detect which modalities are present in the dataframe."""
    cols = list(seq_df.columns)
    imu_acc = [c for c in cols if c in Config.ACC_COLS]
    imu_rot = [c for c in cols if c in Config.ROT_COLS]
    tof_cols = _cols_startswith(cols, TOF_PREFIXES)
    thm_cols = _cols_startswith(cols, THM_PREFIXES)
    present = {
        "imu": len(imu_acc) == 3 and len(imu_rot) == 4,
        "tof": len(tof_cols) > 0,
        "thm": len(thm_cols) > 0,
    }
    return present, {"tof": tof_cols, "thm": thm_cols}


def build_tof_grid_index(tof_cols):
    """Build a mapping from ToF column names to 8x8 grid coordinates."""
    # Example: "tof_r3_c7" or "tof_px27" -> (row, col)
    grid_map = {}
    for c in tof_cols:
        m = re.search(r"r(\d+).*?c(\d+)", c, flags=re.IGNORECASE)
        if m:
            r, col = int(m.group(1)), int(m.group(2))
        else:
            m2 = re.search(r"px(\d+)", c, flags=re.IGNORECASE)
            if m2:
                k = int(m2.group(1))  # 0..63
                r, col = divmod(k, 8)
            else:
                r = col = None  # Unknown -> treat as flat later
        grid_map[c] = (r, col)
    # Only use 8x8 grid if all coordinates can be determined
    ok = all(v[0] is not None for v in grid_map.values())
    return grid_map if ok else None


def _safe_nan_to_num(a):
    """Convert array to float and replace non-finite values with NaN."""
    a = np.asarray(a, float)
    a[~np.isfinite(a)] = np.nan
    return a


def tof_frame_aggregates(seq_df: pd.DataFrame, tof_cols: list, invalid_val=-1.0):
    """Extract frame-level aggregates from ToF data."""
    if not tof_cols:
        return None  # No modality
    A = _safe_nan_to_num(seq_df[tof_cols].values)  # (T, P)
    valid = (A != invalid_val) & np.isfinite(A)
    valid_count = valid.sum(axis=1).astype(float)

    # Statistics only on valid values
    def safe_stat(fn, fill=0.0):
        with np.errstate(all="ignore"):
            x = fn(np.where(valid, A, np.nan), axis=1)
        x = np.nan_to_num(x, nan=fill)
        return x

    agg = {
        "mean": safe_stat(np.nanmean),
        "std": safe_stat(np.nanstd),
        "min": safe_stat(np.nanmin, fill=np.inf),
        "max": safe_stat(np.nanmax, fill=-np.inf),
        "valid_ratio": np.nan_to_num(valid_count / A.shape[1], nan=0.0),
    }
    agg["min"][~np.isfinite(agg["min"])] = 0.0
    agg["max"][~np.isfinite(agg["max"])] = 0.0
    return agg  # Each key has (T,) vector


def tof_spatial_features_per_timestep(
    A_t: np.ndarray, grid_map: dict, invalid_val=-1.0
):
    """Extract spatial features from a single timestep of ToF data."""
    # A_t: (P,) single timestep, grid_map: {col: (r,c)}
    # Reconstruct 8x8 grid
    M = np.full((8, 8), np.nan, float)
    for j, (r, c) in ((j, grid_map[col]) for j, col in enumerate(grid_map.keys())):
        if r is not None:
            M[r, c] = A_t[j]
    V = np.isfinite(M) & (M != invalid_val)
    if V.sum() == 0:
        return dict(cx=0, cy=0, mu20=0, mu02=0, mu11=0, ecc=0, lr_asym=0, ud_asym=0)
    # Center of mass
    yy, xx = np.indices(M.shape)
    W = np.where(V, M, np.nan)  # Can use weights or 1.0
    W = np.nan_to_num(W, nan=0.0)
    s = W.sum()
    cx = float((W * xx).sum() / (s + 1e-9))
    cy = float((W * yy).sum() / (s + 1e-9))
    # Central moments
    dx, dy = xx - cx, yy - cy
    mu20 = float((W * dx * dx).sum() / (s + 1e-9))
    mu02 = float((W * dy * dy).sum() / (s + 1e-9))
    mu11 = float((W * dx * dy).sum() / (s + 1e-9))
    # Eccentricity (simple)
    ecc = float(((mu20 - mu02) ** 2 + 4 * mu11**2) ** 0.5 / (mu20 + mu02 + 1e-9))
    # Left/right and up/down asymmetry (mean difference)
    left = np.nanmean(np.where(V[:, :4], M[:, :4], np.nan))
    right = np.nanmean(np.where(V[:, 4:], M[:, 4:], np.nan))
    up = np.nanmean(np.where(V[:4, :], M[:4, :], np.nan))
    down = np.nanmean(np.where(V[4:, :], M[4:, :], np.nan))
    lr_asym = float(np.nan_to_num(left - right))
    ud_asym = float(np.nan_to_num(up - down))
    return dict(
        cx=cx,
        cy=cy,
        mu20=mu20,
        mu02=mu02,
        mu11=mu11,
        ecc=ecc,
        lr_asym=lr_asym,
        ud_asym=ud_asym,
    )


def summarize_series_features(series_dict: dict, fs: float, prefix: str) -> dict:
    """Summarize time series into statistical and frequency features."""
    feats = {}
    for name, x in series_dict.items():
        feats.update(extract_statistical_features(x, f"{prefix}_{name}"))
        feats.update(extract_peak_features(x, f"{prefix}_{name}_peak", fs=fs))
        feats.update(extract_autocorrelation_features(x, prefix=f"{prefix}_{name}_ac"))
        feats.update(
            extract_frequency_features(
                x, fs=fs, prefix=f"{prefix}_{name}_freq", compute_zcr=False
            )
        )
    return feats


def thermal_frame_aggregates(seq_df: pd.DataFrame, thm_cols: list):
    """Extract frame-level aggregates from thermal data."""
    if not thm_cols:
        return None
    A = _safe_nan_to_num(seq_df[thm_cols].values)  # (T, C)
    valid = np.isfinite(A)
    valid_count = valid.sum(axis=1).astype(float)

    def safe_stat(fn, fill=0.0):
        with np.errstate(all="ignore"):
            x = fn(np.where(valid, A, np.nan), axis=1)
        return np.nan_to_num(x, nan=fill)

    agg = {
        "mean": safe_stat(np.nanmean),
        "std": safe_stat(np.nanstd),
        "min": safe_stat(np.nanmin, fill=np.inf),
        "max": safe_stat(np.nanmax, fill=-np.inf),
        "valid_ratio": np.nan_to_num(valid_count / A.shape[1], nan=0.0),
    }
    agg["min"][~np.isfinite(agg["min"])] = 0.0
    agg["max"][~np.isfinite(agg["max"])] = 0.0

    # Hotspot ratio (values exceeding mean+kσ at each timestep)
    k = 1.0
    thr = agg["mean"][:, None] + k * (agg["std"][:, None])
    hotspot = (A > np.nan_to_num(thr, nan=np.inf)).sum(axis=1) / (A.shape[1] + 1e-9)
    agg["hotspot_ratio"] = np.nan_to_num(hotspot, nan=0.0)
    return agg


def xmod_features(imu_series: dict, tof_series: dict | None, thm_series: dict | None):
    """Extract cross-modality features (correlations and ratios)."""
    feats = {}
    # Example: linear acceleration and angular velocity magnitudes (computed in IMU)
    lin = imu_series.get("linear_acc_mag")  # (T,)
    omg = imu_series.get("angular_vel_mag")  # (T,)
    if tof_series is not None:
        tmin = tof_series.get("min")  # (T,)
        tratio = tof_series.get("valid_ratio")  # (T,)
        if lin is not None and tmin is not None and len(lin) > 1 and len(tmin) > 1:
            feats["xmod_corr_linear_to_tofmin"] = float(np.corrcoef(lin, tmin)[0, 1])
        if omg is not None and tratio is not None and len(omg) > 1 and len(tratio) > 1:
            feats["xmod_corr_omega_to_tofvalid"] = float(np.corrcoef(omg, tratio)[0, 1])
    if thm_series is not None and lin is not None:
        tmean = thm_series.get("mean")
        if tmean is not None and len(lin) > 1 and len(tmean) > 1:
            feats["xmod_corr_linear_to_thmmean"] = float(np.corrcoef(lin, tmean)[0, 1])
    # Replace NaN with 0
    for k, v in list(feats.items()):
        if not np.isfinite(v):
            feats[k] = 0.0
    return feats


def extract_tof_features(
    seq_df: pd.DataFrame, fs: float, tof_cols: list = None
) -> dict:
    """Extract ToF features from sequence dataframe."""
    if tof_cols is None:
        _, cols = detect_modalities(seq_df)
        tof_cols = cols["tof"]
    feats = {}
    feats["mod_present_tof"] = 1 if (len(tof_cols) > 0) else 0
    if not tof_cols:
        return feats

    agg = tof_frame_aggregates(seq_df, tof_cols)
    if agg is None:
        return feats

    # Missing indicators
    feats["tof_valid_ratio_mean"] = float(np.mean(agg["valid_ratio"]))
    feats.update(summarize_series_features(agg, fs, prefix="tof_seq"))

    # 8x8 spatial features (if possible)
    grid_map = build_tof_grid_index(tof_cols)
    if grid_map is not None:
        # Extract spatial features for each timestep
        A = seq_df[tof_cols].values
        series_spatial = {
            k: []
            for k in ["cx", "cy", "mu20", "mu02", "mu11", "ecc", "lr_asym", "ud_asym"]
        }
        for t in range(len(seq_df)):
            f = tof_spatial_features_per_timestep(A[t], grid_map)
            for k in series_spatial:
                series_spatial[k].append(f[k])
        series_spatial = {k: np.asarray(v) for k, v in series_spatial.items()}
        feats.update(
            summarize_series_features(series_spatial, fs, prefix="tof_spatial")
        )
    return feats


def extract_thm_features(
    seq_df: pd.DataFrame, fs: float, thm_cols: list = None
) -> dict:
    """Extract thermal features from sequence dataframe."""
    if thm_cols is None:
        _, cols = detect_modalities(seq_df)
        thm_cols = cols["thm"]
    feats = {}
    feats["mod_present_thm"] = 1 if (len(thm_cols) > 0) else 0
    if not thm_cols:
        return feats
    agg = thermal_frame_aggregates(seq_df, thm_cols)
    if agg is None:
        return feats
    feats["thm_valid_ratio_mean"] = float(np.mean(agg["valid_ratio"]))
    feats.update(summarize_series_features(agg, fs, prefix="thm_seq"))
    return feats


def apply_modality_dropout(
    features_df: pd.DataFrame, dropout_prob: float = 0.5, seed: int = None
) -> pd.DataFrame:
    """Apply modality dropout to ToF and Thermal features for training robustness."""
    # Use local RNG to avoid affecting global random state
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    df_copy = features_df.copy()

    # Randomly decide whether to drop ToF/THM for each sample
    n_samples = len(df_copy)
    drop_tof = rng.random(n_samples) < dropout_prob
    drop_thm = rng.random(n_samples) < dropout_prob

    # Find ToF and THM feature columns
    tof_cols = [
        col
        for col in df_copy.columns
        if col.startswith("tof_") and col != "mod_present_tof"
    ]
    thm_cols = [
        col
        for col in df_copy.columns
        if col.startswith("thm_") and col != "mod_present_thm"
    ]
    xmod_tof_cols = [
        col
        for col in df_copy.columns
        if col.startswith("xmod_") and "tof" in col.lower()
    ]
    xmod_thm_cols = [
        col
        for col in df_copy.columns
        if col.startswith("xmod_") and "thm" in col.lower()
    ]

    # Apply dropout using vectorized operations
    if tof_cols:
        df_copy.loc[drop_tof, tof_cols] = 0
        if "mod_present_tof" in df_copy.columns:
            df_copy.loc[drop_tof, "mod_present_tof"] = 0
        if xmod_tof_cols:
            df_copy.loc[drop_tof, xmod_tof_cols] = 0

    if thm_cols:
        df_copy.loc[drop_thm, thm_cols] = 0
        if "mod_present_thm" in df_copy.columns:
            df_copy.loc[drop_thm, "mod_present_thm"] = 0
        if xmod_thm_cols:
            df_copy.loc[drop_thm, xmod_thm_cols] = 0

    return df_copy


def extract_xmod_features_for_union(
    imu_series: dict, tof_agg: dict | None, thm_agg: dict | None
) -> dict:
    """Extract cross-modality features for the union of modalities."""
    # imu_series should contain linear_acc_mag, angular_vel_mag from extract_features
    return xmod_features(
        {
            "linear_acc_mag": imu_series.get("linear_acc_mag"),
            "angular_vel_mag": imu_series.get("angular_vel_mag"),
        },
        tof_series=tof_agg,
        thm_series=thm_agg,
    )


# === DL frame-level feature extraction ===
def build_frame_features(sequence: pl.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame of shape (T, C) for DL, with fixed column order.
    """
    seq_df = sequence.to_pandas()
    dt, fs = infer_dt_and_fs(seq_df)

    # IMU 必須列の存在保証
    for c in Config.ACC_COLS:
        if c not in seq_df.columns:
            seq_df[c] = 0.0
    available_rot_cols = [c for c in Config.ROT_COLS if c in seq_df.columns]
    if available_rot_cols:
        rot = build_full_quaternion(seq_df, available_rot_cols)
        rot = handle_quaternion_missing_values(rot)
        rot = fix_quaternion_sign(rot)
    else:
        rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(seq_df), 1))

    acc = seq_df[Config.ACC_COLS].ffill().bfill().to_numpy(dtype=float)
    world_acc = compute_world_acceleration(acc, rot)
    linear_acc = compute_linear_acceleration(world_acc, fs=fs)
    omega = compute_angular_velocity(rot, dt=dt)

    # magnitudes
    lin_mag = np.linalg.norm(linear_acc, axis=1, keepdims=True)
    omg_mag = np.linalg.norm(omega, axis=1, keepdims=True)

    # ToF/THM frame aggregates
    _, mod_cols = detect_modalities(seq_df)
    tof_agg = tof_frame_aggregates(seq_df, mod_cols["tof"]) if mod_cols["tof"] else None
    thm_agg = (
        thermal_frame_aggregates(seq_df, mod_cols["thm"]) if mod_cols["thm"] else None
    )

    feats = {}
    feats["linear_acc_x"] = linear_acc[:, 0]
    feats["linear_acc_y"] = linear_acc[:, 1]
    feats["linear_acc_z"] = linear_acc[:, 2]
    feats["omega_x"] = omega[:, 0]
    feats["omega_y"] = omega[:, 1]
    feats["omega_z"] = omega[:, 2]
    feats["linear_acc_mag"] = lin_mag[:, 0]
    feats["omega_mag"] = omg_mag[:, 0]

    # ToF
    if tof_agg is not None:
        feats["tof_mean"] = tof_agg["mean"]
        feats["tof_std"] = tof_agg["std"]
        feats["tof_min"] = tof_agg["min"]
        feats["tof_max"] = tof_agg["max"]
        feats["tof_valid_ratio"] = tof_agg["valid_ratio"]
    else:
        for k in ["mean", "std", "min", "max", "valid_ratio"]:
            feats[f"tof_{k}"] = np.zeros(len(seq_df))

    # THM
    if thm_agg is not None:
        feats["thm_mean"] = thm_agg["mean"]
        feats["thm_std"] = thm_agg["std"]
        feats["thm_min"] = thm_agg["min"]
        feats["thm_max"] = thm_agg["max"]
        feats["thm_valid_ratio"] = thm_agg["valid_ratio"]
    else:
        for k in ["mean", "std", "min", "max", "valid_ratio"]:
            feats[f"thm_{k}"] = np.zeros(len(seq_df))

    frame_df = pd.DataFrame(feats)
    frame_df.replace([np.inf, -np.inf], 0, inplace=True)
    frame_df.fillna(0, inplace=True)
    return frame_df


def build_frame_feature_store(train_df: pl.DataFrame, cols_to_select: list[str]):
    """Frame feature store を事前構築（1seq = 1file）"""
    os.makedirs(DLConfig.FRAME_FEATURE_DIR, exist_ok=True)
    grouped = train_df.select(pl.col(cols_to_select)).group_by(
        "sequence_id", maintain_order=True
    )
    for _, seq in grouped:
        sid = int(seq["sequence_id"][0])
        out = os.path.join(DLConfig.FRAME_FEATURE_DIR, f"{sid}.parquet")
        if os.path.exists(out):
            continue
        df = build_frame_features(seq)
        df.to_parquet(out, index=False)  # pandas の to_parquet
    print(f"✓ Frame feature store built at {DLConfig.FRAME_FEATURE_DIR}")


# === DL preprocessing functions ===
def compute_scaler_stats(
    frame_dfs: list[pd.DataFrame],
) -> dict[str, tuple[float, float]]:
    # 入力: 学習fold内の全 sequence の frame_df リスト
    concat = pd.concat(frame_dfs, axis=0, ignore_index=True)
    stats = {}
    for c in concat.columns:
        x = concat[c].values.astype(np.float64)
        mu = float(np.mean(x))
        sd = float(np.std(x) + 1e-8)
        stats[c] = (mu, sd)
    return stats


def apply_standardize(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = df.copy()
    for c, (mu, sd) in stats.items():
        if c in out.columns:
            out[c] = (out[c].astype(np.float32) - mu) / sd
    return out


def decide_pad_len(lengths: list[int], fixed: int | None, pctl: int = 95) -> int:
    if fixed is not None:
        return int(fixed)
    return int(np.percentile(lengths, pctl))


def pad_and_mask(x: np.ndarray, pad_len: int) -> tuple[np.ndarray, np.ndarray]:
    # x: (T,C) -> (pad_len, C), mask: (pad_len,) 1=valid, 0=pad
    T, C = x.shape
    out = np.zeros((pad_len, C), dtype=np.float32)
    msk = np.zeros((pad_len,), dtype=np.float32)
    t = min(T, pad_len)
    out[:t] = x[:t]
    msk[:t] = 1.0
    return out, msk


# === Torch Dataset and collate function ===
if TORCH_AVAILABLE:

    class TorchDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            sequences: list[pl.DataFrame],
            labels: np.ndarray | None,
            scaler_stats: dict,
            pad_len: int,
            train_mode: bool = False,
            modality_dropout_prob: float = 0.2,
        ):
            self.sequences = sequences
            self.labels = labels
            self.scaler_stats = scaler_stats
            self.pad_len = pad_len
            self.train_mode = train_mode
            self.modality_dropout_prob = modality_dropout_prob

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            frame_df = FRAME_CACHE.get(self.sequences[idx])  # pandas.DataFrame
            frame_df = apply_standardize(
                frame_df, self.scaler_stats
            )  # pandas.DataFrame

            # Apply modality dropout during training (ToF / THM) - use **pandas** assignment
            if self.train_mode and np.random.random() < self.modality_dropout_prob:
                if np.random.random() < 0.5:
                    cols = [c for c in frame_df.columns if c.startswith("tof_")]
                else:
                    cols = [c for c in frame_df.columns if c.startswith("thm_")]
                if cols:
                    frame_df.loc[:, cols] = 0.0  # pandas 列代入

            x = frame_df.to_numpy(dtype=np.float32)
            x, m = pad_and_mask(x, self.pad_len)
            m = (m > 0.5).astype(np.float32)  # Bool mask to 0/1
            y = -1 if self.labels is None else int(self.labels[idx])
            return x, m, y

    def collate_batch(batch):
        xs, ms, ys = zip(*batch)
        x = torch.from_numpy(np.stack(xs, 0))  # (B, T, C)
        m = torch.from_numpy(np.stack(ms, 0))  # (B, T)
        y = torch.tensor(ys, dtype=torch.long)
        return x, m, y

    # === DL Model (Conv → BiLSTM → GRU → Attention) ===
    class TemporalAttention(nn.Module):
        def __init__(self, d_model: int):
            super().__init__()
            self.proj = nn.Linear(d_model, 1)

        def forward(self, h, mask):
            # h: (B,T,D), mask: (B,T) 1=valid
            logit = self.proj(h).squeeze(-1)  # (B,T)
            # Use -65504 instead of -1e9 for float16 compatibility
            logit = logit.masked_fill(mask == 0, -65504)  # padを弾く
            w = torch.softmax(logit, dim=1)  # (B,T)
            pooled = torch.bmm(w.unsqueeze(1), h).squeeze(1)  # (B,D)
            return pooled, w

    class TimeSeriesNet(nn.Module):
        def __init__(
            self, in_ch: int, num_classes: int, hidden: int = 128, dropout: float = 0.2
        ):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(in_ch, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.MaxPool1d(2),
                nn.Conv1d(128, 256, kernel_size=5, padding=2),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.MaxPool1d(2),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.MaxPool1d(2),  # 3段目追加
                nn.Dropout(dropout),
            )
            self.bilstm = nn.LSTM(
                256, hidden, num_layers=1, batch_first=True, bidirectional=True
            )
            self.gru = nn.GRU(
                2 * hidden, hidden, num_layers=1, batch_first=True, bidirectional=True
            )
            self.attn = TemporalAttention(2 * hidden)
            self.head = nn.Sequential(
                nn.Linear(2 * hidden, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

        def forward(self, x, mask):
            # x: (B,T,C) -> Conv1dは (B,C,T)
            x = x.transpose(1, 2)
            x = self.conv(x)  # (B, 256, T')
            x = x.transpose(1, 2)  # (B, T', 256)

            # === FIX: マスクも MaxPool と同等に3回ダウンサンプリングして T' に整合させる ===
            m = mask  # (B, T)
            for _ in range(3):  # conv 内の MaxPool1d を3回適用しているため
                m = F.max_pool1d(m.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
            m = m[:, : x.size(1)]  # 念のため長さを一致させる

            h, _ = self.bilstm(x)
            h, _ = self.gru(h)
            pooled, _ = self.attn(h, m)
            logits = self.head(pooled)
            return logits

    # === Training utilities ===
    def soft_ce_loss(logits, targets, smoothing=0.05, n_classes=18):
        with torch.no_grad():
            true_dist = torch.zeros_like(logits).fill_(smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
        log_prob = F.log_softmax(logits, dim=1)
        return -(true_dist * log_prob).sum(dim=1).mean()

    def focal_loss(logits, targets, alpha=0.25, gamma=2.0, n_classes=18):
        """Focal loss for addressing class imbalance"""
        p = torch.softmax(logits, dim=1)
        y = F.one_hot(targets, n_classes).float()
        pt = (p * y).sum(dim=1).clamp_min(1e-8)
        loss = -alpha * (1 - pt) ** gamma * torch.log(pt)
        return loss.mean()

    # Select loss function based on environment variable
    USE_FOCAL_LOSS = os.getenv("USE_FOCAL_LOSS", "False").lower() == "true"

    def compute_torch_metrics(y_true, y_pred):
        # Binary: 0-7 vs 8-17（既存と揃える）
        bin_true = (y_true <= 7).astype(int)
        bin_pred = (y_pred <= 7).astype(int)
        binary_f1 = f1_score(bin_true, bin_pred, zero_division=0.0)
        # Macro F1（BFRB内：0..7 のみ評価する現行式に揃えるなら調整可）
        macro_f1 = f1_score(
            np.where(y_true <= 7, y_true, 99),
            np.where(y_pred <= 7, y_pred, 99),
            average="macro",
            zero_division=0.0,
        )
        return binary_f1, macro_f1, 0.5 * (binary_f1 + macro_f1)

    # === Torch training helpers (NEW) ===
    from tqdm import tqdm

    class EarlyStopper:
        def __init__(self, patience=6, min_delta=1e-4):
            self.patience = patience
            self.min_delta = min_delta
            self.best = -float("inf")
            self.bad = 0

        def step(self, metric: float) -> bool:
            if metric > self.best + self.min_delta:
                self.best = metric
                self.bad = 0
                return False
            self.bad += 1
            return self.bad >= self.patience

    def _gpu_mem_gb():
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**3)
        return 0.0

    def _log_jsonl(path, obj):
        if not LogConfig.SAVE_JSONL:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # === Torch training function ===
    def train_torch_models(train_df: pl.DataFrame, train_demographics: pl.DataFrame):
        _ = train_demographics  # Not used but kept for API consistency
        base_cols = ["sequence_id", "subject", "phase", "gesture"]
        all_cols = train_df.columns
        sensor_cols = (
            [c for c in all_cols if c in Config.ACC_COLS + Config.ROT_COLS]
            + _cols_startswith(all_cols, TOF_PREFIXES)
            + _cols_startswith(all_cols, THM_PREFIXES)
        )
        cols_to_select = base_cols + sensor_cols
        grouped = train_df.select(pl.col(cols_to_select)).group_by(
            "sequence_id", maintain_order=True
        )

        seq_list, y_list, subj_list, lengths = [], [], [], []
        for _, seq in grouped:
            seq_list.append(seq)
            y_list.append(GESTURE_MAPPER[seq["gesture"][0]])
            subj_list.append(seq["subject"][0])
            lengths.append(len(seq))

        pad_len = decide_pad_len(
            lengths, DLConfig.FIXED_PAD_LEN, DLConfig.PAD_LEN_PERCENTILE
        )
        os.makedirs(DLConfig.TORCH_OUT_DIR, exist_ok=True)

        # OOF predictions for Torch
        n_classes = len(GESTURE_MAPPER)
        oof_torch = np.zeros((len(seq_list), n_classes), dtype=np.float32)

        cv = StratifiedGroupKFold(
            n_splits=DLConfig.N_FOLDS, shuffle=True, random_state=DLConfig.SEED
        )
        fold_weights, models_meta = [], []
        device = "cuda" if torch.cuda.is_available() else "cpu"

        for fold, (tr_idx, va_idx) in enumerate(
            cv.split(seq_list, np.array(y_list), np.array(subj_list))
        ):
            print(f"\n--- Torch Fold {fold + 1}/{DLConfig.N_FOLDS} ---")
            best_path = os.path.join(
                DLConfig.TORCH_OUT_DIR, DLConfig.WEIGHT_TMPL.format(fold)
            )

            # 既に最良重みがあれば学習をスキップ（任意）
            if CheckpointConfig.SKIP_TORCH_FOLD_IF_BEST_EXISTS and os.path.exists(
                best_path
            ):
                print(
                    f"✓ Found existing best weights for fold {fold} at {best_path} — skip training"
                )
                best_score = float(
                    _load_json(CheckpointConfig.TORCH_STATE_JSON, {})
                    .get("best_scores", {})
                    .get(str(fold), -1.0)
                )
                if best_score < 0:
                    # スコアが未記録でも重みは利用可能。暫定で 1.0 を採用
                    best_score = 1.0
                fold_weights.append(best_score)
                # スケーラ統計・メタは再構築して保存に必要
                tr_frames = [build_frame_features(seq_list[i]) for i in tr_idx]
                scaler_stats = compute_scaler_stats(tr_frames)
                models_meta.append(
                    {"scaler_stats": scaler_stats, "weight_path": best_path}
                )
                continue

            # === fold固有のスケーラ統計
            tr_frames = [build_frame_features(seq_list[i]) for i in tr_idx]
            scaler_stats = compute_scaler_stats(tr_frames)

            ds_tr = TorchDataset(
                [seq_list[i] for i in tr_idx],
                np.array(y_list)[tr_idx],
                scaler_stats,
                pad_len,
                train_mode=True,
                modality_dropout_prob=0.2,
            )
            ds_va = TorchDataset(
                [seq_list[i] for i in va_idx],
                np.array(y_list)[va_idx],
                scaler_stats,
                pad_len,
                train_mode=False,
            )

            # DataLoader作成関数（OOMフォールバック付き）
            def make_loader(dataset, batch_size, shuffle=True):
                kwargs = dict(
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=DLConfig.NUM_WORKERS,
                    collate_fn=collate_batch,
                    pin_memory=True,
                    persistent_workers=(DLConfig.NUM_WORKERS > 0),
                )
                # prefetch_factor は num_workers>0 の時だけ渡す
                if DLConfig.NUM_WORKERS > 0:
                    kwargs["prefetch_factor"] = 2
                return torch.utils.data.DataLoader(dataset, **kwargs)

            # OOMフォールバック付きDataLoader作成
            bs = DLConfig.BATCH_SIZE
            dl_tr, dl_va = None, None
            for attempt in [bs, bs // 2, bs // 4, bs // 8]:
                if attempt < 64:
                    attempt = 64  # 最小バッチサイズ
                try:
                    dl_tr = make_loader(ds_tr, attempt, shuffle=True)
                    dl_va = make_loader(ds_va, attempt, shuffle=False)
                    print(f"✓ Using batch_size={attempt}")
                    break
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) and attempt > 64:
                        print(
                            f"OOM at batch={attempt}, retrying with {attempt // 2}..."
                        )
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise

            if dl_tr is None or dl_va is None:
                raise RuntimeError(
                    "Failed to create DataLoaders even with minimal batch size"
                )

            n_classes = len(GESTURE_MAPPER)
            in_ch = tr_frames[0].shape[1]
            model = TimeSeriesNet(
                in_ch, n_classes, hidden=128, dropout=DLConfig.DROPOUT
            )
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=DLConfig.LR, weight_decay=DLConfig.WEIGHT_DECAY
            )
            total_steps = DLConfig.MAX_EPOCHS * max(1, len(dl_tr))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=DLConfig.LR,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy="cos",
            )
            scaler = torch.amp.GradScaler("cuda", enabled=DLConfig.AMP)

            # === Resume (epoch 再開)
            epoch_start, best_score = 0, -1.0
            last_ckpt_path = CheckpointConfig.TORCH_EPOCH_CKPT_TMPL.format(fold)

            # EarlyStopper の初期化
            early = EarlyStopper(
                patience=int(os.getenv("TORCH_ES_PATIENCE", "6")),
                min_delta=float(os.getenv("TORCH_ES_MIN_DELTA", "1e-4")),
            )
            if bool(int(os.getenv("RESUME_TORCH", "1"))) and os.path.exists(
                last_ckpt_path
            ):
                print(f"↻ Resuming fold {fold} from epoch checkpoint: {last_ckpt_path}")
                ckpt = torch.load(last_ckpt_path, map_location=device)
                state = ckpt["model_state"]
                if any(k.startswith("module.") for k in state.keys()):
                    state = {k.replace("module.", "", 1): v for k, v in state.items()}
                model.load_state_dict(state, strict=True)
                optimizer.load_state_dict(ckpt["optim_state"])
                scheduler.load_state_dict(ckpt["sched_state"])
                scaler.load_state_dict(ckpt["scaler_state"])
                epoch_start = int(ckpt["epoch"]) + 1
                best_score = float(ckpt.get("best_score", -1.0))

            # === Train with detailed progress
            for epoch in range(epoch_start, DLConfig.MAX_EPOCHS):
                t0 = time.time()
                model.train()
                running = 0.0
                nstep = 0
                torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

                USE_TQDM = bool(
                    int(os.getenv("USE_TQDM", "0"))
                )  # Default: tqdm off for Kaggle
                if USE_TQDM:
                    iterator = tqdm(
                        dl_tr,
                        total=len(dl_tr),
                        desc=f"[Torch] fold {fold} epoch {epoch + 1}",
                        leave=False,
                    )
                else:
                    iterator = dl_tr

                for batch_idx, (xb, mb, yb) in enumerate(iterator):
                    xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)

                    # 勾配蓄積：最初のバッチまたは蓄積ステップごとにゼロ化
                    if batch_idx % DLConfig.ACCUM_STEPS == 0:
                        optimizer.zero_grad(set_to_none=True)

                    with torch.amp.autocast("cuda", enabled=DLConfig.AMP):
                        logits = model(xb, mb)
                        if USE_FOCAL_LOSS:
                            loss = focal_loss(logits, yb, n_classes=n_classes)
                        else:
                            loss = soft_ce_loss(
                                logits,
                                yb,
                                smoothing=DLConfig.LABEL_SMOOTHING,
                                n_classes=n_classes,
                            )
                        # 勾配蓄積のためlossを分割
                        loss = loss / DLConfig.ACCUM_STEPS

                    scaler.scale(loss).backward()

                    # 蓄積ステップごとにoptimizer更新
                    if (batch_idx + 1) % DLConfig.ACCUM_STEPS == 0 or (
                        batch_idx + 1
                    ) == len(dl_tr):
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()

                    running += float(loss.detach().item()) * DLConfig.ACCUM_STEPS
                    nstep += 1

                    # Existing tqdm postfix
                    if USE_TQDM and (nstep % LogConfig.LOG_EVERY_STEPS == 0):
                        lr = scheduler.get_last_lr()[0]
                        iterator.set_postfix(
                            loss=f"{running / nstep:.4f}", lr=f"{lr:.2e}"
                        )

                    # NEW: Standard output logging
                    if (nstep % LogConfig.PRINT_EVERY_STEPS == 0) or (nstep == 1):
                        lr = scheduler.get_last_lr()[0]
                        print(
                            f"[Torch] fold={fold} epoch={epoch + 1}/{DLConfig.MAX_EPOCHS} "
                            f"step={nstep}/{len(dl_tr)} loss={running / nstep:.4f} lr={lr:.2e} "
                            f"max_mem={_gpu_mem_gb():.2f}GB",
                            flush=True,
                        )

                # --- valid
                model.eval()
                all_pred, all_true, all_prob = [], [], []
                with torch.no_grad():
                    for xb, mb, yb in dl_va:
                        xb, mb = xb.to(device), mb.to(device)
                        with torch.amp.autocast("cuda", enabled=DLConfig.AMP):
                            logits = model(xb, mb)
                            prob = torch.softmax(logits, dim=1)
                        pred = prob.argmax(dim=1).cpu().numpy()
                        all_pred.append(pred)
                        all_true.append(yb.numpy())
                        all_prob.append(prob.cpu().numpy())
                y_true = np.concatenate(all_true)
                y_pred = np.concatenate(all_pred)
                y_prob = np.concatenate(all_prob)
                bF1, mF1, score = compute_torch_metrics(y_true, y_pred)
                epoch_time = time.time() - t0
                max_mem = _gpu_mem_gb()

                print(
                    f"  [fold {fold}] epoch {epoch + 1}/{DLConfig.MAX_EPOCHS} "
                    f"| loss={running / max(nstep, 1):.4f} | score={score:.4f} "
                    f"(BinF1={bF1:.4f}, MacroF1={mF1:.4f}) | time={epoch_time:.1f}s | max_mem={max_mem:.2f}GB"
                )

                _log_jsonl(
                    os.path.join(LogConfig.OUT_DIR, "torch_progress.jsonl"),
                    {
                        "fold": fold,
                        "epoch": epoch,
                        "train_loss": running / max(nstep, 1),
                        "bin_f1": bF1,
                        "macro_f1": mF1,
                        "score": score,
                        "secs": epoch_time,
                        "max_mem_gb": max_mem,
                        "lr": scheduler.get_last_lr()[0],
                    },
                )

                # Save best
                if score > best_score:
                    best_score = score
                    # Save OOF predictions for best model
                    oof_torch[va_idx] = y_prob
                    state = (
                        model.module.state_dict()
                        if isinstance(model, nn.DataParallel)
                        else model.state_dict()
                    )
                    torch.save(
                        {"state_dict": state, "in_ch": in_ch, "n_classes": n_classes},
                        best_path,
                    )
                    print(f"  ↳ New best! saved: {best_path}")

                # Save epoch checkpoint
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": (
                            model.module.state_dict()
                            if isinstance(model, nn.DataParallel)
                            else model.state_dict()
                        ),
                        "optim_state": optimizer.state_dict(),
                        "sched_state": scheduler.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "best_score": best_score,
                        "best_path": best_path,
                    },
                    last_ckpt_path,
                )

                # Early stopping check
                if early.step(score):
                    print(
                        f"  ↯ Early stopping at epoch {epoch + 1} (best={early.best:.4f})"
                    )
                    break

            fold_weights.append(best_score)
            models_meta.append({"scaler_stats": scaler_stats, "weight_path": best_path})

            # Torch state json 更新
            ts = _load_json(CheckpointConfig.TORCH_STATE_JSON, default={})
            bs = ts.get("best_scores", {})
            bs[str(fold)] = float(best_score)
            ts["best_scores"] = bs
            ts["pad_len"] = pad_len
            _save_json(CheckpointConfig.TORCH_STATE_JSON, ts)

        # 重み正規化＋バンドル保存
        denom = max(float(np.sum(fold_weights)), 1e-12)
        fold_w = (np.array(fold_weights) / denom).tolist()
        bundle = {
            "pad_len": pad_len,
            "feature_order": list(tr_frames[0].columns),
            "folds": [
                {"weight": fold_w[i], **models_meta[i]} for i in range(len(models_meta))
            ],
            "gesture_mapper": GESTURE_MAPPER,
            "reverse_gesture_mapper": REVERSE_GESTURE_MAPPER,
        }
        joblib.dump(bundle, os.path.join(DLConfig.TORCH_OUT_DIR, DLConfig.BUNDLE_NAME))

        # Save Torch OOF predictions
        os.makedirs(os.path.join(Config.OUTPUT_PATH, "oof"), exist_ok=True)
        np.save(
            os.path.join(Config.OUTPUT_PATH, "oof", "oof_torch_proba.npy"), oof_torch
        )

        print("✓ Torch training done. Saved to:", DLConfig.TORCH_OUT_DIR)

    # === Keras training function ===
    def train_keras_models(train_df: pl.DataFrame, train_demographics: pl.DataFrame):
        _ = train_demographics  # Not used but kept for API consistency
        if not KERAS_AVAILABLE:
            print("⚠️ Keras not available. Skip Keras training.")
            return

        # Set random seed for reproducibility
        tf.random.set_seed(KerasConfig.SEED)

        os.makedirs(KerasConfig.OUT_DIR, exist_ok=True)

        base_cols = ["sequence_id", "subject", "phase", "gesture"]
        all_cols = train_df.columns
        sensor_cols = (
            [c for c in all_cols if c in Config.ACC_COLS + Config.ROT_COLS]
            + _cols_startswith(all_cols, TOF_PREFIXES)
            + _cols_startswith(all_cols, THM_PREFIXES)
        )
        cols_to_select = base_cols + sensor_cols
        grouped = train_df.select(pl.col(cols_to_select)).group_by(
            "sequence_id", maintain_order=True
        )

        seq_list, y_list, subj_list, lengths = [], [], [], []
        for _, seq in grouped:
            seq_list.append(seq)
            y_list.append(GESTURE_MAPPER[seq["gesture"][0]])
            subj_list.append(seq["subject"][0])
            lengths.append(len(seq))

        pad_len = decide_pad_len(
            lengths, KerasConfig.FIXED_PAD_LEN, KerasConfig.PAD_LEN_PERCENTILE
        )
        n_classes = len(GESTURE_MAPPER)
        cv = StratifiedGroupKFold(
            n_splits=KerasConfig.N_FOLDS, shuffle=True, random_state=KerasConfig.SEED
        )
        oof_proba = np.zeros((len(seq_list), n_classes), dtype=np.float32)

        feat_order = list(build_frame_features(seq_list[0]).columns)  # 列順固定

        # Helper function to convert sequence to Keras tensor
        def make_keras_tensor(
            sequence_pl: pl.DataFrame, stats: dict, pad_len: int, feat_order: list
        ) -> tuple[np.ndarray, np.ndarray]:
            frame_df = (
                build_frame_features(sequence_pl).reindex(columns=feat_order).fillna(0)
            )
            x_std = apply_standardize(frame_df, stats).to_numpy(np.float32)  # (T, C)
            x_pad, m_pad = pad_and_mask(x_std, pad_len)  # (L, C), (L,)
            return x_pad, m_pad

        fold_scaler_stats = []  # Store scaler stats for each fold

        for fold, (tr_idx, va_idx) in enumerate(
            cv.split(seq_list, np.array(y_list), np.array(subj_list))
        ):
            print(f"\n--- Keras Fold {fold + 1}/{KerasConfig.N_FOLDS} ---")

            # Check if fold already completed
            weight_path = os.path.join(
                KerasConfig.OUT_DIR, KerasConfig.WEIGHT_TMPL.format(fold)
            )
            if (
                os.path.exists(weight_path)
                and CheckpointConfig.KERAS_EARLY_EXIT_IF_BEST_EXISTS
            ):
                print(
                    f"✓ Found existing weights for fold {fold} at {weight_path} — skip training"
                )
                # Still need to calculate scaler stats for bundle
                tr_frames = [build_frame_features(seq_list[i]) for i in tr_idx]
                scaler_stats = compute_scaler_stats(tr_frames)
                fold_scaler_stats.append(scaler_stats)
                continue

            # scaler fit
            tr_frames = [build_frame_features(seq_list[i]) for i in tr_idx]
            scaler_stats = compute_scaler_stats(tr_frames)
            fold_scaler_stats.append(scaler_stats)

            # numpy へ事前変換（高速）
            def to_xy(idxs, train_mode=False, modality_dropout_prob=0.2):
                X, M, Y = [], [], []
                for i in idxs:
                    x, m = make_keras_tensor(
                        seq_list[i], scaler_stats, pad_len, feat_order
                    )

                    # Apply modality dropout during training (ToF and THM columns)
                    if train_mode and np.random.random() < modality_dropout_prob:
                        # Find ToF and THM column indices
                        tof_indices = [
                            j
                            for j, col in enumerate(feat_order)
                            if col.startswith("tof_")
                        ]
                        thm_indices = [
                            j
                            for j, col in enumerate(feat_order)
                            if col.startswith("thm_")
                        ]

                        # Randomly drop ToF or THM modality
                        if np.random.random() < 0.5 and tof_indices:
                            # Drop ToF columns
                            x[:, tof_indices] = 0.0
                        elif thm_indices:
                            # Drop THM columns
                            x[:, thm_indices] = 0.0

                    X.append(x)
                    M.append(m)
                    Y.append(y_list[i])
                X = np.stack(X).astype(np.float32)  # (N, L, C)
                M = np.stack(M).astype(np.float32)  # (N, L)
                Y = keras.utils.to_categorical(np.array(Y), num_classes=n_classes)
                return X, M, Y

            Xtr, Mtr, Ytr = to_xy(tr_idx, train_mode=True, modality_dropout_prob=0.2)
            Xva, Mva, Yva = to_xy(va_idx, train_mode=False)

            # Calculate class weights for handling imbalance
            from sklearn.utils.class_weight import compute_class_weight

            train_labels = np.array([y_list[i] for i in tr_idx])
            class_weights = compute_class_weight(
                "balanced", classes=np.unique(train_labels), y=train_labels
            )
            class_weight_dict = {i: w for i, w in enumerate(class_weights)}

            # モデル（2ブランチ）
            model = build_keras_two_branch(
                input_shape=(pad_len, Xtr.shape[-1]),
                n_classes=n_classes,
                dropout=KerasConfig.DROPOUT,
            )

            cbs = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=KerasConfig.EARLY_STOPPING_PATIENCE,
                    restore_best_weights=True,
                    verbose=1,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=KerasConfig.REDUCE_LR_PATIENCE,
                    min_lr=1e-5,
                    verbose=1,
                ),
                keras.callbacks.ModelCheckpoint(
                    weight_path, monitor="val_loss", save_best_only=True, verbose=1
                ),
                keras.callbacks.CSVLogger(
                    os.path.join(KerasConfig.OUT_DIR, f"fold{fold:02d}_train.csv"),
                    append=True,
                ),
                ProgressCallback(
                    fold, os.path.join(LogConfig.OUT_DIR, "keras_progress.jsonl")
                ),
            ]

            hist = model.fit(
                x={"x": Xtr, "mask": Mtr},
                y=Ytr,
                validation_data=({"x": Xva, "mask": Mva}, Yva),
                epochs=KerasConfig.MAX_EPOCHS,
                batch_size=KerasConfig.BATCH_SIZE,
                verbose=2,
                callbacks=cbs,
                class_weight=class_weight_dict,
            )

            # OOF proba
            proba_va = model.predict(
                {"x": Xva, "mask": Mva}, batch_size=KerasConfig.BATCH_SIZE, verbose=0
            )
            oof_proba[va_idx] = proba_va

            # fold メタ保存
            state = _load_json(KerasConfig.STATE_JSON, default={})
            best_scores = state.get("best_scores", {})
            best_scores[str(fold)] = float(np.max(hist.history.get("val_acc", [0.0])))
            state["best_scores"] = best_scores
            state["pad_len"] = pad_len
            _save_json(KerasConfig.STATE_JSON, state)

        # fold 重み（バリデーション acc などから）を正規化
        best_scores = _load_json(KerasConfig.STATE_JSON, default={}).get(
            "best_scores", {}
        )
        fold_scores = [
            float(best_scores.get(str(i), 1.0)) for i in range(KerasConfig.N_FOLDS)
        ]
        fold_weights = (
            np.array(fold_scores) / max(np.sum(fold_scores), 1e-12)
        ).tolist()

        # bundle 保存
        bundle = {
            "pad_len": pad_len,
            "feature_order": feat_order,
            "folds": [
                {
                    "weight": fold_weights[i],
                    "scaler_stats": fold_scaler_stats[i],
                    "weight_path": os.path.join(
                        KerasConfig.OUT_DIR, KerasConfig.WEIGHT_TMPL.format(i)
                    ),
                }
                for i in range(KerasConfig.N_FOLDS)
            ],
            "gesture_mapper": GESTURE_MAPPER,
            "reverse_gesture_mapper": REVERSE_GESTURE_MAPPER,
        }
        joblib.dump(bundle, os.path.join(KerasConfig.OUT_DIR, KerasConfig.BUNDLE_NAME))

        # Save OOF predictions
        os.makedirs(os.path.join(Config.OUTPUT_PATH, "oof"), exist_ok=True)
        np.save(
            os.path.join(Config.OUTPUT_PATH, "oof", "oof_keras_proba.npy"), oof_proba
        )
        np.save(os.path.join(Config.OUTPUT_PATH, "oof", "y_true.npy"), np.array(y_list))

        print("✓ Keras training done. Saved to:", KerasConfig.OUT_DIR)

    # === Keras inference function ===
    _KERAS_RUNTIME = {"bundle": None, "fold_models": []}

    def _load_keras_bundle_and_models():
        if _KERAS_RUNTIME["bundle"] is not None:
            return
        bp = EnsembleConfig.KERAS_BUNDLE_PATH
        if not (KERAS_AVAILABLE and os.path.exists(bp)):
            print(f"ℹ️ Keras bundle not found or Keras unavailable: {bp}")
            _KERAS_RUNTIME["bundle"] = None
            return
        bundle = joblib.load(bp)
        _KERAS_RUNTIME["bundle"] = bundle

        if EnsembleConfig.LOAD_KERAS_FOLDS_IN_MEMORY and KERAS_AVAILABLE:
            _KERAS_RUNTIME["fold_models"] = []
            for f in bundle["folds"]:
                model = keras.models.load_model(
                    f["weight_path"],
                    custom_objects={"KerasTemporalAttention": KerasTemporalAttention},
                )
                _KERAS_RUNTIME["fold_models"].append(model)
            print(
                f"✓ Loaded {len(_KERAS_RUNTIME['fold_models'])} Keras models into memory"
            )

    def predict_keras_proba(
        sequence: pl.DataFrame, demographics: pl.DataFrame
    ) -> np.ndarray | None:
        _ = demographics  # Not used but kept for API consistency
        _load_keras_bundle_and_models()
        bundle = _KERAS_RUNTIME["bundle"]
        if bundle is None:
            return None

        pad_len = bundle["pad_len"]
        feat_order = bundle["feature_order"]
        frame_df = FRAME_CACHE.get(sequence).reindex(columns=feat_order).fillna(0)
        n_classes = len(bundle["reverse_gesture_mapper"])
        proba_accum = np.zeros(n_classes, dtype=np.float64)

        for i, f in enumerate(bundle["folds"]):
            stats = f["scaler_stats"]
            x_std = apply_standardize(frame_df, stats).to_numpy(np.float32)
            x_pad, m_pad = pad_and_mask(x_std, pad_len)
            X = {"x": x_pad[None, ...], "mask": m_pad[None, ...]}

            if (
                EnsembleConfig.LOAD_KERAS_FOLDS_IN_MEMORY
                and _KERAS_RUNTIME["fold_models"]
            ):
                model = _KERAS_RUNTIME["fold_models"][i]
            else:
                model = keras.models.load_model(
                    f["weight_path"],
                    custom_objects={"KerasTemporalAttention": KerasTemporalAttention},
                )
            proba = model.predict(X, verbose=0)[0]
            proba_accum += float(f["weight"]) * proba

        s = proba_accum.sum()
        return proba_accum / s if s > 0 else proba_accum

    # === Ensemble weight optimization ===
    def optimize_ensemble_weights(
        oof_list: list[np.ndarray], y_true: np.ndarray, trials=1024, seed=42
    ):
        rng = np.random.default_rng(seed)

        def score_w(w):
            w = np.maximum(w, 0)
            w = w / max(w.sum(), 1e-12)
            p = sum(w[i] * oof_list[i] for i in range(len(oof_list)))
            y_pred = np.argmax(p, axis=1)
            bin_f1 = f1_score(
                (y_true <= 7).astype(int), (y_pred <= 7).astype(int), zero_division=0.0
            )
            macro_f1 = f1_score(
                np.where(y_true <= 7, y_true, 99),
                np.where(y_pred <= 7, y_pred, 99),
                average="macro",
                zero_division=0.0,
            )
            return 0.5 * (bin_f1 + macro_f1)

        best = (-1, None)
        for _ in range(trials):
            w = rng.random(len(oof_list))
            s = score_w(w)
            if s > best[0]:
                best = (s, w / w.sum())
        return best  # (score, weights)

    # === Torch inference function ===
    def predict_torch(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
        # Note: demographics is not used in DL model (frame-level features only)
        _ = demographics  # Not used but kept for API consistency
        bundle_path = os.path.join(DLConfig.TORCH_OUT_DIR, DLConfig.BUNDLE_NAME)
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"Torch bundle not found: {bundle_path}")
        bundle = joblib.load(bundle_path)
        pad_len = bundle["pad_len"]
        feat_order = bundle["feature_order"]

        # frame features -> standardize -> pad
        frame_df = FRAME_CACHE.get(sequence)
        # 列順を合わせる（訓練時の順）
        frame_df = frame_df.reindex(columns=feat_order).fillna(0)
        # fold ごとに scaler が違う点に注意
        device = "cuda" if torch.cuda.is_available() else "cpu"

        n_classes = len(bundle["reverse_gesture_mapper"])
        proba_accum = np.zeros(n_classes, dtype=np.float64)

        for f in bundle["folds"]:
            stats = f["scaler_stats"]
            x_std = apply_standardize(frame_df, stats).to_numpy(np.float32)
            x_pad, m_pad = pad_and_mask(x_std, pad_len)
            xb = torch.from_numpy(x_pad[None, ...]).to(device)
            mb = torch.from_numpy(m_pad[None, ...]).to(device)

            ckpt = torch.load(f["weight_path"], map_location=device)
            state = ckpt["state_dict"]
            # remove 'module.' prefix if present
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            model = TimeSeriesNet(
                in_ch=ckpt["in_ch"],
                num_classes=ckpt["n_classes"],
                hidden=128,
                dropout=DLConfig.DROPOUT,
            )
            model.load_state_dict(state, strict=True)
            model = model.to(device)
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=DLConfig.AMP):
                prob = torch.softmax(model(xb, mb), dim=1).cpu().numpy()[0]
            proba_accum += f["weight"] * prob

        final_cls = int(np.argmax(proba_accum))
        return bundle["reverse_gesture_mapper"][final_cls]


# === Keras Model Definitions ===
if KERAS_AVAILABLE:
    # Configure Mixed Precision if enabled
    if KerasSpeedConfig.MIXED_PRECISION:
        try:
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy("mixed_float16")
            print("✓ Keras mixed precision enabled (float16 compute / float32 vars)")
        except Exception as e:
            print(f"⚠️ Failed to enable mixed precision: {e}")

    # Progress callback for logging
    class ProgressCallback(keras.callbacks.Callback):
        def __init__(self, fold, out_jsonl):
            super().__init__()
            self.fold = fold
            self.t0 = None
            self.out_jsonl = out_jsonl

        def on_train_begin(self, logs=None):
            self.t0 = time.time()

        def on_epoch_end(self, epoch, logs=None):
            _ = logs  # Not used but required by Keras API
            lr_attr = None
            if hasattr(self.model.optimizer, "lr"):
                try:
                    lr_attr = float(self.model.optimizer.lr.numpy())
                except Exception:
                    pass
            if lr_attr is None and hasattr(self.model.optimizer, "learning_rate"):
                try:
                    lr_attr = float(self.model.optimizer.learning_rate.numpy())
                except Exception:
                    lr_attr = 0.0
            rec = {
                "fold": self.fold,
                "epoch": int(epoch),
                "time_from_start_sec": float(time.time() - self.t0),
                "loss": float(logs.get("loss", 0)),
                "acc": float(logs.get("acc", 0)),
                "val_loss": float(logs.get("val_loss", 0)),
                "val_acc": float(logs.get("val_acc", 0)),
                "lr": lr_attr if lr_attr is not None else 0.0,
            }
            _log_jsonl(self.out_jsonl, rec)

    class KerasTemporalAttention(layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.dense = layers.Dense(1)

        def call(self, h, mask):
            # h: (B, T, D), mask: (B, T)
            logit = self.dense(h)[:, :, 0]  # (B, T)
            min_val = tf.cast(-65504.0, logit.dtype)  # fp16でも安全な最小値
            logit = tf.where(
                tf.equal(mask, 1.0),
                logit,
                tf.fill(tf.shape(logit), min_val),
            )
            w = tf.nn.softmax(logit, axis=1)  # (B, T)
            pooled = tf.matmul(w[:, None, :], h)  # (B, 1, D)
            return tf.squeeze(pooled, axis=1)  # (B, D)

    def build_keras_two_branch(
        input_shape: tuple[int, int], n_classes: int, dropout: float = 0.2
    ) -> keras.Model:
        x_in = layers.Input(shape=input_shape, name="x")  # (L, C)
        m_in = layers.Input(shape=(input_shape[0],), name="mask")  # (L,)

        # IMU/ToF/THM チャネルを分けたければここで split（まずは単一枝でOK→2枝拡張）
        x = x_in

        # Residual SE-CNN stack（簡易版）
        def conv_block(x, ch, k):
            y = layers.Conv1D(ch, k, padding="same")(x)
            y = layers.BatchNormalization()(y)
            y = layers.ReLU()(y)
            y = layers.Conv1D(ch, k, padding="same")(y)
            y = layers.BatchNormalization()(y)
            # Squeeze-Excitation (簡易)
            se = layers.GlobalAveragePooling1D()(y)
            se = layers.Dense(ch // 4, activation="relu")(se)
            se = layers.Dense(ch, activation="sigmoid")(se)
            y = layers.Multiply()([y, layers.Reshape((1, ch))(se)])
            # Residual
            if x.shape[-1] != ch:
                x = layers.Conv1D(ch, 1, padding="same")(x)
            y = layers.Add()([x, y])
            y = layers.ReLU()(y)
            y = layers.MaxPooling1D(2)(y)
            y = layers.Dropout(dropout)(y)
            return y

        h = conv_block(x, 128, 7)
        h = conv_block(h, 256, 5)
        h = conv_block(h, 256, 3)

        # RNN
        h = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(h)
        h = layers.Bidirectional(layers.GRU(128, return_sequences=True))(h)

        # マスクを downsample（MaxPool を3回通したぶん）
        m = m_in
        for _ in range(3):
            m = tf.nn.max_pool1d(m[:, :, None], ksize=2, strides=2, padding="VALID")[
                :, :, 0
            ]
            # 長さが合わない場合のガード
            m = m[:, : tf.shape(h)[1]]

        # Attention
        pooled = KerasTemporalAttention()(h, m)

        # Head
        z = layers.Dense(256, activation="relu")(pooled)
        z = layers.Dropout(dropout)(z)
        z = layers.Dense(128, activation="relu")(z)
        # Use float32 for output layer (important for mixed precision)
        y = layers.Dense(n_classes, activation="softmax", dtype="float32")(z)

        model = keras.Model(inputs=[x_in, m_in], outputs=y)
        opt = keras.optimizers.Adam(learning_rate=KerasConfig.LR)
        model.compile(
            optimizer=opt,
            loss=keras.losses.CategoricalCrossentropy(
                label_smoothing=KerasConfig.LABEL_SMOOTHING
            ),
            metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
            jit_compile=False,  # XLAは環境依存のため既定OFF
        )
        return model

    def make_keras_tensor(
        sequence_pl: pl.DataFrame, stats: dict, pad_len: int, feat_order: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        frame_df = (
            build_frame_features(sequence_pl).reindex(columns=feat_order).fillna(0)
        )
        x_std = apply_standardize(frame_df, stats).to_numpy(np.float32)  # (T, C)
        x_pad, m_pad = pad_and_mask(x_std, pad_len)  # (L, C), (L,)
        return x_pad, m_pad


# === Save model bundle helper ===
def save_model_bundle(
    models, X_train, cv_scores, output_dir: str, filename: str, tof_grid_map=None
) -> str:
    # Normalize fold weights（ゼロ除算対策）
    denom = max(float(np.sum(cv_scores)), 1e-12)
    fold_weights = np.array(cv_scores) / denom

    model_data = {
        "models": models,
        "feature_names": list(X_train.columns),
        "gesture_mapper": GESTURE_MAPPER,
        "reverse_gesture_mapper": REVERSE_GESTURE_MAPPER,
        "cv_scores": cv_scores,
        "fold_weights": fold_weights.tolist(),
        "mean_cv_score": float(np.mean(cv_scores)) if len(cv_scores) else 0.0,
        "config": {
            "n_folds": Config.N_FOLDS,
            "seed": Config.SEED,
            "lgbm_params": Config.LGBM_PARAMS,
        },
        # Modality metadata
        "modalities": {"imu": True, "tof_possible": True, "thm_possible": True},
        "col_patterns": {
            "tof_prefixes": list(TOF_PREFIXES),
            "thm_prefixes": list(THM_PREFIXES),
        },
        "tof_grid_map": tof_grid_map,  # Will be None if ToF columns can't be mapped to grid
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    joblib.dump(model_data, out_path)
    print(f"✓ Models saved to {out_path}")
    print(f"✓ File size: {os.path.getsize(out_path) / 1024 / 1024:.2f} MB")
    return out_path


# === Align helper for inference (use training-time columns in the same order) ===
def align_features_for_inference(
    result_df: pd.DataFrame, feature_names: list
) -> pd.DataFrame:
    for col in feature_names:
        if col not in result_df.columns:
            result_df[col] = 0
    # extra cols are dropped by selecting in order
    result_df = result_df[feature_names].fillna(0)
    # Convert to float32 for memory efficiency and consistency
    return result_df.astype(np.float32)


def extract_statistical_features(data: np.ndarray, prefix: str) -> dict:
    """Extract statistical features from 1D time series."""
    features = {}

    # Basic statistics
    features[f"{prefix}_mean"] = np.mean(data)
    features[f"{prefix}_std"] = np.std(data)
    features[f"{prefix}_var"] = np.var(data)
    features[f"{prefix}_min"] = np.min(data)
    features[f"{prefix}_max"] = np.max(data)
    features[f"{prefix}_median"] = np.median(data)
    features[f"{prefix}_q25"] = np.percentile(data, 25)
    features[f"{prefix}_q75"] = np.percentile(data, 75)
    features[f"{prefix}_iqr"] = features[f"{prefix}_q75"] - features[f"{prefix}_q25"]
    features[f"{prefix}_range"] = features[f"{prefix}_max"] - features[f"{prefix}_min"]

    # Boundary features
    features[f"{prefix}_first"] = data[0] if len(data) > 0 else 0
    features[f"{prefix}_last"] = data[-1] if len(data) > 0 else 0
    features[f"{prefix}_delta"] = (
        features[f"{prefix}_last"] - features[f"{prefix}_first"]
    )

    # Higher order moments
    if len(data) > 1 and np.std(data) > 1e-8:
        features[f"{prefix}_skew"] = pd.Series(data).skew()
        features[f"{prefix}_kurt"] = pd.Series(data).kurtosis()
    else:
        features[f"{prefix}_skew"] = 0
        features[f"{prefix}_kurt"] = 0

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


def extract_features(
    sequence: pl.DataFrame, demographics: pl.DataFrame
) -> pd.DataFrame:
    """Extract features from IMU sequence."""
    # Convert to pandas
    seq_df = sequence.to_pandas()
    demo_df = demographics.to_pandas()

    # Infer sampling rate
    dt, fs = infer_dt_and_fs(seq_df)

    # Ensure acc columns exist (pad with zeros if missing)
    for c in Config.ACC_COLS:
        if c not in seq_df.columns:
            seq_df[c] = 0.0

    # Handle missing values
    acc_data = seq_df[Config.ACC_COLS].copy()
    acc_data = acc_data.ffill().bfill().fillna(0)

    # Build full quaternion array (handles missing columns)
    available_rot_cols = [col for col in Config.ROT_COLS if col in seq_df.columns]
    if len(available_rot_cols) > 0:
        rot_data_clean = build_full_quaternion(seq_df, available_rot_cols)
        rot_data_clean = handle_quaternion_missing_values(rot_data_clean)
        rot_data_clean = fix_quaternion_sign(rot_data_clean)
    else:
        # No rotation data available - use identity quaternions
        rot_data_clean = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(seq_df), 1))

    # Compute world acceleration
    world_acc_data = compute_world_acceleration(acc_data.values, rot_data_clean)

    # Compute linear acceleration (gravity removed)
    linear_acc_data = compute_linear_acceleration(world_acc_data, fs=fs)

    # Compute angular velocity
    angular_velocity = compute_angular_velocity(rot_data_clean, dt=dt)

    # Initialize features
    features = {}

    # Sequence metadata
    features["sequence_length"] = len(seq_df)

    # Demographics features with safe numeric conversion
    if len(demo_df) > 0:
        demo_row = demo_df.iloc[0]

        def _to_num(x, default=0.0):
            """Safely convert value to numeric."""
            try:
                return float(x)
            except Exception:
                return default

        features["age"] = _to_num(demo_row.get("age", 0))
        features["adult_child"] = _to_num(demo_row.get("adult_child", 0))
        features["sex"] = _to_num(demo_row.get("sex", 0))
        features["handedness"] = _to_num(demo_row.get("handedness", 0))
        features["height_cm"] = _to_num(demo_row.get("height_cm", 0))
        features["shoulder_to_wrist_cm"] = _to_num(
            demo_row.get("shoulder_to_wrist_cm", 0)
        )
        features["elbow_to_wrist_cm"] = _to_num(demo_row.get("elbow_to_wrist_cm", 0))
    else:
        # When no demographics data, still create keys with zero values for consistency
        for k in [
            "age",
            "adult_child",
            "sex",
            "handedness",
            "height_cm",
            "shoulder_to_wrist_cm",
            "elbow_to_wrist_cm",
        ]:
            features[k] = 0.0

    # Extract statistical features for each axis
    for i, axis in enumerate(["x", "y", "z"]):
        # Device acceleration
        features.update(
            extract_statistical_features(acc_data.values[:, i], f"acc_{axis}")
        )
        # World acceleration
        features.update(
            extract_statistical_features(world_acc_data[:, i], f"world_acc_{axis}")
        )
        # Linear acceleration (gravity removed)
        features.update(
            extract_statistical_features(linear_acc_data[:, i], f"linear_acc_{axis}")
        )
        # Angular velocity
        if i < angular_velocity.shape[1]:
            features.update(
                extract_statistical_features(
                    angular_velocity[:, i], f"angular_vel_{axis}"
                )
            )

    # Rotation features
    for i, comp in enumerate(["w", "x", "y", "z"]):
        features.update(
            extract_statistical_features(rot_data_clean[:, i], f"rot_{comp}")
        )

    # Magnitude features
    acc_magnitude = np.linalg.norm(acc_data.values, axis=1)
    world_acc_magnitude = np.linalg.norm(world_acc_data, axis=1)
    linear_acc_magnitude = np.linalg.norm(linear_acc_data, axis=1)
    angular_vel_magnitude = np.linalg.norm(angular_velocity, axis=1)

    features.update(extract_statistical_features(acc_magnitude, "acc_magnitude"))
    features.update(
        extract_statistical_features(world_acc_magnitude, "world_acc_magnitude")
    )
    features.update(
        extract_statistical_features(linear_acc_magnitude, "linear_acc_magnitude")
    )
    features.update(
        extract_statistical_features(angular_vel_magnitude, "angular_vel_magnitude")
    )

    # Difference between device and world acceleration
    acc_world_diff = acc_magnitude - world_acc_magnitude
    features.update(extract_statistical_features(acc_world_diff, "acc_world_diff"))

    # Add jerk features for different acceleration types
    features.update(extract_jerk_features(acc_data.values, prefix="acc_jerk", dt=dt))
    features.update(
        extract_jerk_features(world_acc_data, prefix="world_acc_jerk", dt=dt)
    )
    features.update(
        extract_jerk_features(linear_acc_data, prefix="linear_acc_jerk", dt=dt)
    )

    # Angular velocity energy (rotation energy)
    angular_energy = np.mean(angular_velocity**2, axis=1)
    features.update(extract_statistical_features(angular_energy, "angular_energy"))

    # Euler angles features
    try:
        # Convert quaternions to Euler angles
        rot_scipy = rot_data_clean[
            :, [1, 2, 3, 0]
        ]  # Convert to scipy format (x, y, z, w)
        r = R.from_quat(rot_scipy)
        euler_angles = r.as_euler("xyz", degrees=False)  # roll, pitch, yaw in radians

        # Unwrap angles to avoid discontinuities at ±π
        euler_angles_unwrapped = np.unwrap(euler_angles, axis=0)

        # Extract statistical features for each Euler angle
        for i, angle_name in enumerate(["roll", "pitch", "yaw"]):
            # Linear statistics on unwrapped angles
            features.update(
                extract_statistical_features(
                    euler_angles_unwrapped[:, i], f"euler_{angle_name}"
                )
            )

            # Circular statistics on original angles
            s = np.sin(euler_angles[:, i])
            c = np.cos(euler_angles[:, i])
            features[f"euler_{angle_name}_mean_circ"] = np.arctan2(
                np.mean(s), np.mean(c)
            )
            features[f"euler_{angle_name}_R"] = np.hypot(
                np.mean(s), np.mean(c)
            )  # Concentration
    except Exception:
        # If conversion fails, add zero features
        for angle_name in ["roll", "pitch", "yaw"]:
            for stat in [
                "mean",
                "std",
                "var",
                "min",
                "max",
                "median",
                "q25",
                "q75",
                "iqr",
                "range",
                "first",
                "last",
                "delta",
                "skew",
                "kurt",
                "diff_mean",
                "diff_std",
                "n_changes",
                "seg1_mean",
                "seg1_std",
                "seg2_mean",
                "seg2_std",
                "seg3_mean",
                "seg3_std",
                "seg1_to_seg2",
                "seg2_to_seg3",
            ]:
                features[f"euler_{angle_name}_{stat}"] = 0

    # Correlation features
    data_dict = {
        "world_acc_x": world_acc_data[:, 0],
        "world_acc_y": world_acc_data[:, 1],
        "world_acc_z": world_acc_data[:, 2],
        "linear_acc_mag": linear_acc_magnitude,
        "angular_vel_mag": angular_vel_magnitude,
    }
    features.update(extract_correlation_features(data_dict))

    # Peak features for key signals
    features.update(extract_peak_features(acc_magnitude, prefix="acc_mag_peak", fs=fs))
    features.update(
        extract_peak_features(world_acc_magnitude, prefix="world_acc_mag_peak", fs=fs)
    )
    features.update(
        extract_peak_features(linear_acc_magnitude, prefix="linear_acc_mag_peak", fs=fs)
    )
    features.update(
        extract_peak_features(
            angular_vel_magnitude, prefix="angular_vel_mag_peak", fs=fs
        )
    )

    # Autocorrelation features
    features.update(
        extract_autocorrelation_features(acc_magnitude, prefix="acc_mag_autocorr")
    )
    features.update(
        extract_autocorrelation_features(
            world_acc_magnitude, prefix="world_acc_mag_autocorr"
        )
    )
    features.update(
        extract_autocorrelation_features(
            linear_acc_magnitude, prefix="linear_acc_mag_autocorr"
        )
    )
    features.update(
        extract_autocorrelation_features(
            angular_vel_magnitude, prefix="angular_vel_mag_autocorr"
        )
    )

    # Gradient histogram features
    features.update(
        extract_gradient_histogram(acc_magnitude, n_bins=10, prefix="acc_mag_grad_hist")
    )
    features.update(
        extract_gradient_histogram(
            world_acc_magnitude, n_bins=10, prefix="world_acc_mag_grad_hist"
        )
    )
    features.update(
        extract_gradient_histogram(
            linear_acc_magnitude, n_bins=10, prefix="linear_acc_mag_grad_hist"
        )
    )
    features.update(
        extract_gradient_histogram(
            angular_vel_magnitude, n_bins=10, prefix="angular_vel_mag_grad_hist"
        )
    )

    # Frequency domain features
    features.update(
        extract_frequency_features(
            acc_magnitude, fs=fs, prefix="acc_mag_freq", compute_zcr=False
        )
    )
    features.update(
        extract_frequency_features(
            world_acc_magnitude, fs=fs, prefix="world_acc_mag_freq", compute_zcr=False
        )
    )
    features.update(
        extract_frequency_features(
            linear_acc_magnitude, fs=fs, prefix="linear_acc_mag_freq", compute_zcr=False
        )
    )
    features.update(
        extract_frequency_features(
            angular_vel_magnitude,
            fs=fs,
            prefix="angular_vel_mag_freq",
            compute_zcr=False,
        )
    )

    # Also add frequency features for individual axes
    for i, axis in enumerate(["x", "y", "z"]):
        if i < world_acc_data.shape[1]:
            features.update(
                extract_frequency_features(
                    world_acc_data[:, i], fs=fs, prefix=f"world_acc_{axis}_freq"
                )
            )

    # Pose invariant features
    features.update(extract_pose_invariant_features(world_acc_data))

    # Temporal pyramid features for key signals
    features.update(
        extract_temporal_pyramid_features(
            linear_acc_magnitude, prefix="linear_acc_mag_pyramid"
        )
    )
    features.update(
        extract_temporal_pyramid_features(
            world_acc_magnitude, prefix="world_acc_mag_pyramid"
        )
    )

    # Tail window features for key signals
    features.update(
        extract_tail_window_features(linear_acc_magnitude, prefix="linear_acc_mag_tail")
    )
    features.update(
        extract_tail_window_features(world_acc_magnitude, prefix="world_acc_mag_tail")
    )
    features.update(
        extract_tail_window_features(
            angular_vel_magnitude, prefix="angular_vel_mag_tail"
        )
    )

    # === ToF and Thermal Features ===
    # Detect modalities once
    _, mod_cols = detect_modalities(seq_df)

    # Extract ToF features
    tof_features = extract_tof_features(seq_df, fs, tof_cols=mod_cols["tof"])
    features.update(tof_features)

    # Extract Thermal features
    thm_features = extract_thm_features(seq_df, fs, thm_cols=mod_cols["thm"])
    features.update(thm_features)

    # Extract cross-modality features
    # First prepare IMU series for cross-modality
    imu_series = {
        "linear_acc_mag": linear_acc_magnitude,
        "angular_vel_mag": angular_vel_magnitude,
    }

    # Get ToF and THM aggregates for cross-modality (if they exist)
    tof_agg = None
    if mod_cols["tof"]:
        tof_agg = tof_frame_aggregates(seq_df, mod_cols["tof"])

    thm_agg = None
    if mod_cols["thm"]:
        thm_agg = thermal_frame_aggregates(seq_df, mod_cols["thm"])

    # Extract cross-modality features
    xmod_feats = extract_xmod_features_for_union(imu_series, tof_agg, thm_agg)
    features.update(xmod_feats)

    # Add IMU modality flag (always present in this dataset)
    features["mod_present_imu"] = 1

    # Convert to DataFrame
    result_df = pd.DataFrame([features])
    result_df = result_df.fillna(0)

    return result_df


print("✓ Feature extraction function defined")


def validate_features(
    features_df: pd.DataFrame, feature_names: list = None, verbose: bool = True
) -> bool:
    """Validate extracted features for data integrity."""
    is_valid = True

    # Check 1: No NaN or Inf values
    if features_df.isnull().any().any():
        if verbose:
            print("❌ Warning: NaN values found in features")
            print(features_df.columns[features_df.isnull().any()].tolist())
        is_valid = False

    if np.isinf(features_df.values).any():
        if verbose:
            print("❌ Warning: Inf values found in features")
        is_valid = False

    # Check 2: IMU modality should always be present
    if "mod_present_imu" in features_df.columns:
        if not (features_df["mod_present_imu"] == 1).all():
            if verbose:
                print("❌ Warning: IMU modality not present in some samples")
            is_valid = False

    # Check 3: Feature names match expected (if provided)
    if feature_names is not None:
        missing_cols = set(feature_names) - set(features_df.columns)
        if missing_cols:
            if verbose:
                print(f"❌ Warning: Missing expected features: {missing_cols}")
            is_valid = False

    # Check 4: Modality consistency
    tof_cols = [
        c
        for c in features_df.columns
        if c.startswith("tof_") and c != "mod_present_tof"
    ]
    thm_cols = [
        c
        for c in features_df.columns
        if c.startswith("thm_") and c != "mod_present_thm"
    ]

    for idx in range(len(features_df)):
        # If mod_present_tof is 0, all ToF features should be 0
        if (
            "mod_present_tof" in features_df.columns
            and features_df.iloc[idx]["mod_present_tof"] == 0
        ):
            if tof_cols and (features_df.iloc[idx][tof_cols] != 0).any():
                if verbose and idx == 0:  # Only warn once
                    print("❌ Warning: ToF features non-zero when modality is absent")
                is_valid = False

        # Same for thermal
        if (
            "mod_present_thm" in features_df.columns
            and features_df.iloc[idx]["mod_present_thm"] == 0
        ):
            if thm_cols and (features_df.iloc[idx][thm_cols] != 0).any():
                if verbose and idx == 0:  # Only warn once
                    print(
                        "❌ Warning: Thermal features non-zero when modality is absent"
                    )
                is_valid = False

    if verbose and is_valid:
        print("✓ Feature validation passed")

    return is_valid


def _bool_env(name, default=0):
    try:
        return bool(int(os.getenv(name, str(default))))
    except Exception:
        return bool(default)


def build_run_plan(n_folds=Config.N_FOLDS):
    """再利用／再開／学習の方針を一括決定してログに使える dict を返す。"""
    plan = {}

    # 1) features cache
    feat_path = PATHS.FEATURES_CACHE
    plan["features"] = {
        "path": feat_path,
        "exists": feat_path is not None and os.path.exists(feat_path),
        "action": "reuse" if (feat_path and os.path.exists(feat_path)) else "build",
        "reason": "cache found"
        if (feat_path and os.path.exists(feat_path))
        else "no cache",
    }

    # 2) LGBM（pretrained bundle / fold再開 / 学習）
    lgbm_bundle = PATHS.LGBM_BUNDLE  # 一元化
    lgbm_bundle_exists = bool(lgbm_bundle) and os.path.exists(lgbm_bundle)
    fold_paths = [
        os.path.join(
            CheckpointConfig.CKPT_DIR, CheckpointConfig.LGBM_FOLD_TMPL.format(i)
        )
        for i in range(n_folds)
    ]
    some_fold_exists = any(os.path.exists(p) for p in fold_paths)
    if _bool_env("FORCE_TRAIN_LGBM", 0):
        lgbm_action, reason = "train", "FORCE_TRAIN_LGBM=1"
    elif lgbm_bundle_exists:
        lgbm_action, reason = "reuse", "pretrained bundle found"
    elif some_fold_exists:
        lgbm_action, reason = "resume", "found fold checkpoints"
    else:
        lgbm_action, reason = "train", "no bundle/checkpoints"
    plan["lgbm"] = {
        "bundle_path": lgbm_bundle,
        "bundle_exists": lgbm_bundle_exists,
        "fold_paths": fold_paths,
        "has_fold_ckpt": some_fold_exists,
        "action": lgbm_action,
        "reason": reason,
    }

    # 3) Torch
    torch_bundle = PATHS.TORCH_BUNDLE
    torch_bundle_exists = bool(torch_bundle) and os.path.exists(torch_bundle)
    torch_best_paths = [
        os.path.join(DLConfig.TORCH_OUT_DIR, DLConfig.WEIGHT_TMPL.format(i))
        for i in range(DLConfig.N_FOLDS)
    ]
    torch_has_best = any(os.path.exists(p) for p in torch_best_paths)
    if (not TORCH_AVAILABLE) or (not Config.USE_TORCH):
        torch_action, reason = "skip", "torch unavailable or disabled"
    elif _bool_env("FORCE_TRAIN_TORCH", 0):
        torch_action, reason = "train", "FORCE_TRAIN_TORCH=1"
    elif torch_bundle_exists:
        torch_action, reason = "reuse", "bundle found"
    elif torch_has_best:
        torch_action, reason = "resume", "found per-fold best weights"
    else:
        torch_action, reason = "train", "no bundle/weights"
    plan["torch"] = {
        "bundle_path": torch_bundle,
        "bundle_exists": torch_bundle_exists,
        "has_best": torch_has_best,
        "action": torch_action,
        "reason": reason,
    }

    # 4) Keras
    keras_bundle = PATHS.KERAS_BUNDLE
    keras_bundle_exists = bool(keras_bundle) and os.path.exists(keras_bundle)
    keras_best_paths = [
        os.path.join(KerasConfig.OUT_DIR, KerasConfig.WEIGHT_TMPL.format(i))
        for i in range(KerasConfig.N_FOLDS)
    ]
    keras_has_best = any(os.path.exists(p) for p in keras_best_paths)
    if (not KERAS_AVAILABLE) or (not Config.USE_KERAS):
        keras_action, reason = "skip", "keras unavailable or disabled"
    elif _bool_env("FORCE_TRAIN_KERAS", 0):
        keras_action, reason = "train", "FORCE_TRAIN_KERAS=1"
    elif keras_bundle_exists:
        keras_action, reason = "reuse", "bundle found"
    elif keras_has_best:
        keras_action, reason = "resume", "found per-fold best weights"
    else:
        keras_action, reason = "train", "no bundle/weights"
    plan["keras"] = {
        "bundle_path": keras_bundle,
        "bundle_exists": keras_bundle_exists,
        "has_best": keras_has_best,
        "action": keras_action,
        "reason": reason,
    }
    return plan


def print_plan(plan):
    def _line(name, entry):
        icon = {
            "reuse": "✓",
            "resume": "↻",
            "train": "🛠",
            "build": "🛠",
            "skip": "⏭",
        }.get(entry["action"], "?")
        print(
            f"[PLAN] {name:7s}: {icon} {entry['action']:6s} — {entry.get('reason', '')}"
        )
        if "path" in entry:
            print(
                f"        path   : {entry['path']} (exists={entry.get('exists', False)})"
            )
        if "bundle_path" in entry:
            print(
                f"        bundle : {entry['bundle_path']} (exists={entry.get('bundle_exists', False)})"
            )

    print("========== RUN PLAN ==========")
    _line("features", plan["features"])
    _line("lgbm", plan["lgbm"])
    _line("torch", plan["torch"])
    _line("keras", plan["keras"])
    print("==============================")


# ============================================================================


# === MAIN ENTRY POINT ===
def main():
    """Main entry point for training and inference."""
    global RUNTIME_MODEL_PATH

    # === NEW: CheckPointPaths を使って実行プランを固める ===
    plan = build_run_plan(n_folds=Config.N_FOLDS)
    print_plan(plan)

    # features cache の最終採用パスを全体設定へ反映（以降の処理はこのパスを見る）
    CheckpointConfig.FEATURES_CACHE = plan["features"]["path"]
    os.makedirs(os.path.dirname(CheckpointConfig.FEATURES_CACHE), exist_ok=True)

    # これ以降のフラグは plan に準拠（resume も学習扱いでOK：内部で fold毎にスキップ済）
    RUN_LGBM_TRAINING = plan["lgbm"]["action"] in ("train", "resume")
    RUN_TORCH_TRAINING = plan["torch"]["action"] in ("train", "resume")
    RUN_KERAS_TRAINING = plan["keras"]["action"] in ("train", "resume")
    RUN_ANY_TRAINING = RUN_LGBM_TRAINING or RUN_TORCH_TRAINING or RUN_KERAS_TRAINING

    # ここで先に決め打ち（後で学習したら上書き）
    if plan["lgbm"]["action"] == "reuse":
        RUNTIME_MODEL_PATH = plan["lgbm"]["bundle_path"]
    else:
        RUNTIME_MODEL_PATH = os.path.join(Config.OUTPUT_PATH, Config.MODEL_FILENAME)

    if RUN_ANY_TRAINING:
        # ------------------ LOAD TRAIN DATA ------------------
        print("Loading training data...")
        train_df = pl.read_csv(Config.TRAIN_PATH)
        train_demographics = pl.read_csv(Config.TRAIN_DEMOGRAPHICS_PATH)

        print(f"✓ Train shape: {train_df.shape}")
        print(f"✓ Demographics shape: {train_demographics.shape}")

        # Get all available columns (including ToF/THM if present)
        base_cols = ["sequence_id", "subject", "phase", "gesture"]

        # Detect all sensor columns in training data
        all_cols = train_df.columns
        sensor_cols = []

        # Add IMU columns
        sensor_cols.extend([c for c in all_cols if c in Config.ACC_COLS])
        sensor_cols.extend([c for c in all_cols if c in Config.ROT_COLS])

        # Add ToF columns if present
        tof_cols_found = _cols_startswith(all_cols, TOF_PREFIXES)
        tof_grid_map = None
        if tof_cols_found:
            print(f"✓ Found {len(tof_cols_found)} ToF columns")
            sensor_cols.extend(tof_cols_found)
            # Try to build ToF grid map for metadata
            tof_grid_map = build_tof_grid_index(tof_cols_found)
            if tof_grid_map:
                print("✓ ToF columns can be mapped to 8x8 grid")
        else:
            print("ℹ️ No ToF columns found in training data")

        # Add Thermal columns if present
        thm_cols_found = _cols_startswith(all_cols, THM_PREFIXES)
        if thm_cols_found:
            print(f"✓ Found {len(thm_cols_found)} Thermal columns")
            sensor_cols.extend(thm_cols_found)
        else:
            print("ℹ️ No Thermal columns found in training data")

        # Combine all columns
        cols_to_select = base_cols + sensor_cols
        print(f"✓ Using {len(sensor_cols)} sensor columns total")

        # ------------------ FEATURE EXTRACTION with CACHE ------------------
        print("Extracting features for training sequences (with cache)...")

        features_cache_path = CheckpointConfig.FEATURES_CACHE
        cached = CheckpointConfig.RESUME and os.path.exists(features_cache_path)

        if cached:
            cache = joblib.load(features_cache_path)
            X_train = cache["X_train"]
            y_train = cache["y_train"]
            subjects = cache["subjects"]
            tof_grid_map = cache.get("tof_grid_map", None)
            print(
                f"✓ Loaded cached features: {X_train.shape} from {features_cache_path}"
            )
        else:
            train_features_list = []
            train_labels = []
            train_subjects = []

            unique_sequences = train_df["sequence_id"].unique()
            n_sequences = len(unique_sequences)
            print(f"Total sequences to process: {n_sequences}")

            train_sequences = train_df.select(pl.col(cols_to_select)).group_by(
                "sequence_id", maintain_order=True
            )

            t0 = time.time()
            for i, (_, sequence_data) in enumerate(train_sequences):
                subject_id = sequence_data["subject"][0]
                subject_demographics = train_demographics.filter(
                    pl.col("subject") == subject_id
                )

                features = extract_features(sequence_data, subject_demographics)
                train_features_list.append(features)

                gesture = sequence_data["gesture"][0]
                label = GESTURE_MAPPER[gesture]
                train_labels.append(label)
                train_subjects.append(subject_id)

                # Enhanced progress logging
                done = i + 1
                if (
                    (done % LogConfig.FEATURE_LOG_EVERY == 0)
                    or (done == 1)
                    or (done == n_sequences)
                ):
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 1e-9)
                    eta = (n_sequences - done) / max(rate, 1e-9)
                    print(
                        f"Processing sequence {done}/{n_sequences}  "
                        f"{rate:.1f}/s  ETA {eta / 60:.1f} min",
                        flush=True,
                    )

            assert len(train_features_list) == n_sequences, (
                f"Feature extraction failed: {len(train_features_list)} != {n_sequences}"
            )
            print(f"✓ Successfully processed all {n_sequences} sequences")

            X_train = pd.concat(train_features_list, ignore_index=True)
            y_train = np.array(train_labels)
            subjects = np.array(train_subjects)

            print(f"✓ Features extracted: {X_train.shape}")
            print(f"✓ Number of classes: {len(np.unique(y_train))}")

            # Cleaning / standardize dtype
            print("Cleaning and standardizing features...")
            X_train = X_train.reindex(columns=sorted(X_train.columns))
            X_train.replace([np.inf, -np.inf], 0, inplace=True)
            X_train.fillna(0, inplace=True)
            X_train = X_train.astype(np.float32)

            # Validate features
            print("Validating extracted features...")
            validate_features(X_train, verbose=True)

            joblib.dump(
                {
                    "X_train": X_train,
                    "y_train": y_train,
                    "subjects": subjects,
                    "tof_grid_map": tof_grid_map,
                },
                features_cache_path,
            )
            print(f"✓ Saved features cache to {features_cache_path}")

        # ------------------ CV TRAINING with CHECKPOINT ------------------
        if RUN_LGBM_TRAINING:
            print("Training LightGBM models with cross-validation (with checkpoint)...")

            cv = StratifiedGroupKFold(
                n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
            )
            models, cv_scores = [], []

            # OOF predictions for LGBM
            n_classes = len(GESTURE_MAPPER)
            oof_lgbm = np.zeros((len(X_train), n_classes), dtype=np.float32)

            # 既存状態の読み込み
            lgbm_state = _load_json(
                CheckpointConfig.LGBM_STATE_JSON,
                default={"model_paths": {}, "cv_scores": {}, "completed_folds": []},
            )
            completed = set(int(k) for k in lgbm_state.get("model_paths", {}).keys())

            for fold, (train_idx, val_idx) in enumerate(
                cv.split(X_train, y_train, subjects)
            ):
                model_path = os.path.join(
                    CheckpointConfig.CKPT_DIR,
                    CheckpointConfig.LGBM_FOLD_TMPL.format(fold),
                )

                if (
                    CheckpointConfig.RESUME
                    and (fold in completed)
                    and os.path.exists(model_path)
                ):
                    print(f"↻ Resuming: loading fold {fold} model from {model_path}")
                    model = joblib.load(model_path)
                    models.append(model)
                    cv_scores.append(float(lgbm_state["cv_scores"].get(str(fold), 0.0)))
                    continue

                print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")
                X_fold_train = (
                    X_train.iloc[train_idx].reset_index(drop=True).astype(np.float32)
                )
                X_fold_val = (
                    X_train.iloc[val_idx].reset_index(drop=True).astype(np.float32)
                )
                y_fold_train = y_train[train_idx]
                y_fold_val = y_train[val_idx]

                if Config.USE_MODALITY_DROPOUT:
                    print(
                        f"Applying modality dropout with p={Config.MODALITY_DROPOUT_PROB}"
                    )
                    X_fold_train = apply_modality_dropout(
                        X_fold_train,
                        dropout_prob=Config.MODALITY_DROPOUT_PROB,
                        seed=Config.SEED + fold,
                    )

                print(f"Train size: {len(X_fold_train)}, Val size: {len(X_fold_val)}")
                model = LGBMClassifier(**Config.LGBM_PARAMS)

                model.fit(
                    X_fold_train,
                    y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    eval_names=["valid"],
                    eval_metric="multi_logloss",
                    callbacks=[
                        log_evaluation(period=50),
                        early_stopping(stopping_rounds=100, verbose=True),
                    ],
                )

                # Fold 保存（checkpoint）
                joblib.dump(model, model_path)
                print(f"✓ Saved fold {fold} model to {model_path}")

                models.append(model)
                val_preds = model.predict(X_fold_val)

                # Get probabilities for OOF
                val_proba = model.predict_proba(X_fold_val)  # (n_val, n_classes_local)
                # Map local class IDs back to global IDs
                val_proba_full = np.zeros(
                    (len(X_fold_val), n_classes), dtype=np.float32
                )
                for local_j, cls_id in enumerate(model.classes_):
                    val_proba_full[:, int(cls_id)] = val_proba[:, local_j]
                oof_lgbm[val_idx] = val_proba_full

                binary_f1 = f1_score(
                    np.where(y_fold_val <= 7, 1, 0),
                    np.where(val_preds <= 7, 1, 0),
                    zero_division=0.0,
                )
                macro_f1 = f1_score(
                    np.where(y_fold_val <= 7, y_fold_val, 99),
                    np.where(val_preds <= 7, val_preds, 99),
                    average="macro",
                    zero_division=0.0,
                )
                score = 0.5 * (binary_f1 + macro_f1)
                cv_scores.append(score)
                print(
                    f"Fold {fold + 1} Score: {score:.4f} (Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f})"
                )

                # JSON 状態更新
                lgbm_state["model_paths"][str(fold)] = model_path
                lgbm_state["cv_scores"][str(fold)] = float(score)
                lgbm_state["completed_folds"] = sorted(
                    list(set(lgbm_state["model_paths"].keys())), key=int
                )
                _save_json(CheckpointConfig.LGBM_STATE_JSON, lgbm_state)

            print("\n✓ Cross-validation complete!")
            print(
                f"Overall CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}"
            )

            # Save LGBM OOF predictions
            os.makedirs(os.path.join(Config.OUTPUT_PATH, "oof"), exist_ok=True)
            np.save(
                os.path.join(Config.OUTPUT_PATH, "oof", "oof_lgbm_proba.npy"), oof_lgbm
            )
            np.save(os.path.join(Config.OUTPUT_PATH, "oof", "y_true.npy"), y_train)

            # ------------------ SAVE MODEL BUNDLE ------------------
            RUNTIME_MODEL_PATH = save_model_bundle(
                models=models,
                X_train=X_train,
                cv_scores=cv_scores,
                output_dir=Config.OUTPUT_PATH,
                filename=Config.MODEL_FILENAME,
                tof_grid_map=tof_grid_map,  # Pass ToF grid mapping if available
            )

            # （任意）特徴量重要度の保存（LGBM学習を実行したときのみ）
            if len(models) > 0:
                feature_importance = pd.DataFrame(
                    {
                        "feature": X_train.columns,
                        "importance": np.mean(
                            [m.feature_importances_ for m in models], axis=0
                        ),
                    }
                ).sort_values("importance", ascending=False)
                print("\nTop 20 Most Important Features:")
                print(feature_importance.head(20))
                feature_importance.to_csv(
                    os.path.join(Config.OUTPUT_PATH, "feature_importance.csv"),
                    index=False,
                )
            print("\n✓ LGBM Training complete!")
    else:
        if plan["lgbm"]["action"] == "reuse":
            print(
                "✓ Skipping LGBM training. Using pretrained model at:",
                Config.MODEL_PATH,
            )
            RUNTIME_MODEL_PATH = Config.MODEL_PATH
        else:
            # plan 上ここに来ない想定だが、保険で明示エラーにする
            raise FileNotFoundError(
                f"No LGBM bundle found. Plan decided '{plan['lgbm']['action']}'. "
                "Set Config.MODEL_PATH=None to train or provide a valid bundle path."
            )

    # ==== Frame feature store 構築 (DL用) ====
    if RUN_TORCH_TRAINING or RUN_KERAS_TRAINING:
        print("Building frame feature store (if missing)...")
        build_frame_feature_store(train_df, cols_to_select)

    # ==== Torch training (if needed) ====
    if RUN_TORCH_TRAINING:
        print(
            "\nStarting Torch training..."
            if plan["torch"]["action"] == "train"
            else "\nResuming Torch training..."
        )
        train_torch_models(train_df, train_demographics)
        print("✓ Torch training complete")
    elif TORCH_AVAILABLE and Config.USE_TORCH:
        if plan["torch"]["action"] == "reuse":
            print(
                f"✓ Skipping Torch training. Using pretrained bundle at: {EnsembleConfig.TORCH_BUNDLE_PATH}"
            )
        elif plan["torch"]["action"] == "skip":
            print("⏭ Torch disabled/unavailable. Skipping.")
        else:
            print(f"⚠️ Torch bundle not found at: {EnsembleConfig.TORCH_BUNDLE_PATH}")
    elif not TORCH_AVAILABLE:
        msg = "PyTorch is not available. Skipping Torch training."
        if EnsembleConfig.FAIL_IF_TORCH_MISSING:
            raise RuntimeError(msg)
        print("⚠️ " + msg)

    # ==== Keras training (if needed) ====
    if RUN_KERAS_TRAINING:
        print(
            "\nStarting Keras training..."
            if plan["keras"]["action"] == "train"
            else "\nResuming Keras training..."
        )
        train_keras_models(train_df, train_demographics)
        print("✓ Keras training complete")
    elif KERAS_AVAILABLE and Config.USE_KERAS:
        if plan["keras"]["action"] == "reuse":
            print(
                f"✓ Skipping Keras training. Using pretrained bundle at: {EnsembleConfig.KERAS_BUNDLE_PATH}"
            )
        elif plan["keras"]["action"] == "skip":
            print("⏭ Keras disabled/unavailable. Skipping.")
        else:
            print(f"⚠️ Keras bundle not found at: {EnsembleConfig.KERAS_BUNDLE_PATH}")
    else:
        print("ℹ️ Keras not available or disabled. Skipping Keras training.")

    # ==== Ensemble weight optimization (optional) ====
    oof_dir = os.path.join(Config.OUTPUT_PATH, "oof")
    if os.path.exists(oof_dir):
        print("\n=== Optimizing ensemble weights from OOF predictions ===")
        oof_files = {
            "lgbm": os.path.join(oof_dir, "oof_lgbm_proba.npy"),
            "torch": os.path.join(oof_dir, "oof_torch_proba.npy"),
            "keras": os.path.join(oof_dir, "oof_keras_proba.npy"),
        }
        y_true_file = os.path.join(oof_dir, "y_true.npy")

        oof_list = []
        oof_names = []
        for name, path in oof_files.items():
            if os.path.exists(path):
                oof_list.append(np.load(path))
                oof_names.append(name)
                print(f"✓ Loaded OOF predictions from {name}")

        if len(oof_list) >= 2 and os.path.exists(y_true_file):
            y_true = np.load(y_true_file)
            best_score, best_weights = optimize_ensemble_weights(oof_list, y_true)

            print(f"\n✓ Optimized ensemble weights (score={best_score:.4f}):")
            for i, name in enumerate(oof_names):
                print(f"  {name}: {best_weights[i]:.3f}")

            # Save optimized weights
            weights_dict = {
                name: float(best_weights[i]) for i, name in enumerate(oof_names)
            }
            os.makedirs(os.path.join(Config.OUTPUT_PATH, "ensemble"), exist_ok=True)
            _save_json(
                os.path.join(Config.OUTPUT_PATH, "ensemble", "weights.json"),
                weights_dict,
            )
            print("✓ Saved optimized weights to ensemble/weights.json")
        else:
            print("⚠️ Need at least 2 OOF predictions to optimize weights")

    else:
        # ------------------ SKIP ALL TRAINING ------------------
        # No training data needed, but need to set RUNTIME_MODEL_PATH for LGBM
        if not RUN_LGBM_TRAINING:
            path = Config.MODEL_PATH

            if os.path.isdir(path):
                # MODEL_PATH is a directory, try to find the model file
                candidate = os.path.join(path, Config.MODEL_FILENAME)
                if os.path.exists(candidate):
                    RUNTIME_MODEL_PATH = candidate
                    print(f"✓ Found LGBM model file in directory: {RUNTIME_MODEL_PATH}")
                else:
                    raise FileNotFoundError(
                        f"MODEL_PATH is a directory but {Config.MODEL_FILENAME} not found: {candidate}"
                    )
            else:
                # MODEL_PATH is assumed to be a file path
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Specified MODEL_PATH does not exist: {path}\n"
                        f"Upload your pkl to a Kaggle dataset and set the absolute path."
                    )
                RUNTIME_MODEL_PATH = path

            print(
                f"✓ Skipping LGBM training. Using pretrained model at: {RUNTIME_MODEL_PATH}"
            )

            if not os.path.exists(RUNTIME_MODEL_PATH):
                raise FileNotFoundError(
                    f"Specified MODEL_PATH does not exist: {RUNTIME_MODEL_PATH}\n"
                    f"Upload your pkl to a Kaggle dataset and set the absolute path."
                )

        # Check for Torch bundle
        if not RUN_TORCH_TRAINING and TORCH_AVAILABLE and Config.USE_TORCH:
            tbp = EnsembleConfig.TORCH_BUNDLE_PATH
            if os.path.exists(tbp):
                print(f"✓ Using Torch bundle at: {tbp}")
            else:
                print(f"⚠️ Torch bundle not found at: {tbp}")

        # Check for Keras bundle
        if not RUN_KERAS_TRAINING and KERAS_AVAILABLE and Config.USE_KERAS:
            kbp = EnsembleConfig.KERAS_BUNDLE_PATH
            if os.path.exists(kbp):
                print(f"✓ Using Keras bundle at: {kbp}")
            else:
                print(f"⚠️ Keras bundle not found at: {kbp}")

    # # CMI BFRB Detection - Multi-Modal LightGBM Inference

    # === Deferred loading for inference ===
    # Store runtime model path globally for later use
    print(f"✓ Runtime model path set to: {RUNTIME_MODEL_PATH}")


# ====== Inference bundle deferred loading state ======
class _InferState:
    def __init__(self):
        self.lgbm_models = None
        self.feature_names = None
        self.reverse_gesture_mapper = None
        self.fold_weights = None
        self.loaded = False


INFER = _InferState()


def _load_lgbm_bundle(path):
    """Load LGBM model bundle on demand."""
    if INFER.loaded:
        return
    print("Loading LGBM model bundle for inference...")
    model_data = joblib.load(path)
    INFER.lgbm_models = model_data["models"]
    INFER.feature_names = model_data["feature_names"]
    INFER.reverse_gesture_mapper = model_data["reverse_gesture_mapper"]

    if "fold_weights" in model_data and len(model_data["fold_weights"]) == len(
        INFER.lgbm_models
    ):
        INFER.fold_weights = np.array(model_data["fold_weights"])
        print(f"✓ Using fold weights: {INFER.fold_weights}")
    else:
        INFER.fold_weights = np.ones(len(INFER.lgbm_models)) / max(
            len(INFER.lgbm_models), 1
        )
        print("✓ Using equal weights (no fold weights found)")

    print(f"✓ Loaded {len(INFER.lgbm_models)} LGBM models")
    print(f"✓ Number of features: {len(INFER.feature_names)}")
    if "mean_cv_score" in model_data:
        print(f"✓ CV Score (recorded): {model_data['mean_cv_score']:.4f}")

    # Log per-fold classes for debugging
    print(
        "Per-fold classes:",
        [list(getattr(m, "classes_", [])) for m in INFER.lgbm_models],
    )
    INFER.loaded = True


def predict_lgbm_proba(
    sequence: pl.DataFrame, demographics: pl.DataFrame
) -> np.ndarray:
    """Return probability distribution over classes from LGBM models."""
    _load_lgbm_bundle(RUNTIME_MODEL_PATH)
    raw_features = extract_features(sequence, demographics)
    X = align_features_for_inference(raw_features, INFER.feature_names)
    n_classes_global = len(INFER.reverse_gesture_mapper)
    proba_accum = np.zeros(n_classes_global, dtype=np.float64)
    for i, model in enumerate(INFER.lgbm_models):
        proba = model.predict_proba(X)[0]  # (local_n_classes,)
        proba_full = np.zeros(n_classes_global, dtype=np.float64)
        for local_idx, cls_id in enumerate(model.classes_):
            proba_full[int(cls_id)] = proba[local_idx]
        proba_accum += proba_full * float(INFER.fold_weights[i])
    # Normalize (avoid floating point errors)
    s = proba_accum.sum()
    return proba_accum / s if s > 0 else proba_accum


# Global cache for Torch models
_TORCH_RUNTIME = {"bundle": None, "fold_models": []}


def _load_torch_bundle_and_models():
    """Load Torch bundle and optionally cache models in memory."""
    if _TORCH_RUNTIME["bundle"] is not None:
        return
    bundle_path = EnsembleConfig.TORCH_BUNDLE_PATH
    if not (TORCH_AVAILABLE and os.path.exists(bundle_path)):
        msg = f"Torch bundle not found or torch unavailable: {bundle_path}"
        if EnsembleConfig.FAIL_IF_TORCH_MISSING:
            raise FileNotFoundError(msg)
        print("⚠️ " + msg)
        _TORCH_RUNTIME["bundle"] = None
        return
    bundle = joblib.load(bundle_path)
    _TORCH_RUNTIME["bundle"] = bundle
    if EnsembleConfig.LOAD_TORCH_FOLDS_IN_MEMORY and TORCH_AVAILABLE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _TORCH_RUNTIME["fold_models"] = []
        for f in bundle["folds"]:
            ckpt = torch.load(f["weight_path"], map_location=device)
            state = ckpt["state_dict"]
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            model = TimeSeriesNet(
                in_ch=ckpt["in_ch"],
                num_classes=ckpt["n_classes"],
                hidden=128,
                dropout=DLConfig.DROPOUT,
            )
            model.load_state_dict(state, strict=True)
            model = model.to(device).eval()
            _TORCH_RUNTIME["fold_models"].append(model)
        print(f"✓ Loaded {len(_TORCH_RUNTIME['fold_models'])} Torch models into memory")


def predict_torch_proba(
    sequence: pl.DataFrame, demographics: pl.DataFrame
) -> np.ndarray:
    """Return probability distribution over classes from Torch models."""
    _ = demographics  # Not used but kept for API consistency
    _load_torch_bundle_and_models()
    bundle = _TORCH_RUNTIME["bundle"]
    if bundle is None:
        return None  # Fallback indicator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pad_len = bundle["pad_len"]
    feat_order = bundle["feature_order"]
    frame_df = FRAME_CACHE.get(sequence).reindex(columns=feat_order).fillna(0)
    n_classes = len(bundle["reverse_gesture_mapper"])
    proba_accum = np.zeros(n_classes, dtype=np.float64)

    for i, f in enumerate(bundle["folds"]):
        stats = f["scaler_stats"]
        x_std = apply_standardize(frame_df, stats).to_numpy(np.float32)
        x_pad, m_pad = pad_and_mask(x_std, pad_len)
        xb = torch.from_numpy(x_pad[None, ...]).to(device)
        mb = torch.from_numpy(m_pad[None, ...]).to(device)

        if EnsembleConfig.LOAD_TORCH_FOLDS_IN_MEMORY and _TORCH_RUNTIME["fold_models"]:
            model = _TORCH_RUNTIME["fold_models"][i]
        else:
            ckpt = torch.load(f["weight_path"], map_location=device)
            state = ckpt["state_dict"]
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            model = TimeSeriesNet(
                in_ch=ckpt["in_ch"],
                num_classes=ckpt["n_classes"],
                hidden=128,
                dropout=DLConfig.DROPOUT,
            )
            model.load_state_dict(state, strict=True)
            model = model.to(device).eval()

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=DLConfig.AMP):
            prob = torch.softmax(model(xb, mb), dim=1).cpu().numpy()[0]
        proba_accum += float(f["weight"]) * prob

    # Normalize
    s = proba_accum.sum()
    return proba_accum / s if s > 0 else proba_accum


# === REPLACE: Prediction function with ensemble ===
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Prediction function for CMI inference server.
    Uses ensemble of LGBM, Torch, and Keras models.
    """
    # LGBM probability
    try:
        proba_lgbm = predict_lgbm_proba(sequence, demographics)
    except Exception as e:
        print(f"⚠️ LGBM prediction error: {e}")
        proba_lgbm = None

    # Torch probability
    proba_torch = None
    if TORCH_AVAILABLE:
        try:
            proba_torch = predict_torch_proba(sequence, demographics)
        except Exception as e:
            print(f"⚠️ Torch prediction error: {e}")

    # Keras probability
    proba_keras = None
    if KERAS_AVAILABLE:
        try:
            proba_keras = predict_keras_proba(sequence, demographics)
        except Exception as e:
            print(f"⚠️ Keras prediction error: {e}")

    # Ensemble weights
    w_l, w_t, w_k = (
        EnsembleConfig.W_LGBM,
        EnsembleConfig.W_TORCH,
        EnsembleConfig.W_KERAS,
    )

    # Combine probabilities
    weights = []
    probas = []
    if proba_lgbm is not None:
        weights.append(w_l)
        probas.append(proba_lgbm)
    if proba_torch is not None:
        weights.append(w_t)
        probas.append(proba_torch)
    if proba_keras is not None:
        weights.append(w_k)
        probas.append(proba_keras)

    if not probas:
        # All failed - return default
        print("⚠️ WARNING: All models failed prediction. Returning default gesture.")
        return "Text on phone"

    # Temperature scaling function
    def apply_temperature_scaling(proba, temperature=1.5):
        """Apply temperature scaling to reduce overconfidence"""
        if temperature == 1.0:
            return proba
        # Apply temperature scaling
        log_proba = np.log(np.clip(proba, 1e-8, 1.0))
        log_proba = log_proba / temperature
        # Stabilize and normalize
        log_proba = log_proba - log_proba.max()
        scaled_proba = np.exp(log_proba)
        return scaled_proba / scaled_proba.sum()

    # Apply temperature scaling to each model's predictions
    TEMPERATURE = float(os.getenv("ENSEMBLE_TEMPERATURE", "1.5"))
    if TEMPERATURE != 1.0:
        probas = [apply_temperature_scaling(p, TEMPERATURE) for p in probas]

    # Weighted average
    W = np.array(weights, dtype=np.float64)
    W = W / max(W.sum(), 1e-12)
    final_proba = np.sum([W[i] * probas[i] for i in range(len(probas))], axis=0)

    final_class = int(np.argmax(final_proba))
    # Load LGBM bundle if not loaded (for reverse mapping)
    _load_lgbm_bundle(RUNTIME_MODEL_PATH)
    return INFER.reverse_gesture_mapper[final_class]


print("✓ Prediction function defined")


if __name__ == "__main__":
    main()

    # === Quick sanity test after main() execution ===
    print("Testing prediction function with dummy data...")
    test_sequence = pl.DataFrame(
        {
            "acc_x": np.random.randn(120),
            "acc_y": np.random.randn(120),
            "acc_z": np.random.randn(120),
            "rot_w": np.random.randn(120),
            "rot_x": np.random.randn(120),
            "rot_y": np.random.randn(120),
            "rot_z": np.random.randn(120),
        }
    )
    test_demographics = pl.DataFrame(
        {
            "age": [25],
            "adult_child": [1],
            "sex": [0],
            "handedness": [1],
            "height_cm": [170],
            "shoulder_to_wrist_cm": [50],
            "elbow_to_wrist_cm": [30],
        }
    )
    print(f"✓ Test prediction: {predict(test_sequence, test_demographics)}")

    # Initialize CMI inference server
    print("Initializing CMI inference server...")
    inference_server = cmi.CMIInferenceServer(predict)
    print("✓ Inference server initialized")

    # Run inference based on environment
    print("Starting inference...")
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        # Competition environment: serve predictions
        print("Running in competition environment...")
        inference_server.serve()
    else:
        # Local testing: run on test data
        print("Running in local testing mode...")
        inference_server.run_local_gateway(
            data_paths=(
                "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv",
                "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv",
            )
        )
        print("\n✓ Inference complete!")
        print("✓ submission.parquet has been generated")
