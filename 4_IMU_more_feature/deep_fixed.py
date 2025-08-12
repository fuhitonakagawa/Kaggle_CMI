# ====================================================================================================
# CMI BFRB Detection - Deep Learning with 1D-CNN + BiLSTM + Attention
# Version: 6.1 - Fixed Learning Rate Schedule Conflict
# Architecture: Residual SE-CNN + BiLSTM + GRU + Attention (Based on LB 0.77 model)
# Score Target: 0.770+ (Binary F1: 0.94+, Macro F1: 0.60+)
# ====================================================================================================

import json
import os
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, Tuple, List, Optional
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import tensorflow as tf
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout,
    Bidirectional, LSTM, GlobalAveragePooling1D, Dense, Multiply, Reshape,
    Lambda, Concatenate, GRU, GaussianNoise, Flatten
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence, to_categorical

warnings.filterwarnings("ignore")


# ====================================================================================================
# GPU CONFIGURATION
# ====================================================================================================

def configure_gpu():
    """Configure GPU for optimal performance with Metal (Mac) or CUDA support."""
    # Check for available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # For M1/M2 Macs with Metal
            if sys.platform == 'darwin':
                print("✓ Metal GPU detected (Mac)")
                # Metal doesn't require memory growth settings
                # Disable mixed precision for stability
                # policy = tf.keras.mixed_precision.Policy('mixed_float16')
                # tf.keras.mixed_precision.set_global_policy(policy)
                # print("  Mixed precision: enabled (float16 compute, float32 variables)")
            else:
                # For NVIDIA GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                # Enable mixed precision for NVIDIA GPUs
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print(f"✓ CUDA GPU configured: {len(gpus)} device(s) available")
                print("  Mixed precision: enabled")
            return True
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")
            return False
    else:
        print("⚠️ No GPU found, using CPU (training will be slower)")
        return False


def seed_everything(seed=42):
    """Set all random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


# Configure environment
GPU_AVAILABLE = configure_gpu()
seed_everything(42)
print("✓ Environment configured successfully")

# ====================================================================================================
# CONFIGURATION
# ====================================================================================================

CONFIG = {
    "data_path": "cmi-detect-behavior-with-sensor-data/",
    "n_folds": 5,
    "random_state": 42,
    "sample_rate": 20,  # Hz
    
    # Sequence processing
    "sequence_max_len": 500,  # Pad/truncate sequences to this length
    "pad_percentile": 95,     # Percentile for determining pad length
    
    # Deep Learning parameters
    "batch_size": 64,
    "epochs": 150,
    "patience": 40,
    "lr_init": 5e-4,
    "weight_decay": 3e-3,
    "mixup_alpha": 0.4,
    
    # Model architecture
    "cnn_filters": [64, 128, 256],
    "lstm_units": 128,
    "gru_units": 128,
    "attention_units": 128,
    "dropout_rate": 0.4,
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

print(f"✓ Configuration loaded ({len(GESTURE_MAPPER)} gesture classes)")

# ====================================================================================================
# TENSOR MANIPULATIONS FOR ATTENTION
# ====================================================================================================

def time_sum(x):
    """Sum over time dimension."""
    return K.sum(x, axis=1)

def squeeze_last_axis(x):
    """Squeeze last axis."""
    return tf.squeeze(x, axis=-1)

def expand_last_axis(x):
    """Expand last axis."""
    return tf.expand_dims(x, axis=-1)

# ====================================================================================================
# MODEL COMPONENTS
# ====================================================================================================

def se_block(x, reduction=8):
    """Squeeze-and-Excitation block for channel attention."""
    ch = x.shape[-1]
    se = GlobalAveragePooling1D()(x)
    se = Dense(ch // reduction, activation='relu')(se)
    se = Dense(ch, activation='sigmoid')(se)
    se = Reshape((1, ch))(se)
    return Multiply()([x, se])


def residual_se_cnn_block(x, filters, kernel_size, pool_size=2, drop=0.3, wd=1e-4):
    """Residual CNN block with Squeeze-and-Excitation."""
    shortcut = x
    
    # Two conv layers
    for _ in range(2):
        x = Conv1D(filters, kernel_size, padding='same', use_bias=False,
                   kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
    # SE block
    x = se_block(x)
    
    # Residual connection
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', use_bias=False,
                          kernel_regularizer=l2(wd))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = add([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size)(x)
    x = Dropout(drop)(x)
    
    return x


def attention_layer(inputs):
    """Attention mechanism for sequence weighting."""
    score = Dense(1, activation='tanh')(inputs)
    score = Lambda(squeeze_last_axis)(score)
    weights = Activation('softmax')(score)
    weights = Lambda(expand_last_axis)(weights)
    context = Multiply()([inputs, weights])
    context = Lambda(time_sum)(context)
    return context

# ====================================================================================================
# WORLD ACCELERATION AND ANGULAR FEATURES
# ====================================================================================================

def remove_gravity_from_acc(acc_data, rot_data):
    """Remove gravity component from accelerometer data using quaternion rotation."""
    if isinstance(acc_data, pd.DataFrame):
        acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    else:
        acc_values = acc_data

    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)
    
    gravity_world = np.array([0, 0, 9.81])

    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :] 
            continue

        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError:
            linear_accel[i, :] = acc_values[i, :]
             
    return linear_accel


def calculate_angular_velocity(rot: np.ndarray, sample_rate: int = 20) -> np.ndarray:
    """Calculate angular velocity from quaternion data."""
    n_samples = len(rot)
    angular_vel = np.zeros((n_samples, 3))
    time_delta = 1.0 / sample_rate
    
    # Convert to scipy format (x,y,z,w)
    rot_scipy = rot[:, [1, 2, 3, 0]]
    
    for i in range(n_samples - 1):
        try:
            r_t = R.from_quat(rot_scipy[i])
            r_t_plus = R.from_quat(rot_scipy[i + 1])
            delta_rot = r_t.inv() * r_t_plus
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except (ValueError, ZeroDivisionError):
            angular_vel[i, :] = 0
    
    if n_samples > 1:
        angular_vel[-1, :] = angular_vel[-2, :]
    
    return angular_vel

# ====================================================================================================
# MIXUP DATA AUGMENTATION
# ====================================================================================================

class MixupGenerator(Sequence):
    """MixUp data augmentation generator for regularization."""
    
    def __init__(self, X, y, batch_size, alpha=0.2):
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.alpha = alpha
        self.indices = np.arange(len(X))
        
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, i):
        idx = self.indices[i*self.batch_size:(i+1)*self.batch_size]
        Xb, yb = self.X[idx], self.y[idx]
        
        # MixUp
        lam = np.random.beta(self.alpha, self.alpha)
        perm = np.random.permutation(len(Xb))
        X_mix = lam * Xb + (1-lam) * Xb[perm]
        y_mix = lam * yb + (1-lam) * yb[perm]
        
        return X_mix, y_mix
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# ====================================================================================================
# CUSTOM LEARNING RATE SCHEDULER CALLBACK
# ====================================================================================================

class CustomLRScheduler(Callback):
    """Custom learning rate scheduler that combines cosine decay with manual reduction on plateau."""
    
    def __init__(self, initial_lr, first_decay_steps, patience=10, factor=0.5, min_lr=1e-6, verbose=1):
        super().__init__()
        self.initial_lr = initial_lr
        self.first_decay_steps = first_decay_steps
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.verbose = verbose
        self.wait = 0
        self.best = float('inf')
        self.current_lr = initial_lr
        
    def on_epoch_begin(self, epoch, logs=None):
        # Cosine decay
        if epoch < self.first_decay_steps:
            self.current_lr = self.initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / self.first_decay_steps))
        K.set_value(self.model.optimizer.learning_rate, self.current_lr)
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('val_loss')
        if current is None:
            return
            
        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(K.get_value(self.model.optimizer.learning_rate))
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    K.set_value(self.model.optimizer.learning_rate, new_lr)
                    self.current_lr = new_lr
                    if self.verbose > 0:
                        print(f'\nEpoch {epoch+1}: reducing learning rate from {old_lr:.2e} to {new_lr:.2e}.')
                    self.wait = 0

# ====================================================================================================
# DEEP LEARNING MODEL ARCHITECTURE
# ====================================================================================================

def build_two_branch_model(pad_len, imu_dim, tof_dim, n_classes, wd=1e-4):
    """
    Two-branch architecture for IMU and TOF/Thermal data.
    Based on LB 0.77 architecture with BiLSTM + GRU + Attention.
    """
    # Input layer
    inp = Input(shape=(pad_len, imu_dim + tof_dim))
    
    # Split into IMU and TOF branches
    imu = Lambda(lambda t: t[:, :, :imu_dim])(inp)
    tof = Lambda(lambda t: t[:, :, imu_dim:])(inp)
    
    # IMU deep branch with residual SE-CNN blocks
    x1 = residual_se_cnn_block(imu, 64, 3, drop=0.1, wd=wd)
    x1 = residual_se_cnn_block(x1, 128, 5, drop=0.1, wd=wd)
    
    # TOF/Thermal lighter branch (simpler CNN)
    x2 = Conv1D(64, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(tof)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2)
    x2 = Dropout(0.2)(x2)
    
    x2 = Conv1D(128, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2)
    x2 = Dropout(0.2)(x2)
    
    # Merge branches
    merged = Concatenate()([x1, x2])
    
    # Bidirectional LSTM branch
    xa = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    
    # Bidirectional GRU branch
    xb = Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    
    # Gaussian noise branch for regularization
    xc = GaussianNoise(0.09)(merged)
    xc = Dense(16, activation='elu')(xc)
    
    # Combine all branches
    x = Concatenate()([xa, xb, xc])
    x = Dropout(0.4)(x)
    
    # Attention mechanism
    x = attention_layer(x)
    
    # Dense layers for classification
    for units, drop in [(256, 0.5), (128, 0.3)]:
        x = Dense(units, use_bias=False, kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(drop)(x)
    
    # Output layer
    out = Dense(n_classes, activation='softmax', kernel_regularizer=l2(wd))(x)
    
    return Model(inp, out)


def build_imu_only_model(pad_len, imu_dim, n_classes, wd=1e-4):
    """
    Single-branch model for IMU-only sequences.
    Uses the same architecture as the IMU branch in two-branch model.
    """
    # Input layer
    inp = Input(shape=(pad_len, imu_dim))
    
    # Residual SE-CNN blocks
    x = residual_se_cnn_block(inp, 64, 3, drop=0.1, wd=wd)
    x = residual_se_cnn_block(x, 128, 5, drop=0.2, wd=wd)
    x = residual_se_cnn_block(x, 256, 3, drop=0.3, wd=wd)
    
    # BiLSTM
    xa = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(wd)))(x)
    
    # BiGRU
    xb = Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=l2(wd)))(x)
    
    # Combine
    x = Concatenate()([xa, xb])
    x = Dropout(0.4)(x)
    
    # Attention
    x = attention_layer(x)
    
    # Dense layers
    x = Dense(256, use_bias=False, kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128, use_bias=False, kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # Output
    out = Dense(n_classes, activation='softmax', kernel_regularizer=l2(wd))(x)
    
    return Model(inp, out)

# ====================================================================================================
# DATA PREPARATION FOR DEEP LEARNING
# ====================================================================================================

def prepare_sequences_for_dl(df, demo_df, sequence_ids, gesture_mapper=None):
    """
    Prepare sequences for deep learning model.
    Extracts raw time series data with engineered features.
    """
    sequences = []
    labels = []
    groups = []
    
    # Group by sequence
    for seq_id in sequence_ids:
        seq_data = df[df['sequence_id'] == seq_id].copy()
        
        if len(seq_data) == 0:
            continue
        
        # Get subject info
        subject_id = seq_data['subject'].iloc[0]
        
        # Basic IMU features
        acc_data = seq_data[['acc_x', 'acc_y', 'acc_z']].fillna(0).values
        rot_data = seq_data[['rot_w', 'rot_x', 'rot_y', 'rot_z']].ffill().bfill().fillna(1).values
        
        # Remove gravity to get linear acceleration
        linear_acc = remove_gravity_from_acc(acc_data, rot_data[:, [1, 2, 3, 0]])
        
        # Calculate engineered features
        acc_mag = np.sqrt(np.sum(acc_data**2, axis=1))
        linear_acc_mag = np.sqrt(np.sum(linear_acc**2, axis=1))
        rot_angle = 2 * np.arccos(np.clip(rot_data[:, 0], -1, 1))
        
        # Calculate derivatives (jerk and angular velocity)
        acc_mag_jerk = np.diff(acc_mag, prepend=acc_mag[0])
        linear_acc_mag_jerk = np.diff(linear_acc_mag, prepend=linear_acc_mag[0])
        rot_angle_vel = np.diff(rot_angle, prepend=rot_angle[0])
        
        # Angular velocity from quaternions
        angular_vel = calculate_angular_velocity(rot_data, CONFIG['sample_rate'])
        angular_vel_mag = np.linalg.norm(angular_vel, axis=1)
        
        # Combine IMU features
        imu_features = np.column_stack([
            linear_acc,  # 3 features
            rot_data[:, 1:],  # 3 features (x, y, z components)
            acc_mag.reshape(-1, 1),  # 1 feature
            linear_acc_mag.reshape(-1, 1),  # 1 feature
            rot_angle.reshape(-1, 1),  # 1 feature
            acc_mag_jerk.reshape(-1, 1),  # 1 feature
            linear_acc_mag_jerk.reshape(-1, 1),  # 1 feature
            rot_angle_vel.reshape(-1, 1),  # 1 feature
            angular_vel,  # 3 features
            angular_vel_mag.reshape(-1, 1),  # 1 feature
        ])  # Total: 16 IMU features
        
        # TOF features (aggregate statistics)
        tof_features = []
        for i in range(1, 6):
            tof_cols = [f"tof_{i}_v{p}" for p in range(64)]
            if all(col in seq_data.columns for col in tof_cols):
                tof_sensor_data = seq_data[tof_cols].replace(-1, np.nan)
                tof_mean = tof_sensor_data.mean(axis=1).fillna(0).values
                tof_std = tof_sensor_data.std(axis=1).fillna(0).values
                tof_min = tof_sensor_data.min(axis=1).fillna(0).values
                tof_max = tof_sensor_data.max(axis=1).fillna(0).values
                tof_features.extend([tof_mean, tof_std, tof_min, tof_max])
            else:
                # No TOF data - add zeros
                tof_features.extend([
                    np.zeros(len(seq_data)),
                    np.zeros(len(seq_data)),
                    np.zeros(len(seq_data)),
                    np.zeros(len(seq_data))
                ])
        
        # Thermal features
        thm_features = []
        for i in range(1, 6):
            thm_col = f"thm_{i}"
            if thm_col in seq_data.columns:
                thm_data = seq_data[thm_col].fillna(0).values
                thm_features.append(thm_data)
            else:
                thm_features.append(np.zeros(len(seq_data)))
        
        # Stack all TOF and thermal features
        tof_thm_array = np.column_stack(tof_features + thm_features) if (tof_features or thm_features) else np.zeros((len(seq_data), 1))
        
        # Combine all features
        all_features = np.concatenate([imu_features, tof_thm_array], axis=1)
        
        sequences.append(all_features)
        
        # Get label if available
        if gesture_mapper and 'gesture' in seq_data.columns:
            gesture = seq_data['gesture'].iloc[0]
            labels.append(gesture_mapper[gesture])
        
        groups.append(subject_id)
    
    return sequences, labels, groups


def pad_sequences_custom(sequences, maxlen=None, padding='post', truncating='post', dtype='float32'):
    """Custom padding function for sequences."""
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    
    num_samples = len(sequences)
    num_features = sequences[0].shape[1] if len(sequences[0].shape) > 1 else 1
    
    padded = np.zeros((num_samples, maxlen, num_features), dtype=dtype)
    
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), maxlen)
        
        if truncating == 'post':
            trunc_seq = seq[:seq_len]
        else:
            trunc_seq = seq[-seq_len:]
        
        if padding == 'post':
            padded[i, :seq_len] = trunc_seq
        else:
            padded[i, -seq_len:] = trunc_seq
    
    return padded

# ====================================================================================================
# TRAINING PIPELINE
# ====================================================================================================

def train_deep_learning_model():
    """Main training pipeline for deep learning model."""
    print("=" * 70)
    print("CMI BFRB Detection - Deep Learning Training")
    print("=" * 70)
    
    # Check if running in Kaggle
    is_kaggle = os.path.exists("/kaggle/input")
    
    # Load data
    print("Loading data...")
    if is_kaggle:
        train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
        demo_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
    else:
        train_df = pd.read_csv(CONFIG["data_path"] + "train.csv")
        demo_df = pd.read_csv(CONFIG["data_path"] + "train_demographics.csv")
    
    print(f"Loaded {len(train_df)} samples from {train_df['sequence_id'].nunique()} sequences")
    
    # Get unique sequences
    sequence_ids = train_df['sequence_id'].unique()
    
    # Prepare sequences for deep learning
    print("Preparing sequences for deep learning...")
    sequences, labels, groups = prepare_sequences_for_dl(
        train_df, demo_df, sequence_ids, GESTURE_MAPPER
    )
    
    # Calculate padding length
    seq_lengths = [len(seq) for seq in sequences]
    pad_len = int(np.percentile(seq_lengths, CONFIG['pad_percentile']))
    print(f"Sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, pad_len={pad_len}")
    
    # Pad sequences
    X = pad_sequences_custom(sequences, maxlen=pad_len, padding='post', truncating='post')
    y = to_categorical(labels, num_classes=len(GESTURE_MAPPER))
    groups = np.array(groups)
    
    print(f"Padded sequences shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_reshaped = scaler.fit_transform(X_reshaped)
    X = X_reshaped.reshape(X.shape)
    
    # Determine feature dimensions
    imu_dim = 16  # Fixed based on our feature engineering
    tof_thm_dim = X.shape[-1] - imu_dim
    
    print(f"IMU features: {imu_dim}, TOF/THM features: {tof_thm_dim}")
    
    # Split data
    print("Splitting data for training and validation...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=CONFIG['random_state'], stratify=labels
    )
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(len(GESTURE_MAPPER)),
        y=labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    # Build model
    print("Building model...")
    if tof_thm_dim > 0:
        model = build_two_branch_model(
            pad_len, imu_dim, tof_thm_dim, 
            len(GESTURE_MAPPER), wd=CONFIG['weight_decay']
        )
        print("✓ Two-branch model created (IMU + TOF/THM)")
    else:
        model = build_imu_only_model(
            pad_len, imu_dim,
            len(GESTURE_MAPPER), wd=CONFIG['weight_decay']
        )
        print("✓ IMU-only model created")
    
    # Use fixed learning rate instead of schedule
    # This avoids the conflict with ReduceLROnPlateau
    optimizer = Adam(learning_rate=CONFIG['lr_init'])
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    # Callbacks - using custom LR scheduler instead of ReduceLROnPlateau
    steps_per_epoch = len(X_tr) // CONFIG['batch_size']
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=CONFIG['patience'],
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        CustomLRScheduler(
            initial_lr=CONFIG['lr_init'],
            first_decay_steps=15,  # epochs for cosine decay
            patience=10,
            factor=0.5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    # Create data generator with MixUp
    train_gen = MixupGenerator(X_tr, y_tr, batch_size=CONFIG['batch_size'], alpha=CONFIG['mixup_alpha'])
    
    # Train model
    print("\nStarting training...")
    print(f"Epochs: {CONFIG['epochs']}, Batch size: {CONFIG['batch_size']}")
    print(f"MixUp alpha: {CONFIG['mixup_alpha']}")
    
    history = model.fit(
        train_gen,
        epochs=CONFIG['epochs'],
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating model...")
    val_preds = model.predict(X_val).argmax(axis=1)
    val_true = y_val.argmax(axis=1)
    
    # Calculate metrics
    binary_f1 = f1_score(
        val_true < 8,
        val_preds < 8,
        average='binary'
    )
    
    # Macro F1 for BFRB classes
    bfrb_mask = val_true < 8
    if bfrb_mask.any():
        macro_f1 = f1_score(
            val_true[bfrb_mask],
            val_preds[bfrb_mask],
            average='macro'
        )
    else:
        macro_f1 = 0
    
    combined_score = (binary_f1 + macro_f1) / 2
    
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Binary F1 (BFRB detection): {binary_f1:.4f}")
    print(f"Macro F1 (BFRB classification): {macro_f1:.4f}")
    print(f"Combined Score: {combined_score:.4f}")
    print(f"{'='*60}")
    
    # Save model and scaler
    model.save('gesture_deep_model.h5')
    joblib.dump(scaler, 'scaler.pkl')
    np.save('pad_len.npy', pad_len)
    np.save('imu_dim.npy', imu_dim)
    np.save('tof_thm_dim.npy', tof_thm_dim)
    
    print("\n✓ Model saved successfully!")
    
    # Save training results
    results = {
        'binary_f1': float(binary_f1),
        'macro_f1': float(macro_f1),
        'combined_score': float(combined_score),
        'pad_len': int(pad_len),
        'imu_dim': int(imu_dim),
        'tof_thm_dim': int(tof_thm_dim),
        'timestamp': datetime.now().isoformat()
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, scaler, pad_len

# ====================================================================================================
# INFERENCE
# ====================================================================================================

def predict_for_submission(sequence: pl.DataFrame, demographics: pl.DataFrame, 
                          model, scaler, pad_len, imu_dim, tof_thm_dim):
    """Prediction function for Kaggle submission."""
    try:
        # Convert to pandas
        seq_df = sequence.to_pandas() if isinstance(sequence, pl.DataFrame) else sequence
        demo_df = demographics.to_pandas() if isinstance(demographics, pl.DataFrame) else demographics
        
        # Prepare single sequence
        sequences, _, _ = prepare_sequences_for_dl(
            seq_df, demo_df, 
            seq_df['sequence_id'].unique() if 'sequence_id' in seq_df.columns else ['unknown'],
            None
        )
        
        if not sequences:
            return "Text on phone"  # Default prediction
        
        # Pad sequence
        X = pad_sequences_custom(sequences, maxlen=pad_len, padding='post', truncating='post')
        
        # Normalize
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_reshaped = scaler.transform(X_reshaped)
        X = X_reshaped.reshape(X.shape)
        
        # Predict
        pred_probs = model.predict(X, verbose=0)
        pred_class = pred_probs.argmax(axis=1)[0]
        
        return REVERSE_GESTURE_MAPPER[pred_class]
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Text on phone"  # Default prediction

# ====================================================================================================
# MAIN EXECUTION
# ====================================================================================================

if __name__ == "__main__":
    # Check if running in Kaggle environment
    is_kaggle = os.path.exists("/kaggle/input")
    
    if is_kaggle:
        print("Running in Kaggle environment")
        CONFIG["data_path"] = "/kaggle/input/cmi-detect-behavior-with-sensor-data/"
        
        # Check if pre-trained model exists
        if os.path.exists("/kaggle/input/pretrained-deep-model/gesture_deep_model.h5"):
            print("Loading pre-trained model...")
            
            # Load model and parameters
            custom_objects = {
                'time_sum': time_sum,
                'squeeze_last_axis': squeeze_last_axis,
                'expand_last_axis': expand_last_axis,
                'se_block': se_block,
                'residual_se_cnn_block': residual_se_cnn_block,
                'attention_layer': attention_layer,
            }
            
            model = load_model(
                "/kaggle/input/pretrained-deep-model/gesture_deep_model.h5",
                custom_objects=custom_objects
            )
            scaler = joblib.load("/kaggle/input/pretrained-deep-model/scaler.pkl")
            pad_len = int(np.load("/kaggle/input/pretrained-deep-model/pad_len.npy"))
            imu_dim = int(np.load("/kaggle/input/pretrained-deep-model/imu_dim.npy"))
            tof_thm_dim = int(np.load("/kaggle/input/pretrained-deep-model/tof_thm_dim.npy"))
            
            print("✓ Pre-trained model loaded successfully")
        else:
            print("Training new model...")
            model, scaler, pad_len = train_deep_learning_model()
            imu_dim = 16  # Fixed
            tof_thm_dim = model.input_shape[-1] - imu_dim
        
        # Create prediction function
        def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
            return predict_for_submission(
                sequence, demographics, model, scaler, pad_len, imu_dim, tof_thm_dim
            )
        
        # Test prediction
        print("\nTesting prediction function...")
        test_seq = pl.DataFrame({
            'acc_x': np.random.randn(100),
            'acc_y': np.random.randn(100),
            'acc_z': np.random.randn(100),
            'rot_w': np.ones(100),
            'rot_x': np.zeros(100),
            'rot_y': np.zeros(100),
            'rot_z': np.zeros(100),
            'subject': ['test'] * 100,
            'sequence_id': ['test_seq'] * 100,
        })
        test_demo = pl.DataFrame({
            'age': [25],
            'adult_child': [1],
            'sex': [0],
            'handedness': [1]
        })
        
        test_result = predict(test_seq, test_demo)
        print(f"Test prediction: {test_result}")
        assert test_result in GESTURE_MAPPER, "Invalid prediction"
        print("✓ Test passed!")
        
        # Initialize inference server
        sys.path.append("/kaggle/input/cmi-detect-behavior-with-sensor-data")
        try:
            import kaggle_evaluation.cmi_inference_server
            
            print("\nInitializing CMI inference server...")
            inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
            print("✓ Inference server initialized")
            
            # Run inference
            if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
                print("Competition environment - serving predictions...")
                inference_server.serve()
            else:
                print("Local testing mode...")
                inference_server.run_local_gateway(
                    data_paths=(
                        CONFIG["data_path"] + "test.csv",
                        CONFIG["data_path"] + "test_demographics.csv",
                    )
                )
                print("✓ Inference complete!")
                
        except ImportError as e:
            print(f"Could not import CMI inference server: {e}")
            print("This is normal if running locally without the CMI package.")
    else:
        # Local training and testing
        print("Running in local environment")
        
        # Check if model exists
        if os.path.exists("gesture_deep_model.h5"):
            print("Found existing model. Loading...")
            
            custom_objects = {
                'time_sum': time_sum,
                'squeeze_last_axis': squeeze_last_axis,
                'expand_last_axis': expand_last_axis,
                'se_block': se_block,
                'residual_se_cnn_block': residual_se_cnn_block,
                'attention_layer': attention_layer,
            }
            
            model = load_model("gesture_deep_model.h5", custom_objects=custom_objects)
            scaler = joblib.load("scaler.pkl")
            pad_len = int(np.load("pad_len.npy"))
            imu_dim = int(np.load("imu_dim.npy"))
            tof_thm_dim = int(np.load("tof_thm_dim.npy"))
            
            print("✓ Model loaded successfully")
        else:
            print("Training new model...")
            model, scaler, pad_len = train_deep_learning_model()
            imu_dim = 16
            tof_thm_dim = model.input_shape[-1] - imu_dim
            print("✓ Training completed!")
        
        print("\n" + "=" * 70)
        print("Model ready for inference!")
        print("To use this model for Kaggle submission:")
        print("1. Upload this script to Kaggle")
        print("2. Run it to train the model (or upload pre-trained model)")
        print("3. The notebook will generate submission.parquet")
        print("=" * 70)