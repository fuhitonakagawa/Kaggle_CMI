# %% [markdown]
# # CMI BFRB Detection - IMU-only LightGBM Training
#
# This notebook trains the IMU-only LightGBM baseline model for BFRB detection.

# Import required libraries
import os
import sys
import warnings

import joblib
import kaggle_evaluation.cmi_inference_server
import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

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


np.random.seed(Config.SEED)
print("✓ Configuration loaded")

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


print("✓ Feature engineering functions defined")


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

    # Handle missing values
    acc_data = seq_df[Config.ACC_COLS].copy()
    acc_data = acc_data.ffill().bfill().fillna(0)

    rot_data = seq_df[Config.ROT_COLS].copy()
    rot_data = rot_data.ffill().bfill()

    # Handle quaternion missing values
    rot_data_clean = handle_quaternion_missing_values(rot_data.values)

    # Compute world acceleration
    world_acc_data = compute_world_acceleration(acc_data.values, rot_data_clean)

    # Initialize features
    features = {}

    # Sequence metadata
    features["sequence_length"] = len(seq_df)

    # Demographics features
    if len(demo_df) > 0:
        demo_row = demo_df.iloc[0]
        features["age"] = demo_row.get("age", 0)
        features["adult_child"] = demo_row.get("adult_child", 0)
        features["sex"] = demo_row.get("sex", 0)
        features["handedness"] = demo_row.get("handedness", 0)
        features["height_cm"] = demo_row.get("height_cm", 0)
        features["shoulder_to_wrist_cm"] = demo_row.get("shoulder_to_wrist_cm", 0)
        features["elbow_to_wrist_cm"] = demo_row.get("elbow_to_wrist_cm", 0)

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

    # Rotation features
    for i, comp in enumerate(["w", "x", "y", "z"]):
        features.update(
            extract_statistical_features(rot_data_clean[:, i], f"rot_{comp}")
        )

    # Magnitude features
    acc_magnitude = np.linalg.norm(acc_data.values, axis=1)
    world_acc_magnitude = np.linalg.norm(world_acc_data, axis=1)

    features.update(extract_statistical_features(acc_magnitude, "acc_magnitude"))
    features.update(
        extract_statistical_features(world_acc_magnitude, "world_acc_magnitude")
    )

    # Difference between device and world acceleration
    acc_world_diff = acc_magnitude - world_acc_magnitude
    features.update(extract_statistical_features(acc_world_diff, "acc_world_diff"))

    # Convert to DataFrame
    result_df = pd.DataFrame([features])
    result_df = result_df.fillna(0)

    return result_df


print("✓ Feature extraction function defined")

# Load data
print("Loading training data...")
train_df = pl.read_csv(Config.TRAIN_PATH)
train_demographics = pl.read_csv(Config.TRAIN_DEMOGRAPHICS_PATH)

print(f"✓ Train shape: {train_df.shape}")
print(f"✓ Demographics shape: {train_demographics.shape}")

# Get IMU columns (common between train and test)
imu_cols = (
    ["sequence_id", "subject", "phase", "gesture"] + Config.ACC_COLS + Config.ROT_COLS
)
print(f"✓ Using {len(imu_cols)} IMU columns")

# Prepare training data
print("Extracting features for training sequences...")

train_features_list = []
train_labels = []
train_subjects = []

# Get unique sequences count
unique_sequences = train_df["sequence_id"].unique()
n_sequences = len(unique_sequences)
print(f"Total sequences to process: {n_sequences}")

# Group by sequence_id
train_sequences = train_df.select(pl.col(imu_cols)).group_by(
    "sequence_id", maintain_order=True
)

for i, (sequence_id, sequence_data) in enumerate(train_sequences):
    if i % 1000 == 0:
        print(f"Processing sequence {i + 1}/{n_sequences}")

    # Get sequence ID
    seq_id_val = sequence_id[0] if isinstance(sequence_id, tuple) else sequence_id

    # Get demographics
    subject_id = sequence_data["subject"][0]
    subject_demographics = train_demographics.filter(pl.col("subject") == subject_id)

    # Extract features
    features = extract_features(sequence_data, subject_demographics)
    train_features_list.append(features)

    # Get label
    gesture = sequence_data["gesture"][0]
    label = GESTURE_MAPPER[gesture]
    train_labels.append(label)
    train_subjects.append(subject_id)

# Combine features
X_train = pd.concat(train_features_list, ignore_index=True)
y_train = np.array(train_labels)
subjects = np.array(train_subjects)

print(f"✓ Features extracted: {X_train.shape}")
print(f"✓ Number of classes: {len(np.unique(y_train))}")

# %% [code] {"execution":{"iopub.status.busy":"2025-08-12T14:58:13.747441Z","iopub.execute_input":"2025-08-12T14:58:13.747718Z","iopub.status.idle":"2025-08-12T15:05:02.512385Z","shell.execute_reply.started":"2025-08-12T14:58:13.747696Z","shell.execute_reply":"2025-08-12T15:05:02.511578Z"}}
# Train models with cross-validation
print("Training LightGBM models with cross-validation...")

cv = StratifiedGroupKFold(
    n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
)
models = []
oof_predictions = np.zeros(len(y_train))
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train, subjects)):
    print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")

    # Split data
    X_fold_train = X_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_train = y_train[train_idx]
    y_fold_val = y_train[val_idx]

    print(f"Train size: {len(X_fold_train)}, Val size: {len(X_fold_val)}")

    # Train model
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

    # Store model
    models.append(model)

    # Predictions
    val_preds = model.predict(X_fold_val)
    oof_predictions[val_idx] = val_preds

    # Calculate score
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

print("\n✓ Cross-validation complete!")
print(f"Overall CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Save models and metadata
print("Saving models...")

# Prepare model data
model_data = {
    "models": models,
    "feature_names": list(X_train.columns),
    "gesture_mapper": GESTURE_MAPPER,
    "reverse_gesture_mapper": REVERSE_GESTURE_MAPPER,
    "cv_scores": cv_scores,
    "mean_cv_score": np.mean(cv_scores),
    "config": {
        "n_folds": Config.N_FOLDS,
        "seed": Config.SEED,
        "lgbm_params": Config.LGBM_PARAMS,
    },
}

# Save to file
model_path = os.path.join(Config.OUTPUT_PATH, "imu_lgbm_model.pkl")
joblib.dump(model_data, model_path)

print(f"✓ Models saved to {model_path}")
print(f"✓ File size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

# Also save feature importance
feature_importance = pd.DataFrame(
    {
        "feature": X_train.columns,
        "importance": np.mean([model.feature_importances_ for model in models], axis=0),
    }
).sort_values("importance", ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance.head(20))

feature_importance.to_csv(
    os.path.join(Config.OUTPUT_PATH, "feature_importance.csv"), index=False
)
print("\n✓ Training complete!")


# # CMI BFRB Detection - IMU-only LightGBM Inference

# Load trained model
print("Loading trained model...")
model_path = "/kaggle/working/imu_lgbm_model.pkl"  # Update this path based on where your model is saved
model_data = joblib.load(model_path)

models = model_data["models"]
feature_names = model_data["feature_names"]
reverse_gesture_mapper = model_data["reverse_gesture_mapper"]
config = model_data["config"]

print(f"✓ Loaded {len(models)} models")
print(f"✓ Number of features: {len(feature_names)}")
print(f"✓ CV Score: {model_data['mean_cv_score']:.4f}")

# Define feature extraction functions (same as training)
ACC_COLS = ["acc_x", "acc_y", "acc_z"]
ROT_COLS = ["rot_w", "rot_x", "rot_y", "rot_z"]


def extract_features(
    sequence: pl.DataFrame, demographics: pl.DataFrame
) -> pd.DataFrame:
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
    features["sequence_length"] = len(seq_df)

    # Demographics features
    if len(demo_df) > 0:
        demo_row = demo_df.iloc[0]
        features["age"] = demo_row.get("age", 0)
        features["adult_child"] = demo_row.get("adult_child", 0)
        features["sex"] = demo_row.get("sex", 0)
        features["handedness"] = demo_row.get("handedness", 0)
        features["height_cm"] = demo_row.get("height_cm", 0)
        features["shoulder_to_wrist_cm"] = demo_row.get("shoulder_to_wrist_cm", 0)
        features["elbow_to_wrist_cm"] = demo_row.get("elbow_to_wrist_cm", 0)
    else:
        # Default values if demographics not available
        features["age"] = 0
        features["adult_child"] = 0
        features["sex"] = 0
        features["handedness"] = 0
        features["height_cm"] = 0
        features["shoulder_to_wrist_cm"] = 0
        features["elbow_to_wrist_cm"] = 0

    # Extract statistical features for each axis
    for i, axis in enumerate(["x", "y", "z"]):
        if i < acc_data.shape[1]:
            # Device acceleration
            features.update(
                extract_statistical_features(acc_data.values[:, i], f"acc_{axis}")
            )
            # World acceleration
            features.update(
                extract_statistical_features(world_acc_data[:, i], f"world_acc_{axis}")
            )

    # Rotation features
    for i, comp in enumerate(["w", "x", "y", "z"]):
        if i < rot_data_clean.shape[1]:
            features.update(
                extract_statistical_features(rot_data_clean[:, i], f"rot_{comp}")
            )

    # Magnitude features
    acc_magnitude = np.linalg.norm(acc_data.values, axis=1)
    world_acc_magnitude = np.linalg.norm(world_acc_data, axis=1)

    features.update(extract_statistical_features(acc_magnitude, "acc_magnitude"))
    features.update(
        extract_statistical_features(world_acc_magnitude, "world_acc_magnitude")
    )

    # Difference between device and world acceleration
    acc_world_diff = acc_magnitude - world_acc_magnitude
    features.update(extract_statistical_features(acc_world_diff, "acc_world_diff"))

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


print("✓ Feature extraction function defined")


# Define prediction function for CMI inference server
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Prediction function for CMI inference server.
    Takes a single sequence and returns the predicted gesture name.
    """
    try:
        # Extract features
        features = extract_features(sequence, demographics)

        # Get predictions from all models
        predictions = []
        probabilities = []

        for model in models:
            # Get prediction probabilities
            pred_proba = model.predict_proba(features)
            probabilities.append(pred_proba[0])

            # Get predicted class
            pred_class = np.argmax(pred_proba, axis=1)[0]
            predictions.append(pred_class)

        # Ensemble: average probabilities
        avg_proba = np.mean(probabilities, axis=0)
        final_prediction = np.argmax(avg_proba)

        # Convert to gesture name
        gesture_name = reverse_gesture_mapper[final_prediction]

        return gesture_name

    except Exception as e:
        print(f"Prediction error: {e}")
        # Return default prediction in case of error
        return "Text on phone"


print("✓ Prediction function defined")

# Test the prediction function with a small example
print("Testing prediction function...")

# Create dummy data for testing
test_sequence = pl.DataFrame(
    {
        "acc_x": np.random.randn(100),
        "acc_y": np.random.randn(100),
        "acc_z": np.random.randn(100),
        "rot_w": np.random.randn(100),
        "rot_x": np.random.randn(100),
        "rot_y": np.random.randn(100),
        "rot_z": np.random.randn(100),
    }
)

test_demographics = pl.DataFrame(
    {
        "age": [25],
        "adult_child": [1],
        "sex": [0],
        "handedness": [1],
        "height_cm": [175],
        "shoulder_to_wrist_cm": [50],
        "elbow_to_wrist_cm": [30],
    }
)

# Test prediction
test_result = predict(test_sequence, test_demographics)
print(f"✓ Test prediction: {test_result}")

# Import the CMI inference server
sys.path.append("/kaggle/input/cmi-detect-behavior-with-sensor-data")

# Initialize CMI inference server
print("Initializing CMI inference server...")

inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

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
