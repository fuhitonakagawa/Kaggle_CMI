# %% [markdown]
# **If you support the notebook, please upvote—I’d really appreciate it!**
# **Also, if you have any questions, feel free to ask—I’ll be happy to answer. 😊**
# ### UPDATE: Enhanced IMU Signal Processing with Gravity Removal & Dataset Split Adjustment
#
# This version builds upon the enhancements from my previous notebook: **[LB 0.76 | IMU+THM/TOF | TF BiLSTM+GRU+Attention](https://www.kaggle.com/code/rktqwe/lb-0-76-imu-thm-tof-tf-bilstm-gru-attention)**. The key changes in *this* update are:
#
# 1.  **Gravity Component Removal from Accelerometer Data (Key Improvement):**
#     *   **Purpose:** Raw accelerometer data measures both the device's linear acceleration (actual movement) and the constant pull of gravity. By isolating the linear acceleration, we aim to provide the model with cleaner, more motion-specific signals, potentially improving its ability to distinguish between subtle gestures.
#     *   **Method:**
#         *   We utilize quaternion rotation data (`rot_x`, `rot_y`, `rot_z`, `rot_w`) to determine the device's orientation in 3D space.
#         *   Using `scipy.spatial.transform.Rotation`, we project a standard gravity vector (`[0, 0, 9.81] m/s²`) from the world frame into the sensor's coordinate frame for each timestamp.
#         *   This estimated gravity component in the sensor's frame is then subtracted from the raw `acc_x`, `acc_y`, `acc_z` readings.
#     *   **New Features Derived:**
#         *   `linear_acc_x`, `linear_acc_y`, `linear_acc_z`: Accelerometer readings with gravity removed. These now form the base acceleration features.
#         *   `linear_acc_mag`: Magnitude of the linear acceleration.
#         *   `linear_acc_mag_jerk`: Derivative (jerk) of the linear acceleration magnitude.
#     *   **Impact:** The `imu_cols_base` now uses these `linear_acc_x/y/z` features instead of the raw `acc_x/y/z`. The original `acc_mag` (which includes gravity) is retained as it might still offer complementary information.
#
# 2.  **Train/Validation Split Adjustment:**
#     *   The `train_test_split` ratio for creating training and validation sets was changed from `test_size=0.1` (90% train, 10% validation) to `test_size=0.2` (80% train, 20% validation). This provides a larger validation set for more robust evaluation during training.
#
#
#
# **This solution was developed by building upon the foundational work of two key contributors:**
#
# - **CMI25 | IMU+THM/TOF |TF BiLSTM+GRU+Attention|LB.75** by [yukiZ](https://www.kaggle.com/code/hideyukizushi/cmi25-imu-thm-tof-tf-bilstm-gru-attention-lb-75)
# - **imu-tof** by [Erich von Mainstein](https://www.kaggle.com/code/vonmainstein/imu-tof)
#
# **Special recognition for the exceptional exploratory data analysis:**
#
# - **Sensor Pulse🧠| Viz & EDA for BFRB Detection** by [Tarun Mishra](https://www.kaggle.com/code/tarundirector/sensor-pulse-viz-eda-for-bfrb-detection)
#
# ### Implemented Enhancements
#
# #### 1. IMU Signal Processing Optimization (Core Improvement)
#
# **Implemented Features:**
# - Derived kinematic metrics:
#   - Acceleration vector magnitude (`acc_mag`)
#   - Scalar rotation angle (`rot_angle`)
#   - Acceleration derivative (`acc_mag_jerk`)
#   - Angular velocity (`rot_angle_vel`)
# - Initial learning rate reduction from `1e-3` to `5e-4`
# - Implemented cosine decay scheduling:
#   - Parameters: `first_decay_steps = 15 × steps_per_epoch`
#
# **Unsuccessful Experiments (Reverted):**
# - Further LR Reduction: `LR_INIT = 2-e4` worsened score.
# - Dropout Rate Adjustments: No improvement over baseline dropout.
# - TOF/THEM Branch Architecture Change: Replacing simpler CNNs with `residual_se_cnn_blocks` was detrimental.

# %% [markdown]
# ### 》》》**Importing the necessary Libraries**

# %%
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# %% [markdown]
# ### 》》》**Fix Seed**
# %%
import random

import polars as pl
import tensorflow as tf
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Activation,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GaussianNoise,
    GlobalAveragePooling1D,
    Input,
    Lambda,
    MaxPooling1D,
    Multiply,
    Reshape,
    add,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence, pad_sequences, to_categorical


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


seed_everything(seed=42)

# %% [markdown]
# ### 》》》**Configuration**

# %%
# (Competition metric will only be imported when TRAINing)
TRAIN = True  # ← set to True when you want to train
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
PRETRAINED_DIR = Path("/kaggle/input/pretrained-model")  # used when TRAIN=False
EXPORT_DIR = Path("./")  # artefacts will be saved here
BATCH_SIZE = 64
PAD_PERCENTILE = 95
LR_INIT = 5e-4
WD = 3e-3
MIXUP_ALPHA = 0.4
EPOCHS = 160
PATIENCE = 40


print("▶ imports ready · tensorflow", tf.__version__)

# %% [markdown]
# ### 》》》**Utility Functions**


# %%
# Tensor Manipulations
def time_sum(x):
    return K.sum(x, axis=1)


def squeeze_last_axis(x):
    return tf.squeeze(x, axis=-1)


def expand_last_axis(x):
    return tf.expand_dims(x, axis=-1)


def se_block(x, reduction=8):
    ch = x.shape[-1]
    se = GlobalAveragePooling1D()(x)
    se = Dense(ch // reduction, activation="relu")(se)
    se = Dense(ch, activation="sigmoid")(se)
    se = Reshape((1, ch))(se)
    return Multiply()([x, se])


# Residual CNN Block with SE
def residual_se_cnn_block(x, filters, kernel_size, pool_size=2, drop=0.3, wd=1e-4):
    shortcut = x
    for _ in range(2):
        x = Conv1D(
            filters,
            kernel_size,
            padding="same",
            use_bias=False,
            kernel_regularizer=l2(wd),
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    x = se_block(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(
            filters, 1, padding="same", use_bias=False, kernel_regularizer=l2(wd)
        )(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = add([x, shortcut])
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size)(x)
    x = Dropout(drop)(x)
    return x


def attention_layer(inputs):
    score = Dense(1, activation="tanh")(inputs)
    score = Lambda(squeeze_last_axis)(score)
    weights = Activation("softmax")(score)
    weights = Lambda(expand_last_axis)(weights)
    context = Multiply()([inputs, weights])
    context = Lambda(time_sum)(context)
    return context


# %% [markdown]
# ### 》》》**Data Helpers**

# %%
# Normalizes and cleans the time series sequence.


def preprocess_sequence(
    df_seq: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler
):
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    return scaler.transform(mat).astype("float32")


# MixUp the data argumentation in order to regularize the neural network.


class MixupGenerator(Sequence):
    def __init__(self, X, y, batch_size, alpha=0.2):
        self.X, self.y = X, y
        self.batch = batch_size
        self.alpha = alpha
        self.indices = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch))

    def __getitem__(self, i):
        idx = self.indices[i * self.batch : (i + 1) * self.batch]
        Xb, yb = self.X[idx], self.y[idx]
        lam = np.random.beta(self.alpha, self.alpha)
        perm = np.random.permutation(len(Xb))
        X_mix = lam * Xb + (1 - lam) * Xb[perm]
        y_mix = lam * yb + (1 - lam) * yb[perm]
        return X_mix, y_mix

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# %%
def remove_gravity_from_acc(acc_data, rot_data):
    if isinstance(acc_data, pd.DataFrame):
        acc_values = acc_data[["acc_x", "acc_y", "acc_z"]].values
    else:
        acc_values = acc_data

    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[["rot_x", "rot_y", "rot_z", "rot_w"]].values
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


# %% [markdown]
# ### 》》》**Model Definition - Two Branch Architecture**


# %%
def build_two_branch_model(pad_len, imu_dim, tof_dim, n_classes, wd=1e-4):
    inp = Input(shape=(pad_len, imu_dim + tof_dim))
    imu = Lambda(lambda t: t[:, :, :imu_dim])(inp)
    tof = Lambda(lambda t: t[:, :, imu_dim:])(inp)

    # IMU deep branch
    x1 = residual_se_cnn_block(imu, 64, 3, drop=0.1, wd=wd)
    x1 = residual_se_cnn_block(x1, 128, 5, drop=0.1, wd=wd)

    # TOF/Thermal lighter branch
    x2 = Conv1D(64, 3, padding="same", use_bias=False, kernel_regularizer=l2(wd))(tof)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)
    x2 = MaxPooling1D(2)(x2)
    x2 = Dropout(0.2)(x2)
    x2 = Conv1D(128, 3, padding="same", use_bias=False, kernel_regularizer=l2(wd))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)
    x2 = MaxPooling1D(2)(x2)
    x2 = Dropout(0.2)(x2)

    merged = Concatenate()([x1, x2])

    xa = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(wd)))(
        merged
    )
    xb = Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=l2(wd)))(
        merged
    )
    xc = GaussianNoise(0.09)(merged)
    xc = Dense(16, activation="elu")(xc)

    x = Concatenate()([xa, xb, xc])
    x = Dropout(0.4)(x)
    x = attention_layer(x)

    for units, drop in [(256, 0.5), (128, 0.3)]:
        x = Dense(units, use_bias=False, kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(drop)(x)

    out = Dense(n_classes, activation="softmax", kernel_regularizer=l2(wd))(x)
    return Model(inp, out)


tmp_model = build_two_branch_model(127, 7, 325, 18)

# %% [markdown]
# ### 》》》**Training / Inference Pipeline**

# %%
if TRAIN:
    print("▶ TRAIN MODE – loading dataset …")
    df = pd.read_csv(RAW_DIR / "train.csv")

    train_dem_df = pd.read_csv(RAW_DIR / "train_demographics.csv")
    df_for_groups = pd.merge(df.copy(), train_dem_df, on="subject", how="left")

    le = LabelEncoder()
    df["gesture_int"] = le.fit_transform(df["gesture"])
    np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)
    gesture_classes = le.classes_

    print("  Calculating base engineered IMU features (magnitude, angle)...")
    df["acc_mag"] = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)
    df["rot_angle"] = 2 * np.arccos(df["rot_w"].clip(-1, 1))

    print(
        "  Calculating engineered IMU derivatives (jerk, angular velocity) for original acc_mag..."
    )
    df["acc_mag_jerk"] = df.groupby("sequence_id")["acc_mag"].diff().fillna(0)
    df["rot_angle_vel"] = df.groupby("sequence_id")["rot_angle"].diff().fillna(0)

    print("  Removing gravity and calculating linear acceleration features...")

    linear_accel_list = []
    for _, group in df.groupby("sequence_id"):
        acc_data_group = group[["acc_x", "acc_y", "acc_z"]]
        rot_data_group = group[["rot_x", "rot_y", "rot_z", "rot_w"]]
        linear_accel_group = remove_gravity_from_acc(acc_data_group, rot_data_group)
        linear_accel_list.append(
            pd.DataFrame(
                linear_accel_group,
                columns=["linear_acc_x", "linear_acc_y", "linear_acc_z"],
                index=group.index,
            )
        )

    df_linear_accel = pd.concat(linear_accel_list)
    df = pd.concat([df, df_linear_accel], axis=1)

    df["linear_acc_mag"] = np.sqrt(
        df["linear_acc_x"] ** 2 + df["linear_acc_y"] ** 2 + df["linear_acc_z"] ** 2
    )
    df["linear_acc_mag_jerk"] = (
        df.groupby("sequence_id")["linear_acc_mag"].diff().fillna(0)
    )

    meta_cols = {...}

    imu_cols_base = ["linear_acc_x", "linear_acc_y", "linear_acc_z"]
    imu_cols_base.extend(
        [
            c
            for c in df.columns
            if c.startswith("rot_") and c not in ["rot_angle", "rot_angle_vel"]
        ]
    )

    imu_engineered_features = [
        "acc_mag",
        "rot_angle",
        "acc_mag_jerk",
        "rot_angle_vel",
        "linear_acc_mag",
        "linear_acc_mag_jerk",
    ]
    imu_cols = imu_cols_base + imu_engineered_features
    imu_cols = list(dict.fromkeys(imu_cols))

    thm_cols_original = [c for c in df.columns if c.startswith("thm_")]

    tof_aggregated_cols_template = []
    for i in range(1, 6):
        tof_aggregated_cols_template.extend(
            [f"tof_{i}_mean", f"tof_{i}_std", f"tof_{i}_min", f"tof_{i}_max"]
        )

    final_feature_cols = imu_cols + thm_cols_original + tof_aggregated_cols_template
    imu_dim_final = len(imu_cols)
    tof_thm_aggregated_dim_final = len(thm_cols_original) + len(
        tof_aggregated_cols_template
    )

    print(
        f"  IMU (incl. engineered & derivatives) {imu_dim_final} | THM + Aggregated TOF {tof_thm_aggregated_dim_final} | total {len(final_feature_cols)} features"
    )
    np.save(EXPORT_DIR / "feature_cols.npy", np.array(final_feature_cols))

    print("  Building sequences with aggregated TOF and preparing data for scaler...")
    seq_gp = df.groupby("sequence_id")

    all_steps_for_scaler_list = []
    X_list_unscaled, y_list_int_for_stratify, lens = [], [], []

    for seq_id, seq_df_orig in seq_gp:
        seq_df = seq_df_orig.copy()

        for i in range(1, 6):
            pixel_cols_tof = [f"tof_{i}_v{p}" for p in range(64)]
            tof_sensor_data = seq_df[pixel_cols_tof].replace(-1, np.nan)
            seq_df[f"tof_{i}_mean"] = tof_sensor_data.mean(axis=1)
            seq_df[f"tof_{i}_std"] = tof_sensor_data.std(axis=1)
            seq_df[f"tof_{i}_min"] = tof_sensor_data.min(axis=1)
            seq_df[f"tof_{i}_max"] = tof_sensor_data.max(axis=1)

        mat_unscaled = (
            seq_df[final_feature_cols]
            .ffill()
            .bfill()
            .fillna(0)
            .values.astype("float32")
        )

        all_steps_for_scaler_list.append(mat_unscaled)
        X_list_unscaled.append(mat_unscaled)
        y_list_int_for_stratify.append(seq_df["gesture_int"].iloc[0])
        lens.append(len(mat_unscaled))

    print("  Fitting StandardScaler...")
    all_steps_concatenated = np.concatenate(all_steps_for_scaler_list, axis=0)
    scaler = StandardScaler().fit(all_steps_concatenated)
    joblib.dump(scaler, EXPORT_DIR / "scaler.pkl")
    del all_steps_for_scaler_list, all_steps_concatenated

    print("  Scaling and padding sequences...")
    X_scaled_list = [scaler.transform(x_seq) for x_seq in X_list_unscaled]
    del X_list_unscaled

    pad_len = int(np.percentile(lens, PAD_PERCENTILE))
    np.save(EXPORT_DIR / "sequence_maxlen.npy", pad_len)

    X = pad_sequences(
        X_scaled_list,
        maxlen=pad_len,
        padding="post",
        truncating="post",
        dtype="float32",
    )
    del X_scaled_list

    y_int_for_stratify = np.array(y_list_int_for_stratify)
    y = to_categorical(y_int_for_stratify, num_classes=len(le.classes_))

    print("  Splitting data and preparing for training...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=82, stratify=y_int_for_stratify
    )

    cw_vals = compute_class_weight(
        "balanced", classes=np.arange(len(le.classes_)), y=y_int_for_stratify
    )
    class_weight = dict(enumerate(cw_vals))

    model = build_two_branch_model(
        pad_len, imu_dim_final, tof_thm_aggregated_dim_final, len(le.classes_), wd=WD
    )

    steps = len(X_tr) // BATCH_SIZE
    lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
        5e-4, first_decay_steps=15 * steps
    )

    model.compile(
        optimizer=Adam(lr_sched),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    train_gen = MixupGenerator(X_tr, y_tr, batch_size=BATCH_SIZE, alpha=MIXUP_ALPHA)
    cb = EarlyStopping(
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
        monitor="val_accuracy",
        mode="max",
    )

    print("  Starting model training...")
    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=[cb],
        verbose=1,
    )

    model.save(EXPORT_DIR / "gesture_two_branch_mixup.h5")
    print("✔ Training done – artefacts saved in", EXPORT_DIR)

    from cmi_2025_metric_copy_for_import import CompetitionMetric

    preds_val = model.predict(X_val).argmax(1)
    true_val_int = y_val.argmax(1)

    h_f1 = CompetitionMetric().calculate_hierarchical_f1(
        pd.DataFrame({"gesture": le.classes_[true_val_int]}),
        pd.DataFrame({"gesture": le.classes_[preds_val]}),
    )
    print("Hold‑out H‑F1 =", round(h_f1, 4))

else:
    print("▶ INFERENCE MODE – loading artefacts from", PRETRAINED_DIR)
    final_feature_cols = np.load(
        PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True
    ).tolist()
    pad_len = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
    scaler = joblib.load(PRETRAINED_DIR / "scaler.pkl")
    gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

    temp_imu_cols = [
        c for c in final_feature_cols if c.startswith("acc_") or c.startswith("rot_")
    ]
    imu_dim_final = len(temp_imu_cols)
    tof_thm_aggregated_dim_final = len(final_feature_cols) - imu_dim_final

    custom_objs = {
        "time_sum": time_sum,
        "squeeze_last_axis": squeeze_last_axis,
        "expand_last_axis": expand_last_axis,
        "se_block": se_block,
        "residual_se_cnn_block": residual_se_cnn_block,
        "attention_layer": attention_layer,
    }
    model = load_model(
        PRETRAINED_DIR / "gesture_two_branch_mixup.h5",
        compile=False,
        custom_objects=custom_objs,
    )
    print("  Model, scaler, feature_cols, pad_len loaded – ready for evaluation")

# %% [markdown]
# ### 》》》**Predict**


# %%
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    df_seq = sequence.to_pandas()

    df_seq["acc_mag"] = np.sqrt(
        df_seq["acc_x"] ** 2 + df_seq["acc_y"] ** 2 + df_seq["acc_z"] ** 2
    )
    df_seq["rot_angle"] = 2 * np.arccos(df_seq["rot_w"].clip(-1, 1))
    df_seq["acc_mag_jerk"] = df_seq["acc_mag"].diff().fillna(0)
    df_seq["rot_angle_vel"] = df_seq["rot_angle"].diff().fillna(0)

    acc_cols_for_gravity_removal = ["acc_x", "acc_y", "acc_z"]
    rot_cols_for_gravity_removal = ["rot_x", "rot_y", "rot_z", "rot_w"]

    if not all(
        col in df_seq.columns
        for col in acc_cols_for_gravity_removal + rot_cols_for_gravity_removal
    ):
        print(
            "Warning: Missing raw acc/rot columns for gravity removal in predict for sequence. Using raw acc as linear."
        )
        df_seq["linear_acc_x"] = df_seq.get("acc_x", 0)
        df_seq["linear_acc_y"] = df_seq.get("acc_y", 0)
        df_seq["linear_acc_z"] = df_seq.get("acc_z", 0)
    else:
        acc_data_seq = df_seq[acc_cols_for_gravity_removal]
        rot_data_seq = df_seq[rot_cols_for_gravity_removal]
        linear_accel_seq_arr = remove_gravity_from_acc(acc_data_seq, rot_data_seq)

        df_seq["linear_acc_x"] = linear_accel_seq_arr[:, 0]
        df_seq["linear_acc_y"] = linear_accel_seq_arr[:, 1]
        df_seq["linear_acc_z"] = linear_accel_seq_arr[:, 2]

    df_seq["linear_acc_mag"] = np.sqrt(
        df_seq["linear_acc_x"] ** 2
        + df_seq["linear_acc_y"] ** 2
        + df_seq["linear_acc_z"] ** 2
    )
    df_seq["linear_acc_mag_jerk"] = df_seq["linear_acc_mag"].diff().fillna(0)

    for i in range(1, 6):
        pixel_cols_tof = [f"tof_{i}_v{p}" for p in range(64)]
        if not all(col in df_seq.columns for col in pixel_cols_tof):
            print(
                f"Warning: Missing some TOF pixel columns for tof_{i} in predict. Filling aggregates with 0."
            )
            df_seq[f"tof_{i}_mean"] = 0
            df_seq[f"tof_{i}_std"] = 0
            df_seq[f"tof_{i}_min"] = 0
            df_seq[f"tof_{i}_max"] = 0
            continue

        tof_sensor_data = df_seq[pixel_cols_tof].replace(-1, np.nan)
        df_seq[f"tof_{i}_mean"] = tof_sensor_data.mean(axis=1)
        df_seq[f"tof_{i}_std"] = tof_sensor_data.std(axis=1)
        df_seq[f"tof_{i}_min"] = tof_sensor_data.min(axis=1)
        df_seq[f"tof_{i}_max"] = tof_sensor_data.max(axis=1)

    if "tof_range_across_sensors" in final_feature_cols:
        tof_mean_cols_for_contrast = [
            f"tof_{i}_mean" for i in range(1, 6) if f"tof_{i}_mean" in df_seq.columns
        ]
        thm_cols_for_contrast = [
            f"thm_{i}" for i in range(1, 6) if f"thm_{i}" in df_seq.columns
        ]

        if tof_mean_cols_for_contrast:
            tof_values_for_contrast = df_seq[tof_mean_cols_for_contrast]
            df_seq["tof_range_across_sensors"] = tof_values_for_contrast.max(
                axis=1
            ) - tof_values_for_contrast.min(axis=1)
            df_seq["tof_std_across_sensors"] = tof_values_for_contrast.std(axis=1)
        else:
            df_seq["tof_range_across_sensors"] = 0
            df_seq["tof_std_across_sensors"] = 0

        if thm_cols_for_contrast:
            thm_values_for_contrast = df_seq[thm_cols_for_contrast]
            df_seq["thm_range_across_sensors"] = thm_values_for_contrast.max(
                axis=1
            ) - thm_values_for_contrast.min(axis=1)
            df_seq["thm_std_across_sensors"] = thm_values_for_contrast.std(axis=1)
        else:
            df_seq["thm_range_across_sensors"] = 0
            df_seq["thm_std_across_sensors"] = 0

    df_seq_final_features = pd.DataFrame(index=df_seq.index)
    for col_name in final_feature_cols:
        if col_name in df_seq.columns:
            df_seq_final_features[col_name] = df_seq[col_name]
        else:
            print(
                f"CRITICAL ERROR IN PREDICT: Feature '{col_name}' expected by model (from final_feature_cols) was NOT generated in df_seq. Filling with 0. THIS IS LIKELY A BUG."
            )
            df_seq_final_features[col_name] = 0

    mat_unscaled = (
        df_seq_final_features.ffill().bfill().fillna(0).values.astype("float32")
    )

    mat_scaled = scaler.transform(mat_unscaled)

    pad_input = pad_sequences(
        [mat_scaled], maxlen=pad_len, padding="post", truncating="post", dtype="float32"
    )

    idx = int(model.predict(pad_input, verbose=0).argmax(1)[0])
    return str(gesture_classes[idx])


# %% [markdown]
# ### 》》》**Submit Inference server**

# %%
import kaggle_evaluation.cmi_inference_server

inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv",
            "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv",
        )
    )

# %%
