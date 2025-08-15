# %% [markdown]
# <h1 style="color: #6cb4e4;  text-align: center;  padding: 0.25em;  border-top: solid 2.5px #6cb4e4;  border-bottom: solid 2.5px #6cb4e4;  background: -webkit-repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);  background: repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);height:45px;">
# <b>
# OnlyInfKernel
# </b></h1>

# %% [markdown]
# ### **ℹ️INFO**
# * First, we would like to thank our participants for sharing their excellent baselines.
#     * [Two‑Branch Human‑Activity‑Recognition Pipeline (IMU + Thermopile/TOF  + SE‑CNN + BiLSTM + Attentio](https://www.kaggle.com/code/vonmainstein/imu-tof)
#
# ### **ℹ️IMU+THM/TOF Great Related works(Published order)**
# * Thanks for sharing your implementation of the new feature.
#     * LB.75 [CMI25 | IMU+THM/TOF |TF BiLSTM+GRU+Attention|LB.75](https://www.kaggle.com/code/hideyukizushi/cmi25-imu-thm-tof-tf-bilstm-gru-attention-lb-75)
#     * LB.76 [IMU Signal Processing Optimization](https://www.kaggle.com/code/rktqwe/lb-0-76-imu-thm-tof-tf-bilstm-gru-attention)
#     * LB.77 [Gravity Component Removal from Accelerometer Data](https://www.kaggle.com/code/rktqwe/lb-0-77-linear-accel-tf-bilstm-gru-attention)
#     * LB.78 [New Feature: Angular Velocity from Quaternion Derivatives](https://www.kaggle.com/code/nksusth/lb-0-78-quaternions-tf-bilstm-gru-attention)
#     * LB.80 [IMU × TOF/THM Fusion with Physics‑FE & MixUp|LB0.8](https://www.kaggle.com/code/pepushi/imu-tof-thm-fusion-with-physics-fe-mixup-lb0-8)
#
# ---
#
# ### **ℹ️[LB.80 2025/06/28]MyUpdate**
# * In previous notebooks that used "IMU+TOF" as features, the validation strategy was TTS, which raised concerns about its robustness.
# For this reason, we will release an implementation that simply blends five models and a model with a high ValidationScore.
#
# ```
# predictions = []
# for model in models:
#     idx = int(model.predict(pad_input, verbose=0).argmax(1)[0])
#     predictions.append(idx)
#
# idx = max(set(predictions), key=predictions.count)
# return str(gesture_classes[idx])
# ```
#
# * The validationScore for the local training  model included in this notebook is
#     * [ModelWeight](https://www.kaggle.com/datasets/hideyukizushi/20250627-cmi-b-102-b-105)
#
# ```
# 0.891134700273056
# 0.8912659261884439
# 0.8914825129445727
# 0.8915471835009202
# 0.8922128108549205
# ```
#
# ### **ℹ️[LB.82 2025/07/09]MyUpdate**
# * blend use SGKF split model(The model is attached in a notebook and is publicly available.)
#     * [ModelWeight](https://www.kaggle.com/datasets/hideyukizushi/cmi-d-111)
#
# ```
# # OOF(CMI Metrics)
# 0.8120966859982701
# ```

# %% [markdown]
# <h1 style="color: #6cb4e4;  text-align: center;  padding: 0.25em;  border-top: solid 2.5px #6cb4e4;  border-bottom: solid 2.5px #6cb4e4;  background: -webkit-repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);  background: repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);height:45px;">
# <b>
# Model Detail
# </b></h1>

# %%
from PIL import Image

im1 = Image.open(
    "/kaggle/input/cmi25-imu-thmtof-tf-bilstm-gru-attentionlb-xx/CMI-Model.png"
)


def get_concat_h_multi_resize(im_list, resample=Image.BICUBIC):
    min_height = min(im.height for im in im_list)
    im_list_resize = [
        im.resize(
            (int(im.width * min_height / im.height), min_height), resample=resample
        )
        for im in im_list
    ]
    total_width = sum(im.width for im in im_list_resize)
    dst = Image.new("RGB", (total_width, min_height))
    pos_x = 0
    for im in im_list_resize:
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    return dst


get_concat_h_multi_resize([im1])

# %% [markdown]
# <h1 style="color: #6cb4e4;  text-align: center;  padding: 0.25em;  border-top: solid 2.5px #6cb4e4;  border-bottom: solid 2.5px #6cb4e4;  background: -webkit-repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);  background: repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);height:45px;">
# <b>
# InfPipeline
# </b></h1>

# %% [markdown]
# # 》》》**Importing the necessary Libraries**

# %%
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# %% [markdown]
# # 》》》**Fix Seed**
# %%
import random

import polars as pl
import tensorflow as tf
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
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
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence, pad_sequences


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
# # 》》》**Configuration**

# %%
# (Competition metric will only be imported when TRAINing)
TRAIN = False  # ← set to True when you want to train
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
PRETRAINED_DIR = Path("/kaggle/input/cmi-d-111")
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
# # 》》》**Utility Functions**


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
# # 》》》**Data Helpers**

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


def calculate_angular_velocity_from_quat(
    rot_data, time_delta=1 / 200
):  # Assuming 200Hz sampling rate
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[["rot_x", "rot_y", "rot_z", "rot_w"]].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))

    for i in range(num_samples - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i + 1]

        if (
            np.all(np.isnan(q_t))
            or np.all(np.isclose(q_t, 0))
            or np.all(np.isnan(q_t_plus_dt))
            or np.all(np.isclose(q_t_plus_dt, 0))
        ):
            continue

        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)

            # Calculate the relative rotation
            delta_rot = rot_t.inv() * rot_t_plus_dt

            # Convert delta rotation to angular velocity vector
            # The rotation vector (Euler axis * angle) scaled by 1/dt
            # is a good approximation for small delta_rot
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            # If quaternion is invalid, angular velocity remains zero
            pass

    return angular_vel


# %%
def calculate_angular_distance(rot_data):
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[["rot_x", "rot_y", "rot_z", "rot_w"]].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_dist = np.zeros(num_samples)

    for i in range(num_samples - 1):
        q1 = quat_values[i]
        q2 = quat_values[i + 1]

        if (
            np.all(np.isnan(q1))
            or np.all(np.isclose(q1, 0))
            or np.all(np.isnan(q2))
            or np.all(np.isclose(q2, 0))
        ):
            angular_dist[i] = 0  # Или np.nan, в зависимости от желаемого поведения
            continue
        try:
            # Преобразование кватернионов в объекты Rotation
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)

            # Вычисление углового расстояния: 2 * arccos(|real(p * q*)|)
            # где p* - сопряженный кватернион q
            # В scipy.spatial.transform.Rotation, r1.inv() * r2 дает относительное вращение.
            # Угол этого относительного вращения - это и есть угловое расстояние.
            relative_rotation = r1.inv() * r2

            # Угол rotation vector соответствует угловому расстоянию
            # Норма rotation vector - это угол в радианах
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0  # В случае недействительных кватернионов
            pass

    return angular_dist


# %% [markdown]
# # 》》》**Model Definition - Two Branch Architecture**


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
# <h1 style="color: #6cb4e4;  text-align: center;  padding: 0.25em;  border-top: solid 2.5px #6cb4e4;  border-bottom: solid 2.5px #6cb4e4;  background: -webkit-repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);  background: repeating-linear-gradient(-45deg, #f0f8ff, #f0f8ff 3px,#e9f4ff 3px, #e9f4ff 7px);height:45px;">
# <b>
# Blending Model
# </b></h1>

# %% [markdown]
# # 》》》**LoadBlendingModels**

# %%
print("▶ INFERENCE MODE – loading artefacts from", PRETRAINED_DIR)
final_feature_cols = np.load(
    PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True
).tolist()
pad_len = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
scaler = joblib.load(PRETRAINED_DIR / "scaler.pkl")
gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)


custom_objs = {
    "time_sum": time_sum,
    "squeeze_last_axis": squeeze_last_axis,
    "expand_last_axis": expand_last_axis,
    "se_block": se_block,
    "residual_se_cnn_block": residual_se_cnn_block,
    "attention_layer": attention_layer,
}

# ----------------------------------------------------------------- #
# Load any Models
# * is 2 Train Model Load
# ----------------------------------------------------------------- #

models = []
print("  Loading models for ensemble inference...")
for fold in range(10):
    MODEL_DIR = "/kaggle/input/cmi-d-111"

    model_path = f"{MODEL_DIR}/D-111_{fold}.h5"
    print(">>>LoadModel>>>", model_path)
    model = load_model(model_path, compile=False, custom_objects=custom_objs)
    models.append(model)
print("-" * 50)

for fold in range(10):
    MODEL_DIR = "/kaggle/input/cmi-d-111"

    model_path = f"{MODEL_DIR}/v0629_{fold}.h5"
    print(">>>LoadModel>>>", model_path)
    model = load_model(model_path, compile=False, custom_objects=custom_objs)
    models.append(model)
print("-" * 50)
print(f"[INFO]NumUseModels:{len(models)}")

# %% [markdown]
# # 》》》**Predict**


# %%
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    df_seq = sequence.to_pandas()
    linear_accel = remove_gravity_from_acc(df_seq, df_seq)
    df_seq["linear_acc_x"], df_seq["linear_acc_y"], df_seq["linear_acc_z"] = (
        linear_accel[:, 0],
        linear_accel[:, 1],
        linear_accel[:, 2],
    )
    df_seq["linear_acc_mag"] = np.sqrt(
        df_seq["linear_acc_x"] ** 2
        + df_seq["linear_acc_y"] ** 2
        + df_seq["linear_acc_z"] ** 2
    )
    df_seq["linear_acc_mag_jerk"] = df_seq["linear_acc_mag"].diff().fillna(0)
    angular_vel = calculate_angular_velocity_from_quat(df_seq)
    df_seq["angular_vel_x"], df_seq["angular_vel_y"], df_seq["angular_vel_z"] = (
        angular_vel[:, 0],
        angular_vel[:, 1],
        angular_vel[:, 2],
    )
    df_seq["angular_distance"] = calculate_angular_distance(df_seq)

    for i in range(1, 6):
        pixel_cols = [f"tof_{i}_v{p}" for p in range(64)]
        tof_data = df_seq[pixel_cols].replace(-1, np.nan)
        (
            df_seq[f"tof_{i}_mean"],
            df_seq[f"tof_{i}_std"],
            df_seq[f"tof_{i}_min"],
            df_seq[f"tof_{i}_max"],
        ) = (
            tof_data.mean(axis=1),
            tof_data.std(axis=1),
            tof_data.min(axis=1),
            tof_data.max(axis=1),
        )

    mat_unscaled = (
        df_seq[final_feature_cols].ffill().bfill().fillna(0).values.astype("float32")
    )
    mat_scaled = scaler.transform(mat_unscaled)
    pad_input = pad_sequences(
        [mat_scaled], maxlen=pad_len, padding="post", truncating="post", dtype="float32"
    )

    all_preds = [
        model.predict(pad_input, verbose=0)[0] for model in models
    ]  # 主出力のみ取得
    avg_pred = np.mean(all_preds, axis=0)
    return str(gesture_classes[avg_pred.argmax()])


# %% [markdown]
# # 》》》**Submit Inference server**

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
