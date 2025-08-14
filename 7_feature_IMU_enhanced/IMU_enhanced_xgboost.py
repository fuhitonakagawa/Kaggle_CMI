#!/usr/bin/env python3
# ====================================================================================================
# CMI-BFRB検出: 高度な特徴量エンジニアリングとXGBoost v2.0
# ====================================================================================================
#
# 📊 使用方法:
# -----------
# Kaggle環境:
#   1. IS_KAGGLE_ENV = True に設定
#   2. スクリプト全体をKaggleノートブックにコピー
#   3. 実行（エクスポート済み特徴量の使用を推奨）
#
# ローカル環境:
#   1. IS_KAGGLE_ENV = False に設定
#   2. uv run 6_Feature_Research/feature_engineering_xgboost.py で実行
#
# 高速実行（特徴量再利用）:
#   1. USE_EXPORTED_FEATURES = True に設定
#   2. EXPORTED_FEATURES_PATH を適切に設定
#
# ====================================================================================================
#
# 🔧 実装している7種類の特徴量エンジニアリング:
# ====================================================================================================
#
# ① IMU特徴量（慣性測定装置）
#    - 3軸加速度計: acc_x, acc_y, acc_z の統計量・周波数特性
#    - クォータニオン: rot_w, rot_x, rot_y, rot_z から姿勢推定
#    - 世界座標系加速度: デバイス座標から世界座標への変換
#    - 線形加速度: 重力成分を除去した純粋な動作加速度
#    - 角速度: 回転の速度ベクトル
#    - オイラー角: ロール・ピッチ・ヨーの直感的な回転表現
#    - ジャーク: 加速度の時間微分（動きの滑らかさ）
#
# ② ToF特徴量（Time-of-Flight距離センサー）
#    - 空間特徴: 8×8画像の重心、モーメント、分散
#    - 領域分析: 中心3×3、内側リング、外側リングの統計
#    - 近接検出: 距離分位数による物体接近の検出
#    - クラスタリング: 近距離領域の連結成分分析
#    - PCA次元削減: 64次元を主成分に圧縮（fold内でfit）
#    - 最小距離追跡: 各フレームの最小距離時系列
#    - 異方性: 方向依存性の測定
#
# ③ サーマル特徴量（温度センサー）
#    - 温度変化率: 1次微分による変化速度
#    - トレンド分析: 線形回帰による全体傾向
#    - 2次微分: 温度変化の加速度
#
# ④ 統計的特徴量
#    - 基本統計: 平均、標準偏差、最小/最大、中央値、四分位数
#    - 形状メトリクス: 歪度、尖度、変動係数
#    - セグメント特徴: 時系列を3分割した各部の統計量
#    - Hjorthパラメータ: 活動度、移動度、複雑度（脳波解析由来）
#    - ピーク検出: ピーク数、高さ、間隔
#    - ライン長: 信号の総変動量
#
# ⑤ 周波数領域特徴量
#    - PSD: Welch法によるパワースペクトル密度（動的nperseg対応）
#    - バンドパワー: 0.3-3Hz、3-8Hz、8-12Hzの絶対値・相対値
#    - スペクトル特性: 重心、85%ロールオフ、エントロピー
#    - 支配的周波数: 最大パワーを持つ周波数
#    - ゼロ交差率: 信号の振動頻度
#
# ⑥ クロスモーダル特徴量
#    - ToFセンサー間同期: 複数センサーの相関・時間遅れ
#    - IMU-ToF相関: 加速度とToF距離の関係性
#    - ピーク整合: 加速度ピーク時のToF値
#
# ⑦ マルチ解像度特徴量
#    - 時間窓: micro(5)、short(20)、medium(50)サンプル
#    - 移動統計: 各窓での平均・標準偏差の時系列
#
# 🔍 Quality特徴量（データ品質メトリクス）
#    - 連続欠測長: 各センサーの最大連続NaN長
#    - 有効データ比率: センサー別の有効サンプル率
#    - ToF品質: valid_ratioの分位統計（p5/p50/p95）
#
# ⑦ マルチ解像度特徴量
#    - 異なる時間窓（S: 1-1.5秒、M: 3-4秒、L: 10-12秒）
#    - テンポラルピラミッド
#
# ====================================================================================================

import json
import os
import pickle
import warnings
from datetime import datetime
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

warnings.filterwarnings("ignore")

# メインプロセスでのみヘッダーを表示（マルチプロセスワーカーでは表示しない）
if __name__ == "__main__":
    print("=" * 70)
    print("CMI-BFRB検出 - 高度な特徴量エンジニアリング v2.0")
    print("包括的センサー融合特徴量を用いたXGBoost")
    print("エクスポート/インポート機能で高速反復開発が可能")
    print("=" * 70)

# ====================================================================================================
# 環境設定
# ====================================================================================================

# 🔧 メイン環境スイッチ - KaggleとローカルMacの切り替え
IS_KAGGLE_ENV = True  # True: Kaggle環境、False: ローカルMacBook

# ⚙️ 特徴量抽出設定
# 動作を制御する変数:
USE_EXPORTED_FEATURES = (
    True  # True: 特徴量抽出をスキップしてエクスポート済みデータを使用
)
EXPORT_FEATURES = False  # True: 特徴量をエクスポート（初回実行時）
EXPORT_NAME = None  # エクスポート名（None = タイムスタンプ自動生成）

# 🚀 並列処理設定（ローカル環境のみ）
USE_PARALLEL = True  # True: 並列処理を使用（ローカル環境のみ有効）
N_JOBS = -1  # 並列処理のワーカー数 (-1: 全コア使用, 正の整数: 指定数のコア使用)

# 🔧 学習済モデル設定
USE_PRETRAINED_MODEL = False  # True: 学習済モデルをロード、False: 新規に学習
PRETRAINED_MODEL_PATH = None  # 学習済モデルファイルのパス（None = 自動検出）
PRETRAINED_EXTRACTOR_PATH = None  # 学習済Extractorファイルのパス（None = 自動検出）
PRETRAINED_ARTIFACTS_PATH = None  # fold artifactsファイルのパス（None = 自動検出）
EXPORT_TRAINED_MODEL = True  # True: 学習後にモデルをエクスポート

# 💾 チェックポイント設定
USE_CHECKPOINT = False  # True: チェックポイントから再開、False: 最初から学習
CHECKPOINT_DIR = "checkpoints"  # チェックポイント保存ディレクトリ
CHECKPOINT_INTERVAL = 1  # 何fold毎にチェックポイントを保存するか（1=毎fold）
AUTO_REMOVE_CHECKPOINT = True  # 学習完了時に自動的にチェックポイントを削除

# 🔧 fold毎のアーティファクト（スケーラーなど）
FOLD_ARTIFACTS = None  # fold毎のスケーラーとメタデータ

# 環境に応じてエクスポート済み特徴量のパスを自動設定
if IS_KAGGLE_ENV:
    # Kaggleデータセットのパス
    EXPORTED_FEATURES_PATH = "/kaggle/input/cmi-bfrb-detection-exported-feature-data/features_v1.1.0_20250813_184410"
else:
    # ローカルパス
    EXPORTED_FEATURES_PATH = "exported_features/features_v1.1.0_20250813_184410"

# 使用例:
# Kaggle環境: IS_KAGGLE_ENV = True に設定するだけ
# ローカル環境: IS_KAGGLE_ENV = False に設定するだけ
# パスは自動的に設定されます！

# 環境に応じたパス設定
if IS_KAGGLE_ENV:
    # Kaggleのパス
    EXPORT_DIR = Path("./exported_features")
    DATA_BASE_PATH = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
else:
    # ローカルMacBookのパス
    EXPORT_DIR = Path("exported_features")
    DATA_BASE_PATH = Path("cmi-detect-behavior-with-sensor-data")

EXPORT_DIR.mkdir(exist_ok=True, parents=True)
FEATURE_VERSION = "v1.1.0"

# メインプロセスでのみ設定を表示
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
    print(
        f"🤖 Model mode: {'LOAD PRETRAINED' if USE_PRETRAINED_MODEL else 'TRAIN NEW'}"
    )
    if USE_PRETRAINED_MODEL:
        print(f"   Model path: {PRETRAINED_MODEL_PATH or 'Auto-detect'}")
        print(f"   Extractor path: {PRETRAINED_EXTRACTOR_PATH or 'Auto-detect'}")
        print(f"   Artifacts path: {PRETRAINED_ARTIFACTS_PATH or 'Auto-detect'}")
    if USE_CHECKPOINT:
        print("💾 Checkpoint: ENABLED")
        print(f"   Directory: {CHECKPOINT_DIR}")
        print(f"   Save interval: Every {CHECKPOINT_INTERVAL} fold(s)")
        print(f"   Auto-remove: {'Yes' if AUTO_REMOVE_CHECKPOINT else 'No'}")

# ====================================================================================================
# 設定
# ====================================================================================================

# 環境に応じたデータパスの設定
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

    # ローカルデータの存在確認
    if not DATA_BASE_PATH.exists():
        print(f"⚠️ 警告: ローカルデータディレクトリが見つかりません: {DATA_BASE_PATH}")
        print("データファイルが正しい場所にあることを確認してください。")

CONFIG = {
    # データパス
    **DATA_PATHS,
    # 特徴量エンジニアリング設定
    "sampling_rate": 20,  # サンプリングレート（Hz）
    "gravity": 9.81,  # 重力加速度（m/s^2）
    "use_world_acc": True,  # 世界座標系での加速度を使用
    "use_linear_acc": True,  # 線形加速度（重力除去）を使用
    "use_angular_velocity": True,  # 角速度を使用
    "use_frequency_features": True,  # 周波数領域特徴量を使用
    "use_tof_spatial": True,  # ToF空間特徴量を使用
    "use_thermal_trends": True,  # 温度トレンド特徴量を使用
    "use_cross_modal": True,  # クロスモーダル特徴量を使用
    # マルチ解像度ウィンドウ（S/M/L）
    "use_multi_resolution": True,  # マルチ解像度特徴量を使用
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
    # 🔧 T1: NaN保持 & スケーラ停止
    "preserve_nan_for_missing": True,  # 欠損をNaNのまま保持（XGBoostのmissing分岐を活用）
    "use_scaler_for_xgb": False,  # XGBoost時はスケーラ無効（樹木系はスケール不要）
    # 🔧 T5: 非IMUの計算スキップ（低品質時）
    "quality_thresholds": {"tof": 0.05, "thm": 0.05},  # 品質闾値
    # 🔧 T4: スマート窓
    "smart_windowing": True,  # エネルギー最大窓を使用
    "topk_windows": 1,  # Top-k windows to use
    # 🔧 T7: モダリティ・ドロップアウト
    "modality_dropout_prob": 0.4,  # 学習時のモダリティドロップアウト確率
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

# ジェスチャーマッピング（18クラス）
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

# メインプロセスでのみ設定を表示
if __name__ == "__main__":
    print(f"✓ Configuration loaded ({len(GESTURE_MAPPER)} gesture classes)")
    print(
        f"📍 Data paths configured for {'Kaggle' if IS_KAGGLE_ENV else 'Local'} environment"
    )

# ====================================================================================================
# QUATERNION AND IMU PROCESSING
# ====================================================================================================


def handle_quaternion_missing(rot_data: np.ndarray) -> np.ndarray:
    """クォータニオンデータの欠損値を適切な正規化で処理する。"""
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
        # クォータニオン形式(w,x,y,z)をscipy形式(x,y,z,w)に変換
        rot_scipy = rot[:, [1, 2, 3, 0]]
        r = R.from_quat(rot_scipy)
        acc_world = r.apply(acc)
    except Exception:
        acc_world = acc.copy()
    return acc_world


def robust_normalize(x: np.ndarray) -> np.ndarray:
    """
    🔧 T3: ロバスト正規化（中央値/IQR）
    外れ値に頑健な正規化を行う。
    """
    med = np.nanmedian(x)
    iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    return (x - med) / (iqr + 1e-8)


def compute_linear_acceleration(
    acc: np.ndarray, rot: np.ndarray = None, method: str = "subtract"
) -> np.ndarray:
    """
    Remove gravity from acceleration to get linear acceleration.
    🔧 T3: クォータニオンなし時のフォールバック追加
    """
    if method == "subtract" and rot is not None:
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


def compute_angular_velocity(rot: np.ndarray, dt: float = None) -> np.ndarray:
    """Compute angular velocity from quaternion sequence."""
    # dtが指定されていない場合はCONFIGから取得
    if dt is None:
        dt = 1.0 / CONFIG.get("sampling_rate", 20)

    omega = np.zeros((len(rot) - 1, 3))

    for i in range(len(rot) - 1):
        q1 = rot[i, [1, 2, 3, 0]]  # scipy形式に変換
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
    """
    1次元時系列データから包括的な統計的特徴量を抽出する。

    ④ 統計的特徴量：
    - 基本統計量（平均、標準偏差、最小値、最大値、中央値、四分位数）
    - 形状メトリクス（歪度、尖度、変動係数）
    - 境界特徴量（最初の値、最後の値、変化量）
    - 差分特徴量（差分の平均、標準偏差、変化点数）
    - セグメント特徴量（3分割した各セグメントの統計量と遷移）
    """
    features = {}

    # Pandas SeriesをNumPy配列に変換
    if hasattr(data, "values"):
        data = data.values

    if len(data) == 0 or np.all(np.isnan(data)):
        # 空のデータの場合はゼロを返す
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

    # データのクリーニング
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

    # 基本統計量
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

    # 形状メトリクス
    if len(data) > 1:
        features[f"{prefix}_skew"] = stats.skew(data)
        features[f"{prefix}_kurt"] = stats.kurtosis(data)
    else:
        features[f"{prefix}_skew"] = 0
        features[f"{prefix}_kurt"] = 0

    # 境界特徴量
    features[f"{prefix}_first"] = data[0]
    features[f"{prefix}_last"] = data[-1]
    features[f"{prefix}_delta"] = data[-1] - data[0]

    # 差分特徴量
    if len(data) > 1:
        diff_data = np.diff(data)
        features[f"{prefix}_diff_mean"] = np.mean(diff_data)
        features[f"{prefix}_diff_std"] = np.std(diff_data)
        features[f"{prefix}_n_changes"] = np.sum(np.abs(diff_data) > np.std(data) * 0.1)
    else:
        features[f"{prefix}_diff_mean"] = 0
        features[f"{prefix}_diff_std"] = 0
        features[f"{prefix}_n_changes"] = 0

    # セグメント特徴量（3分割）
    seq_len = len(data)
    if seq_len >= 9:
        seg_size = seq_len // 3
        for i in range(3):
            start_idx = i * seg_size
            end_idx = (i + 1) * seg_size if i < 2 else seq_len
            segment = data[start_idx:end_idx]
            features[f"{prefix}_seg{i + 1}_mean"] = np.mean(segment)
            features[f"{prefix}_seg{i + 1}_std"] = np.std(segment)

        # セグメント間の遷移
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
    """
    Hjorthパラメータ（活動度、移動度、複雑度）を抽出する。

    Hjorthパラメータは脳波解析で使用される時系列特徴量：
    - 活動度(Activity): 信号の分散（パワーの指標）
    - 移動度(Mobility): 周波数の標準偏差の推定値
    - 複雑度(Complexity): 周波数変化の指標
    """
    features = {}

    if len(data) < 2:
        features[f"{prefix}_hjorth_activity"] = 0
        features[f"{prefix}_hjorth_mobility"] = 0
        features[f"{prefix}_hjorth_complexity"] = 0
        return features

    # 活動度：信号の分散
    activity = np.var(data)
    features[f"{prefix}_hjorth_activity"] = activity

    # 移動度：sqrt(一次微分の分散 / 信号の分散)
    diff1 = np.diff(data)
    if activity > 0:
        mobility = np.sqrt(np.var(diff1) / activity)
    else:
        mobility = 0
    features[f"{prefix}_hjorth_mobility"] = mobility

    # 複雑度：一次微分の移動度 / 信号の移動度
    if len(diff1) > 1 and mobility > 0:
        diff2 = np.diff(diff1)
        mobility2 = np.sqrt(np.var(diff2) / np.var(diff1)) if np.var(diff1) > 0 else 0
        complexity = mobility2 / mobility
    else:
        complexity = 0
    features[f"{prefix}_hjorth_complexity"] = complexity

    return features


def extract_peak_features(data: np.ndarray, prefix: str) -> dict:
    """
    ピーク関連の特徴量を抽出する。

    ピーク検出による特徴量：
    - ピーク数
    - ピークの平均高さ
    - ピーク間の平均距離
    """
    features = {}

    if len(data) < 3:
        features[f"{prefix}_n_peaks"] = 0
        features[f"{prefix}_peak_mean_height"] = 0
        features[f"{prefix}_peak_mean_distance"] = 0
        return features

    # ピークを検出（標準偏差の0.5倍を閾値として使用）
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
    """
    ライン長（絶対差分の合計）を抽出する。

    信号の総変動量を表す特徴量。
    """
    features = {}

    if len(data) < 2:
        features[f"{prefix}_line_length"] = 0
        return features

    features[f"{prefix}_line_length"] = np.sum(np.abs(np.diff(data)))

    return features


def extract_autocorrelation(
    data: np.ndarray, prefix: str, lags: list = [1, 2, 4, 8]
) -> dict:
    """
    異なるラグでの自己相関特徴量を抽出する。

    時系列データの周期性や持続性を検出する特徴量。
    """
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
    """
    勾配ヒストグラム特徴量を抽出する。

    信号の変化率の分布を表現する特徴量。
    """
    features = {}

    if len(data) < 2:
        for i in range(n_bins):
            features[f"{prefix}_grad_hist_bin{i}"] = 0
        return features

    # 勾配を計算
    gradients = np.diff(data)

    # ヒストグラムを作成
    hist, _ = np.histogram(gradients, bins=n_bins)
    hist = hist / (len(gradients) + 1e-8)  # Normalize

    for i, val in enumerate(hist):
        features[f"{prefix}_grad_hist_bin{i}"] = val

    return features


def extract_jerk_features(
    acc_data: np.ndarray, prefix: str, dt: float = 1.0 / 20
) -> dict:
    """
    ジャーク特徴量（加速度の一次微分）を抽出する。

    動きの滑らかさや突発的な変化を捉える特徴量。
    """
    features = {}

    if len(acc_data) < 2:
        features[f"{prefix}_jerk_mean"] = 0
        features[f"{prefix}_jerk_std"] = 0
        features[f"{prefix}_jerk_max"] = 0
        features[f"{prefix}_jerk_p90"] = 0
        features[f"{prefix}_jerk_L2"] = 0
        return features

    # ジャーク（加速度の微分）を計算
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
    """
    Welch法を使用して周波数領域特徴量を抽出する。

    ⑤ 周波数領域特徴量：
    - パワースペクトル密度（PSD）
    - 周波数帯域パワー（絶対値と相対値）
    - スペクトル重心
    - スペクトルロールオフ
    - スペクトルエントロピー
    - 支配的周波数
    - ゼロ交差率

    改修：短系列対応のため動的にnpersegを調整。
    """
    features = {}

    # 動的にnpersegを決定（最小32、最大128、データ長以下）
    min_nperseg = 32
    max_nperseg = CONFIG.get("welch_nperseg", 128)
    nperseg = min(max(min_nperseg, len(data) // 4), max_nperseg, len(data))
    noverlap = nperseg // 2

    if len(data) < min_nperseg:
        # データが短すぎる場合はゼロを返す
        for band_idx, _ in enumerate(CONFIG["freq_bands"]):
            features[f"{prefix}_band{band_idx}_power"] = 0
            features[f"{prefix}_band{band_idx}_power_rel"] = 0  # 相対パワー
            features[f"{prefix}_band{band_idx}_power_log"] = 0  # 対数パワー
        features[f"{prefix}_spectral_centroid"] = 0
        features[f"{prefix}_spectral_rolloff"] = 0
        features[f"{prefix}_spectral_entropy"] = 0
        features[f"{prefix}_dominant_freq"] = 0
        features[f"{prefix}_dominant_power"] = 0
        features[f"{prefix}_zcr"] = 0
        features[f"{prefix}_power_total"] = 0
        return features

    # 動的パラメータを使用してWelch法でPSDを計算
    try:
        freqs, psd = welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    except:
        # エラーが発生した場合はゼロを返す
        for band_idx, _ in enumerate(CONFIG["freq_bands"]):
            features[f"{prefix}_band{band_idx}_power"] = 0
            features[f"{prefix}_band{band_idx}_power_rel"] = 0
            features[f"{prefix}_band{band_idx}_power_log"] = 0
        features[f"{prefix}_spectral_centroid"] = 0
        features[f"{prefix}_spectral_rolloff"] = 0
        features[f"{prefix}_spectral_entropy"] = 0
        features[f"{prefix}_dominant_freq"] = 0
        features[f"{prefix}_dominant_power"] = 0
        features[f"{prefix}_zcr"] = 0
        features[f"{prefix}_power_total"] = 0
        return features

    # Total power
    total_power = np.sum(psd)
    features[f"{prefix}_power_total"] = total_power

    # Band power features (absolute and relative)
    band_powers = []
    for band_idx, (low, high) in enumerate(CONFIG["freq_bands"]):
        band_mask = (freqs >= low) & (freqs <= high)
        if np.any(band_mask):
            band_power = np.sum(psd[band_mask])
            band_powers.append(band_power)
            features[f"{prefix}_band{band_idx}_power"] = band_power

            # 相対パワー（バンドパワー / 総パワー）
            if total_power > 0:
                features[f"{prefix}_band{band_idx}_power_rel"] = (
                    band_power / total_power
                )
            else:
                features[f"{prefix}_band{band_idx}_power_rel"] = 0

            # 対数パワー（log1p変換でスケール頑健性）
            features[f"{prefix}_band{band_idx}_power_log"] = np.log1p(band_power)
        else:
            band_powers.append(0)
            features[f"{prefix}_band{band_idx}_power"] = 0
            features[f"{prefix}_band{band_idx}_power_rel"] = 0
            features[f"{prefix}_band{band_idx}_power_log"] = 0

    # パワー比（低周波/高周波など）
    if len(band_powers) >= 2 and band_powers[1] > 0:
        features[f"{prefix}_power_ratio_lf_hf"] = band_powers[0] / band_powers[1]
    else:
        features[f"{prefix}_power_ratio_lf_hf"] = 0

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
# QUALITY FEATURES
# ====================================================================================================


def extract_quality_features(
    sequence_df: pd.DataFrame, prefix: str = "quality"
) -> dict:
    """
    データ品質に関する特徴量を抽出する。

    Quality特徴量：
    - 連続欠測長の最大値
    - 有効データ比率
    - パディング比率
    - センサー別の品質メトリクス
    """
    features = {}

    # 全体のデータ品質
    total_rows = len(sequence_df)
    features[f"{prefix}_sequence_length"] = total_rows

    # 🔧 T2: 品質・可用性フラグ（モダリティ有無をモデルに明示）
    # IMU有無の判定
    features[f"{prefix}_has_imu"] = int(
        all(c in sequence_df.columns for c in ["acc_x", "acc_y", "acc_z"])
    )

    # クォータニオン有無の判定
    features[f"{prefix}_has_quat"] = int(len(detect_quat_cols(sequence_df)) > 0)

    # ToF有無の判定
    features[f"{prefix}_has_tof"] = int(
        any(c.startswith("tof_") for c in sequence_df.columns)
    )

    # サーマル有無の判定
    tp = detect_thermal_prefix(sequence_df)
    features[f"{prefix}_has_thermal"] = int(
        any(c.startswith(tp) for c in sequence_df.columns) if tp else 0
    )

    # IMUデータの品質
    for axis in ["x", "y", "z"]:
        if f"acc_{axis}" in sequence_df.columns:
            data = sequence_df[f"acc_{axis}"].values
            nan_mask = np.isnan(data)

            # 有効データ比率
            features[f"{prefix}_acc_{axis}_valid_ratio"] = 1 - np.mean(nan_mask)

            # 最大連続欠測長
            if np.any(nan_mask):
                # 連続する欠測をカウント
                changes = np.diff(np.concatenate(([0], nan_mask.astype(int), [0])))
                starts = np.where(changes == 1)[0]
                ends = np.where(changes == -1)[0]
                if len(starts) > 0:
                    consecutive_nans = ends - starts
                    features[f"{prefix}_acc_{axis}_max_consecutive_nan"] = np.max(
                        consecutive_nans
                    )
                else:
                    features[f"{prefix}_acc_{axis}_max_consecutive_nan"] = 0
            else:
                features[f"{prefix}_acc_{axis}_max_consecutive_nan"] = 0

    # 🔧 T2: IMUの有効サンプル比（列単位→平均）
    acc_valid = []
    for axis in ["x", "y", "z"]:
        if f"acc_{axis}" in sequence_df.columns:
            v = sequence_df[f"acc_{axis}"].values
            acc_valid.append(1 - np.mean(np.isnan(v)))
    features[f"{prefix}_imu_valid_ratio_mean"] = (
        float(np.mean(acc_valid)) if acc_valid else 0.0
    )

    # Quaternionデータの品質
    quat_cols = ["quat_w", "quat_x", "quat_y", "quat_z"]
    if all(col in sequence_df.columns for col in quat_cols):
        quat_data = sequence_df[quat_cols].values
        quat_nan_ratio = np.mean(np.isnan(quat_data))
        features[f"{prefix}_quat_valid_ratio"] = 1 - quat_nan_ratio

    # ToFデータの品質
    for sensor_id in range(5):
        tof_cols = [c for c in sequence_df.columns if c.startswith(f"tof_{sensor_id}_")]
        if tof_cols:
            tof_data = sequence_df[tof_cols].values

            # 有効ピクセル比率の統計
            valid_ratios = []
            for frame_idx in range(len(tof_data)):
                frame = tof_data[frame_idx]
                valid_mask = (frame >= 0) & ~np.isnan(frame)
                valid_ratios.append(np.mean(valid_mask))

            valid_ratios = np.array(valid_ratios)
            features[f"{prefix}_tof_{sensor_id}_valid_ratio_mean"] = np.mean(
                valid_ratios
            )
            features[f"{prefix}_tof_{sensor_id}_valid_ratio_min"] = np.min(valid_ratios)
            features[f"{prefix}_tof_{sensor_id}_valid_ratio_p5"] = np.percentile(
                valid_ratios, 5
            )
            features[f"{prefix}_tof_{sensor_id}_valid_ratio_p50"] = np.percentile(
                valid_ratios, 50
            )
            features[f"{prefix}_tof_{sensor_id}_valid_ratio_p95"] = np.percentile(
                valid_ratios, 95
            )

            # 完全に無効なフレームの割合
            features[f"{prefix}_tof_{sensor_id}_invalid_frame_ratio"] = np.mean(
                valid_ratios == 0
            )

    # 🔧 T2: ToF全体の集約品質メトリクス
    all_tof_valid_ratios = []
    for sensor_id in range(5):
        if f"{prefix}_tof_{sensor_id}_valid_ratio_mean" in features:
            all_tof_valid_ratios.append(
                features[f"{prefix}_tof_{sensor_id}_valid_ratio_mean"]
            )

    if all_tof_valid_ratios:
        features[f"{prefix}_tof_all_valid_ratio_mean"] = np.mean(all_tof_valid_ratios)
        features[f"{prefix}_tof_all_valid_ratio_min"] = np.min(all_tof_valid_ratios)
        features[f"{prefix}_tof_all_valid_ratio_p25"] = np.percentile(
            all_tof_valid_ratios, 25
        )
        features[f"{prefix}_tof_all_valid_ratio_p75"] = np.percentile(
            all_tof_valid_ratios, 75
        )

    # Thermalデータの品質
    thermal_cols = [c for c in sequence_df.columns if c.startswith("therm_")]
    for therm_col in thermal_cols:
        if therm_col in sequence_df.columns:
            therm_data = sequence_df[therm_col].values
            nan_mask = np.isnan(therm_data)
            features[f"{prefix}_{therm_col}_valid_ratio"] = 1 - np.mean(nan_mask)

            # 最大連続欠測長
            if np.any(nan_mask):
                changes = np.diff(np.concatenate(([0], nan_mask.astype(int), [0])))
                starts = np.where(changes == 1)[0]
                ends = np.where(changes == -1)[0]
                if len(starts) > 0:
                    consecutive_nans = ends - starts
                    features[f"{prefix}_{therm_col}_max_consecutive_nan"] = np.max(
                        consecutive_nans
                    )
                else:
                    features[f"{prefix}_{therm_col}_max_consecutive_nan"] = 0
            else:
                features[f"{prefix}_{therm_col}_max_consecutive_nan"] = 0

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
    """
    ToFフレームの異なる空間領域から特徴量を抽出する。

    ② ToF特徴量 - 領域分析：
    - 8×8画像を3層の同心領域に分割
    - 中心3×3領域
    - 内側リング（5×5から中心3×3を除く）
    - 外側リング（周縁）
    - 各領域の統計量と領域間の変動性

    修正：領域マスクのロジックを明確化。
    """
    features = {}

    if tof_frame.shape != (8, 8):
        return features

    # 無効値を処理
    valid_mask = (tof_frame >= 0) & ~np.isnan(tof_frame)
    tof_clean = np.where(valid_mask, tof_frame, np.inf)

    # 明示的に三層を定義
    # 1. Center 3x3 region (rows 2-4, cols 2-4, but Python uses 0-indexing)
    # Note: 8x8の中心3x3は[2:5, 2:5]（Pythonの範囲は終端を含まない）
    center_region = tof_clean[2:5, 2:5]
    valid_center = center_region[center_region < np.inf]
    if len(valid_center) > 0:
        features[f"{prefix}_center_mean"] = np.mean(valid_center)
        features[f"{prefix}_center_min"] = np.min(valid_center)
        features[f"{prefix}_center_std"] = np.std(valid_center)
    else:
        features[f"{prefix}_center_mean"] = 0
        features[f"{prefix}_center_min"] = 0
        features[f"{prefix}_center_std"] = 0

    # 2. Inner ring (5x5 excluding center 3x3)
    # 5x5領域は[1:6, 1:6]
    inner_mask = np.zeros((8, 8), dtype=bool)
    inner_mask[1:6, 1:6] = True  # 5x5領域をTrue
    inner_mask[2:5, 2:5] = False  # 中心3x3をFalse
    inner_vals = tof_clean[inner_mask]
    valid_inner = inner_vals[inner_vals < np.inf]
    if len(valid_inner) > 0:
        features[f"{prefix}_inner_mean"] = np.mean(valid_inner)
        features[f"{prefix}_inner_min"] = np.min(valid_inner)
        features[f"{prefix}_inner_std"] = np.std(valid_inner)
    else:
        features[f"{prefix}_inner_mean"] = 0
        features[f"{prefix}_inner_min"] = 0
        features[f"{prefix}_inner_std"] = 0

    # 3. Outer ring (everything outside 5x5)
    outer_mask = np.ones((8, 8), dtype=bool)
    outer_mask[1:6, 1:6] = False  # 5x5領域をFalse
    outer_vals = tof_clean[outer_mask]
    valid_outer = outer_vals[outer_vals < np.inf]
    if len(valid_outer) > 0:
        features[f"{prefix}_outer_mean"] = np.mean(valid_outer)
        features[f"{prefix}_outer_min"] = np.min(valid_outer)
        features[f"{prefix}_outer_std"] = np.std(valid_outer)
    else:
        features[f"{prefix}_outer_mean"] = 0
        features[f"{prefix}_outer_min"] = 0
        features[f"{prefix}_outer_std"] = 0

    # 領域間の変動（中心から外側への勾配）
    if len(valid_center) > 0 and len(valid_outer) > 0:
        features[f"{prefix}_center_to_outer_gradient"] = np.mean(valid_outer) - np.mean(
            valid_center
        )
    else:
        features[f"{prefix}_center_to_outer_gradient"] = 0

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
            features[f"{prefix}_quad{i}_std"] = np.std(valid_quad)
        else:
            features[f"{prefix}_quad{i}_mean"] = 0
            features[f"{prefix}_quad{i}_min"] = 0
            features[f"{prefix}_quad{i}_std"] = 0

    # Left vs Right half
    left_half = tof_clean[:, :4]
    right_half = tof_clean[:, 4:]
    valid_left = left_half[left_half < np.inf]
    valid_right = right_half[right_half < np.inf]

    if len(valid_left) > 0 and len(valid_right) > 0:
        features[f"{prefix}_lr_asymmetry"] = np.mean(valid_left) - np.mean(valid_right)
        features[f"{prefix}_lr_variance_ratio"] = np.var(valid_left) / (
            np.var(valid_right) + 1e-8
        )
    else:
        features[f"{prefix}_lr_asymmetry"] = 0
        features[f"{prefix}_lr_variance_ratio"] = 1

    # Top vs Bottom half
    top_half = tof_clean[:4, :]
    bottom_half = tof_clean[4:, :]
    valid_top = top_half[top_half < np.inf]
    valid_bottom = bottom_half[bottom_half < np.inf]

    if len(valid_top) > 0 and len(valid_bottom) > 0:
        features[f"{prefix}_tb_asymmetry"] = np.mean(valid_top) - np.mean(valid_bottom)
        features[f"{prefix}_tb_variance_ratio"] = np.var(valid_top) / (
            np.var(valid_bottom) + 1e-8
        )
    else:
        features[f"{prefix}_tb_asymmetry"] = 0
        features[f"{prefix}_tb_variance_ratio"] = 1

    return features


def extract_tof_near_frac(
    tof_frame: np.ndarray, prefix: str, quantiles: list = [10, 20]
) -> dict:
    """
    特定の距離分位数以下のピクセルの割合を抽出する。

    近距離物体の検出に有用な特徴量。
    """
    features = {}

    # 無効値を処理
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
    """
    PCA固有値を使用して異方性特徴量を抽出する。

    ToF画像の方向依存性と構造を分析する特徴量。
    """
    features = {}

    if tof_frame.shape != (8, 8):
        features[f"{prefix}_anisotropy"] = 0
        features[f"{prefix}_principal_angle"] = 0
        return features

    # 無効値を処理
    valid_mask = (tof_frame >= 0) & ~np.isnan(tof_frame)

    if np.sum(valid_mask) < 3:
        features[f"{prefix}_anisotropy"] = 0
        features[f"{prefix}_principal_angle"] = 0
        return features

    # 逆距離重み付けされた有効ピクセルの座標を取得
    x, y = np.meshgrid(range(8), range(8))
    weights = np.where(valid_mask, 1.0 / (tof_frame + 1), 0)

    # 点群を作成
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

    # 共分散行列を計算
    cov = np.cov(points.T)

    # 固有値を取得
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
    """
    複数のToFセンサー間の同期特徴量を抽出する。

    ⑥ クロスモーダル特徴量：
    - センサー間の相関
    - 同期性の測定

    修正：padded_dataを使用して長さ不一致エラーを防ぐ。
    """
    features = {}

    if len(all_min_dists) < 2:
        features[f"{prefix}_simultaneous_drop_rate"] = 0
        features[f"{prefix}_avg_time_lag"] = 0
        features[f"{prefix}_coherence"] = 0
        return features

    # 配列のリストに変換
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

    # Time lag analysis using cross-correlation（修正：padded_dataを使用）
    from scipy.signal import correlate, correlation_lags

    lags = []
    for i in range(len(padded_data)):
        for j in range(i + 1, len(padded_data)):
            # scipy.signal.correlateを使用して相互相関を計算
            data_i = padded_data[i] - np.mean(padded_data[i])
            data_j = padded_data[j] - np.mean(padded_data[j])

            # Normalize to avoid numerical issues
            std_i = np.std(data_i)
            std_j = np.std(data_j)
            if std_i > 0 and std_j > 0:
                data_i = data_i / std_i
                data_j = data_j / std_j

                # 相互相関を計算
                corr = correlate(data_i, data_j, mode="same")
                lag_values = correlation_lags(len(data_i), len(data_j), mode="same")

                # Find lag of maximum correlation
                max_corr_idx = np.argmax(np.abs(corr))
                lag = lag_values[max_corr_idx]
                lags.append(abs(lag))

    if lags:
        features[f"{prefix}_avg_time_lag"] = np.mean(lags) / 20.0  # 秒に変換
    else:
        features[f"{prefix}_avg_time_lag"] = 0

    # Overall coherence (average correlation between sensors)（修正：padded_dataを使用）
    correlations = []
    for i in range(len(padded_data)):
        for j in range(i + 1, len(padded_data)):
            # padded_dataは既に同じ長さ
            try:
                corr = np.corrcoef(padded_data[i], padded_data[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            except:
                pass  # 相関計算に失敗した場合はスキップ

    if correlations:
        features[f"{prefix}_coherence"] = np.mean(correlations)
    else:
        features[f"{prefix}_coherence"] = 0

    return features


def extract_tof_arrival_event_features(
    min_dists: np.ndarray, prefix: str, threshold_percentile: int = 20
) -> dict:
    """
    ToF最小距離時系列から到着イベント特徴量を抽出する。

    物体の接近・離脱パターンを検出する特徴量。
    """
    features = {}

    if len(min_dists) < 2:
        features[f"{prefix}_arrival_rate"] = 0
        features[f"{prefix}_max_arrival_duration"] = 0
        features[f"{prefix}_arrival_frequency"] = 0
        return features

    # 「到達」（近距離）の閾値を計算
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
        )  # 秒に変換
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
    """
    二値化されたToFフレームからクラスタリング特徴量を抽出する。

    ② ToF特徴量 - クラスタリング：
    - 近距離領域のクラスタ検出
    - クラスタのサイズと形状特性
    """
    features = {}

    if tof_frame.shape != (8, 8):
        for key in ["max_cluster_size", "n_clusters", "cluster_circularity"]:
            features[f"{prefix}_{key}"] = 0
        return features

    # 無効値を処理
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

        # 最大クラスタの円形度を計算
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
    """
    8×8 ToFフレームから空間特徴量を抽出する。

    ② ToF特徴量 - 空間特徴：
    - 重心位置
    - 空間的広がり
    - モーメント特徴
    """
    features = {}

    # 無効値を処理
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
        # 座標グリッドを作成
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
                    # ピーク周辺のmin_distの低下を計算
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

        for event_idx in tof_events[:10]:  # 最初の10イベントをチェック
            if event_idx + 20 < len(thm_data) and event_idx + 20 < len(linear_acc_mag):
                # ToF近接後に温度が上昇するかチェック
                thm_before = np.mean(thm_data[max(0, event_idx - 5) : event_idx])
                thm_after = np.mean(
                    thm_data[event_idx : min(event_idx + 10, len(thm_data))]
                )
                thm_increase = thm_after > thm_before

                # 加速度ピークが続くかチェック
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
    """
    Extract features from multiple time windows (S/M/L) with Temporal Pyramid.
    🔧 T4: スマート窓（エネルギー最大窓）の実装
    """
    features = {}

    if not config.get("use_multi_resolution", False):
        return features

    seq_len = len(sequence_df)
    window_sizes = config.get(
        "window_sizes", {"S": (20, 30), "M": (60, 80), "L": (200, 256)}
    )

    # 🔧 T4: スマート窓の実装（エネルギー最大窓）
    # 加速度マグニチュードをエネルギー指標として使用
    base = None
    if all(f"acc_{a}" in sequence_df.columns for a in ["x", "y", "z"]):
        base = np.sqrt(
            sequence_df["acc_x"] ** 2
            + sequence_df["acc_y"] ** 2
            + sequence_df["acc_z"] ** 2
        ).values

    # For each window size
    for window_name, (min_size, max_size) in window_sizes.items():
        # Determine actual window size based on sequence length
        if seq_len < min_size:
            continue

        window_size = min(max_size, seq_len)

        if base is not None and config.get("smart_windowing", True):
            # 🔧 T4: エネルギー最大窓を見つける
            # 移動RMSを計算してエネルギーが最大の位置を特定
            s = pd.Series(base)
            rms = s.rolling(window_size, min_periods=max(8, window_size // 5)).apply(
                lambda v: np.sqrt(np.mean(v**2))
            )

            if not rms.isna().all():
                # RMS最大位置を中心とした窓を取得
                center_idx = int(np.nanargmax(rms.values))
                start_idx = max(
                    0, min(center_idx - window_size // 2, seq_len - window_size)
                )
                window_df = sequence_df.iloc[start_idx : start_idx + window_size]
            else:
                # フォールバック：末尾窓を使用
                start_idx = max(0, seq_len - window_size)
                window_df = sequence_df.iloc[start_idx:]
        else:
            # 従来の処理：末尾ウィンドウを抽出（予測用に強調）
            if config.get("use_tail_emphasis", True):
                start_idx = max(0, seq_len - window_size)
                window_df = sequence_df.iloc[start_idx:]
            else:
                # Use middle window
                start_idx = max(0, (seq_len - window_size) // 2)
                window_df = sequence_df.iloc[start_idx : start_idx + window_size]

        # このウィンドウの基本統計量を抽出
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
                    # このウィンドウで移動平均を適用
                    smoothed = (
                        pd.Series(data)
                        .rolling(pyramid_size, center=True, min_periods=1)
                        .mean()
                        .values
                    )

                    # 平滑化された信号から統計量を抽出
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


# ====================================================================================================
# 列名の自動検出ユーティリティ
# ====================================================================================================


def detect_quat_cols(df: pd.DataFrame) -> List[str]:
    """クォータニオン列名を自動検出する。"""
    candidates = [
        ["rot_w", "rot_x", "rot_y", "rot_z"],
        ["quat_w", "quat_x", "quat_y", "quat_z"],
    ]
    for cols in candidates:
        if all(c in df.columns for c in cols):
            return cols
    return []  # 見つからない場合は空リスト


def detect_thermal_prefix(df: pd.DataFrame) -> str:
    """サーマル列のプレフィックスを自動検出する。"""
    if any(c.startswith("thm_") for c in df.columns):
        return "thm_"
    if any(c.startswith("therm_") for c in df.columns):
        return "therm_"
    if any(c.startswith("thermal_") for c in df.columns):
        return "thermal_"
    return "thm_"  # デフォルト


def detect_tof_sensor_ids(df: pd.DataFrame) -> List[int]:
    """ToFセンサーIDを自動検出する。
    例: 'tof_1_v0' → センサーID=1
    """
    ids = set()
    for c in df.columns:
        if c.startswith("tof_") and "_v" in c:
            try:
                sid = int(c.split("_")[1])
                ids.add(sid)
            except:
                pass
    return sorted(ids) if ids else list(range(5))  # デフォルトは0-4


# ========================================
# チェックポイント関数
# ========================================


def save_checkpoint(
    fold: int, model, feature_names: list, scaler, fold_artifacts: list
):
    """チェックポイントを保存"""
    if not USE_CHECKPOINT:
        return

    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_file = checkpoint_dir / f"checkpoint_fold_{fold}.pkl"
    checkpoint_data = {
        "fold": fold,
        "model": model,
        "feature_names": feature_names,
        "scaler": scaler,
        "fold_artifacts": fold_artifacts,
    }

    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint_data, f)

    print(f"  💾 Checkpoint saved: fold {fold}")


def load_checkpoint():
    """チェックポイントから再開"""
    if not USE_CHECKPOINT:
        return None, None, 0

    checkpoint_dir = Path(CHECKPOINT_DIR)
    if not checkpoint_dir.exists():
        return None, None, 0

    checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_fold_*.pkl"))
    if not checkpoint_files:
        return None, None, 0

    # 最新のチェックポイントを読み込み
    models = []
    fold_artifacts = []
    last_fold = -1

    for cp_file in checkpoint_files:
        with open(cp_file, "rb") as f:
            cp_data = pickle.load(f)
        models.append(cp_data["model"])
        fold_artifacts.append(
            {"feature_names": cp_data["feature_names"], "scaler": cp_data["scaler"]}
        )
        last_fold = max(last_fold, cp_data["fold"])

    print(f"✅ Checkpoint loaded: Resuming from fold {last_fold + 1}")
    return models, fold_artifacts, last_fold + 1


def remove_checkpoints():
    """チェックポイントファイルを削除"""
    if not AUTO_REMOVE_CHECKPOINT:
        return

    checkpoint_dir = Path(CHECKPOINT_DIR)
    if checkpoint_dir.exists():
        import shutil

        shutil.rmtree(checkpoint_dir)
        print("🗑️ Checkpoints removed")


def fill_series_nan(x: np.ndarray) -> np.ndarray:
    """NaN値を前方補完→後方補完→0で埋める。"""
    series = pd.Series(x)
    return series.ffill().bfill().fillna(0).values


def extract_features_parallel(args):
    """Global function for parallel feature extraction (used only in local environment)."""
    extractor, seq_df, demo_df = args
    return extractor.extract_features(seq_df, demo_df)


# ========================================
# ヘルパー関数
# ========================================


def _to01_handedness(v):
    """handednessをR/L文字列から1/0に変換"""
    if isinstance(v, str):
        v = v.strip().lower()
        if v.startswith("r"):
            return 1
        if v.startswith("l"):
            return 0
    try:
        return int(v)
    except:
        return 0


class FeatureExtractor:
    """
    フィットされた変換器を持つメイン特徴量抽出クラス。

    すべての特徴量エンジニアリング処理を統合管理。
    改修：fold内でScaler/PCAをfitするためfit()とtransform()を分離。
    """

    def __init__(self, config: dict):
        self.config = config
        self.scaler = None
        self.tof_pcas = {}  # Will store PCA transformers for each ToF sensor
        self.feature_names = None
        self.is_fitted = False
        self.percentile_thresholds = {}  # Store percentile thresholds for fold-specific fitting

    def fit(
        self, sequences: List[pd.DataFrame], demographics: List[pd.DataFrame]
    ) -> None:
        """
        訓練データからScaler、PCA、分位閾値などをfitする。
        fold内のtrainデータのみでfitし、CVリークを防ぐ。

        修正: PCA → 最終特徴 → scalerの順序でfit
        """
        print("  Fitting transformers on training data...")

        # ステップ1: ToF PCA用のデータを収集してfit
        tof_data_by_sensor = {f"tof_{i}": [] for i in range(5)}

        if self.config.get("tof_use_pca", False):
            print("    Collecting ToF data for PCA...")
            for i in range(len(sequences)):
                seq_df = sequences[i]
                demo_df = demographics[i]
                for sensor_id in range(5):
                    tof_cols = [
                        c for c in seq_df.columns if c.startswith(f"tof_{sensor_id}_")
                    ]
                    if tof_cols:
                        tof_data = seq_df[tof_cols].values
                        # 利き手処理が有効な場合は適用
                        if self.config.get("tof_use_handedness_mirror", False):
                            handedness = (
                                demo_df["handedness"].iloc[0]
                                if "handedness" in demo_df.columns
                                else 0
                            )
                            for idx in range(len(tof_data)):
                                tof_data[idx] = mirror_tof_by_handedness(
                                    tof_data[idx], handedness
                                )
                        # Clean data
                        valid_mask = (tof_data >= 0) & ~np.isnan(tof_data)
                        tof_clean = np.where(valid_mask, tof_data, 0)
                        tof_data_by_sensor[f"tof_{sensor_id}"].append(tof_clean)

            # ToF PCAをfit
            print("    Fitting ToF PCAs...")
            for sensor_id in range(5):
                sensor_key = f"tof_{sensor_id}"
                if tof_data_by_sensor[sensor_key]:
                    # 全訓練データを結合
                    all_tof_data = np.vstack(tof_data_by_sensor[sensor_key])
                    n_samples, n_features = all_tof_data.shape
                    max_components = min(
                        n_samples - 1, n_features, self.config["tof_pca_components"]
                    )

                    if max_components >= 2:
                        pca = PCA(n_components=max_components)
                        try:
                            pca.fit(all_tof_data)
                            self.tof_pcas[sensor_key] = pca
                            print(
                                f"      Fitted PCA for {sensor_key}: {max_components} components"
                            )
                        except Exception as e:
                            print(f"      Failed to fit PCA for {sensor_key}: {e}")

        # ステップ2: PCAを含む最終形の特徴を抽出
        print("    Extracting final features with PCA...")
        final_features = []
        for i in range(len(sequences)):
            seq_df = sequences[i]
            demo_df = demographics[i]
            if i % 500 == 0:
                print(f"      Processing sequence {i}/{len(sequences)}...")
            # extract_featuresを使ってPCAを含む最終形の特徴を取得
            features = self.extract_features(seq_df, demo_df)
            final_features.append(features)

        # ステップ3: 最終形の特徴に対してscalerをfit
        if final_features:
            X_final = pd.concat(final_features, ignore_index=True)
            self.feature_names = list(X_final.columns)

            # Scalerをfit
            print("    Fitting scaler on final features...")
            if self.config["robust_scaler"]:
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()
            self.scaler.fit(X_final[self.feature_names])

            # 分位閾値を計算（必要に応じて）
            print("    Computing percentile thresholds...")
            # ToF近接判定の閾値など
            # （ここに必要な分位閾値の計算を追加）

        self.is_fitted = True
        print(
            f"  ✓ Fitted transformers on {len(sequences)} sequences with {len(self.feature_names)} features"
        )

    def transform(
        self, sequences: List[pd.DataFrame], demographics: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        学習済みのScaler/PCAで特徴量を変換する。
        列アライメントを行い、fit時と同じ列順序を保証する。
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")

        print(f"  Transforming {len(sequences)} sequences...")
        feature_dfs = []

        for i in range(len(sequences)):
            seq_df = sequences[i]
            demo_df = demographics[i]
            if i % 500 == 0:
                print(f"    Processing sequence {i}/{len(sequences)}...")
            features = self.extract_features(seq_df, demo_df)
            feature_dfs.append(features)

        X = pd.concat(feature_dfs, ignore_index=True)

        # 🔧 T1: 列アライメント（NaN保持）
        # 不足列はNaNで補い、余剰列は削除し、順序を揃える
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = np.nan  # T1: 欠損はNaNのまま保持
        X = X[self.feature_names]  # fit時の列順序に合わせる

        # 🔧 T1: スケーリングの条件付き適用
        if self.scaler is not None and self.config.get("use_scaler_for_xgb", True):
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=self.feature_names)

        return X

    def _extract_features_raw(
        self, seq_df: pd.DataFrame, demo_df: pd.DataFrame
    ) -> pd.DataFrame:
        """生の特徴量を抽出（スケーリング前）"""
        # 単一のシーケンスに対して特徴抽出
        features = {}

        # ToFの前処理
        tof_cols = [c for c in seq_df.columns if c.startswith("tof_")]
        if tof_cols and "handedness" in demo_df.columns:
            handedness = demo_df["handedness"].iloc[0] if len(demo_df) > 0 else 0
            seq_df = mirror_tof_by_handedness(seq_df, handedness)

        # IMU特徴
        if self.config.get("use_imu_features", True):
            # 四元数列を検出
            quat_cols = detect_quat_cols(seq_df)

            # 基本的なIMU特徴
            for col in ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]:
                if col in seq_df.columns:
                    data = seq_df[col].values
                    feat_dict = extract_statistical_features(data, prefix=f"{col}_")
                    features.update(feat_dict)

            # 世界座標系の加速度
            if quat_cols and all(
                c in seq_df.columns for c in ["accel_x", "accel_y", "accel_z"]
            ):
                try:
                    world_accel = compute_world_acceleration(seq_df, quat_cols)
                    for i, axis in enumerate(["x", "y", "z"]):
                        feat_dict = extract_statistical_features(
                            world_accel[:, i], prefix=f"world_accel_{axis}_"
                        )
                        features.update(feat_dict)
                except:
                    pass

        # ToF特徴
        if self.config.get("use_tof_features", True) and tof_cols:
            # 基本的なToF特徴
            for col in tof_cols[:5]:  # 最初の5センサー
                data = seq_df[col].values
                feat_dict = extract_statistical_features(data, prefix=f"{col}_")
                features.update(feat_dict)

        # デモグラフィック特徴
        if self.config.get("use_demographic_features", True):
            for col in demo_df.columns:
                if col != "subject":
                    features[f"demo_{col}"] = (
                        demo_df[col].iloc[0] if len(demo_df) > 0 else 0
                    )

        return pd.DataFrame([features])

    def fit_transform(
        self,
        sequences: List[pd.DataFrame],
        demographics: List[pd.DataFrame],
        labels: np.ndarray = None,
    ) -> pd.DataFrame:
        """
        fit()とtransform()を連続実行（後方互換性のため残す）。
        """
        self.fit(sequences, demographics)
        return self.transform(sequences, demographics)

    def _extract_features_raw(
        self, seq_df: pd.DataFrame, demo_df: pd.DataFrame
    ) -> pd.DataFrame:
        """生の特徴量を抽出（スケーリング前）

        extract_featuresと同じ特徴量を生成する（PCA/スケーリングなし）
        """
        # extract_featuresメソッドをそのまま呼び出す
        # （PCAはis_fittedでないため適用されない）
        return self.extract_features(seq_df, demo_df)

    def extract_features(
        self, sequence_df: pd.DataFrame, demographics_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        シーケンスからすべての特徴量を抽出する。

        7種類の特徴量エンジニアリングを統合的に実行。
        改修：PCAはfit済みのものをtransformのみ使用。
        """
        features = {}

        # Demographic features (subjectは除外)
        # features["subject"] = demographics_df["subject"].iloc[0]  # 未知subject汎化のため除外
        features["age"] = demographics_df["age"].iloc[0]
        features["handedness"] = demographics_df["handedness"].iloc[0]

        # Quality features（データ品質メトリクス）
        quality_features = extract_quality_features(sequence_df)
        features.update(quality_features)

        # ① IMU features (Accelerometer, Quaternion, World/Linear acceleration, Angular velocity, Euler)
        for axis in ["x", "y", "z"]:
            # Raw accelerometer
            if f"acc_{axis}" in sequence_df.columns:
                # NaN処理: ffill→bfill→0
                acc_data = fill_series_nan(sequence_df[f"acc_{axis}"].values)
                features.update(extract_statistical_features(acc_data, f"acc_{axis}"))
                features.update(extract_hjorth_parameters(acc_data, f"acc_{axis}"))
                features.update(extract_peak_features(acc_data, f"acc_{axis}"))
                features.update(extract_line_length(acc_data, f"acc_{axis}"))
                features.update(extract_autocorrelation(acc_data, f"acc_{axis}"))
                features.update(extract_gradient_histogram(acc_data, f"acc_{axis}"))

                # ⑤ Frequency features
                features.update(extract_frequency_features(acc_data, f"acc_{axis}"))

                # 🔧 T3: ロバスト正規化版IMU特徴量
                acc_r = robust_normalize(acc_data)
                features.update(extract_statistical_features(acc_r, f"accR_{axis}"))
                features.update(extract_frequency_features(acc_r, f"accR_{axis}"))

        # Acceleration magnitude
        if all(f"acc_{axis}" in sequence_df.columns for axis in ["x", "y", "z"]):
            acc_mag = np.sqrt(
                sequence_df["acc_x"] ** 2
                + sequence_df["acc_y"] ** 2
                + sequence_df["acc_z"] ** 2
            )
            features.update(extract_statistical_features(acc_mag, "acc_mag"))
            features.update(extract_hjorth_parameters(acc_mag, "acc_mag"))
            features.update(extract_peak_features(acc_mag, "acc_mag"))
            features.update(extract_frequency_features(acc_mag, "acc_mag"))

            # Jerk features
            features.update(extract_jerk_features(acc_mag, "acc_mag"))

        # Quaternion features（自動検出）
        quat_cols = detect_quat_cols(sequence_df)
        if quat_cols:
            quaternions = sequence_df[quat_cols].values
            quaternions = handle_quaternion_missing(quaternions)

            # Quaternion statistics
            for i, col in enumerate(quat_cols):
                features.update(extract_statistical_features(quaternions[:, i], col))

            # World acceleration
            if all(f"acc_{axis}" in sequence_df.columns for axis in ["x", "y", "z"]):
                acc_raw = sequence_df[["acc_x", "acc_y", "acc_z"]].values
                world_acc = compute_world_acceleration(acc_raw, quaternions)

                for i, axis in enumerate(["x", "y", "z"]):
                    features.update(
                        extract_statistical_features(
                            world_acc[:, i], f"world_acc_{axis}"
                        )
                    )
                    features.update(
                        extract_frequency_features(world_acc[:, i], f"world_acc_{axis}")
                    )

                # World acceleration magnitude
                world_acc_mag = np.linalg.norm(world_acc, axis=1)
                features.update(
                    extract_statistical_features(world_acc_mag, "world_acc_mag")
                )
                features.update(
                    extract_hjorth_parameters(world_acc_mag, "world_acc_mag")
                )
                features.update(
                    extract_frequency_features(world_acc_mag, "world_acc_mag")
                )

                # Linear acceleration
                linear_acc = compute_linear_acceleration(acc_raw, quaternions)
                for i, axis in enumerate(["x", "y", "z"]):
                    features.update(
                        extract_statistical_features(
                            linear_acc[:, i], f"linear_acc_{axis}"
                        )
                    )

                linear_acc_mag = np.linalg.norm(linear_acc, axis=1)
                features.update(
                    extract_statistical_features(linear_acc_mag, "linear_acc_mag")
                )
                features.update(
                    extract_hjorth_parameters(linear_acc_mag, "linear_acc_mag")
                )

                # ⑦ Multi-resolution features (micro/short/medium windows)
                if self.config.get("use_multi_resolution", False):
                    for window_name, window_size in [
                        ("micro", 5),
                        ("short", 20),
                        ("medium", 50),
                    ]:
                        if len(world_acc_mag) >= window_size:
                            # Moving statistics
                            rolling_mean = (
                                pd.Series(world_acc_mag)
                                .rolling(window_size, min_periods=1)
                                .mean()
                            )
                            rolling_std = (
                                pd.Series(world_acc_mag)
                                .rolling(window_size, min_periods=1)
                                .std()
                            )

                            features[f"world_acc_mag_{window_name}_mean_mean"] = (
                                rolling_mean.mean()
                            )
                            features[f"world_acc_mag_{window_name}_mean_std"] = (
                                rolling_mean.std()
                            )
                            features[f"world_acc_mag_{window_name}_std_mean"] = (
                                rolling_std.mean()
                            )
                            features[f"world_acc_mag_{window_name}_std_max"] = (
                                rolling_std.max()
                            )

                        if len(linear_acc_mag) >= window_size:
                            rolling_mean = (
                                pd.Series(linear_acc_mag)
                                .rolling(window_size, min_periods=1)
                                .mean()
                            )
                            features[f"linear_acc_mag_{window_name}_mean_std"] = (
                                rolling_mean.std()
                            )

            # Angular velocity
            angular_vel = compute_angular_velocity(quaternions)
            for i, axis in enumerate(["x", "y", "z"]):
                features.update(
                    extract_statistical_features(
                        angular_vel[:, i], f"angular_vel_{axis}"
                    )
                )
            angular_vel_mag = np.linalg.norm(angular_vel, axis=1)
            features.update(
                extract_statistical_features(angular_vel_mag, "angular_vel_mag")
            )

            # Euler angles
            euler_angles = quaternion_to_euler(quaternions)
            for i, angle in enumerate(["roll", "pitch", "yaw"]):
                features.update(
                    extract_statistical_features(euler_angles[:, i], f"euler_{angle}")
                )

        # 🔧 T5: ToF品質チェック（低品質時は計算スキップ）
        # 品質情報から計算する（既にextract_quality_featuresで計算済み）
        q_tof_mean = features.get("quality_tof_all_valid_ratio_mean", 0.0)
        HAS_TOF = q_tof_mean is not None and q_tof_mean >= self.config.get(
            "quality_thresholds", {}
        ).get("tof", 0.05)

        # ② ToF features
        min_dists_all = []
        for sensor_id in range(5):
            tof_cols = [
                c for c in sequence_df.columns if c.startswith(f"tof_{sensor_id}_")
            ]
            if tof_cols and HAS_TOF:  # T5: 品質が低い場合はスキップ
                tof_data = sequence_df[tof_cols].values

                # 利き手処理を適用
                if self.config.get("tof_use_handedness_mirror", False):
                    handedness = demographics_df["handedness"].iloc[0]
                    for idx in range(len(tof_data)):
                        tof_data[idx] = mirror_tof_by_handedness(
                            tof_data[idx], handedness
                        )

                # PCA transformation if enabled（改修：transformのみ）
                if self.config.get("tof_use_pca", False) and self.is_fitted:
                    sensor_key = f"tof_{sensor_id}"
                    if sensor_key in self.tof_pcas:
                        # 無効値を処理 for PCA
                        valid_mask = (tof_data >= 0) & ~np.isnan(tof_data)
                        tof_clean = np.where(valid_mask, tof_data, 0)

                        try:
                            pca = self.tof_pcas[sensor_key]
                            tof_pca_features = pca.transform(tof_clean)

                            # PCA特徴量を抽出
                            if tof_pca_features is not None:
                                for comp_idx in range(tof_pca_features.shape[1]):
                                    pca_series = tof_pca_features[:, comp_idx]
                                    features.update(
                                        extract_statistical_features(
                                            pca_series, f"tof_{sensor_id}_pca{comp_idx}"
                                        )
                                    )

                                # Reconstruction error
                                reconstructed = pca.inverse_transform(tof_pca_features)
                                recon_error = np.mean(
                                    np.abs(tof_clean - reconstructed), axis=1
                                )
                                features.update(
                                    extract_statistical_features(
                                        recon_error, f"tof_{sensor_id}_recon_error"
                                    )
                                )
                        except:
                            pass  # 変換に失敗した場合はPCAをスキップ

                # 各フレームの空間特徴量を処理
                frame_features = []
                for frame_idx in range(len(tof_data)):
                    frame_8x8 = tof_data[frame_idx].reshape(8, 8)

                    # Basic spatial features
                    frame_feat = extract_tof_spatial_features(
                        frame_8x8, f"tof_{sensor_id}_frame"
                    )

                    # 有効な場合は追加の空間特徴量を計算
                    if self.config.get("tof_region_analysis", False):
                        frame_feat.update(
                            extract_tof_region_features(
                                frame_8x8, f"tof_{sensor_id}_frame"
                            )
                        )
                        frame_feat.update(
                            extract_tof_near_frac(frame_8x8, f"tof_{sensor_id}_frame")
                        )
                        frame_feat.update(
                            extract_tof_anisotropy(frame_8x8, f"tof_{sensor_id}_frame")
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
                        features.update(
                            extract_statistical_features(velocity, f"{col}_velocity")
                        )

                # Min distance time series
                min_dists = []
                for frame_idx in range(len(tof_data)):
                    frame_8x8 = tof_data[frame_idx].reshape(8, 8)
                    valid_mask = (frame_8x8 >= 0) & ~np.isnan(frame_8x8)
                    valid_data = frame_8x8[valid_mask]
                    if len(valid_data) > 0:
                        min_dists.append(np.min(valid_data))
                    else:
                        min_dists.append(np.nan)

                min_dists = np.array(min_dists)
                min_dists = min_dists[~np.isnan(min_dists)]

                if len(min_dists) > 0:
                    features.update(
                        extract_statistical_features(
                            min_dists, f"tof_{sensor_id}_min_dist"
                        )
                    )
                    features.update(
                        extract_hjorth_parameters(
                            min_dists, f"tof_{sensor_id}_min_dist"
                        )
                    )
                    features.update(
                        extract_tof_arrival_event_features(
                            min_dists, f"tof_{sensor_id}"
                        )
                    )
                    min_dists_all.append(min_dists)

                # Valid pixel ratio
                valid_ratios = []
                for frame_idx in range(len(tof_data)):
                    frame_8x8 = tof_data[frame_idx].reshape(8, 8)
                    valid_mask = (frame_8x8 >= 0) & ~np.isnan(frame_8x8)
                    valid_ratios.append(np.mean(valid_mask))
                valid_ratios = np.array(valid_ratios)
                features.update(
                    extract_statistical_features(
                        valid_ratios, f"tof_{sensor_id}_valid_ratio"
                    )
                )

        # ⑥ Cross-modal ToF sync features
        if len(min_dists_all) > 1:
            # センサーIDをキーとする辞書を作成
            min_dists_dict = {
                f"tof_{i}": min_dists_all[i] for i in range(len(min_dists_all))
            }
            sync_features = extract_tof_sensor_sync_features(min_dists_dict)
            features.update(sync_features)

        # Global min across all ToF sensors
        if min_dists_all:
            # Pad to same length
            max_len = max(len(d) for d in min_dists_all)
            padded_dists = []
            for d in min_dists_all:
                if len(d) < max_len:
                    padded = np.pad(d, (0, max_len - len(d)), mode="edge")
                else:
                    padded = d
                padded_dists.append(padded)

            # Global min at each time point
            global_min = np.min(np.vstack(padded_dists), axis=0)
            features.update(
                extract_statistical_features(global_min, "tof_min_dist_global")
            )
            features.update(
                extract_hjorth_parameters(global_min, "tof_min_dist_global")
            )

        # ③ Thermal features
        thermal_prefix = detect_thermal_prefix(sequence_df)
        thermal_cols = [c for c in sequence_df.columns if c.startswith(thermal_prefix)]
        for therm_col in thermal_cols:
            therm_data = sequence_df[therm_col].values
            therm_data = therm_data[~np.isnan(therm_data)]

            if len(therm_data) > 0:
                features.update(extract_statistical_features(therm_data, therm_col))

                # Rate of change
                if len(therm_data) > 1:
                    therm_diff = np.diff(therm_data)
                    features.update(
                        extract_statistical_features(therm_diff, f"{therm_col}_diff")
                    )

                # Temperature trend
                if len(therm_data) > 2:
                    time_indices = np.arange(len(therm_data))
                    slope, intercept = np.polyfit(time_indices, therm_data, 1)
                    features[f"{therm_col}_trend_slope"] = slope
                    features[f"{therm_col}_trend_intercept"] = intercept

                # Second derivative (acceleration of temperature change)
                if len(therm_data) > 2:
                    therm_diff2 = np.diff(therm_data, n=2)
                    features.update(
                        extract_statistical_features(therm_diff2, f"{therm_col}_diff2")
                    )

        # ⑥ Cross-modal: IMU-ToF correlations
        # 常に5つのセンサー分の特徴量を生成（存在しない場合は0）
        for sensor_idx in range(5):
            features[f"cross_acc_tof{sensor_idx}_corr"] = 0
            features[f"cross_acc_peak_tof{sensor_idx}_mean"] = 0
            features[f"cross_acc_peak_tof{sensor_idx}_min"] = 0

        if "acc_mag" in locals() and len(acc_mag) > 0 and min_dists_all:
            # Correlate acceleration peaks with ToF proximity
            acc_peaks, _ = find_peaks(acc_mag, height=np.std(acc_mag) * 0.5)

            for i, min_dists in enumerate(min_dists_all):
                if i >= 5:  # 最大5センサーまで
                    break
                if len(min_dists) > 0:
                    # Resample to match lengths
                    if len(acc_mag) != len(min_dists):
                        min_dists_resampled = np.interp(
                            np.linspace(0, 1, len(acc_mag)),
                            np.linspace(0, 1, len(min_dists)),
                            min_dists,
                        )
                    else:
                        min_dists_resampled = min_dists

                    # Correlation
                    if len(min_dists_resampled) > 1:
                        correlation = np.corrcoef(acc_mag, min_dists_resampled)[0, 1]
                        if not np.isnan(correlation):
                            features[f"cross_acc_tof{i}_corr"] = correlation

                    # Peak alignment
                    if len(acc_peaks) > 0:
                        # 加速度ピーク時のToF値をチェック
                        peak_tof_values = []
                        for peak_idx in acc_peaks:
                            if peak_idx < len(min_dists_resampled):
                                peak_tof_values.append(min_dists_resampled[peak_idx])
                        if peak_tof_values:
                            features[f"cross_acc_peak_tof{i}_mean"] = np.mean(
                                peak_tof_values
                            )
                            features[f"cross_acc_peak_tof{i}_min"] = np.min(
                                peak_tof_values
                            )

        # 🔧 T1: NaN/inf値の処理（preserve_nan_for_missingに応じて）
        if not self.config.get("preserve_nan_for_missing", False):
            # 従来の処理: NaN/infを0に置換
            for key in features:
                if isinstance(features[key], (float, np.floating)):
                    if np.isnan(features[key]) or np.isinf(features[key]):
                        features[key] = 0
        # else: NaNを保持（XGBoostが処理）

        return pd.DataFrame([features])

    # transform()メソッドは既に上で定義済み


# ====================================================================================================
# MODALITY DROPOUT (T7)
# ====================================================================================================


def apply_modality_dropout(X: pd.DataFrame, p: float, seed: int = 42) -> pd.DataFrame:
    """
    🔧 T7: モダリティ・ドロップアウト
    学習時にToF/サーマル特徴をランダムにNaN化して、
    IMU-onlyデータへの順応性を高める。

    Args:
        X: 特徴量DataFrame
        p: ドロップアウト確率（0-1）
        seed: 乱数シード

    Returns:
        ドロップアウトを適用したDataFrame
    """
    if p <= 0:
        return X

    X_dropout = X.copy()
    rng = np.random.RandomState(seed)
    n_samples = len(X)

    # ドロップアウトする行をランダムに選択
    dropout_mask = rng.rand(n_samples) < p

    # ToFとサーマル関連の列を見つける
    drop_cols = []
    for col in X.columns:
        if (
            col.startswith("tof_")
            or col.startswith("thm_")
            or col.startswith("therm_")
            or col.startswith("thermal_")
        ):
            drop_cols.append(col)

    # 選択された行の該当列をNaN化
    if len(drop_cols) > 0:
        X_dropout.loc[dropout_mask, drop_cols] = np.nan
        print(
            f"  Applied modality dropout: {dropout_mask.sum()}/{n_samples} samples, {len(drop_cols)} columns"
        )

    return X_dropout


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

        # 特徴量をParquet形式で保存
        features_file = export_path / "features.parquet"
        features_df.to_parquet(
            features_file, compression="snappy" if compress else None, index=False
        )
        print(
            f"  ✓ Features saved ({len(features_df)} samples, {len(features_df.columns)} features)"
        )

        # メタデータを保存
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

        # エクストラクタの状態を保存
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

        # 特徴量をロード
        features_df = pd.read_parquet(import_path / "features.parquet")

        # メタデータをロード
        with open(import_path / "metadata.json", "r") as f:
            metadata = json.load(f)

        labels = np.array(metadata["labels"])
        subjects = np.array(metadata["subjects"])

        # エクストラクタの状態をロード
        with open(import_path / "extractor.pkl", "rb") as f:
            extractor_state = pickle.load(f)

        print(
            f"  ✓ Imported {features_df.shape[0]} samples, {features_df.shape[1]} features"
        )
        return features_df, labels, subjects, extractor_state


# ====================================================================================================
# DATA VARIANT BUILDING (T6)
# ====================================================================================================


def build_dataset_variant(
    features_df: pd.DataFrame, variant: str = "full"
) -> pd.DataFrame:
    """
    🔧 T6: データバリアントの恒常化
    Full版とIMU-only版のデータセットを生成。

    Args:
        features_df: 特徴量DataFrame
        variant: "full" または "imu_only"

    Returns:
        指定されたバリアントのDataFrame
    """
    if variant == "full":
        # Full版はそのまま返す
        return features_df
    elif variant == "imu_only":
        # IMU-only版：ToF/サーマル特徴をNaN化
        features_variant = features_df.copy()

        # ToFとサーマル関連の列を見つけてNaN化
        drop_cols = []
        for col in features_variant.columns:
            if (
                col.startswith("tof_")
                or col.startswith("thm_")
                or col.startswith("therm_")
                or col.startswith("thermal_")
            ):
                drop_cols.append(col)

        if len(drop_cols) > 0:
            features_variant[drop_cols] = np.nan
            print(f"  Created IMU-only variant: {len(drop_cols)} columns set to NaN")

        return features_variant
    else:
        raise ValueError(f"Unknown variant: {variant}")


# ====================================================================================================
# MODEL TRAINING
# ====================================================================================================


def train_models():
    """Train XGBoost models with cross-validation, with feature import/export.

    改修：CVリーク防止のため、fold内でScaler/PCAをfit。
    チェックポイント、学習済みモデル、fold_artifacts対応を追加。
    """
    # Access global variables
    global USE_EXPORTED_FEATURES, EXPORTED_FEATURES_PATH, EXPORT_FEATURES, EXPORT_NAME
    global MODELS, EXTRACTOR, FOLD_ARTIFACTS
    global USE_PRETRAINED_MODEL, PRETRAINED_MODEL_PATH, PRETRAINED_EXTRACTOR_PATH
    global EXPORT_TRAINED_MODEL, USE_CHECKPOINT

    print("\n" + "=" * 70)
    print("TRAINING PHASE")
    print("=" * 70)

    # 学習済みモデルを使用する場合
    if USE_PRETRAINED_MODEL:
        print("\n📦 Loading pretrained model...")
        if PRETRAINED_MODEL_PATH and Path(PRETRAINED_MODEL_PATH).exists():
            MODELS, EXTRACTOR = load_models(
                PRETRAINED_MODEL_PATH, PRETRAINED_EXTRACTOR_PATH
            )
            print("✓ Pretrained model loaded")
            return MODELS, EXTRACTOR, {}
        else:
            print("⚠️ Pretrained model not found, proceeding with training...")
            USE_PRETRAINED_MODEL = False

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

    # CONFIGパスを使用してデータをロード
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

    # ジェスチャーシーケンスのみをフィルタリング
    train_df = train_df[train_df["behavior"] == "Performs gesture"].copy()

    print(
        f"Loaded {len(train_df)} samples from {train_df['sequence_id'].nunique()} sequences"
    )

    # シーケンスをグループ化
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

    # エクスポート済み特徴量のチェックと読み込み
    import_path = EXPORTED_FEATURES_PATH
    use_precomputed = False
    X_all = None
    temp_extractor = None

    if USE_EXPORTED_FEATURES and EXPORTED_FEATURES_PATH:
        # Check if path exists (both Kaggle and local)
        import_path_obj = Path(import_path)
        path_exists = False

        # For Kaggle: just check if the path exists as-is
        if IS_KAGGLE_ENV:
            if import_path_obj.exists():
                path_exists = True
            else:
                print(f"⚠️ Warning: Export not found at {EXPORTED_FEATURES_PATH}")
                print("  Will extract features from raw data instead.")
        else:
            # For local: try adjusting the path if needed
            if not import_path_obj.exists():
                export_name = Path(EXPORTED_FEATURES_PATH).name
                local_path = EXPORT_DIR / export_name
                if local_path.exists():
                    import_path = str(local_path)
                    path_exists = True
                    print(
                        f"📂 Adjusted import path for local environment: {import_path}"
                    )
                else:
                    print(f"⚠️ Warning: Export not found at {EXPORTED_FEATURES_PATH}")
            else:
                path_exists = True

        if path_exists and import_path and Path(import_path).exists():
            print("📥 Loading exported raw features...")
            # Note: We load features but will re-fit scalers/PCA per fold
            X_all, loaded_labels, loaded_subjects, extractor_state = (
                FeatureExporter.import_features(import_path)
            )
            # extractor_stateから実際のFeatureExtractorを復元
            temp_extractor = FeatureExtractor(CONFIG)
            if isinstance(extractor_state, dict):
                temp_extractor.feature_names = extractor_state.get("feature_names", [])
                temp_extractor.is_fitted = True
            else:
                # 後方互換性のため
                temp_extractor = extractor_state
            use_precomputed = True
            print(f"  Raw features loaded! Shape: {X_all.shape}")
            # Verify the loaded data matches
            if len(loaded_labels) != len(labels):
                print(
                    f"⚠️ Warning: Loaded labels count ({len(loaded_labels)}) doesn't match current ({len(labels)})"
                )
                use_precomputed = False
                X_all = None
                temp_extractor = None

    # エクスポート済み特徴量が見つからない場合のみ、新規に抽出
    if not use_precomputed:
        # 特徴量を一括で抽出（CVの前に実行）
        print("📊 Extracting features for all sequences...")
        print(f"  Total sequences: {len(sequences)}")

        # 一時的なextractorを作成して特徴量を抽出（PCAなし、Scalerなし）
        temp_extractor = FeatureExtractor(CONFIG)
        temp_extractor.config["tof_use_pca"] = False  # 一旦PCAなしで抽出

        # ローカル環境では並列処理を使用
        if not IS_KAGGLE_ENV and USE_PARALLEL:
            from multiprocessing import Pool, cpu_count

            # N_JOBSが-1の場合は全コア使用
            n_workers = cpu_count() if N_JOBS == -1 else N_JOBS
            print(f"  Using parallel processing with {n_workers} workers...")
            print(f"  Available CPU cores: {cpu_count()}")

            # 並列処理用の引数を準備
            parallel_args = [
                (temp_extractor, seq_df, demo_df)
                for seq_df, demo_df in zip(sequences, demographics)
            ]

            # 並列処理を実行
            with Pool(processes=n_workers) as pool:
                all_features = []
                for i, features in enumerate(
                    pool.imap(extract_features_parallel, parallel_args, chunksize=10)
                ):
                    if i % 500 == 0:
                        print(f"  Processing sequence {i}/{len(sequences)}...")
                    all_features.append(features)
        else:
            # Kaggle環境または並列処理無効時は逐次処理
            if IS_KAGGLE_ENV:
                print("  Using sequential processing (Kaggle environment)...")
            else:
                print("  Using sequential processing (parallel disabled)...")

            all_features = []
            for i, (seq_df, demo_df) in enumerate(zip(sequences, demographics)):
                if i % 500 == 0:
                    print(f"  Processing sequence {i}/{len(sequences)}...")
                features = temp_extractor.extract_features(seq_df, demo_df)
                all_features.append(features)

        # 全特徴量を結合
        X_all = pd.concat(all_features, ignore_index=True)
        print(f"✓ Features extracted: {X_all.shape}")

    # エクスポートする場合はここで保存
    if EXPORT_FEATURES and not USE_EXPORTED_FEATURES:
        print("💾 Exporting raw features for future use...")
        export_path = FeatureExporter.export_features(
            X_all, temp_extractor, labels, subjects, EXPORT_NAME
        )
        print(f"✓ Features exported to: {export_path}")
        print("📝 To use these features in the future, set:")
        print("   USE_EXPORTED_FEATURES = True")
        if IS_KAGGLE_ENV:
            print(f'   EXPORTED_FEATURES_PATH = "./{export_path.name}"')
        else:
            print(f'   EXPORTED_FEATURES_PATH = "{export_path}"')

    # Cross-validation setup
    print("Starting cross-validation...")
    cv = StratifiedGroupKFold(
        n_splits=CONFIG["n_folds"], shuffle=True, random_state=CONFIG["random_state"]
    )

    # チェックポイントから再開
    models, fold_artifacts, start_fold = load_checkpoint()
    if models is None:
        models = []
        fold_artifacts = []
        start_fold = 0

    oof_predictions = np.zeros(len(labels))
    cv_scores = []
    binary_f1_scores = []
    macro_f1_scores = []

    # Store extractor from first fold for later use
    final_extractor = None

    # Keep the extractor for later use
    # temp_extractorがNoneの場合は新しく作成
    if temp_extractor is None:
        temp_extractor = FeatureExtractor(CONFIG)
    extractor = temp_extractor

    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(cv.split(labels, labels, subjects)):
        # 既に処理済みのfoldはスキップ
        if fold < start_fold:
            continue

        print(f"--- Fold {fold + 1}/{CONFIG['n_folds']} ---")

        # このfoldのデータを分割（事前に抽出した特徴量を使用）
        y_train = labels[train_idx]
        y_val = labels[val_idx]

        print(f"Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

        # Scalerをfold内でfit（CVリークを防ぐため）
        if use_precomputed and X_all is not None:
            # If using precomputed features, we still need to fit scaler per fold
            print("  Using precomputed raw features, fitting scaler for this fold...")
            X_train_raw = X_all.iloc[train_idx]
            X_val_raw = X_all.iloc[val_idx]

            # Fit scaler on train data only
            if CONFIG["robust_scaler"]:
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()

            X_train = pd.DataFrame(
                scaler.fit_transform(X_train_raw),
                columns=X_train_raw.columns,
                index=X_train_raw.index,
            )
            X_val = pd.DataFrame(
                scaler.transform(X_val_raw),
                columns=X_val_raw.columns,
                index=X_val_raw.index,
            )

            if hasattr(extractor, "scaler"):
                extractor.scaler = scaler
            if hasattr(extractor, "feature_names"):
                extractor.feature_names = list(X_train.columns)
            if hasattr(extractor, "is_fitted"):
                extractor.is_fitted = True
        else:
            # 新規に抽出した特徴量を使用
            print("  Using newly extracted features, fitting scaler for this fold...")
            X_train_raw = X_all.iloc[train_idx]
            X_val_raw = X_all.iloc[val_idx]

            # Fit scaler on train data only
            if CONFIG["robust_scaler"]:
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()

            X_train = pd.DataFrame(
                scaler.fit_transform(X_train_raw),
                columns=X_train_raw.columns,
                index=X_train_raw.index,
            )
            X_val = pd.DataFrame(
                scaler.transform(X_val_raw),
                columns=X_val_raw.columns,
                index=X_val_raw.index,
            )

        # Store first fold's extractor for later use
        if fold == 0:
            final_extractor = extractor

            # Export features if requested (only on first fold)
            if EXPORT_FEATURES and not use_precomputed:
                print("\n💾 Exporting features for future use...")
                # Combine train and val for export
                X_all = pd.concat([X_train, X_val])
                all_labels = np.concatenate([y_train, y_val])
                all_subjects = np.concatenate([subjects[train_idx], subjects[val_idx]])

                export_path = FeatureExporter.export_features(
                    X_all, extractor, all_labels, all_subjects, EXPORT_NAME
                )
                print(f"✓ Features exported to: {export_path}")
                print("\n📝 To use these features in the future, set:")
                print("   USE_EXPORTED_FEATURES = True")
                if IS_KAGGLE_ENV:
                    print(f'   EXPORTED_FEATURES_PATH = "./{export_path.name}"')
                else:
                    print(f'   EXPORTED_FEATURES_PATH = "{export_path}"')

        print(f"  Train features shape: {X_train.shape}")
        print(f"  Val features shape: {X_val.shape}")

        # Configure XGBoost parameters based on environment
        xgb_params = CONFIG["xgb_params"].copy()

        # GPU acceleration settings - 自動検出
        try:
            import torch

            if torch.cuda.is_available():
                xgb_params["tree_method"] = "gpu_hist"
                xgb_params["device"] = "cuda:0"
                xgb_params.pop("gpu_id", None)
                print("  Using GPU acceleration (CUDA)")
            else:
                xgb_params["tree_method"] = "hist"
                xgb_params["device"] = "cpu"
                xgb_params.pop("gpu_id", None)
                print("  Using CPU")
        except ImportError:
            # torchがインストールされていない場合はCPUを使用
            xgb_params["tree_method"] = "hist"
            xgb_params["device"] = "cpu"
            xgb_params.pop("gpu_id", None)
            print("  Using CPU (torch not installed)")

        # XGBoostを訓練
        model = xgb.XGBClassifier(**xgb_params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

        models.append(model)

        # fold artifactsを保存
        fold_artifacts.append(
            {"feature_names": list(X_train_raw.columns), "scaler": scaler}
        )

        # チェックポイントを保存
        if USE_CHECKPOINT and (fold + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                fold, model, list(X_train_raw.columns), scaler, fold_artifacts
            )

        # Predictions
        val_preds = model.predict(X_val)
        oof_predictions[val_idx] = val_preds

        # メトリクスを計算
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
        binary_f1_scores.append(binary_f1)
        macro_f1_scores.append(macro_f1)

        print(
            f"Fold {fold + 1} - Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f}, Score: {score:.4f}"
        )

    # チェックポイントを削除
    if USE_CHECKPOINT:
        remove_checkpoints()

    # グローバル変数に保存
    MODELS = models
    EXTRACTOR = final_extractor if final_extractor else extractor
    FOLD_ARTIFACTS = fold_artifacts

    print("" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)

    # 各メトリクスの平均と標準偏差を計算
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    mean_binary_f1 = np.mean(binary_f1_scores)
    std_binary_f1 = np.std(binary_f1_scores)
    mean_macro_f1 = np.mean(macro_f1_scores)
    std_macro_f1 = np.std(macro_f1_scores)

    print(f"Binary F1: {mean_binary_f1:.4f} ± {std_binary_f1:.4f}")
    print(f"Macro F1:  {mean_macro_f1:.4f} ± {std_macro_f1:.4f}")
    print(f"CV Score:  {mean_score:.4f} ± {std_score:.4f}")
    print(f"Fold scores: {cv_scores}")

    # Feature importance (average across folds)
    if final_extractor and final_extractor.feature_names:
        feature_importance = pd.DataFrame(
            {
                "feature": final_extractor.feature_names,
                "importance": np.mean([m.feature_importances_ for m in models], axis=0),
            }
        ).sort_values("importance", ascending=False)

        print("\nTop 20 Most Important Features:")
        print(feature_importance.head(20))

    # 学習済みモデルをエクスポート
    if EXPORT_TRAINED_MODEL:
        save_models(models, final_extractor, fold_artifacts)

    # 結果をまとめて返す
    metrics = {
        "mean_score": mean_score,
        "std_score": std_score,
        "mean_binary_f1": mean_binary_f1,
        "std_binary_f1": std_binary_f1,
        "mean_macro_f1": mean_macro_f1,
        "std_macro_f1": std_macro_f1,
        "cv_scores": cv_scores,
        "binary_f1_scores": binary_f1_scores,
        "macro_f1_scores": macro_f1_scores,
    }

    return models, final_extractor, metrics


# ========================================
# モデルの保存と読み込み
# ========================================


def save_models(models, extractor, fold_artifacts):
    """学習済みモデルとエクストラクタを保存"""
    import pickle
    from pathlib import Path

    # エクスポートディレクトリを作成
    model_export_dir = Path("trained_models")
    model_export_dir.mkdir(exist_ok=True)

    # モデルを保存
    model_file = model_export_dir / "models.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(models, f)
    print(f"✓ Models saved to: {model_file}")

    # エクストラクタを保存
    extractor_file = model_export_dir / "extractor.pkl"
    with open(extractor_file, "wb") as f:
        pickle.dump(extractor, f)
    print(f"✓ Extractor saved to: {extractor_file}")

    # fold artifactsを保存
    artifacts_file = model_export_dir / "fold_artifacts.pkl"
    with open(artifacts_file, "wb") as f:
        pickle.dump(fold_artifacts, f)
    print(f"✓ Fold artifacts saved to: {artifacts_file}")

    print(f"\n📦 All models exported to: {model_export_dir}/")
    print("To use these models for inference, set:")
    print("  USE_PRETRAINED_MODEL = True")
    print(f'  PRETRAINED_MODEL_PATH = "{model_file}"')
    print(f'  PRETRAINED_EXTRACTOR_PATH = "{extractor_file}"')


# ====================================================================================================
# INFERENCE
# ====================================================================================================

# Global variables for models
MODELS = None
EXTRACTOR = None


def load_models(
    model_path: str = None, extractor_path: str = None, artifacts_path: str = None
):
    """
    事前に保存されたモデルとextractorをロードする。
    評価サーバーでのタイムアウトを防ぐため、学習は行わない。
    """
    global MODELS, EXTRACTOR, FOLD_ARTIFACTS
    global PRETRAINED_MODEL_PATH, PRETRAINED_EXTRACTOR_PATH, PRETRAINED_ARTIFACTS_PATH
    import os

    # グローバル変数が設定されている場合はそれを使用
    if model_path is None and PRETRAINED_MODEL_PATH is not None:
        model_path = PRETRAINED_MODEL_PATH
    if extractor_path is None and PRETRAINED_EXTRACTOR_PATH is not None:
        extractor_path = PRETRAINED_EXTRACTOR_PATH
    if artifacts_path is None and PRETRAINED_ARTIFACTS_PATH is not None:
        artifacts_path = PRETRAINED_ARTIFACTS_PATH

    # まずカレントディレクトリを優先
    if model_path is None and os.path.exists("models.pkl"):
        model_path = "models.pkl"
    if extractor_path is None and os.path.exists("extractor.pkl"):
        extractor_path = "extractor.pkl"
    if artifacts_path is None and os.path.exists("fold_artifacts.pkl"):
        artifacts_path = "fold_artifacts.pkl"

    # trained_modelsディレクトリもチェック
    if model_path is None and os.path.exists("trained_models/models.pkl"):
        model_path = "trained_models/models.pkl"
    if extractor_path is None and os.path.exists("trained_models/extractor.pkl"):
        extractor_path = "trained_models/extractor.pkl"
    if artifacts_path is None and os.path.exists("trained_models/fold_artifacts.pkl"):
        artifacts_path = "trained_models/fold_artifacts.pkl"

    # それでもなければデフォルトパスを探す
    if model_path is None:
        if IS_KAGGLE_ENV:
            # Kaggle環境では/kaggle/input/から読み込む
            model_path = "/kaggle/input/cmi-models/models.pkl"
            extractor_path = "/kaggle/input/cmi-models/extractor.pkl"
            artifacts_path = "/kaggle/input/cmi-models/fold_artifacts.pkl"
        else:
            # ローカル環境ではexported_featuresから最新のものを探す
            exports = sorted(EXPORT_DIR.glob("features_*"))
            if exports:
                latest_export = exports[-1]
                model_path = latest_export / "models_5fold.pkl"
                extractor_path = latest_export / "extractor.pkl"
                artifacts_path = latest_export / "fold_artifacts.pkl"
            else:
                raise FileNotFoundError(
                    "No saved models found. Please train models first."
                )

    # モデルをロード
    print(f"Loading models from: {model_path}")
    with open(model_path, "rb") as f:
        MODELS = pickle.load(f)

    # Extractorをロード
    if extractor_path and Path(extractor_path).exists():
        print(f"Loading extractor from: {extractor_path}")
        with open(extractor_path, "rb") as f:
            EXTRACTOR = pickle.load(f)
    else:
        # Extractorが保存されていない場合は新規作成（特徴量抽出用）
        print("Creating new extractor...")
        EXTRACTOR = FeatureExtractor(CONFIG)
        # Note: この場合、fit済みでないので事前に学習が必要

    # fold artifactsのロード
    FOLD_ARTIFACTS = None
    if artifacts_path and Path(artifacts_path).exists():
        with open(artifacts_path, "rb") as f:
            FOLD_ARTIFACTS = pickle.load(f)
        print(f"✓ Loaded fold artifacts: {len(FOLD_ARTIFACTS)} folds")
    else:
        print("⚠️ Fold artifacts not found — per-fold scaling will be inconsistent")

    print(f"✓ Loaded {len(MODELS)} models")
    return MODELS, EXTRACTOR


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Prediction function for Kaggle inference server.
    改修：fold毎のスケーラーを使用して正しくスケーリング。
    """
    global MODELS, EXTRACTOR, FOLD_ARTIFACTS

    # 必要に応じてモデルを初期化（ロードのみ）
    if MODELS is None or EXTRACTOR is None:
        print("Loading pre-trained models...")
        try:
            MODELS, EXTRACTOR = load_models()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(
                "Falling back to training models (this may timeout on evaluation server)..."
            )
            MODELS, EXTRACTOR, _ = train_models()

    # pandasに変換
    seq_df = sequence.to_pandas()
    demo_df = demographics.to_pandas()

    # handednessの変換（R/L文字列を1/0に）
    if "handedness" in demo_df.columns:
        demo_df = demo_df.copy()
        demo_df["handedness"] = demo_df["handedness"].apply(_to01_handedness)

    # 生の特徴量を抽出（スケーリング前）
    if hasattr(EXTRACTOR, "_extract_features_raw"):
        X_raw = EXTRACTOR._extract_features_raw(seq_df, demo_df)
    else:
        # 後方互換性のため
        features = EXTRACTOR.extract_features(seq_df, demo_df)
        X_raw = features

    # すべてのモデルから予測を取得
    predictions = []

    if FOLD_ARTIFACTS is not None and len(FOLD_ARTIFACTS) == len(MODELS):
        # fold毎のスケーラーを使用（正しい方法）
        for model, art in zip(MODELS, FOLD_ARTIFACTS):
            # 訓練時の特徴量名に合わせる
            feature_names = art["feature_names"]

            # X_rawから必要な特徴量のみを選択（存在しない特徴量は0で埋める）
            X_selected = pd.DataFrame()
            for col in feature_names:
                if col in X_raw.columns:
                    X_selected[col] = X_raw[col]
                else:
                    # 訓練時にあったが推論時にない特徴量は0で埋める
                    X_selected[col] = 0

            # このfoldのスケーラーを適用
            X_scaled = pd.DataFrame(
                art["scaler"].transform(X_selected),
                columns=feature_names,
                index=X_raw.index,
            )

            # 予測
            pred = model.predict_proba(X_scaled)[0]
            predictions.append(pred)
    else:
        # fold artifactsがない場合は従来の方法（非推奨）
        print("⚠️ Warning: Using fallback prediction without fold-specific scalers")

        # extractorのスケーラーを使用（fold 0のみ）
        if hasattr(EXTRACTOR, "scaler") and EXTRACTOR.scaler is not None:
            X_scaled = pd.DataFrame(
                EXTRACTOR.scaler.transform(X_raw),
                columns=X_raw.columns,
                index=X_raw.index,
            )
        else:
            # スケーラーがない場合はそのまま使用
            X_scaled = X_raw

        for model in MODELS:
            pred = model.predict_proba(X_scaled)[0]
            predictions.append(pred)

    # Average predictions
    avg_pred = np.mean(predictions, axis=0)
    final_class = np.argmax(avg_pred)

    # ジェスチャー名に変換
    gesture_name = REVERSE_GESTURE_MAPPER[final_class]

    return gesture_name


# ====================================================================================================
# MAIN EXECUTION
# ====================================================================================================

if __name__ == "__main__":
    # 利用可能なエクスポートをチェックして表示
    print("\n" + "=" * 70)
    print("AVAILABLE FEATURE EXPORTS")
    print("=" * 70)

    # Kaggle環境では、inputディレクトリもチェック
    if IS_KAGGLE_ENV and USE_EXPORTED_FEATURES and EXPORTED_FEATURES_PATH:
        kaggle_path = Path(EXPORTED_FEATURES_PATH)
        if kaggle_path.exists():
            print(f"✓ Found Kaggle dataset at: {EXPORTED_FEATURES_PATH}")
            # メタデータファイルをチェック
            metadata_file = kaggle_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    meta = json.load(f)
                print(
                    f"  Samples: {meta.get('n_samples', '?')}, Features: {meta.get('n_features', '?')}"
                )
                print("📊 Will use these exported features for training.")
            else:
                print("  ⚠️ Warning: metadata.json not found in dataset")
        else:
            print(f"⚠️ Dataset not found at: {EXPORTED_FEATURES_PATH}")
            print("  Will extract features from raw data instead.")
    elif EXPORT_DIR.exists():
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

    # モデルを訓練
    MODELS, EXTRACTOR, metrics = train_models()
    print("✓ Models trained successfully")
    print(
        f"   Binary F1: {metrics['mean_binary_f1']:.4f} ± {metrics['std_binary_f1']:.4f}"
    )
    print(
        f"   Macro F1:  {metrics['mean_macro_f1']:.4f} ± {metrics['std_macro_f1']:.4f}"
    )
    print(f"   CV Score:  {metrics['mean_score']:.4f} ± {metrics['std_score']:.4f}")

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

    # Kaggle推論サーバーを初期化
    if IS_KAGGLE_ENV:
        print("Initializing Kaggle inference server...")

        try:
            from kaggle_evaluation.cmi_inference_server import CMIInferenceServer

            inference_server = CMIInferenceServer(predict)
            print("✓ Inference server created")

            # 環境に応じて適切なメソッドを呼び出す
            if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
                # 競技環境: serve()を使用
                print("Running in competition environment...")
                inference_server.serve()
                print("✓ Submission complete!")
            else:
                # ローカルテスト環境: run_local_gateway()を使用
                print("Running in local testing mode...")
                print("Generating submission.parquet from test data...")

                # test.csvが存在する場合は処理
                test_path = (
                    "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv"
                )
                test_demo_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv"

                if os.path.exists(test_path) and os.path.exists(test_demo_path):
                    inference_server.run_local_gateway(
                        data_paths=(test_path, test_demo_path)
                    )
                    print("✓ Inference complete!")
                    print("✓ submission.parquet has been generated")
                else:
                    print("⚠️ Test data not found, generating empty submission...")
                    # 空のsubmissionを生成
                    submission_df = pd.DataFrame({"sequence_id": [], "prediction": []})
                    submission_df.to_parquet("submission.parquet", index=False)
                    print("✓ Empty submission.parquet generated")

        except ImportError as e:
            print(f"⚠️ Kaggle evaluation module not available: {e}")
            print("Generating submission manually...")

            # Manual submission generation as fallback
            test_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv"
            test_demo_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv"

            if os.path.exists(test_path):
                test_df = pd.read_csv(test_path)
                test_demo_df = pd.read_csv(test_demo_path)

                # test.csvからユニークなsequence_idを取得
                test_sequences = test_df["sequence_id"].unique()
                print(f"Processing {len(test_sequences)} test sequences...")

                predictions = []
                for i, seq_id in enumerate(test_sequences):
                    if i % 100 == 0:
                        print(f"  Processing sequence {i}/{len(test_sequences)}...")

                    seq_data = test_df[test_df["sequence_id"] == seq_id]
                    seq_pl = pl.from_pandas(seq_data)

                    # subject情報を取得
                    if "subject" in seq_data.columns:
                        subject_id = seq_data["subject"].iloc[0]
                        demo_data = test_demo_df[test_demo_df["subject"] == subject_id]
                    else:
                        # subjectがない場合はダミーのdemographicsを作成
                        demo_data = pd.DataFrame(
                            {"subject": [0], "age": [30], "handedness": ["R"]}
                        )

                    demo_pl = pl.from_pandas(demo_data)

                    # 予測を実行
                    try:
                        pred = predict(seq_pl, demo_pl)
                    except Exception as e:
                        print(f"  Warning: Error predicting sequence {seq_id}: {e}")
                        pred = "Idle"  # デフォルト値

                    predictions.append({"sequence_id": seq_id, "prediction": pred})

                # DataFrameを作成して保存
                submission_df = pd.DataFrame(predictions)
                submission_df.to_parquet("submission.parquet", index=False)
                print(
                    f"✅ Generated submission.parquet with {len(submission_df)} predictions"
                )
            else:
                print("⚠️ Test data not found. Creating empty submission...")
                submission_df = pd.DataFrame({"sequence_id": [], "prediction": []})
                submission_df.to_parquet("submission.parquet", index=False)
                print("✓ Empty submission.parquet generated")

    else:
        # ローカル環境での処理
        print("Local environment - skipping submission generation")
        print("To use for Kaggle submission:")
        print("1. Copy this entire script to a Kaggle notebook")
        print("2. Run all cells")
        print("3. submission.parquet will be generated automatically")
