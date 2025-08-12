"""
総合的な特徴量エンジニアリングモジュール
すべての特徴量抽出を統合
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from world_acceleration import (
    handle_quaternion_missing_values,
    compute_world_acceleration,
    compute_orientation_features,
    compute_rotation_energy
)
from frequency_features import (
    extract_fft_features,
    extract_spectral_features,
    extract_wavelet_features,
    extract_periodogram_features
)


def extract_statistical_features(
    data: np.ndarray, 
    prefix: str,
    functions: List[str] = None
) -> Dict[str, float]:
    """
    統計的特徴量を抽出
    
    Args:
        data: 時系列データ
        prefix: 特徴量名のプレフィックス
        functions: 計算する統計関数のリスト
    
    Returns:
        特徴量の辞書
    """
    if functions is None:
        functions = ["mean", "std", "min", "max", "median", "q25", "q75", "iqr", "skew", "kurt"]
    
    features = {}
    
    # 基本統計量
    if "mean" in functions:
        features[f'{prefix}_mean'] = np.mean(data)
    if "std" in functions:
        features[f'{prefix}_std'] = np.std(data)
    if "min" in functions:
        features[f'{prefix}_min'] = np.min(data)
    if "max" in functions:
        features[f'{prefix}_max'] = np.max(data)
    if "median" in functions:
        features[f'{prefix}_median'] = np.median(data)
    
    # パーセンタイル
    if "q25" in functions:
        features[f'{prefix}_q25'] = np.percentile(data, 25)
    if "q75" in functions:
        features[f'{prefix}_q75'] = np.percentile(data, 75)
    if "iqr" in functions:
        features[f'{prefix}_iqr'] = np.percentile(data, 75) - np.percentile(data, 25)
    
    # 高次モーメント
    if "skew" in functions and len(data) > 1:
        features[f'{prefix}_skew'] = stats.skew(data)
    if "kurt" in functions and len(data) > 1:
        features[f'{prefix}_kurt'] = stats.kurtosis(data)
    
    # 追加の統計量
    features[f'{prefix}_range'] = np.max(data) - np.min(data)
    features[f'{prefix}_cv'] = np.std(data) / (np.mean(data) + 1e-10)  # 変動係数
    
    # 時系列の特徴
    if len(data) > 1:
        features[f'{prefix}_first'] = data[0]
        features[f'{prefix}_last'] = data[-1]
        features[f'{prefix}_delta'] = data[-1] - data[0]
        
        # 差分統計量
        diff_data = np.diff(data)
        features[f'{prefix}_diff_mean'] = np.mean(diff_data)
        features[f'{prefix}_diff_std'] = np.std(diff_data)
        features[f'{prefix}_diff_max'] = np.max(np.abs(diff_data))
        
        # 変化点の数
        threshold = np.std(data) * 0.5
        features[f'{prefix}_n_changes'] = np.sum(np.abs(diff_data) > threshold)
        
        # トレンド（時間との相関）
        time_indices = np.arange(len(data))
        corr_coef = np.corrcoef(time_indices, data)[0, 1]
        features[f'{prefix}_trend'] = corr_coef if not np.isnan(corr_coef) else 0
    
    return features


def extract_segment_features(
    data: np.ndarray,
    prefix: str,
    n_segments: int = 3
) -> Dict[str, float]:
    """
    セグメント分割による特徴量抽出
    
    Args:
        data: 時系列データ
        prefix: 特徴量名のプレフィックス
        n_segments: セグメント数
    
    Returns:
        特徴量の辞書
    """
    features = {}
    
    seq_len = len(data)
    if seq_len < n_segments * 3:  # 各セグメントに最低3サンプル必要
        # データが短すぎる場合
        for i in range(n_segments):
            features[f'{prefix}_seg{i+1}_mean'] = np.mean(data)
            features[f'{prefix}_seg{i+1}_std'] = np.std(data)
        return features
    
    seg_size = seq_len // n_segments
    segments = []
    
    for i in range(n_segments):
        start_idx = i * seg_size
        end_idx = (i + 1) * seg_size if i < n_segments - 1 else seq_len
        segment = data[start_idx:end_idx]
        segments.append(segment)
        
        # セグメントごとの統計量
        features[f'{prefix}_seg{i+1}_mean'] = np.mean(segment)
        features[f'{prefix}_seg{i+1}_std'] = np.std(segment)
        features[f'{prefix}_seg{i+1}_max'] = np.max(segment)
        features[f'{prefix}_seg{i+1}_min'] = np.min(segment)
    
    # セグメント間の遷移
    for i in range(n_segments - 1):
        features[f'{prefix}_seg{i+1}_to_{i+2}_diff'] = (
            np.mean(segments[i+1]) - np.mean(segments[i])
        )
    
    # セグメント間の分散比
    segment_vars = [np.var(seg) for seg in segments]
    if np.mean(segment_vars) > 0:
        features[f'{prefix}_segment_var_ratio'] = np.max(segment_vars) / (np.mean(segment_vars) + 1e-10)
    
    return features


def extract_jerk_features(
    acc_data: pd.DataFrame,
    sampling_rate: float = 20.0
) -> Dict[str, float]:
    """
    ジャーク（加速度の変化率）特徴量を抽出
    
    Args:
        acc_data: 加速度データ (acc_x, acc_y, acc_z列を含む)
        sampling_rate: サンプリングレート
    
    Returns:
        特徴量の辞書
    """
    features = {}
    dt = 1.0 / sampling_rate
    
    for axis in ['x', 'y', 'z']:
        col = f'acc_{axis}'
        if col in acc_data.columns:
            acc = acc_data[col].values
            jerk = np.diff(acc) / dt
            
            features[f'jerk_{axis}_mean'] = np.mean(np.abs(jerk))
            features[f'jerk_{axis}_std'] = np.std(jerk)
            features[f'jerk_{axis}_max'] = np.max(np.abs(jerk))
            features[f'jerk_{axis}_energy'] = np.sum(jerk ** 2)
    
    # 総ジャーク
    if all(f'acc_{axis}' in acc_data.columns for axis in ['x', 'y', 'z']):
        jerk_x = np.diff(acc_data['acc_x'].values) / dt
        jerk_y = np.diff(acc_data['acc_y'].values) / dt
        jerk_z = np.diff(acc_data['acc_z'].values) / dt
        
        jerk_magnitude = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)
        features['jerk_magnitude_mean'] = np.mean(jerk_magnitude)
        features['jerk_magnitude_std'] = np.std(jerk_magnitude)
        features['jerk_magnitude_max'] = np.max(jerk_magnitude)
    
    return features


def extract_change_detection_features(
    data: np.ndarray,
    prefix: str,
    penalty: float = 1.0
) -> Dict[str, float]:
    """
    変化点検出による特徴量抽出
    
    Args:
        data: 時系列データ
        prefix: 特徴量名のプレフィックス
        penalty: 変化点検出のペナルティパラメータ
    
    Returns:
        特徴量の辞書
    """
    features = {}
    
    # NaN処理
    data = np.nan_to_num(data, nan=0.0)
    
    if len(data) < 10:
        return {
            f'{prefix}_n_change_points': 0,
            f'{prefix}_mean_segment_length': len(data),
            f'{prefix}_std_segment_length': 0
        }
    
    # 簡易的な変化点検出（差分の閾値ベース）
    diff = np.abs(np.diff(data))
    threshold = np.mean(diff) + penalty * np.std(diff)
    change_points = np.where(diff > threshold)[0]
    
    features[f'{prefix}_n_change_points'] = len(change_points)
    
    # セグメント長の統計
    if len(change_points) > 0:
        segments = np.diff(np.concatenate([[0], change_points, [len(data)]]))
        features[f'{prefix}_mean_segment_length'] = np.mean(segments)
        features[f'{prefix}_std_segment_length'] = np.std(segments)
        features[f'{prefix}_max_segment_length'] = np.max(segments)
        features[f'{prefix}_min_segment_length'] = np.min(segments)
    else:
        features[f'{prefix}_mean_segment_length'] = len(data)
        features[f'{prefix}_std_segment_length'] = 0
        features[f'{prefix}_max_segment_length'] = len(data)
        features[f'{prefix}_min_segment_length'] = len(data)
    
    # 変化率の統計
    if len(change_points) > 1:
        change_intervals = np.diff(change_points)
        features[f'{prefix}_mean_change_interval'] = np.mean(change_intervals)
        features[f'{prefix}_std_change_interval'] = np.std(change_intervals)
    else:
        features[f'{prefix}_mean_change_interval'] = len(data)
        features[f'{prefix}_std_change_interval'] = 0
    
    return features


def extract_comprehensive_features(
    sequence_df: pd.DataFrame,
    demographics_df: Optional[pd.DataFrame] = None,
    config: Optional[dict] = None
) -> pd.DataFrame:
    """
    総合的な特徴量抽出
    
    Args:
        sequence_df: シーケンスデータ
        demographics_df: デモグラフィックデータ
        config: 設定辞書
    
    Returns:
        特徴量データフレーム
    """
    features = {}
    
    # 設定の読み込み
    if config is None:
        config = {
            'features': {
                'statistical': {'enabled': True, 'functions': ["mean", "std", "min", "max"]},
                'world_acceleration': {'enabled': True},
                'frequency': {'enabled': True, 'sampling_rate': 20},
                'segment': {'enabled': True, 'n_segments': 3},
                'change_detection': {'enabled': True, 'penalty': 1},
                'additional': {
                    'jerk': True,
                    'rotation_energy': True,
                    'angle_velocity': True,
                    'orientation_changes': True
                }
            }
        }
    
    feature_config = config.get('features', {})
    
    # IMUカラムの定義
    acc_cols = ['acc_x', 'acc_y', 'acc_z']
    rot_cols = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
    
    # データの前処理
    acc_data = sequence_df[acc_cols].copy()
    acc_data = acc_data.ffill().bfill().fillna(0)
    
    rot_data = sequence_df[rot_cols].copy()
    rot_data = rot_data.ffill().bfill()
    
    # 四元数の欠損値処理と正規化
    rot_data_clean = handle_quaternion_missing_values(rot_data.values)
    
    # 1. World Acceleration特徴量
    if feature_config.get('world_acceleration', {}).get('enabled', True):
        world_acc_data, gravity_vector = compute_world_acceleration(
            acc_data.values, 
            rot_data_clean,
            correct_gravity=True
        )
        
        # World Acceleration統計量
        for i, axis in enumerate(['x', 'y', 'z']):
            world_acc_axis = world_acc_data[:, i]
            stat_features = extract_statistical_features(
                world_acc_axis, 
                f'world_acc_{axis}',
                feature_config.get('statistical', {}).get('functions')
            )
            features.update(stat_features)
        
        # World Acceleration magnitude
        world_acc_magnitude = np.linalg.norm(world_acc_data, axis=1)
        features.update(extract_statistical_features(world_acc_magnitude, 'world_acc_magnitude'))
        
        # Gravity features
        features['gravity_x'] = gravity_vector[0]
        features['gravity_y'] = gravity_vector[1]
        features['gravity_z'] = gravity_vector[2]
        features['gravity_magnitude'] = np.linalg.norm(gravity_vector)
    
    # 2. デバイス加速度統計量
    if feature_config.get('statistical', {}).get('enabled', True):
        for axis in ['x', 'y', 'z']:
            acc_axis = acc_data[f'acc_{axis}'].values
            stat_features = extract_statistical_features(
                acc_axis,
                f'acc_{axis}',
                feature_config.get('statistical', {}).get('functions')
            )
            features.update(stat_features)
        
        # 加速度マグニチュード
        acc_magnitude = np.linalg.norm(acc_data.values, axis=1)
        features.update(extract_statistical_features(acc_magnitude, 'acc_magnitude'))
    
    # 3. 回転特徴量
    orientation_features = compute_orientation_features(rot_data_clean)
    features.update(orientation_features)
    
    # 回転エネルギー
    if feature_config.get('additional', {}).get('rotation_energy', True):
        rot_energy = compute_rotation_energy(rot_data_clean)
        features.update(extract_statistical_features(rot_energy, 'rotation_energy'))
    
    # 4. 周波数領域特徴量
    if feature_config.get('frequency', {}).get('enabled', True):
        sampling_rate = feature_config.get('frequency', {}).get('sampling_rate', 20)
        
        # 各軸の周波数特徴量
        for axis in ['x', 'y', 'z']:
            acc_axis = acc_data[f'acc_{axis}'].values
            
            # FFT特徴量
            fft_features = extract_fft_features(acc_axis, sampling_rate)
            for feat_name, feat_val in fft_features.items():
                features[f'acc_{axis}_{feat_name}'] = feat_val
            
            # スペクトラル特徴量
            spectral_features = extract_spectral_features(acc_axis, sampling_rate)
            for feat_name, feat_val in spectral_features.items():
                features[f'acc_{axis}_{feat_name}'] = feat_val
        
        # マグニチュードの周波数特徴量
        fft_mag_features = extract_fft_features(acc_magnitude, sampling_rate)
        for feat_name, feat_val in fft_mag_features.items():
            features[f'acc_magnitude_{feat_name}'] = feat_val
        
        spectral_mag_features = extract_spectral_features(acc_magnitude, sampling_rate)
        for feat_name, feat_val in spectral_mag_features.items():
            features[f'acc_magnitude_{feat_name}'] = feat_val
        
        # ペリオドグラム
        periodogram_features = extract_periodogram_features(acc_magnitude, sampling_rate)
        features.update({f'acc_magnitude_{k}': v for k, v in periodogram_features.items()})
    
    # 5. セグメント特徴量
    if feature_config.get('segment', {}).get('enabled', True):
        n_segments = feature_config.get('segment', {}).get('n_segments', 3)
        
        # 加速度のセグメント特徴量
        for axis in ['x', 'y', 'z']:
            acc_axis = acc_data[f'acc_{axis}'].values
            seg_features = extract_segment_features(acc_axis, f'acc_{axis}', n_segments)
            features.update(seg_features)
        
        # マグニチュードのセグメント特徴量
        seg_mag_features = extract_segment_features(acc_magnitude, 'acc_magnitude', n_segments)
        features.update(seg_mag_features)
    
    # 6. 変化点検出特徴量
    if feature_config.get('change_detection', {}).get('enabled', True):
        penalty = feature_config.get('change_detection', {}).get('penalty', 1)
        
        for axis in ['x', 'y', 'z']:
            acc_axis = acc_data[f'acc_{axis}'].values
            change_features = extract_change_detection_features(acc_axis, f'acc_{axis}', penalty)
            features.update(change_features)
    
    # 7. ジャーク特徴量
    if feature_config.get('additional', {}).get('jerk', True):
        jerk_features = extract_jerk_features(acc_data)
        features.update(jerk_features)
    
    # 8. シーケンスメタデータ
    features['sequence_length'] = len(sequence_df)
    features['sequence_id'] = sequence_df['sequence_id'].iloc[0] if 'sequence_id' in sequence_df.columns else 'unknown'
    
    # 9. デモグラフィック特徴量
    if demographics_df is not None and not demographics_df.empty:
        demo_row = demographics_df.iloc[0]
        features['age'] = demo_row.get('age', 0)
        features['adult_child'] = demo_row.get('adult_child', 0)
        features['sex'] = demo_row.get('sex', 0)
        features['handedness'] = demo_row.get('handedness', 0)
        features['height_cm'] = demo_row.get('height_cm', 0)
        features['shoulder_to_wrist_cm'] = demo_row.get('shoulder_to_wrist_cm', 0)
        features['elbow_to_wrist_cm'] = demo_row.get('elbow_to_wrist_cm', 0)
    
    # DataFrameに変換
    result_df = pd.DataFrame([features])
    
    # NaN処理
    result_df = result_df.fillna(0)
    
    return result_df