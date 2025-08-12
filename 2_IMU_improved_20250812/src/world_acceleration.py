"""
World Acceleration変換モジュール
デバイス座標系から世界座標系への変換を行う
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Optional


def handle_quaternion_missing_values(rot_data: np.ndarray) -> np.ndarray:
    """
    四元数データの欠損値を処理
    
    四元数の性質: |w|² + |x|² + |y|² + |z|² = 1
    1つの成分が欠損している場合、他の成分から復元可能
    
    Args:
        rot_data: 四元数データ [N, 4] (w, x, y, z)
    
    Returns:
        処理済み四元数データ
    """
    rot_cleaned = rot_data.copy()
    
    for i in range(len(rot_data)):
        row = rot_data[i]
        missing_count = np.isnan(row).sum()
        
        if missing_count == 0:
            # 欠損値なし - 単位四元数に正規化
            norm = np.linalg.norm(row)
            if norm > 1e-8:
                rot_cleaned[i] = row / norm
            else:
                rot_cleaned[i] = [1.0, 0.0, 0.0, 0.0]  # 恒等四元数
                
        elif missing_count == 1:
            # 1つの欠損値 - 単位四元数制約から復元
            missing_idx = np.where(np.isnan(row))[0][0]
            valid_values = row[~np.isnan(row)]
            
            sum_squares = np.sum(valid_values**2)
            if sum_squares <= 1.0:
                missing_value = np.sqrt(max(0, 1.0 - sum_squares))
                # 前の四元数との連続性を考慮して符号を決定
                if i > 0 and not np.isnan(rot_cleaned[i-1, missing_idx]):
                    if rot_cleaned[i-1, missing_idx] < 0:
                        missing_value = -missing_value
                rot_cleaned[i, missing_idx] = missing_value
                rot_cleaned[i, ~np.isnan(row)] = valid_values
            else:
                # 制約を満たさない場合は恒等四元数
                rot_cleaned[i] = [1.0, 0.0, 0.0, 0.0]
        else:
            # 複数の欠損値 - 恒等四元数を使用
            rot_cleaned[i] = [1.0, 0.0, 0.0, 0.0]
    
    return rot_cleaned


def compute_world_acceleration(
    acc: np.ndarray, 
    rot: np.ndarray,
    correct_gravity: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    デバイス座標系から世界座標系への加速度変換
    
    Args:
        acc: デバイス座標系の加速度 [N, 3] (x, y, z)
        rot: 回転四元数 [N, 4] (w, x, y, z)
        correct_gravity: 重力補正を行うか
    
    Returns:
        world_acc: 世界座標系の加速度
        gravity_vector: 推定された重力ベクトル
    """
    try:
        # 四元数フォーマットを[w, x, y, z]から[x, y, z, w]に変換（scipy用）
        rot_scipy = rot[:, [1, 2, 3, 0]]
        
        # 四元数の妥当性チェック
        norms = np.linalg.norm(rot_scipy, axis=1)
        mask = norms < 1e-8
        if np.any(mask):
            rot_scipy[mask] = [0.0, 0.0, 0.0, 1.0]  # 恒等四元数
        
        # Rotationオブジェクトを作成して変換適用
        r = R.from_quat(rot_scipy)
        world_acc = r.apply(acc)
        
        # 重力補正
        gravity_vector = np.array([0.0, 0.0, 0.0])
        if correct_gravity and len(world_acc) > 10:
            # 静止時の加速度から重力ベクトルを推定
            # 加速度の分散が小さい部分を静止状態とみなす
            acc_var = np.var(world_acc, axis=0)
            if np.min(acc_var) < 0.1:  # 静止状態の閾値
                # 最初の10%のデータから重力を推定
                static_portion = int(len(world_acc) * 0.1)
                gravity_vector = np.mean(world_acc[:static_portion], axis=0)
                # 重力を除去（地球の重力は約9.8 m/s²）
                gravity_vector[2] = gravity_vector[2] - 9.8
                world_acc = world_acc - gravity_vector
        
    except Exception as e:
        print(f"World acceleration transformation failed: {e}")
        world_acc = acc.copy()
        gravity_vector = np.array([0.0, 0.0, 0.0])
    
    return world_acc, gravity_vector


def compute_orientation_features(rot: np.ndarray) -> dict:
    """
    回転四元数から向きに関する特徴量を計算
    
    Args:
        rot: 回転四元数 [N, 4] (w, x, y, z)
    
    Returns:
        向き関連の特徴量辞書
    """
    features = {}
    
    try:
        # scipy用フォーマットに変換
        rot_scipy = rot[:, [1, 2, 3, 0]]
        r = R.from_quat(rot_scipy)
        
        # オイラー角に変換
        euler_angles = r.as_euler('xyz', degrees=True)
        
        # 各軸の角度統計量
        for i, axis in enumerate(['roll', 'pitch', 'yaw']):
            angles = euler_angles[:, i]
            features[f'{axis}_mean'] = np.mean(angles)
            features[f'{axis}_std'] = np.std(angles)
            features[f'{axis}_range'] = np.max(angles) - np.min(angles)
            
            # 角度変化率
            angle_diff = np.diff(angles)
            features[f'{axis}_velocity_mean'] = np.mean(np.abs(angle_diff))
            features[f'{axis}_velocity_max'] = np.max(np.abs(angle_diff))
        
        # 向きの安定性（四元数の変化量）
        quat_diff = np.diff(rot, axis=0)
        quat_change = np.linalg.norm(quat_diff, axis=1)
        features['orientation_stability'] = np.mean(quat_change)
        features['orientation_max_change'] = np.max(quat_change)
        
    except Exception as e:
        print(f"Orientation feature extraction failed: {e}")
        # デフォルト値を設定
        for axis in ['roll', 'pitch', 'yaw']:
            features[f'{axis}_mean'] = 0
            features[f'{axis}_std'] = 0
            features[f'{axis}_range'] = 0
            features[f'{axis}_velocity_mean'] = 0
            features[f'{axis}_velocity_max'] = 0
        features['orientation_stability'] = 0
        features['orientation_max_change'] = 0
    
    return features


def compute_rotation_energy(rot: np.ndarray) -> np.ndarray:
    """
    回転エネルギーを計算
    
    Args:
        rot: 回転四元数 [N, 4] (w, x, y, z)
    
    Returns:
        回転エネルギーの時系列
    """
    # 四元数の虚部（x, y, z）のノルムを計算
    imaginary_part = rot[:, 1:4]
    rotation_energy = np.linalg.norm(imaginary_part, axis=1)
    
    return rotation_energy