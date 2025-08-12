"""
後処理モジュール
予測結果の改善と調整
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import uniform_filter1d


# BFRBと非BFRB行動の定義
BFRB_BEHAVIORS = [
    'Above ear - pull hair',
    'Cheek - pinch skin',
    'Eyebrow - pull hair',
    'Eyelash - pull hair',
    'Forehead - pull hairline',
    'Forehead - scratch',
    'Neck - pinch skin',
    'Neck - scratch'
]

NON_BFRB_BEHAVIORS = [
    'Drink from bottle/cup',
    'Feel around in tray and pull out an object',
    'Glasses on/off',
    'Pinch knee/leg skin',
    'Pull air toward your face',
    'Scratch knee/leg skin',
    'Text on phone',
    'Wave hello',
    'Write name in air',
    'Write name on leg'
]

# 混同しやすいジェスチャーのマッピング
GESTURE_CONFUSION_MAP = {
    # スクラッチ系
    'Neck - scratch': ['Forehead - scratch', 'Scratch knee/leg skin'],
    'Forehead - scratch': ['Neck - scratch', 'Scratch knee/leg skin'],
    'Scratch knee/leg skin': ['Neck - scratch', 'Forehead - scratch'],
    
    # 髪を引っ張る系
    'Above ear - pull hair': ['Eyebrow - pull hair', 'Eyelash - pull hair', 'Forehead - pull hairline'],
    'Eyebrow - pull hair': ['Above ear - pull hair', 'Eyelash - pull hair'],
    'Eyelash - pull hair': ['Eyebrow - pull hair', 'Above ear - pull hair'],
    'Forehead - pull hairline': ['Above ear - pull hair', 'Eyebrow - pull hair'],
    
    # 皮膚をつまむ系
    'Cheek - pinch skin': ['Neck - pinch skin', 'Pinch knee/leg skin'],
    'Neck - pinch skin': ['Cheek - pinch skin', 'Pinch knee/leg skin'],
    'Pinch knee/leg skin': ['Cheek - pinch skin', 'Neck - pinch skin'],
    
    # 書く系
    'Write name in air': ['Write name on leg', 'Text on phone'],
    'Write name on leg': ['Write name in air', 'Text on phone'],
    'Text on phone': ['Write name on leg', 'Write name in air'],
}

# 類似度スコア（0-1の範囲で、1が最も類似）
GESTURE_SIMILARITY = {
    ('Neck - scratch', 'Forehead - scratch'): 0.8,
    ('Neck - scratch', 'Scratch knee/leg skin'): 0.6,
    ('Above ear - pull hair', 'Eyebrow - pull hair'): 0.7,
    ('Above ear - pull hair', 'Eyelash - pull hair'): 0.7,
    ('Cheek - pinch skin', 'Neck - pinch skin'): 0.8,
    ('Cheek - pinch skin', 'Pinch knee/leg skin'): 0.5,
    ('Write name in air', 'Write name on leg'): 0.7,
    ('Text on phone', 'Write name on leg'): 0.6,
}


def apply_postprocessing(
    predictions: np.ndarray,
    class_names: List[str],
    config: Optional[Dict] = None
) -> np.ndarray:
    """
    予測結果に後処理を適用
    
    Args:
        predictions: 予測確率 [N, n_classes]
        class_names: クラス名のリスト
        config: 後処理設定
    
    Returns:
        調整後の予測確率
    """
    if config is None:
        config = {
            'confidence_threshold': 0.35,
            'bfrb_boost_factor': 1.25,
            'smooth_window': 5,
            'use_test_time_augmentation': False
        }
    
    predictions_adjusted = predictions.copy()
    
    # 1. BFRB行動の確率ブースト
    if config.get('bfrb_boost_factor', 1.0) != 1.0:
        predictions_adjusted = boost_bfrb_predictions(
            predictions_adjusted, 
            class_names, 
            config['bfrb_boost_factor']
        )
    
    # 2. 低信頼度予測の調整
    if config.get('confidence_threshold', 0) > 0:
        predictions_adjusted = adjust_low_confidence_predictions(
            predictions_adjusted,
            class_names,
            config['confidence_threshold']
        )
    
    # 3. 混同しやすいジェスチャーの調整
    predictions_adjusted = adjust_confused_gestures(predictions_adjusted, class_names)
    
    # 4. スムージング（複数予測がある場合）
    if len(predictions_adjusted) > 1 and config.get('smooth_window', 0) > 1:
        predictions_adjusted = smooth_predictions(
            predictions_adjusted,
            config['smooth_window']
        )
    
    # 5. 再正規化
    predictions_adjusted = normalize_predictions(predictions_adjusted)
    
    return predictions_adjusted


def boost_bfrb_predictions(
    predictions: np.ndarray,
    class_names: List[str],
    boost_factor: float
) -> np.ndarray:
    """BFRB行動の予測確率をブースト"""
    
    predictions_boosted = predictions.copy()
    
    for i, class_name in enumerate(class_names):
        if class_name in BFRB_BEHAVIORS:
            predictions_boosted[:, i] *= boost_factor
    
    return predictions_boosted


def adjust_low_confidence_predictions(
    predictions: np.ndarray,
    class_names: List[str],
    threshold: float
) -> np.ndarray:
    """低信頼度の予測を調整"""
    
    predictions_adjusted = predictions.copy()
    
    for idx in range(len(predictions_adjusted)):
        max_prob = np.max(predictions_adjusted[idx])
        
        if max_prob < threshold:
            # 上位5つの予測を考慮
            top_5_indices = np.argsort(predictions_adjusted[idx])[-5:]
            top_5_classes = [class_names[i] for i in top_5_indices]
            
            # BFRB行動が上位に含まれる場合、その確率を強化
            bfrb_count = sum(1 for cls in top_5_classes if cls in BFRB_BEHAVIORS)
            
            if bfrb_count >= 2:  # 複数のBFRB行動が候補にある
                for i in top_5_indices:
                    if class_names[i] in BFRB_BEHAVIORS:
                        predictions_adjusted[idx, i] *= 1.15
            
            # 非常に低い信頼度の場合、類似ジェスチャーの情報を使用
            if max_prob < threshold * 0.5:
                predictions_adjusted[idx] = use_similarity_information(
                    predictions_adjusted[idx],
                    class_names
                )
    
    return predictions_adjusted


def adjust_confused_gestures(
    predictions: np.ndarray,
    class_names: List[str]
) -> np.ndarray:
    """混同しやすいジェスチャーの調整"""
    
    predictions_adjusted = predictions.copy()
    
    for idx in range(len(predictions_adjusted)):
        pred_idx = np.argmax(predictions_adjusted[idx])
        pred_class = class_names[pred_idx]
        
        if pred_class in GESTURE_CONFUSION_MAP:
            confused_classes = GESTURE_CONFUSION_MAP[pred_class]
            
            for conf_class in confused_classes:
                if conf_class in class_names:
                    conf_idx = class_names.index(conf_class)
                    prob_diff = predictions_adjusted[idx, pred_idx] - predictions_adjusted[idx, conf_idx]
                    
                    # 確率が近い場合の調整
                    if prob_diff < 0.15:
                        # 類似度情報を使用
                        similarity = get_similarity(pred_class, conf_class)
                        
                        if similarity > 0.7:  # 高い類似度
                            # BFRB行動を優先
                            if conf_class in BFRB_BEHAVIORS and pred_class not in BFRB_BEHAVIORS:
                                predictions_adjusted[idx, conf_idx] *= (1.1 + similarity * 0.1)
                            elif pred_class in BFRB_BEHAVIORS and conf_class not in BFRB_BEHAVIORS:
                                predictions_adjusted[idx, pred_idx] *= (1.1 + similarity * 0.1)
    
    return predictions_adjusted


def use_similarity_information(
    prediction: np.ndarray,
    class_names: List[str]
) -> np.ndarray:
    """類似度情報を使用して予測を調整"""
    
    adjusted = prediction.copy()
    
    # 上位3つのクラスを取得
    top_3_indices = np.argsort(prediction)[-3:]
    
    for i in range(len(top_3_indices) - 1):
        for j in range(i + 1, len(top_3_indices)):
            idx1, idx2 = top_3_indices[i], top_3_indices[j]
            class1, class2 = class_names[idx1], class_names[idx2]
            
            similarity = get_similarity(class1, class2)
            if similarity > 0.5:
                # 類似度が高い場合、確率を平均化
                avg_prob = (adjusted[idx1] + adjusted[idx2]) / 2
                weight = similarity * 0.3  # 類似度に応じた重み
                
                adjusted[idx1] = adjusted[idx1] * (1 - weight) + avg_prob * weight
                adjusted[idx2] = adjusted[idx2] * (1 - weight) + avg_prob * weight
    
    return adjusted


def get_similarity(class1: str, class2: str) -> float:
    """2つのクラス間の類似度を取得"""
    
    # 順序を考慮しない
    pair = tuple(sorted([class1, class2]))
    
    if pair in GESTURE_SIMILARITY:
        return GESTURE_SIMILARITY[pair]
    
    # デフォルトの類似度を計算
    if class1 in GESTURE_CONFUSION_MAP.get(class2, []):
        return 0.5
    if class2 in GESTURE_CONFUSION_MAP.get(class1, []):
        return 0.5
    
    return 0.0


def smooth_predictions(
    predictions: np.ndarray,
    window_size: int
) -> np.ndarray:
    """予測結果をスムージング"""
    
    if len(predictions) < window_size:
        return predictions
    
    smoothed = np.zeros_like(predictions)
    
    for i in range(predictions.shape[1]):
        smoothed[:, i] = uniform_filter1d(
            predictions[:, i],
            size=window_size,
            mode='nearest'
        )
    
    return smoothed


def normalize_predictions(predictions: np.ndarray) -> np.ndarray:
    """予測確率を正規化"""
    
    row_sums = predictions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # ゼロ除算を防ぐ
    
    return predictions / row_sums


def apply_test_time_augmentation(
    predictions_list: List[np.ndarray],
    weights: Optional[List[float]] = None
) -> np.ndarray:
    """
    Test Time Augmentation (TTA)を適用
    
    Args:
        predictions_list: 複数の予測結果のリスト
        weights: 各予測の重み
    
    Returns:
        統合された予測結果
    """
    if weights is None:
        weights = [1.0] * len(predictions_list)
    
    # 重み付き平均
    weighted_sum = np.zeros_like(predictions_list[0])
    total_weight = sum(weights)
    
    for pred, weight in zip(predictions_list, weights):
        weighted_sum += pred * weight
    
    return weighted_sum / total_weight


def calibrate_predictions(
    predictions: np.ndarray,
    temperature: float = 1.0
) -> np.ndarray:
    """
    温度スケーリングによる予測の較正
    
    Args:
        predictions: 予測確率
        temperature: 温度パラメータ（> 1で平滑化、< 1で鋭利化）
    
    Returns:
        較正された予測確率
    """
    if temperature == 1.0:
        return predictions
    
    # Logitに変換
    logits = np.log(predictions + 1e-10)
    
    # 温度スケーリング
    scaled_logits = logits / temperature
    
    # Softmax
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    calibrated = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    return calibrated