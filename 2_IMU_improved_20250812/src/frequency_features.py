"""
周波数領域特徴量抽出モジュール
FFT、スペクトログラム、ウェーブレット変換などの特徴量を計算
"""

import numpy as np
from scipy import signal
from scipy.stats import entropy
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def extract_fft_features(
    data: np.ndarray, 
    sampling_rate: float = 20.0,
    n_top_frequencies: int = 5
) -> Dict[str, float]:
    """
    FFTベースの周波数特徴量を抽出
    
    Args:
        data: 時系列データ (1次元配列)
        sampling_rate: サンプリングレート (Hz)
        n_top_frequencies: 上位何個の周波数を特徴量とするか
    
    Returns:
        特徴量の辞書
    """
    features = {}
    
    # NaN処理
    data = np.nan_to_num(data, nan=0.0)
    
    if len(data) < 10:
        # データが短すぎる場合
        return _get_default_fft_features(n_top_frequencies)
    
    # FFT計算
    fft_vals = np.fft.rfft(data)
    fft_power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(len(data), 1/sampling_rate)
    
    # DC成分を除外
    if len(fft_power) > 1:
        fft_power = fft_power[1:]
        freqs = freqs[1:]
    
    # 最大パワーの周波数（主周波数）
    if len(fft_power) > 0:
        max_idx = np.argmax(fft_power)
        features['fft_max_freq'] = freqs[max_idx]
        features['fft_max_power'] = fft_power[max_idx]
    else:
        features['fft_max_freq'] = 0
        features['fft_max_power'] = 0
    
    # パワースペクトラムの統計量
    features['fft_mean_power'] = np.mean(fft_power)
    features['fft_std_power'] = np.std(fft_power)
    features['fft_total_power'] = np.sum(fft_power)
    
    # 上位N個の周波数とそのパワー
    if len(fft_power) >= n_top_frequencies:
        top_indices = np.argsort(fft_power)[-n_top_frequencies:][::-1]
        for i, idx in enumerate(top_indices):
            features[f'fft_top{i+1}_freq'] = freqs[idx]
            features[f'fft_top{i+1}_power'] = fft_power[idx]
    
    # 周波数帯域ごとのパワー
    # 低周波数帯域 (0-2 Hz)
    low_band_mask = (freqs >= 0) & (freqs < 2)
    features['fft_low_band_power'] = np.sum(fft_power[low_band_mask])
    
    # 中周波数帯域 (2-5 Hz)
    mid_band_mask = (freqs >= 2) & (freqs < 5)
    features['fft_mid_band_power'] = np.sum(fft_power[mid_band_mask])
    
    # 高周波数帯域 (5-10 Hz)
    high_band_mask = (freqs >= 5) & (freqs < 10)
    features['fft_high_band_power'] = np.sum(fft_power[high_band_mask])
    
    # パワー比率
    total_power = features['fft_total_power']
    if total_power > 0:
        features['fft_low_band_ratio'] = features['fft_low_band_power'] / total_power
        features['fft_mid_band_ratio'] = features['fft_mid_band_power'] / total_power
        features['fft_high_band_ratio'] = features['fft_high_band_power'] / total_power
    else:
        features['fft_low_band_ratio'] = 0
        features['fft_mid_band_ratio'] = 0
        features['fft_high_band_ratio'] = 0
    
    return features


def extract_spectral_features(
    data: np.ndarray, 
    sampling_rate: float = 20.0
) -> Dict[str, float]:
    """
    スペクトラル特徴量を抽出
    
    Args:
        data: 時系列データ
        sampling_rate: サンプリングレート
    
    Returns:
        特徴量の辞書
    """
    features = {}
    
    # NaN処理
    data = np.nan_to_num(data, nan=0.0)
    
    if len(data) < 10:
        return {
            'spectral_centroid': 0,
            'spectral_rolloff': 0,
            'spectral_bandwidth': 0,
            'spectral_flatness': 0,
            'spectral_entropy': 0,
            'zero_crossing_rate': 0
        }
    
    # FFT計算
    fft_vals = np.fft.rfft(data)
    fft_power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(len(data), 1/sampling_rate)
    
    # スペクトラル重心（周波数の重心）
    if np.sum(fft_power) > 0:
        features['spectral_centroid'] = np.sum(freqs * fft_power) / np.sum(fft_power)
    else:
        features['spectral_centroid'] = 0
    
    # スペクトラルロールオフ（エネルギーの85%が含まれる周波数）
    if np.sum(fft_power) > 0:
        cumsum_power = np.cumsum(fft_power)
        rolloff_idx = np.searchsorted(cumsum_power, 0.85 * cumsum_power[-1])
        features['spectral_rolloff'] = freqs[min(rolloff_idx, len(freqs)-1)]
    else:
        features['spectral_rolloff'] = 0
    
    # スペクトラル帯域幅
    if np.sum(fft_power) > 0 and features['spectral_centroid'] > 0:
        spectral_bandwidth = np.sqrt(
            np.sum(((freqs - features['spectral_centroid']) ** 2) * fft_power) / 
            np.sum(fft_power)
        )
        features['spectral_bandwidth'] = spectral_bandwidth
    else:
        features['spectral_bandwidth'] = 0
    
    # スペクトラルフラットネス（幾何平均と算術平均の比）
    if len(fft_power) > 0 and np.all(fft_power > 0):
        geometric_mean = np.exp(np.mean(np.log(fft_power + 1e-10)))
        arithmetic_mean = np.mean(fft_power)
        if arithmetic_mean > 0:
            features['spectral_flatness'] = geometric_mean / arithmetic_mean
        else:
            features['spectral_flatness'] = 0
    else:
        features['spectral_flatness'] = 0
    
    # スペクトラルエントロピー
    if np.sum(fft_power) > 0:
        normalized_power = fft_power / np.sum(fft_power)
        spectral_entropy = entropy(normalized_power + 1e-10)
        features['spectral_entropy'] = spectral_entropy
    else:
        features['spectral_entropy'] = 0
    
    # ゼロ交差率
    zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
    features['zero_crossing_rate'] = zero_crossings / len(data)
    
    return features


def extract_wavelet_features(
    data: np.ndarray,
    wavelet: str = 'db4',
    level: int = 3
) -> Dict[str, float]:
    """
    ウェーブレット変換による特徴量抽出
    
    Args:
        data: 時系列データ
        wavelet: ウェーブレット種類
        level: 分解レベル
    
    Returns:
        特徴量の辞書
    """
    features = {}
    
    try:
        import pywt
        
        # NaN処理
        data = np.nan_to_num(data, nan=0.0)
        
        if len(data) < 2**level:
            # データが短すぎる場合
            for i in range(level + 1):
                features[f'wavelet_energy_level_{i}'] = 0
                features[f'wavelet_std_level_{i}'] = 0
            return features
        
        # ウェーブレット分解
        coeffs = pywt.wavedec(data, wavelet, level=level)
        
        # 各レベルのエネルギーと統計量
        for i, coeff in enumerate(coeffs):
            energy = np.sum(coeff ** 2)
            features[f'wavelet_energy_level_{i}'] = energy
            features[f'wavelet_std_level_{i}'] = np.std(coeff)
            features[f'wavelet_mean_level_{i}'] = np.mean(np.abs(coeff))
        
        # 総エネルギー
        total_energy = sum([np.sum(c ** 2) for c in coeffs])
        
        # エネルギー比率
        if total_energy > 0:
            for i, coeff in enumerate(coeffs):
                energy = np.sum(coeff ** 2)
                features[f'wavelet_energy_ratio_level_{i}'] = energy / total_energy
        
    except ImportError:
        # pywtがインストールされていない場合
        for i in range(level + 1):
            features[f'wavelet_energy_level_{i}'] = 0
            features[f'wavelet_std_level_{i}'] = 0
    
    return features


def extract_periodogram_features(
    data: np.ndarray,
    sampling_rate: float = 20.0
) -> Dict[str, float]:
    """
    ペリオドグラムによる周期性の特徴量抽出
    
    Args:
        data: 時系列データ
        sampling_rate: サンプリングレート
    
    Returns:
        特徴量の辞書
    """
    features = {}
    
    # NaN処理
    data = np.nan_to_num(data, nan=0.0)
    
    if len(data) < 10:
        return {
            'periodogram_max_freq': 0,
            'periodogram_max_power': 0,
            'periodogram_peak_prominence': 0
        }
    
    try:
        # ペリオドグラム計算
        freqs, power = signal.periodogram(data, fs=sampling_rate)
        
        # DC成分を除外
        if len(power) > 1:
            power = power[1:]
            freqs = freqs[1:]
        
        # 最大パワーの周波数
        if len(power) > 0:
            max_idx = np.argmax(power)
            features['periodogram_max_freq'] = freqs[max_idx]
            features['periodogram_max_power'] = power[max_idx]
            
            # ピークの顕著性（最大値と平均値の比）
            if np.mean(power) > 0:
                features['periodogram_peak_prominence'] = power[max_idx] / np.mean(power)
            else:
                features['periodogram_peak_prominence'] = 0
        else:
            features['periodogram_max_freq'] = 0
            features['periodogram_max_power'] = 0
            features['periodogram_peak_prominence'] = 0
            
    except Exception as e:
        print(f"Periodogram extraction failed: {e}")
        features['periodogram_max_freq'] = 0
        features['periodogram_max_power'] = 0
        features['periodogram_peak_prominence'] = 0
    
    return features


def _get_default_fft_features(n_top_frequencies: int) -> Dict[str, float]:
    """デフォルトのFFT特徴量を返す"""
    features = {
        'fft_max_freq': 0,
        'fft_max_power': 0,
        'fft_mean_power': 0,
        'fft_std_power': 0,
        'fft_total_power': 0,
        'fft_low_band_power': 0,
        'fft_mid_band_power': 0,
        'fft_high_band_power': 0,
        'fft_low_band_ratio': 0,
        'fft_mid_band_ratio': 0,
        'fft_high_band_ratio': 0
    }
    
    for i in range(n_top_frequencies):
        features[f'fft_top{i+1}_freq'] = 0
        features[f'fft_top{i+1}_power'] = 0
    
    return features