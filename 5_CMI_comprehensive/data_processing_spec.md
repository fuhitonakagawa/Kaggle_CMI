# データ処理仕様書
# CMI BFRB Detection - Data Processing Specification

## 1. 入力データ構造

### 1.1 センサーデータ (train.csv / test.csv)
```python
# IMUデータ (必須)
- acc_x, acc_y, acc_z: 加速度センサー (m/s²)
- rot_w, rot_x, rot_y, rot_z: 回転クォータニオン

# TOFデータ (オプション - 50%のテストデータで欠損)
- tof_1_v0 ~ tof_1_v63: TOFセンサー1の64ピクセル
- tof_2_v0 ~ tof_2_v63: TOFセンサー2の64ピクセル
- tof_3_v0 ~ tof_3_v63: TOFセンサー3の64ピクセル
- tof_4_v0 ~ tof_4_v63: TOFセンサー4の64ピクセル
- tof_5_v0 ~ tof_5_v63: TOFセンサー5の64ピクセル

# サーマルデータ (オプション)
- thm_1 ~ thm_5: サーマルセンサー値

# メタデータ
- sequence_id: シーケンス識別子
- subject: 被験者ID
- gesture: ジェスチャーラベル (訓練データのみ)
```

### 1.2 人口統計データ (demographics.csv)
```python
- subject: 被験者ID
- age: 年齢
- gender: 性別
- dominant_hand: 利き手
```

## 2. 特徴量エンジニアリング

### 2.1 IMU基本特徴量

#### 重力除去と線形加速度
```python
def remove_gravity(acc_data, rot_data):
    """
    クォータニオンを使用して重力成分を除去
    """
    gravity_world = np.array([0, 0, 9.81])
    linear_accel = np.zeros_like(acc_data)
    
    for i in range(len(acc_data)):
        # クォータニオンから回転行列を作成
        rotation = R.from_quat(rot_data[i])
        # 重力をセンサー座標系に変換
        gravity_sensor = rotation.apply(gravity_world, inverse=True)
        # 重力成分を除去
        linear_accel[i] = acc_data[i] - gravity_sensor
    
    return linear_accel
```

#### 角速度計算
```python
def calculate_angular_velocity(rot_data, sample_rate=20):
    """
    連続するクォータニオンから角速度を計算
    """
    angular_vel = np.zeros((len(rot_data), 3))
    dt = 1.0 / sample_rate
    
    for i in range(len(rot_data) - 1):
        r_t = R.from_quat(rot_data[i])
        r_t_plus = R.from_quat(rot_data[i + 1])
        # 相対回転を計算
        delta_rot = r_t.inv() * r_t_plus
        # 角速度ベクトルに変換
        angular_vel[i] = delta_rot.as_rotvec() / dt
    
    return angular_vel
```

#### 角距離計算
```python
def calculate_angular_distance(rot_data):
    """
    連続フレーム間の角距離を計算
    """
    angular_dist = np.zeros(len(rot_data))
    
    for i in range(len(rot_data) - 1):
        r1 = R.from_quat(rot_data[i])
        r2 = R.from_quat(rot_data[i + 1])
        # 相対回転の角度を計算
        relative_rotation = r1.inv() * r2
        angle = np.linalg.norm(relative_rotation.as_rotvec())
        angular_dist[i] = angle
    
    return angular_dist
```

### 2.2 統計的特徴量

#### 時間窓統計量
```python
def extract_window_features(signal, window_size=20, step_size=10):
    """
    移動窓での統計量抽出
    """
    features = []
    
    for i in range(0, len(signal) - window_size, step_size):
        window = signal[i:i+window_size]
        
        features.append({
            'mean': np.mean(window),
            'std': np.std(window),
            'max': np.max(window),
            'min': np.min(window),
            'range': np.ptp(window),
            'median': np.median(window),
            'q25': np.percentile(window, 25),
            'q75': np.percentile(window, 75),
            'skew': stats.skew(window),
            'kurtosis': stats.kurtosis(window),
            'zero_crossing': np.sum(np.diff(np.sign(window)) != 0),
            'peak_count': len(find_peaks(window)[0])
        })
    
    return features
```

### 2.3 周波数領域特徴量

#### FFT特徴量
```python
def extract_frequency_features(signal, sample_rate=20):
    """
    FFTベースの周波数特徴量抽出
    """
    # FFT計算
    fft_vals = np.abs(fft(signal))
    freqs = fftfreq(len(signal), 1/sample_rate)
    
    # 正の周波数のみ
    pos_mask = freqs > 0
    fft_vals = fft_vals[pos_mask]
    freqs = freqs[pos_mask]
    
    features = {
        'dominant_freq': freqs[np.argmax(fft_vals)],
        'spectral_centroid': np.sum(freqs * fft_vals) / np.sum(fft_vals),
        'spectral_spread': np.sqrt(np.sum(((freqs - spectral_centroid)**2) * fft_vals) / np.sum(fft_vals)),
        'spectral_entropy': -np.sum((fft_vals/np.sum(fft_vals)) * np.log2(fft_vals/np.sum(fft_vals) + 1e-10)),
        'band_power_0_2': np.sum(fft_vals[(freqs >= 0) & (freqs < 2)]),
        'band_power_2_5': np.sum(fft_vals[(freqs >= 2) & (freqs < 5)]),
        'band_power_5_10': np.sum(fft_vals[(freqs >= 5) & (freqs < 10)])
    }
    
    return features
```

#### Welch PSD特徴量
```python
def extract_psd_features(signal, sample_rate=20):
    """
    Welch法によるパワースペクトル密度特徴量
    """
    freqs, psd = signal.welch(signal, fs=sample_rate, nperseg=min(256, len(signal)))
    
    features = {
        'total_power': np.trapz(psd, freqs),
        'peak_frequency': freqs[np.argmax(psd)],
        'median_frequency': freqs[np.where(np.cumsum(psd) >= np.sum(psd)/2)[0][0]],
        'spectral_edge_95': freqs[np.where(np.cumsum(psd) >= 0.95*np.sum(psd))[0][0]]
    }
    
    return features
```

### 2.4 TOF/Thermal特徴量

#### 空間パターン認識
```python
def extract_tof_spatial_features(tof_data):
    """
    TOFセンサーの空間パターン特徴量
    """
    # 8x8グリッドに再形成
    grid = tof_data.reshape(8, 8)
    
    features = {
        # 中心と周辺の比較
        'center_vs_edge': np.mean(grid[2:6, 2:6]) / (np.mean(grid[:2, :]) + np.mean(grid[6:, :]) + 1e-10),
        
        # 空間的勾配
        'gradient_x': np.mean(np.abs(np.diff(grid, axis=1))),
        'gradient_y': np.mean(np.abs(np.diff(grid, axis=0))),
        
        # ホットスポット検出
        'max_location_x': np.unravel_index(np.argmax(grid), grid.shape)[0],
        'max_location_y': np.unravel_index(np.argmax(grid), grid.shape)[1],
        
        # 空間的分散
        'spatial_variance': np.var(grid),
        
        # 対称性
        'horizontal_symmetry': np.corrcoef(grid[:, :4].flatten(), grid[:, 4:].flatten()[::-1])[0, 1],
        'vertical_symmetry': np.corrcoef(grid[:4, :].flatten(), grid[4:, :].flatten()[::-1])[0, 1]
    }
    
    return features
```

### 2.5 相互相関特徴量

```python
def extract_cross_correlation_features(signals_dict):
    """
    異なるセンサー間の相互相関
    """
    features = {}
    
    # 加速度軸間の相関
    features['corr_acc_xy'] = np.corrcoef(signals_dict['acc_x'], signals_dict['acc_y'])[0, 1]
    features['corr_acc_xz'] = np.corrcoef(signals_dict['acc_x'], signals_dict['acc_z'])[0, 1]
    features['corr_acc_yz'] = np.corrcoef(signals_dict['acc_y'], signals_dict['acc_z'])[0, 1]
    
    # 加速度と角速度の相関
    if 'angular_vel_x' in signals_dict:
        features['corr_acc_angvel_x'] = np.corrcoef(signals_dict['acc_x'], signals_dict['angular_vel_x'])[0, 1]
        features['corr_acc_angvel_y'] = np.corrcoef(signals_dict['acc_y'], signals_dict['angular_vel_y'])[0, 1]
        features['corr_acc_angvel_z'] = np.corrcoef(signals_dict['acc_z'], signals_dict['angular_vel_z'])[0, 1]
    
    return features
```

## 3. データ正規化戦略

### 3.1 特徴量ごとの正規化

```python
class FeatureNormalizer:
    def __init__(self):
        self.scalers = {}
        
    def fit_transform(self, features_dict):
        normalized = {}
        
        for feature_name, values in features_dict.items():
            if feature_name.startswith('acc_'):
                # 加速度: StandardScaler
                scaler = StandardScaler()
                normalized[feature_name] = scaler.fit_transform(values.reshape(-1, 1)).ravel()
                self.scalers[feature_name] = scaler
                
            elif feature_name.startswith('rot_'):
                # クォータニオン: 正規化済み（単位クォータニオン）
                normalized[feature_name] = values
                
            elif feature_name.startswith('tof_'):
                # TOF: RobustScaler（外れ値に強い）
                scaler = RobustScaler()
                normalized[feature_name] = scaler.fit_transform(values.reshape(-1, 1)).ravel()
                self.scalers[feature_name] = scaler
                
            elif feature_name.startswith('thm_'):
                # Thermal: MinMaxScaler
                scaler = MinMaxScaler()
                normalized[feature_name] = scaler.fit_transform(values.reshape(-1, 1)).ravel()
                self.scalers[feature_name] = scaler
                
            else:
                # その他: StandardScaler
                scaler = StandardScaler()
                normalized[feature_name] = scaler.fit_transform(values.reshape(-1, 1)).ravel()
                self.scalers[feature_name] = scaler
        
        return normalized
```

## 4. シーケンス処理

### 4.1 パディング戦略

```python
def adaptive_padding(sequences, percentile=95):
    """
    適応的パディング長の決定
    """
    lengths = [len(seq) for seq in sequences]
    pad_length = int(np.percentile(lengths, percentile))
    
    padded_sequences = []
    for seq in sequences:
        if len(seq) > pad_length:
            # Truncate: 中央部分を保持
            start = (len(seq) - pad_length) // 2
            padded = seq[start:start + pad_length]
        else:
            # Pad: ゼロパディング
            pad_width = pad_length - len(seq)
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            padded = np.pad(seq, ((pad_left, pad_right), (0, 0)), mode='constant')
        
        padded_sequences.append(padded)
    
    return np.array(padded_sequences)
```

### 4.2 セグメンテーション

```python
def segment_sequence(sequence, segment_length=100, overlap=50):
    """
    オーバーラップ付きセグメンテーション
    """
    segments = []
    step = segment_length - overlap
    
    for i in range(0, len(sequence) - segment_length + 1, step):
        segment = sequence[i:i + segment_length]
        segments.append(segment)
    
    return segments
```

## 5. データ拡張

### 5.1 時系列特有の拡張

```python
class TimeSeriesAugmentation:
    def __init__(self):
        self.augmentations = []
    
    def time_warp(self, x, sigma=0.2):
        """時間軸の歪み"""
        time_points = np.arange(len(x))
        warp = np.random.normal(loc=1.0, scale=sigma, size=len(x))
        warped_time = np.cumsum(warp)
        warped_time = warped_time / warped_time[-1] * len(x)
        return np.interp(time_points, warped_time, x)
    
    def magnitude_warp(self, x, sigma=0.2):
        """振幅の歪み"""
        warp = np.random.normal(loc=1.0, scale=sigma, size=x.shape)
        return x * warp
    
    def add_noise(self, x, sigma=0.05):
        """ガウシアンノイズ追加"""
        noise = np.random.normal(0, sigma, x.shape)
        return x + noise
    
    def cutmix(self, x1, x2, alpha=0.5):
        """CutMix for sequences"""
        cut_point = int(len(x1) * alpha)
        mixed = np.copy(x1)
        mixed[cut_point:] = x2[cut_point:]
        return mixed
```

## 6. 欠損値処理

### 6.1 センサー別欠損値処理

```python
def handle_missing_values(df):
    """
    センサータイプ別の欠損値処理
    """
    # IMUデータ: 前方/後方補完
    imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
    df[imu_cols] = df[imu_cols].fillna(method='ffill').fillna(method='bfill')
    
    # TOFデータ: -1を欠損値として扱い、中央値で補完
    tof_cols = [col for col in df.columns if col.startswith('tof_')]
    for col in tof_cols:
        df[col] = df[col].replace(-1, np.nan)
        df[col] = df[col].fillna(df[col].median())
    
    # Thermalデータ: 線形補間
    thm_cols = [col for col in df.columns if col.startswith('thm_')]
    df[thm_cols] = df[thm_cols].interpolate(method='linear')
    
    return df
```

## 7. 最終特徴量セット

### 7.1 特徴量リスト

```python
FEATURE_GROUPS = {
    'imu_raw': [
        'acc_x', 'acc_y', 'acc_z',
        'rot_w', 'rot_x', 'rot_y', 'rot_z'
    ],
    'imu_engineered': [
        'linear_acc_x', 'linear_acc_y', 'linear_acc_z',
        'linear_acc_mag', 'linear_acc_mag_jerk',
        'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
        'angular_vel_mag', 'angular_distance'
    ],
    'imu_statistical': [
        'acc_mean_x', 'acc_std_x', 'acc_max_x', 'acc_min_x',
        'acc_mean_y', 'acc_std_y', 'acc_max_y', 'acc_min_y',
        'acc_mean_z', 'acc_std_z', 'acc_max_z', 'acc_min_z',
        'acc_skew', 'acc_kurtosis', 'acc_zero_crossing'
    ],
    'imu_frequency': [
        'acc_dominant_freq_x', 'acc_spectral_centroid_x',
        'acc_dominant_freq_y', 'acc_spectral_centroid_y',
        'acc_dominant_freq_z', 'acc_spectral_centroid_z',
        'acc_band_power_low', 'acc_band_power_mid', 'acc_band_power_high'
    ],
    'tof_aggregated': [
        'tof_1_mean', 'tof_1_std', 'tof_1_min', 'tof_1_max',
        'tof_2_mean', 'tof_2_std', 'tof_2_min', 'tof_2_max',
        'tof_3_mean', 'tof_3_std', 'tof_3_min', 'tof_3_max',
        'tof_4_mean', 'tof_4_std', 'tof_4_min', 'tof_4_max',
        'tof_5_mean', 'tof_5_std', 'tof_5_min', 'tof_5_max'
    ],
    'tof_spatial': [
        'tof_center_vs_edge', 'tof_gradient_x', 'tof_gradient_y',
        'tof_spatial_variance', 'tof_horizontal_symmetry', 'tof_vertical_symmetry'
    ],
    'thermal': [
        'thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5',
        'thm_mean', 'thm_std', 'thm_trend'
    ],
    'cross_correlation': [
        'corr_acc_xy', 'corr_acc_xz', 'corr_acc_yz',
        'corr_acc_angvel_x', 'corr_acc_angvel_y', 'corr_acc_angvel_z'
    ]
}

# Deep Learning用: 時系列データ
DL_FEATURES = FEATURE_GROUPS['imu_raw'] + FEATURE_GROUPS['imu_engineered']

# Gradient Boosting用: 集約特徴量
GB_FEATURES = (
    FEATURE_GROUPS['imu_statistical'] + 
    FEATURE_GROUPS['imu_frequency'] + 
    FEATURE_GROUPS['tof_aggregated'] + 
    FEATURE_GROUPS['tof_spatial'] +
    FEATURE_GROUPS['thermal'] +
    FEATURE_GROUPS['cross_correlation']
)
```

## 8. 処理パイプライン

```python
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.normalizer = FeatureNormalizer()
        self.augmenter = TimeSeriesAugmentation()
        
    def process_sequence(self, df, is_training=False):
        """
        単一シーケンスの処理
        """
        # 1. 欠損値処理
        df = handle_missing_values(df)
        
        # 2. 基本特徴量抽出
        features = self.extract_base_features(df)
        
        # 3. エンジニアリング特徴量
        features.update(self.extract_engineered_features(df))
        
        # 4. 統計的特徴量
        features.update(self.extract_statistical_features(df))
        
        # 5. 周波数特徴量
        features.update(self.extract_frequency_features(df))
        
        # 6. 正規化
        features = self.normalizer.fit_transform(features)
        
        # 7. データ拡張（訓練時のみ）
        if is_training:
            features = self.apply_augmentation(features)
        
        return features
```

---

**注意事項**:
- すべての特徴量計算は、NaNや無限大を適切に処理する必要がある
- TOF/Thermalデータは欠損の可能性があるため、フォールバック処理を実装
- 計算効率を考慮し、可能な限りベクトル化された演算を使用