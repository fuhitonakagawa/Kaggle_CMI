# 詳細比較レビュー：deep.py vs トップコントリビューターのノートブック

## 📊 実装比較マトリックス

| 機能/手法 | deep.py | LB 0.77 TF | PyTorch CNN | LightGBM World |
|----------|---------|------------|-------------|----------------|
| **基本アーキテクチャ** | LightGBM（未完成） | TF BiLSTM+GRU+Attention ✅ | PyTorch 1D-CNN ✅ | LightGBM ✅ |
| **深層学習実装** | ❌ 未実装 | ✅ 完全実装 | ✅ 完全実装 | N/A |
| **重力除去** | ✅ 実装あり | ✅ 実装あり | ❌ なし | ✅ 世界座標変換 |
| **時系列処理** | ❌ 統計特徴量のみ | ✅ パディング＋直接入力 | ✅ パディング＋直接入力 | ❌ 統計特徴量 |
| **MixUp** | ❌ 未実装 | ✅ 実装あり | ❌ なし | N/A |
| **Two-Branch** | ❌ 未実装 | ✅ IMU/TOF分離 | ❌ なし | N/A |
| **動作確認** | ❌ エラー発生 | ✅ 動作確認済 | ✅ 動作確認済 | ✅ 動作確認済 |

## 🚨 deep.pyの致命的問題

### 1. **コードの矛盾**
```python
# Line 2-12: タイトルと説明
# "Deep Learning with 1D-CNN + BiLSTM + Attention"と謳っている

# Line 565-566: 実際の実装
binary_model = lgb.LGBMClassifier(**self.config["binary_lgbm"])
# → LightGBMを使用（深層学習ではない！）
```

### 2. **未インポートのモジュール**
```python
# 必要だが未インポート：
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
```

### 3. **未定義のパラメータ**
```python
CONFIG = {
    # 深層学習用パラメータは定義されているが未使用
    "cnn_filters": [64, 128, 256],  # 使われていない
    "lstm_units": 128,               # 使われていない
    
    # LightGBM用パラメータが未定義
    # "binary_lgbm": {...},  ← これが必要！
    # "bfrb_lgbm": {...},    ← これが必要！
}
```

## 💡 トップノートブックから学ぶべき実装

### 1. **LB 0.77 (TensorFlow) の優れた実装**

#### 重力除去の実装
```python
def remove_gravity_from_acc(acc_data, rot_data):
    """四元数を使って重力成分を除去"""
    rotation = R.from_quat(quat_values)
    gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
    linear_accel = acc_values - gravity_sensor_frame
    return linear_accel
```

#### Two-Branch Architecture
```python
def build_two_branch_model():
    # IMU専用の深いブランチ
    x1 = residual_se_cnn_block(imu, 64, 3)
    x1 = residual_se_cnn_block(x1, 128, 5)
    
    # TOF/Thermal用の軽いブランチ
    x2 = Conv1D(64, 3)(tof)
    x2 = MaxPooling1D(2)(x2)
    
    # マージして処理
    merged = Concatenate()([x1, x2])
    xa = Bidirectional(LSTM(128))(merged)
    xb = Bidirectional(GRU(128))(merged)
```

#### MixUp実装
```python
class MixupGenerator(Sequence):
    def __getitem__(self, i):
        lam = np.random.beta(self.alpha, self.alpha)
        X_mix = lam * Xb + (1-lam) * Xb[perm]
        y_mix = lam * yb + (1-lam) * yb[perm]
        return X_mix, y_mix
```

### 2. **PyTorch CNNの優れた実装**

#### 1D-CNNアーキテクチャ
```python
class CNN1D(nn.Module):
    def __init__(self):
        self.block1 = nn.Sequential(
            nn.Conv1d(feature_dim, 256, kernel_size=7),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )
```

#### 特徴生成
```python
def feature_gen(arr, gen_index):
    """2つの特徴から新特徴を生成"""
    # acc_x - acc_y → acc_x_y_diff
    gen_arr[:, arr.shape[1]+i] = arr[:, ind[0]] - arr[:, ind[1]]
```

### 3. **LightGBM World Accの優れた実装**

#### 世界座標変換
```python
def compute_world_acceleration(acc, rot):
    """デバイス座標から世界座標へ変換"""
    rot_scipy = rot[:, [1, 2, 3, 0]]  # scipy形式へ
    r = R.from_quat(rot_scipy)
    acc_world = r.apply(acc)
    return acc_world
```

## 🔧 deep.pyの改善提案

### Phase 1: LightGBMベースで動作可能にする
```python
# 1. 必要なインポートを追加
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

# 2. LightGBMパラメータを定義
CONFIG["binary_lgbm"] = {
    'objective': 'binary',
    'n_estimators': 1000,
    'max_depth': 8,
    'learning_rate': 0.025,
    'colsample_bytree': 0.5,
    'subsample': 0.5,
    'random_state': 42,
}

CONFIG["bfrb_lgbm"] = {
    'objective': 'multiclass',
    'num_class': 8,
    'n_estimators': 1000,
    'max_depth': 8,
    'learning_rate': 0.025,
    'random_state': 42,
}
```

### Phase 2: 深層学習モデルの実装
```python
def create_deep_model(sequence_length, n_features, n_classes):
    """実際の深層学習モデルを実装"""
    inputs = tf.keras.Input((sequence_length, n_features))
    
    # 1D-CNN layers
    x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    
    # BiLSTM layer
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True)
    )(x)
    
    # Attention layer
    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=32
    )(x, x)
    
    # Classification head
    x = tf.keras.layers.GlobalAveragePooling1D()(attention)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)
```

### Phase 3: データ処理の改善
```python
def prepare_sequences_for_dl(df, max_len=500):
    """深層学習用に時系列データを準備"""
    sequences = []
    for seq_id in df['sequence_id'].unique():
        seq_data = df[df['sequence_id'] == seq_id]
        
        # IMUデータを取得
        imu_data = seq_data[['acc_x', 'acc_y', 'acc_z', 
                            'rot_w', 'rot_x', 'rot_y', 'rot_z']].values
        
        # 重力除去
        linear_acc = remove_gravity_from_acc(
            imu_data[:, :3], imu_data[:, 3:]
        )
        
        # パディング/トランケート
        if len(imu_data) < max_len:
            padded = np.pad(imu_data, ((0, max_len - len(imu_data)), (0, 0)))
        else:
            padded = imu_data[:max_len]
        
        sequences.append(padded)
    
    return np.array(sequences)
```

## 📈 パフォーマンス比較

| モデル | スコア | 備考 |
|--------|--------|------|
| LB 0.77 TF | 0.77 | BiLSTM+GRU+Attention、重力除去 |
| PyTorch CNN | 0.67-0.71 | 1D-CNN、特徴生成 |
| LightGBM World | 推定0.65-0.70 | 世界座標変換、統計特徴 |
| deep.py | **動作不可** | エラーで実行不可 |

## 🎯 推奨アクションプラン

### 優先度1：即座に修正すべき点
1. ✅ LightGBMのインポート追加
2. ✅ 設定パラメータの定義
3. ✅ SMOTEのインポート追加

### 優先度2：深層学習の実装
1. ✅ TensorFlowまたはPyTorchでモデル実装
2. ✅ 時系列データの前処理
3. ✅ MixUpの実装

### 優先度3：アンサンブル
1. ✅ LightGBM + 深層学習のアンサンブル
2. ✅ Test Time Augmentation
3. ✅ Pseudo-labeling

## 結論

deep.pyは優れた特徴量エンジニアリング（世界座標変換、角速度計算など）を持っていますが、**深層学習の実装が完全に欠如**しています。トップノートブックの実装を参考に、段階的に改善することで、より高いスコアが期待できます。

特に、LB 0.77のTwo-Branch ArchitectureとMixUpの実装は、すぐに適用可能で効果的です。まずはLightGBMベースで動作可能にしてから、深層学習モデルを追加実装することをお勧めします。