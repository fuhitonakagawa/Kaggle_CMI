# 実装ロードマップ
# CMI BFRB Detection - Implementation Roadmap

## 🎯 優先実装項目（スコア向上への影響度順）

### Priority 1: Macro F1改善の核心部分（影響度: ★★★★★）

#### 1.1 階層的分類の実装
```python
# 最優先: これだけでMacro F1が0.6+に改善される可能性
# 実装時間: 2-3時間

class HierarchicalClassifier:
    def __init__(self):
        # Binary: BFRB (0-7) vs Non-BFRB (8-17)
        self.binary_model = None
        # BFRB専用: 8クラス分類
        self.bfrb_model = None
        
    def train(self, X, y):
        # Step 1: Binary分類器を訓練
        y_binary = (y < 8).astype(int)
        self.binary_model = LightGBM(params_binary)
        self.binary_model.fit(X, y_binary)
        
        # Step 2: BFRBサンプルのみで8クラス分類器を訓練
        bfrb_mask = y < 8
        X_bfrb = X[bfrb_mask]
        y_bfrb = y[bfrb_mask]
        self.bfrb_model = LightGBM(params_multiclass)
        self.bfrb_model.fit(X_bfrb, y_bfrb)
```

#### 1.2 Angular Velocity特徴量
```python
# 既存のdeep.pyにはあるが、LightGBMには未実装
# 実装時間: 1時間

def calculate_angular_features(rot_data):
    # Angular velocity
    angular_vel = calculate_angular_velocity(rot_data)
    
    # Angular distance (連続フレーム間の回転角)
    angular_dist = calculate_angular_distance(rot_data)
    
    return {
        'angular_vel_mag': np.linalg.norm(angular_vel, axis=1),
        'angular_dist': angular_dist,
        'angular_vel_x': angular_vel[:, 0],
        'angular_vel_y': angular_vel[:, 1],
        'angular_vel_z': angular_vel[:, 2]
    }
```

### Priority 2: StratifiedGroupKFold検証（影響度: ★★★★☆）

```python
# 現在: train_test_split → 過学習のリスク
# 改善: StratifiedGroupKFold → 適切な汎化性能評価
# 実装時間: 1時間

from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # モデル訓練
    model = train_model(X_train, y_train)
    
    # 検証
    val_pred = model.predict(X_val)
    score = calculate_competition_metric(y_val, val_pred)
```

### Priority 3: 簡易アンサンブル（影響度: ★★★★☆）

```python
# 5モデルの単純平均でも効果大
# 実装時間: 2時間

models = []

# Model 1: LightGBM (IMU only)
models.append(train_lightgbm_imu_only())

# Model 2: LightGBM (Full features)
models.append(train_lightgbm_full())

# Model 3: XGBoost
models.append(train_xgboost())

# Model 4: Deep Learning (既存のdeep.py)
models.append(load_deep_model())

# Model 5: CatBoost
models.append(train_catboost())

# 推論
def ensemble_predict(X):
    predictions = []
    for model in models:
        pred = model.predict_proba(X)
        predictions.append(pred)
    
    # 単純平均
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred.argmax(axis=1)
```

### Priority 4: FFT特徴量（影響度: ★★★☆☆）

```python
# 周期的な動作パターンの捕捉
# 実装時間: 1時間

def extract_fft_features(signal, sample_rate=20):
    fft_vals = np.abs(fft(signal))
    freqs = fftfreq(len(signal), 1/sample_rate)
    
    # 正の周波数のみ
    pos_mask = freqs > 0
    fft_vals = fft_vals[pos_mask]
    freqs = freqs[pos_mask]
    
    return {
        'dominant_freq': freqs[np.argmax(fft_vals)],
        'spectral_energy': np.sum(fft_vals**2),
        'spectral_entropy': -np.sum((fft_vals/np.sum(fft_vals)) * 
                                    np.log2(fft_vals/np.sum(fft_vals) + 1e-10))
    }
```

## 📋 実装チェックリスト（1日で完了可能）

### Morning (3-4 hours)
- [ ] 階層的分類器の実装とテスト
- [ ] Angular velocity/distance特徴量の追加
- [ ] StratifiedGroupKFoldの実装

### Afternoon (3-4 hours)
- [ ] 5モデルアンサンブルの構築
- [ ] FFT特徴量の追加
- [ ] 統合テストと検証

### Evening (1-2 hours)
- [ ] Kaggle提出用コードの準備
- [ ] ローカル検証結果の確認
- [ ] 提出とLBスコア確認

## 🚀 クイックスタートコード

```python
# main.py - 統合実行スクリプト

import numpy as np
import pandas as pd
from pathlib import Path

# 1. データ読み込み
train_df = pd.read_csv('train.csv')
demo_df = pd.read_csv('train_demographics.csv')

# 2. 特徴量エンジニアリング
features = extract_all_features(train_df)

# 3. 階層的分類器の訓練
hierarchical_clf = HierarchicalClassifier()
hierarchical_clf.fit(features, labels, groups=subjects, cv='sgkf')

# 4. アンサンブル
ensemble = EnsembleClassifier([
    hierarchical_clf,
    lightgbm_model,
    xgboost_model,
    deep_model,
    catboost_model
])

# 5. 検証
val_score = ensemble.validate()
print(f"Validation Score: {val_score}")

# 6. 提出ファイル生成
if val_score > 0.8:
    generate_submission(ensemble)
```

## ⚡ 即効性のある改善（30分以内で実装可能）

### 1. Class Weight調整
```python
# 現在: balanced
# 改善: BFRBクラスにより高い重み

class_weights = compute_class_weight('balanced', 
                                    classes=np.unique(y), 
                                    y=y)
# BFRBクラス（0-7）の重みを1.5倍に
class_weights[:8] *= 1.5
```

### 2. 後処理の追加
```python
def post_process_predictions(predictions):
    # Binary分類の信頼度が低い場合は調整
    binary_conf = np.max([predictions[:8].sum(), predictions[8:].sum()])
    
    if binary_conf < 0.7:
        # 信頼度が低い場合は、より保守的な予測に
        predictions = smooth_predictions(predictions)
    
    return predictions
```

### 3. TOF欠損時のフォールバック
```python
def handle_missing_tof(df):
    # TOFが欠損している場合、IMU特徴量を強化
    if df['tof_1_v0'].isna().all():
        # IMU特徴量の重みを増やす
        imu_features = extract_enhanced_imu_features(df)
        return imu_features
    else:
        return extract_all_features(df)
```

## 📊 期待される改善効果

| 実装項目 | 実装時間 | Binary F1改善 | Macro F1改善 | 総合スコア改善 |
|---------|---------|-------------|-------------|--------------|
| 階層的分類 | 3時間 | +0.5% | +20% | +10% |
| Angular特徴量 | 1時間 | +0.2% | +5% | +2.5% |
| SGKF検証 | 1時間 | 0% | +10% | +5% |
| アンサンブル | 2時間 | +0.3% | +15% | +7.5% |
| FFT特徴量 | 1時間 | +0.1% | +3% | +1.5% |
| **合計** | **8時間** | **+1.1%** | **+53%** | **+26.5%** |

### 予測最終スコア
- Binary F1: 0.957 (0.9459 → 0.957)
- Macro F1: 0.726 (0.473 → 0.726)
- **Combined: 0.841** (0.7094 → 0.841)

## 🔥 実装のコツ

1. **並列開発**: 階層的分類とアンサンブルは独立して開発可能
2. **既存コードの活用**: deep.pyの特徴量関数を流用
3. **段階的テスト**: 各コンポーネントを個別に検証してから統合
4. **早期提出**: 完璧を求めず、まず提出してLBフィードバックを得る

## 📝 トラブルシューティング

### よくある問題と解決策

1. **メモリ不足**
   - 解決: データを分割処理、不要な特徴量を削除

2. **訓練時間が長い**
   - 解決: n_estimatorsを減らす、early_stoppingを使用

3. **過学習**
   - 解決: regularizationパラメータを増やす、dropoutを追加

4. **クラス不均衡**
   - 解決: focal_lossを使用、SMOTEでオーバーサンプリング

---

**重要**: まず階層的分類を実装し、動作確認後に他の改善を追加していく。
完璧を求めず、iterativeに改善していくことが重要。