# 📊 IMUモデル改善分析と戦略 v3.0

## 🔍 現状分析

### 2_IMU_improvedモデルの結果
- **全体スコア**: 0.709
- **Binary F1**: 0.942 (非常に高い)
- **Macro F1**: 0.475 (非常に低い)
- **LightGBM**: 0.702 ± 0.020
- **XGBoost**: 0.707 ± 0.017

### 問題点の特定
1. **BFRB検出は優秀** - Binary F1が0.942と高く、BFRBか否かの判定は良好
2. **BFRB内分類が弱い** - Macro F1が0.475と低く、8種類のBFRB行動の区別ができていない
3. **クラス不均衡** - 特定のBFRBクラスに偏っている可能性

### トップノートブックとの比較
- **Top Voted Models**: LB 0.77-0.82
- **Gap**: 約0.07-0.11ポイントの差
- **主な違い**: 深層学習、高度な特徴量、マルチモーダル融合

---

## 🎯 改善戦略

### 1. 二段階分類アプローチ 【優先度: 高】

#### 1.1 Stage 1: Binary Classification (BFRB vs Non-BFRB)
- 現在のBinary F1 (0.942)を維持
- 高信頼度の予測のみを次段階へ

#### 1.2 Stage 2: BFRB Multi-class Classification
- BFRB 8クラスに特化したモデル
- クラスバランシングを重視
- より詳細な特徴量を使用

**実装案:**
```python
# Stage 1: Binary
binary_model = LGBMClassifier(objective='binary')
is_bfrb = binary_model.predict_proba(X)[:, 1] > threshold

# Stage 2: Multi-class (BFRB only)
if is_bfrb:
    bfrb_model = LGBMClassifier(
        objective='multiclass',
        num_class=8,
        class_weight='balanced'  # 重要
    )
    final_class = bfrb_model.predict(X)
```

---

### 2. 特徴量エンジニアリングの強化 【優先度: 高】

#### 2.1 Linear Acceleration (World Accelerationの改良)
- 重力除去の精緻化
- 動的な重力推定
- ローパスフィルタで重力成分を分離

#### 2.2 詳細な動作パターン特徴
- **ジェスチャー固有の周期性**
  - 各BFRBの典型的な周波数帯域を特定
  - バンドパスフィルタで特定周波数を強調
  
- **動作の方向性**
  - 主成分分析(PCA)で主要な動作方向を抽出
  - 各軸の寄与率を特徴量化

#### 2.3 時系列セグメンテーション
- **動的タイムワーピング(DTW)**
  - テンプレートマッチング
  - 類似度スコアを特徴量に
  
- **変化点検出の高度化**
  - PELT (Pruned Exact Linear Time)
  - Bayesian changepoint detection

---

### 3. 深層学習モデルの導入 【優先度: 中】

#### 3.1 1D-CNN + BiLSTM アーキテクチャ
```
Input (sequence_length, 7) → IMU channels
    ↓
1D-CNN (局所パターン抽出)
    ↓
BiLSTM (時系列依存性)
    ↓
Attention (重要な時点に注目)
    ↓
Dense → Output (18 classes)
```

#### 3.2 実装のポイント
- **データ正規化**: LayerNormalization
- **ドロップアウト**: 0.3-0.5
- **残差接続**: スキップコネクション
- **焦点損失**: クラス不均衡対策

---

### 4. データ拡張とサンプリング戦略 【優先度: 高】

#### 4.1 BFRB特化のオーバーサンプリング
- **SMOTE for Time Series**
  - 少数BFRBクラスの合成サンプル生成
  - DTWベースの類似度で近傍選択

#### 4.2 データオーグメンテーション
- **時間軸の変形**
  - Time stretching (0.8x - 1.2x)
  - Time shifting
  
- **振幅の変形**
  - Scaling (0.9x - 1.1x)
  - Jittering (ノイズ追加)
  
- **回転の摂動**
  - 四元数に小さな回転を追加
  - 現実的な手首の動きの範囲内

---

### 5. アンサンブル戦略の見直し 【優先度: 中】

#### 5.1 異種モデルのアンサンブル
```
├── LightGBM (統計特徴量)
├── XGBoost (統計特徴量)
├── 1D-CNN (生波形)
├── BiLSTM (時系列)
└── Random Forest (ロバスト性)
```

#### 5.2 スタッキング
- **Level 1**: 各モデルのOOF予測
- **Level 2**: メタ学習器（LogisticRegression）
- **クラス別の重み調整**

---

## 📈 期待される改善

### ターゲットメトリクス
- **全体スコア**: 0.75-0.78 (現在0.709)
- **Binary F1**: 0.94+ (維持)
- **Macro F1**: 0.56-0.62 (現在0.475)

### 段階的な改善目標
1. **Phase 1** (1週間): 二段階分類 → 0.73
2. **Phase 2** (2週間): 特徴量強化 → 0.75
3. **Phase 3** (3週間): 深層学習追加 → 0.77+

---

## 🚀 実装優先順位

### 即効性の高い改善 (1-2日)
1. ✅ 二段階分類の実装
2. ✅ クラスウェイトの調整
3. ✅ BFRBクラスのオーバーサンプリング

### 中期的な改善 (3-5日)
4. ⬜ Linear Acceleration実装
5. ⬜ 詳細な周波数特徴量
6. ⬜ データオーグメンテーション

### 長期的な改善 (1週間以上)
7. ⬜ 1D-CNN + BiLSTMモデル
8. ⬜ スタッキングアンサンブル
9. ⬜ 最適化とファインチューニング

---

## 💡 具体的な次のアクション

### Step 1: 二段階分類の実装
```python
# 実装ファイル: 3_IMU_two_stage/src/two_stage_classifier.py

class TwoStageClassifier:
    def __init__(self):
        self.binary_model = None  # BFRB vs Non-BFRB
        self.bfrb_model = None    # BFRB 8-class
        self.non_bfrb_model = None # Non-BFRB 10-class
    
    def fit(self, X, y):
        # Stage 1: Binary classification
        y_binary = (y < 8).astype(int)
        self.binary_model.fit(X, y_binary)
        
        # Stage 2a: BFRB classification
        bfrb_mask = y < 8
        self.bfrb_model.fit(X[bfrb_mask], y[bfrb_mask])
        
        # Stage 2b: Non-BFRB classification
        non_bfrb_mask = y >= 8
        self.non_bfrb_model.fit(X[non_bfrb_mask], y[non_bfrb_mask] - 8)
```

### Step 2: クラスバランシング
```python
# 実装ファイル: 3_IMU_two_stage/src/data_balancing.py

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

def balance_bfrb_classes(X, y):
    # BFRBクラスのみを対象
    bfrb_mask = y < 8
    X_bfrb = X[bfrb_mask]
    y_bfrb = y[bfrb_mask]
    
    # SMOTE + Tomek Links
    smt = SMOTETomek(random_state=42)
    X_balanced, y_balanced = smt.fit_resample(X_bfrb, y_bfrb)
    
    return X_balanced, y_balanced
```

### Step 3: 特徴量の追加
```python
# 実装ファイル: 3_IMU_two_stage/src/advanced_features.py

def extract_linear_acceleration(acc, rot, gravity_filter='butterworth'):
    """重力を除去した線形加速度"""
    # 1. ローパスフィルタで重力成分を推定
    gravity = butter_lowpass_filter(acc, cutoff=0.3, fs=20)
    
    # 2. 線形加速度 = 総加速度 - 重力
    linear_acc = acc - gravity
    
    return linear_acc

def extract_gesture_specific_features(data, gesture_templates):
    """ジェスチャー固有のテンプレートマッチング"""
    features = {}
    
    for gesture_name, template in gesture_templates.items():
        # DTW距離を計算
        distance, _ = fastdtw(data, template)
        features[f'dtw_distance_{gesture_name}'] = distance
    
    return features
```

---

## 📊 評価とモニタリング

### メトリクス追跡
- **各Foldごと**: Binary F1, Macro F1, クラス別F1
- **混同行列**: 特に混同しやすいBFRBペアを特定
- **特徴量重要度**: どの特徴がBFRB分類に効いているか

### A/Bテスト
- ベースライン vs 二段階分類
- 統計特徴のみ vs 統計+周波数特徴
- 単一モデル vs アンサンブル

---

## 🎯 最終目標

**LB 0.77+** を達成するための3つの柱：

1. **正確なBFRB検出** (Binary F1 > 0.94)
2. **精密なBFRB分類** (Macro F1 > 0.60)
3. **ロバストなアンサンブル** (CV-LB gap < 0.02)

---

*Document Version: 1.0*  
*Created: 2025-01-13*  
*Author: CMI Competition Team*