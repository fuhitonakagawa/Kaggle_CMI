# CMI BFRB Detection - 包括的改善戦略と実装計画
# Version: 1.0 - Comprehensive Analysis and Strategy
# Target Score: 0.85+ (Binary F1: 0.95+, Macro F1: 0.75+)

## 📊 現状分析

### 現在のスコア
- **2_IMU_improved (LightGBM/XGBoost)**: 
  - Competition Score: 0.7094
  - Binary F1: 0.9459
  - Macro F1: 0.4730
  
- **4_IMU_more_feature (Deep Learning)**:
  - Competition Score: 0.6252
  - Binary F1: 0.9788
  - Macro F1: 0.2716

### 問題点の特定
1. **深層学習モデルの低性能**: Binary F1は高いがMacro F1が極端に低い
2. **クラス不均衡への対処不足**: BFRBクラス間の分類精度が低い
3. **アンサンブル戦略の欠如**: 単一モデルに依存
4. **検証戦略の問題**: Train-Test Split vs StratifiedGroupKFold

## 🎯 改善戦略

### 1. データ処理の改善

#### 1.1 特徴量エンジニアリング強化
```python
# 既存の特徴量
- Linear Acceleration (重力除去済み)
- Angular Velocity (クォータニオンから計算)
- Magnitude features (加速度、角速度)
- Jerk features (加速度の微分)

# 新規追加特徴量
- Angular Distance (連続フレーム間の回転角度)
- Frequency Domain Features (FFT, Welch PSD)
- Statistical Window Features (移動窓統計量)
- Cross-axis Correlations (軸間相関)
- Orientation-invariant Features (方向不変特徴量)
```

#### 1.2 TOF/Thermal特徴量の改善
```python
# 既存: 単純な統計量のみ
# 改善案:
- Spatial Pattern Recognition (空間パターン認識)
- Temporal Dynamics (時間的変化パターン)
- Multi-scale Aggregation (マルチスケール集約)
- Anomaly Detection Features (異常検知特徴量)
```

### 2. モデルアーキテクチャの改善

#### 2.1 ハイブリッドアプローチ
```
┌─────────────────────────────────────┐
│     Hybrid Model Architecture       │
├─────────────────────────────────────┤
│                                     │
│  ┌──────────────┐  ┌──────────────┐│
│  │ Deep Branch  │  │ GBDT Branch  ││
│  │  (CNN+RNN)   │  │ (LightGBM)   ││
│  └──────┬───────┘  └──────┬───────┘│
│         │                  │        │
│  ┌──────▼──────────────────▼──────┐│
│  │    Meta-Learner (Stacking)     ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
```

#### 2.2 深層学習アーキテクチャの改善
```python
# Multi-Head Architecture
- Head 1: Binary Classification (BFRB vs Non-BFRB)
- Head 2: BFRB Type Classification (8 classes)
- Head 3: Full Classification (18 classes)

# Improved Components:
- Multi-Scale CNN (異なるカーネルサイズ)
- Transformer Encoder layers
- Temporal Convolutional Network (TCN)
- Gated Recurrent Units with Skip Connections
```

### 3. 訓練戦略の改善

#### 3.1 階層的学習
```python
# Stage 1: Binary Classification
- BFRB vs Non-BFRB の高精度分類器を訓練
- Class weight: {0: 1.0, 1: 2.0}

# Stage 2: BFRB Subtype Classification
- BFRBサンプルのみで8クラス分類器を訓練
- Focal Loss for handling class imbalance

# Stage 3: End-to-End Fine-tuning
- 全体を通した最適化
- Custom loss: α * Binary_CE + β * Macro_CE
```

#### 3.2 データ拡張戦略
```python
# Time Series Augmentation
- MixUp (α=0.4)
- CutMix for sequences
- Time Warping
- Magnitude Warping
- Random Noise Injection
- Rotation Augmentation (for IMU)
```

### 4. アンサンブル戦略

#### 4.1 モデル多様性の確保
```python
models = [
    # Deep Learning Models
    "CNN_BiLSTM_Attention",     # Base architecture
    "TCN_Transformer",           # Alternative architecture
    "ResNet1D_GRU",             # Residual architecture
    
    # Gradient Boosting Models
    "LightGBM_IMU_only",        # IMU features only
    "LightGBM_Full",            # All features
    "XGBoost_Engineered",       # Advanced features
    "CatBoost_Categorical",     # With categorical encoding
]
```

#### 4.2 重み付け戦略
```python
# Validation-based Weighting
- Binary F1 score weight: 0.5
- Macro F1 score weight: 0.5
- Dynamic weighting based on confidence

# Blending Methods:
1. Simple Average
2. Weighted Average (validation scores)
3. Rank Average
4. Meta-learner Stacking
```

### 5. 検証戦略

#### 5.1 StratifiedGroupKFold
```python
# 5-Fold Cross Validation
- Group by subject_id (no subject leakage)
- Stratify by gesture class
- Maintain class distribution
```

#### 5.2 評価指標の最適化
```python
def custom_metric(y_true, y_pred):
    # Binary F1 (BFRB detection)
    binary_f1 = f1_score(y_true < 8, y_pred < 8)
    
    # Macro F1 (BFRB classification)
    bfrb_mask = y_true < 8
    if bfrb_mask.any():
        macro_f1 = f1_score(
            y_true[bfrb_mask],
            y_pred[bfrb_mask],
            average='macro'
        )
    else:
        macro_f1 = 0
    
    # Competition metric
    return (binary_f1 + macro_f1) / 2
```

## 📋 実装優先順位

### Phase 1: 基盤強化（1-2日）
1. ✅ StratifiedGroupKFold検証の実装
2. ✅ 拡張特徴量エンジニアリング
3. ✅ データ正規化とスケーリングの改善

### Phase 2: モデル開発（2-3日）
1. ⬜ 階層的分類器の実装
2. ⬜ 改善された深層学習アーキテクチャ
3. ⬜ LightGBM/XGBoostの最適化

### Phase 3: アンサンブル（1日）
1. ⬜ 複数モデルの訓練
2. ⬜ 最適な重み付けの探索
3. ⬜ メタ学習器の実装

### Phase 4: 最終調整（1日）
1. ⬜ ハイパーパラメータ最適化
2. ⬜ 閾値調整
3. ⬜ 推論速度の最適化

## 🎯 期待される改善

### スコア目標
- **Binary F1**: 0.95+ (現在: 0.9459)
- **Macro F1**: 0.75+ (現在: 0.4730)
- **Combined Score**: 0.85+ (現在: 0.7094)

### 主要な改善ポイント
1. **Macro F1の大幅改善**: 階層的学習とクラスバランシング
2. **安定性の向上**: StratifiedGroupKFoldによる適切な検証
3. **汎化性能**: アンサンブルによるロバスト性向上

## 🔧 技術的考慮事項

### GPU最適化
- Metal GPU (M1/M2 Mac) サポート
- Mixed Precision Training (可能な場合)
- バッチサイズの最適化

### メモリ管理
- シーケンス長の動的パディング
- データジェネレータの使用
- 効率的な特徴量計算

### 推論最適化
- モデル量子化
- バッチ推論
- 特徴量キャッシング

## 📝 実装チェックリスト

### データ処理
- [ ] 重力除去の実装確認
- [ ] 角速度計算の実装確認
- [ ] FFT特徴量の追加
- [ ] 窓関数統計量の追加

### モデル実装
- [ ] Two-branch architectureの実装
- [ ] Attention mechanismの実装
- [ ] 階層的分類器の実装
- [ ] LightGBMパイプラインの最適化

### 訓練プロセス
- [ ] StratifiedGroupKFoldの実装
- [ ] カスタム損失関数の実装
- [ ] Early Stoppingの設定
- [ ] Learning Rate Schedulingの実装

### 評価と提出
- [ ] ローカル評価パイプライン
- [ ] Kaggle API統合
- [ ] サブミッション生成
- [ ] エラーハンドリング

## 🚀 次のステップ

1. **immediate**: 基本的な特徴量エンジニアリングの実装
2. **next**: 階層的分類器の開発
3. **future**: フルアンサンブルの構築

---

**Note**: この戦略は、既存の実装とKaggleの上位ソリューションの分析に基づいています。
実装時には、各コンポーネントを段階的にテストし、改善を確認しながら進めることが重要です。