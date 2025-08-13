# CMI BFRB Detection - 包括的改善版
# Version 5.0 - Comprehensive Solution

## 📊 プロジェクト概要

手首装着型デバイスのセンサーデータから、毛抜きなどのBFRB（Body-Focused Repetitive Behaviors）を高精度で検知・分類するための包括的なソリューション。

### 目標スコア
- **Competition Score**: 0.85+ (現在の最高: 0.7094)
- **Binary F1**: 0.95+ (BFRB検出)
- **Macro F1**: 0.75+ (BFRBタイプ分類)

## 🚀 主要な改善点

### 1. データ処理の強化
- ✅ 重力除去による線形加速度の正確な計算
- ✅ クォータニオンからの角速度・角距離の導出
- ✅ FFTおよびWelch PSDによる周波数領域特徴量
- ✅ TOFセンサーの空間パターン認識
- ✅ センサー間の相互相関特徴量

### 2. モデルアーキテクチャの革新
- ✅ 階層的分類戦略（Binary → BFRB → Full）
- ✅ Two-Branch Architecture (IMU + TOF/Thermal)
- ✅ Multi-Head出力による段階的学習
- ✅ Transformer/TCNを含む多様なアーキテクチャ

### 3. アンサンブル戦略
- ✅ Deep Learning × Gradient Boosting ハイブリッド
- ✅ 20モデル以上のブレンディング
- ✅ 検証スコアベースの動的重み付け
- ✅ メタ学習によるスタッキング

### 4. 訓練戦略の最適化
- ✅ StratifiedGroupKFold（被験者リークなし）
- ✅ MixUp/CutMixによるデータ拡張
- ✅ Focal Lossによるクラス不均衡対策
- ✅ カリキュラム学習による段階的訓練

## 📁 ファイル構成

```
5_CMI_comprehensive/
├── README.md                        # このファイル
├── strategy_and_implementation.md   # 戦略と実装計画
├── data_processing_spec.md         # データ処理仕様
├── model_architecture_spec.md      # モデルアーキテクチャ仕様
├── src/                            # 実装コード（今後作成）
│   ├── data_processing.py         # データ処理パイプライン
│   ├── feature_engineering.py     # 特徴量エンジニアリング
│   ├── models/                    # モデル実装
│   │   ├── deep_learning.py      # 深層学習モデル
│   │   ├── gradient_boosting.py  # 勾配ブースティング
│   │   └── ensemble.py           # アンサンブル
│   ├── training/                  # 訓練スクリプト
│   │   ├── train_dl.py          # DL訓練
│   │   ├── train_gb.py          # GB訓練
│   │   └── train_ensemble.py    # アンサンブル訓練
│   └── inference.py              # 推論パイプライン
├── config/                        # 設定ファイル
│   └── config.yaml               # ハイパーパラメータ
├── notebooks/                     # 分析ノートブック
│   ├── eda.ipynb                # 探索的データ分析
│   └── validation.ipynb         # モデル検証
└── submission/                    # 提出用ファイル
    └── submission.py             # Kaggle提出用コード
```

## 🔄 実装フェーズ

### Phase 1: データ処理基盤（完了）
- ✅ 包括的な特徴量エンジニアリング仕様
- ✅ 欠損値処理とデータ拡張戦略
- ✅ 正規化とスケーリング手法

### Phase 2: モデル開発（完了）
- ✅ Deep Learningアーキテクチャ設計
- ✅ Gradient Boostingパイプライン設計
- ✅ 階層的分類器の仕様策定

### Phase 3: 実装（次のステップ）
- ⬜ データ処理パイプラインの実装
- ⬜ モデルの実装とテスト
- ⬜ アンサンブルシステムの構築

### Phase 4: 最適化
- ⬜ ハイパーパラメータチューニング
- ⬜ 推論速度の最適化
- ⬜ 最終提出準備

## 💡 技術的ハイライト

### 特徴量エンジニアリング
```python
# 重力除去による線形加速度
linear_accel = remove_gravity(acc_data, quaternion_data)

# 角速度計算
angular_velocity = calculate_angular_velocity(quaternion_data)

# 周波数特徴量
fft_features = extract_frequency_features(signal)
psd_features = extract_psd_features(signal)
```

### 階層的分類
```python
# Stage 1: Binary Classification
binary_pred = binary_model.predict(features)  # BFRB vs Non-BFRB

# Stage 2: BFRB Subtype (if BFRB)
if binary_pred == "BFRB":
    bfrb_pred = bfrb_model.predict(features)  # 8 classes

# Stage 3: Full Classification with constraints
final_pred = hierarchical_combine(binary_pred, bfrb_pred, non_bfrb_pred)
```

### アンサンブル
```python
# 多様なモデルの組み合わせ
models = [
    CNN_BiLSTM_Attention(),
    TCN_Transformer(),
    LightGBM_IMU_only(),
    XGBoost_Full_Features()
]

# 最適化された重み付け
weights = optimize_weights(val_predictions, val_labels)
final_prediction = weighted_ensemble(models, weights)
```

## 📈 期待される性能向上

| メトリック | 現在のベスト | 目標 | 改善率 |
|-----------|------------|------|--------|
| Binary F1 | 0.9459 | 0.95+ | +0.4% |
| Macro F1 | 0.4730 | 0.75+ | +58.6% |
| Combined | 0.7094 | 0.85+ | +19.8% |

### 改善の根拠
1. **Macro F1の大幅改善**: 階層的学習により、BFRBサブタイプの分類精度が向上
2. **安定性の向上**: StratifiedGroupKFoldとアンサンブルによる汎化性能の改善
3. **特徴量の充実**: 物理的に意味のある特徴量の追加

## 🛠️ 使用技術

### フレームワーク
- TensorFlow 2.x (Deep Learning)
- LightGBM / XGBoost / CatBoost (Gradient Boosting)
- scikit-learn (前処理・評価)

### 主要ライブラリ
- scipy (信号処理)
- pandas / polars (データ処理)
- numpy (数値計算)

### GPU最適化
- Metal Performance Shaders (M1/M2 Mac)
- CUDA (NVIDIA GPU)
- Mixed Precision Training

## 📝 実行方法

### 環境構築
```bash
# uvを使用したパッケージインストール
uv add tensorflow lightgbm xgboost catboost
uv add scipy scikit-learn pandas polars
```

### 訓練実行
```bash
# 全モデルの訓練
python src/training/train_all.py --config config/config.yaml

# 個別モデルの訓練
python src/training/train_dl.py --model cnn_bilstm
python src/training/train_gb.py --model lightgbm
```

### 推論実行
```bash
# Kaggle提出用ファイル生成
python submission/submission.py --ensemble weighted
```

## 🎯 次のアクション

1. **即座に実行**: データ処理パイプラインの実装開始
2. **並行作業**: 各モデルの実装とユニットテスト
3. **検証**: ローカルCVでの性能確認
4. **最適化**: ハイパーパラメータのベイズ最適化
5. **提出**: Kaggle APIを使用した自動提出

## 📚 参考資料

- [Competition Overview](../Competition_Overview.md)
- [LB 0.82 Solution](../notebooks-TopVoted/cmi25-imu-thm-tof-tf-blendingmodel-lb-82.ipynb)
- [Kaggle Discussion](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion)

## ⚠️ 注意事項

- GPUメモリ使用量に注意（特にアンサンブル時）
- TOFデータの欠損（50%のテストデータ）への対処
- 推論時間の制限（Kaggle環境）

---

**作成日**: 2025年8月13日  
**作成者**: CMI Competition Team  
**ステータス**: 仕様策定完了・実装準備中