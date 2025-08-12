# IMU改良モデル v2.0.0

CMI - Detect Behavior with Sensor Data コンペティション向けのIMU改良モデル

## 📋 概要

このプロジェクトは、手首装着型デバイスのIMUセンサーデータから毛抜きなどのBFRB（Body-Focused Repetitive Behaviors）を検出・分類するための改良モデルです。

## 🚀 主な改良点

### 1. **World Acceleration変換**
- デバイス座標系から世界座標系への変換
- 四元数を使用した高精度な座標変換
- 重力補正による動作の正規化

### 2. **周波数領域特徴量**
- FFT、スペクトラル特徴量
- ウェーブレット変換
- ペリオドグラム解析
- 周波数帯域ごとのパワー分析

### 3. **高度な時系列特徴**
- 変化点検出
- セグメント分析（beginning, middle, end）
- ジャーク（加速度変化率）
- 回転エネルギー、角速度

### 4. **アンサンブル学習**
- LightGBMとXGBoostの組み合わせ
- 重み付き平均によるアンサンブル
- Out-of-Fold予測の活用

### 5. **強化された後処理**
- BFRB行動の優先度調整
- 混同しやすいジェスチャーの補正
- 信頼度ベースの調整
- Test Time Augmentation (TTA)

## 📁 ディレクトリ構造

```
2_IMU_improved_20250812/
├── config/
│   └── config.yaml         # 設定ファイル
├── src/
│   ├── world_acceleration.py    # World Acceleration変換
│   ├── frequency_features.py    # 周波数特徴量抽出
│   ├── feature_engineering.py   # 総合的な特徴量エンジニアリング
│   ├── postprocessing.py        # 後処理モジュール
│   ├── train_model.py          # モデル訓練スクリプト
│   └── inference.py            # 推論スクリプト
├── models/                 # 訓練済みモデル保存先
├── results/                # 実験結果保存先
├── main.py                 # メイン実行スクリプト
├── requirements.txt        # 必要パッケージ
└── README.md              # このファイル
```

## 🔧 環境設定

### 1. 必要パッケージのインストール

```bash
# uvを使用（推奨）
uv add numpy pandas polars scipy scikit-learn lightgbm xgboost pyyaml

# または requirements.txt から
uv pip install -r requirements.txt
```

### 2. データの準備

`cmi-detect-behavior-with-sensor-data/` ディレクトリに以下のファイルを配置:
- train.csv
- train_demographics.csv
- test.csv
- test_demographics.csv

## 🚀 使用方法

### モデルの訓練

```bash
# 基本的な訓練
python main.py --mode train

# 設定ファイルを指定
python main.py --mode train --config config/config.yaml
```

### 推論の実行

```bash
# 訓練済みモデルで推論
python main.py --mode inference --model_dir models/run_20250812_120000

# 訓練と推論を連続実行
python main.py --mode both
```

## ⚙️ 設定のカスタマイズ

`config/config.yaml` を編集して以下の設定を調整可能:

- **特徴量設定**: 使用する特徴量の選択
- **モデルパラメータ**: LightGBM/XGBoostのハイパーパラメータ
- **アンサンブル設定**: 重み付けや手法の選択
- **後処理設定**: BFRB優先度、信頼度閾値など

## 📊 期待される性能

- **クロスバリデーションスコア**: 0.72-0.75
- **Binary F1 (BFRB vs non-BFRB)**: 0.75-0.78
- **Macro F1 (BFRB内分類)**: 0.68-0.72

## 🎯 改良のポイント

1. **World Acceleration**: デバイスの向きに依存しない特徴量
2. **周波数解析**: ジェスチャーの周期性を捉える
3. **セグメント分析**: 動作の時間的変化を考慮
4. **アンサンブル**: 複数モデルの強みを組み合わせ
5. **後処理**: ドメイン知識を活用した予測の改善

## 📝 注意事項

- IMUセンサーのみを使用（TOF、Thermopileは使用しない）
- テストデータの約50%がIMUのみという前提で最適化
- StratifiedGroupKFoldで被験者リークを防止
- メモリ使用量を考慮してPolarsで高速データ処理

## 🔮 今後の改善案

- [ ] ディープラーニングモデル（1D-CNN, LSTM）の追加
- [ ] より高度な時系列特徴（DTW, Shapelet）
- [ ] 半教師あり学習の活用
- [ ] アクティブラーニングによるデータ拡張
- [ ] AutoMLによるハイパーパラメータ最適化

## 📄 ライセンス

このプロジェクトはKaggleコンペティション用です。

## 👥 作成者

2025年1月12日作成

---

**Version**: 2.0.0  
**Last Updated**: 2025-01-12