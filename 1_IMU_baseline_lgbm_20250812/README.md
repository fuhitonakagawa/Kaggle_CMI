# IMU-only LightGBM Baseline for CMI BFRB Detection

## 概要
このベースライン実装は、CMI (Child Mind Institute) のBFRB検知コンペティションのためのIMU専用LightGBMモデルです。

### 主な特徴
- 🚀 **World Acceleration変換**: デバイス座標系から世界座標系への変換により、手首の向きの違いを正規化
- 📊 **包括的な統計特徴量**: 時系列データから多様な統計量を抽出
- 🔄 **StratifiedGroupKFold交差検証**: 被験者単位でのリークを防ぎながらクラス分布を保持
- 🎯 **階層型評価指標**: Binary F1とMacro F1を組み合わせた競技用メトリック

## プロジェクト構造
```
1_IMU_baseline_lgbm_20250112/
├── config.yaml              # ハイパーパラメータ設定
├── main.py                  # メイン実行スクリプト
├── src/
│   ├── data_loader.py       # データ読み込み
│   ├── feature_engineering.py # 特徴量エンジニアリング
│   ├── model.py             # LightGBMモデル
│   ├── train.py             # 訓練ループ
│   └── evaluate.py          # 評価指標
└── results/                 # 結果保存ディレクトリ
```

## セットアップ

### 依存関係のインストール
```bash
# プロジェクトルートで実行
uv sync
```

## 実行方法

### 1. ベースライン訓練の実行
```bash
cd 1_IMU_baseline_lgbm_20250112
uv run python main.py
```

### 2. 設定の変更
`config.yaml`を編集してハイパーパラメータを調整：
- `n_folds`: 交差検証の分割数
- `learning_rate`: 学習率
- `n_estimators`: ブースティングラウンド数
- `use_world_acceleration`: World Acceleration特徴量の使用有無

## 主要コンポーネント

### データローダー (`data_loader.py`)
- Polarsを使用した高速データ読み込み
- IMU専用カラムの自動検出
- シーケンス単位でのデータ準備

### 特徴量エンジニアリング (`feature_engineering.py`)
- **World Acceleration**: クォータニオンを使用した座標変換
- **統計特徴量**: 平均、標準偏差、歪度、尖度など
- **セグメント特徴量**: 時系列を3分割（開始・中間・終了）して特徴抽出
- **差分特徴量**: 時間変化を捉える特徴量

### モデル (`model.py`)
- LightGBMによる勾配ブースティング
- アンサンブル予測（多数決）
- 特徴量重要度の計算

### 評価 (`evaluate.py`)
- **競技用メトリック**: (Binary F1 + Macro F1) / 2
- Binary F1: BFRBか否かの2値分類
- Macro F1: BFRB内での詳細分類

## 出力

訓練実行後、以下が`results/run_YYYYMMDD_HHMMSS/`に保存されます：
- `models/`: 各foldの訓練済みモデル
- `oof_predictions.npy`: Out-of-fold予測
- `cv_results.json`: 交差検証スコア
- `feature_importance.csv`: 特徴量重要度
- `submission.csv`: テストデータの予測結果

## パフォーマンス目安
- 交差検証スコア: 約0.65-0.75（データとパラメータに依存）
- 訓練時間: 約10-20分（5-fold CV、CPUの場合）

## 改善のアイデア
1. **特徴量の追加**:
   - 周波数領域特徴（FFT、パワースペクトル）
   - 自己相関特徴
   - ウェーブレット変換

2. **モデルの改良**:
   - XGBoostやCatBoostの追加
   - ニューラルネットワーク（1D-CNN）との アンサンブル
   - スタッキング

3. **後処理**:
   - 閾値の最適化
   - Test Time Augmentation (TTA)

## 注意事項
- テストデータの約50%はIMUセンサーのみのため、IMU特化モデルが重要
- 被験者間でのデータリークを避けるため、GroupKFoldを使用
- メモリ使用量が大きい場合は、バッチ処理を検討

## ライセンス
このコードはコンペティション用に作成されました。