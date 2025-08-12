# CMI BFRB Detection - Submission Guide

## 概要

このディレクトリには、CMI BFRB コンペティション用の IMU-only LightGBM モデルのサブミッション用ファイルが含まれています。

## ファイル構成

- `train_notebook.ipynb`: モデル訓練用ノートブック
- `inference_notebook.ipynb`: 推論・サブミッション生成用ノートブック
- `inference.py`: 推論スクリプト（ノートブックの Python 版）

## サブミッション手順

### 1. モデルの訓練

#### Kaggle Notebook 上での実行:

1. Kaggle で新しいノートブックを作成
2. `train_notebook.ipynb`の内容をコピー
3. 以下のデータセットを追加:
   - `cmi-detect-behavior-with-sensor-data` (コンペティションデータ)
4. ノートブックを実行してモデルを訓練
5. 出力される`imu_lgbm_model.pkl`を保存

### 2. モデルのアップロード

1. 訓練済みモデル(`imu_lgbm_model.pkl`)を Kaggle Dataset としてアップロード
2. データセット名を`imu-lgbm-model`として保存（または適切な名前を付ける）

### 3. 推論とサブミッション生成

#### 方法 A: Notebook 版を使用

1. Kaggle で新しいノートブックを作成
2. `inference_notebook.ipynb`の内容をコピー
3. 以下のデータセットを追加:
   - `cmi-detect-behavior-with-sensor-data` (コンペティションデータ)
   - `imu-lgbm-model` (訓練済みモデル)
4. モデルパスを更新:
   ```python
   model_path = '/kaggle/input/imu-lgbm-model/imu_lgbm_model.pkl'
   ```
5. ノートブックを実行
6. `submission.parquet`が生成される

#### 方法 B: Python Script 版を使用

1. Kaggle で新しいノートブックを作成
2. Code cell に`inference.py`の内容をコピー
3. データセットを追加（方法 A と同じ）
4. モデルパスを更新（必要に応じて）
5. スクリプトを実行

### 4. サブミット

1. 生成された`submission.parquet`をダウンロード
2. コンペティションページの"Submit Predictions"から提出

## 重要な注意事項

### CMI Inference Server について

- このコンペティションでは、通常の CSV ではなく**評価 API（CMIInferenceServer）**を使用します
- `predict()`関数は**1 シーケンスごと**に呼ばれます
- 出力は**Parquet 形式**である必要があります

### 予測関数の仕様

```python
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    # sequenceとdemographicsから特徴量を抽出
    # モデルで予測
    # gesture名（文字列）を返す
    return 'Above ear - pull hair'  # 例
```

### ジェスチャークラス

全 18 クラス（0-7: BFRB、8-17: 非 BFRB）

- クラス名は**大文字小文字・スペースまで完全一致**させる必要があります
- デフォルト予測: 'Text on phone'

## トラブルシューティング

### エラー: "No module named 'kaggle_evaluation'"

- `/kaggle/input/cmi-detect-behavior-with-sensor-data`を sys.path に追加

### エラー: "Model file not found"

- モデルパスを確認
- データセットが正しく追加されているか確認

### submission.parquet が生成されない

- `KAGGLE_IS_COMPETITION_RERUN`環境変数を確認
- ローカルテストモードで実行されているか確認

## パフォーマンス目安

- CV Score: 約 0.65-0.75（データとパラメータに依存）
- 推論時間: 1 シーケンスあたり < 1 秒

## 改善のアイデア

1. **ハイパーパラメータ調整**: config 内の LGBM パラメータを調整
2. **特徴量追加**: 周波数領域特徴、自己相関特徴など
3. **アンサンブル**: XGBoost や CatBoost との組み合わせ
4. **後処理**: 予測確率の閾値調整
