# CMI BFRB Detection - Submission Guide

## 概要

このディレクトリには、CMI BFRB コンペティション用の IMU-only LightGBM モデルのサブミッション用ファイルが含まれています。

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
