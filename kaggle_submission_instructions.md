# Kaggle提出時のエラー解決方法

## 問題
「Your submission notebook may not have started the inference server」エラーが発生

## 原因
Kaggleの推論サーバーが正しく初期化されていない、または`predict`関数が正しく登録されていない

## 解決方法

### 方法1: メインスクリプトの修正
`6_Feature_Research/feature_engineering_xgboost.py`に以下の修正を適用済み：

1. **`save_models`関数を追加** - モデルを推論用に保存
2. **`predict`関数のエラーハンドリング強化** - タイムアウトを防ぐ
3. **推論サーバーの初期化を改善** - より詳細なログ出力

### 方法2: Kaggleノートブックで直接実行する際の注意点

1. **環境変数の設定を確認**
```python
IS_KAGGLE_ENV = True  # 必ずTrueに設定
```

2. **推論サーバーの初期化を確認**
スクリプトの最後で以下が実行されることを確認：
```python
if IS_KAGGLE_ENV:
    from kaggle_evaluation.cmi_inference_server import CMIInferenceServer
    inference_server = CMIInferenceServer(predict)
    inference_server.serve()
```

3. **モデルの保存と読み込み**
推論時にタイムアウトしないよう、事前に訓練済みモデルを保存：
```python
save_models(MODELS, EXTRACTOR)
```

### 方法3: 補助スクリプトの使用（推奨）

`kaggle_submission_fix.py`を作成済み。このスクリプトは：
- グローバル変数を確実に初期化
- モデルのロードを適切に処理
- エラー時にデフォルト予測を返す

**使い方：**
1. メインスクリプトを実行してモデルを訓練
2. `kaggle_submission_fix.py`の内容をノートブックの最後に追加
3. 実行

### デバッグのヒント

1. **ログを確認**
   - `[PREDICT]`タグのメッセージを確認
   - `[INFERENCE]`タグのメッセージを確認

2. **モデルファイルの存在確認**
   - `models.pkl`と`extractor.pkl`が生成されているか確認

3. **タイムアウトの回避**
   - 特徴量抽出済みのデータを使用（`USE_EXPORTED_FEATURES = True`）
   - 事前訓練済みモデルを使用

### テスト方法

ローカルでテスト：
```bash
# IS_KAGGLE_ENV = False に設定して実行
uv run 6_Feature_Research/feature_engineering_xgboost.py
```

Kaggleでテスト：
1. ノートブックに全コードをコピー
2. `IS_KAGGLE_ENV = True`に設定
3. 実行して`submission.parquet`が生成されることを確認

## トラブルシューティング

### エラー: モデルが見つからない
→ `train_models()`を実行してから`save_models()`を呼ぶ

### エラー: タイムアウト
→ エクスポート済み特徴量を使用（`USE_EXPORTED_FEATURES = True`）

### エラー: 推論サーバーが起動しない
→ `kaggle_submission_fix.py`の内容を使用

## 最終チェックリスト

- [ ] `IS_KAGGLE_ENV = True`に設定
- [ ] エクスポート済み特徴量のパスが正しい
- [ ] `predict`関数が定義されている
- [ ] 推論サーバーの初期化コードがある
- [ ] エラーハンドリングが適切
- [ ] ログが正常に出力されている