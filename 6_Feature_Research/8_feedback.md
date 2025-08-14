結論から言うと、**コード上の原因が濃厚**です。とくに下記の 2 点は精度を大きく落とします。

---

## まず押さえるべき“致命傷”2 つ

### 1) 学習時は fold ごとに別スケーラを fit→ 推論時は **1 つのスケーラだけ**で全モデルに入力している

- 学習ループ内で毎 fold `RobustScaler/StandardScaler` を **新しく fit** し、そのスケール空間で XGBoost を学習しています（`train_models()` 内）。
- ところが推論では `EXTRACTOR.transform()` の **単一のスケーラ**（しかも fold0 のもの）で特徴量を 1 回だけ変換し、**全 fold モデルに同じ特徴量を食わせて**います（`predict()` → `EXTRACTOR.transform()`）。
- 結果、**4/5 のモデルは自分が学習したスケール分布と違う入力**を受け取り、スプリット閾値がズレて予測が崩れます。LB が CV より大きく落ちる典型です。

> 対応策：
>
> - **fold ごとにスケーラを保存**しておき、推論では「生特徴量」→「fold 固有スケーラで変換」→「該当 fold モデルで推論」を行い、最後に平均化してください。
> - あるいは、最初から **`Pipeline([('scaler', ...), ('xgb', ...)])`** で学習・保存し、推論時はそのパイプラインをそのまま呼ぶのが一番安全です（スケーラとモデルの結合を保証）。

---

### 2) モデル読み込みのデフォルトパスが不整合 → `predict()` で **失敗すると全件 "Wave hello" を返す** 可能性

- `load_models()` はデフォルトで **`/kaggle/input/cmi-models/models.pkl`** を探しますが、実際には学習時に **カーネルの作業ディレクトリへ保存**（`models.pkl`）しています。
- もし推論プロセスが再起動・別プロセス化等でグローバル変数 `MODELS` が失われると、`predict()` 内で `load_models()` が呼ばれ、**見つからずに例外 → 学習トライ → 失敗 → `"Wave hello"` 固定のフォールバック**が発火します。
- これが起きると LB は壊滅的に落ちます（0.3〜0.4 台まで下がることが多い）。

> 対応策：
>
> - `load_models()` は**作業ディレクトリの `./models.pkl` / `./extractor.pkl` を最優先で探す**ようにしてください。
> - フォールバックの `"Wave hello"` は消す（あるいは確実に到達しないようにパスを直す）。
> - 可能なら **`CMIInferenceServer` 作成前に必ず `MODELS` と fold スケーラ（後述）をロード**しておき、`predict()` でロード処理に入らない設計にします。

---

## そのほかの“効いてくる”ポイント

- **ToF PCA の整合性**
  学習でエクスポート済み特徴を使うパスでは `tof_use_pca=False` で生特徴量を使っているのに、推論側 `EXTRACTOR` が `tof_use_pca=True` かつ `tof_pcas` を持っていると、**推論で余計な列が出たり分布がズレたり**します。列合わせはしているものの、**スケーリングと組み合わさると予測が不安定**になります。
  → **学習と推論で `tof_use_pca` と PCA 器の有無を必ず一致**させてください。

- **handedness の型**
  `mirror_tof_by_handedness()` は 1/0 を想定しているのに、デモグラは文字（"R"/"L"）の可能性があります。
  → `handedness` は推論・学習の両方で **正規化関数**（"R"/"Right"→1、"L"/"Left"→0、数値はそのまま）を通してから使うのが安全です。

---

## 最小修正で直すなら（具体的パッチ案）

### A. モデルとスケーラを fold 単位で保存 → 推論で fold ごとに適用

1. **学習ループ**で fold 用のスケーラ情報を保存

```python
# train_models() の fold ループ内
fold_artifacts = []  # ループ外で初期化

# ... スケーラをfitした直後
fold_artifacts.append({
    "feature_names": list(X_train_raw.columns),
    "scaler": scaler
})
```

2. **保存**（既存のモデル保存に加えて）

```python
# EXPORT_TRAINED_MODEL の保存ブロックに追加
artifact_file = model_export_dir / "fold_artifacts.pkl"
with open(artifact_file, "wb") as f:
    pickle.dump(fold_artifacts, f)
print(f"✓ Fold artifacts saved to: {artifact_file}")
```

3. **読み込み**（`load_models()` を拡張）

```python
def load_models(model_path=None, extractor_path=None, artifacts_path=None):
    # まずカレントディレクトリを優先
    if model_path is None and os.path.exists("models.pkl"):
        model_path = "models.pkl"
    if extractor_path is None and os.path.exists("extractor.pkl"):
        extractor_path = "extractor.pkl"
    if artifacts_path is None and os.path.exists("fold_artifacts.pkl"):
        artifacts_path = "fold_artifacts.pkl"

    # それでも無ければ /kaggle/input/... を探す（既存のロジック）

    # モデルとEXTRACTORのロード（既存）
    # ...

    # fold アーティファクトのロード
    global FOLD_ARTIFACTS
    FOLD_ARTIFACTS = None
    if artifacts_path and os.path.exists(artifacts_path):
        with open(artifacts_path, "rb") as f:
            FOLD_ARTIFACTS = pickle.load(f)
        print(f"✓ Loaded fold artifacts: {len(FOLD_ARTIFACTS)} folds")
    else:
        print("⚠️ Fold artifacts not found — per-fold scaling will be inconsistent")
    return MODELS, EXTRACTOR
```

4. **推論**（`predict()` を fold ごとスケール → 推論に変更）
   生特徴量を一度だけ作って、fold ごとに専用スケーラを適用します。

```python
def _to01_handedness(v):
    if isinstance(v, str):
        v = v.strip().lower()
        if v.startswith("r"): return 1
        if v.startswith("l"): return 0
    try:
        return int(v)
    except:
        return 0

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    global MODELS, EXTRACTOR, FOLD_ARTIFACTS
    if MODELS is None or EXTRACTOR is None:
        MODELS, EXTRACTOR = load_models()  # 上の修正でカレントを先に見る
    # handedness の正規化（必要なら）
    demo_df = demographics.to_pandas().copy()
    if "handedness" in demo_df.columns:
        demo_df["handedness"] = demo_df["handedness"].map(_to01_handedness)

    seq_df = sequence.to_pandas()

    # ここが重要：生の特徴量を1回だけ作る（スケールしない）
    # ※ _extract_features_raw は既に定義されているメソッド
    X_raw = EXTRACTOR._extract_features_raw(seq_df, demo_df)

    predictions = []
    if FOLD_ARTIFACTS is None:
        # 互換性確保：最後の手段として従来の transform を使う（非推奨）
        X = EXTRACTOR.transform([seq_df], [demo_df])
        for model in MODELS:
            predictions.append(model.predict_proba(X)[0])
    else:
        for model, art in zip(MODELS, FOLD_ARTIFACTS):
            X = X_raw.copy()
            # 列合わせ
            for col in art["feature_names"]:
                if col not in X.columns:
                    X[col] = 0
            X = X[art["feature_names"]]
            # fold専用スケーラで変換
            Xs = art["scaler"].transform(X)
            # 予測
            predictions.append(model.predict_proba(Xs)[0])

    avg_pred = np.mean(predictions, axis=0)
    final_class = np.argmax(avg_pred)
    return REVERSE_GESTURE_MAPPER[final_class]
```

> これで「fold 0 のスケーラで全モデルに入力する」という根本問題は解消します。

---

### B. `load_models()` のパスとフォールバックを修正（最小限）

- **作業ディレクトリの `./models.pkl` と `./extractor.pkl` を最優先**で探し、見つからなければはじめて `/kaggle/input/...` を探す。
- `"Wave hello"` 固定のフォールバックは削除（または `raise` に変更）。

---

### C. ToF PCA と handedness の整合性

- 学習時に `tof_use_pca=False` で作ったモデルを使うなら、**推論側 `EXTRACTOR.config['tof_use_pca']` も False** に統一。
- handedness は **数値化ユーティリティ**を入れて学習・推論で同じ前処理を保証。

---

## すぐできる検証（原因切り分け）

1. \*\*“推論と同じコードパス”\*\*で、学習データを 1–2 シーケンス通してみてください。

   - いまの `predict()` を使い、学習データのシーケンスを与えて予測 →CV の fold 対応の正解と F1 を簡易計算。
   - ここで急落するなら、**推論パスのスケーリング不一致**が確定です。

2. `predict()` 内のモデルロードで、**どのパスからロードされたか必ず print** して確認。

   - `/kaggle/input/...` へ飛んでいたら要修正。
   - 予防として、`MODELS is None` に入らないよう **サーバ初期化前に確実にロード**しておくのが無難です。

3. 推論直後の \*\*特徴量の分布（平均・標準偏差）\*\*を fold0 と fold1〜4 で比べる。

   - 同じ入力に対し、fold ごとの `scaler` で標準化した後の平均が 0 から大きくズレる列が多いはずです（不一致の兆候）。

---

## まとめ

- **fold 別スケーリングを推論でも再現しない**のが最大の原因候補です。これだけで LB が 0.2〜0.4 近く落ちることは珍しくありません。
- さらに **モデルロードのパス不一致**により、`predict()` がフォールバックに落ちると LB は 0.3〜0.4 台に張り付きます。
- 上記パッチ（fold アーティファクトの保存・適用、ロード優先順の修正、PCA/handedness の整合）で、**CV 相当のスコアに近づく可能性が高い**です。

必要なら、上のパッチを反映した完全版の該当関数（`train_models()` の保存部と `predict()`、`load_models()`）を書き起こします。
