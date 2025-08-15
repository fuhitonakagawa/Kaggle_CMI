
## 最も重大な原因：エクスポート済み特徴量読み込み時のラベル不整合

### 何が起きているか

* スクリプトは既定で **エクスポート済み特徴量を使用** します（`USE_EXPORTED_FEATURES = True`）。

  ```python
  USE_EXPORTED_FEATURES = (
      True  # True: 特徴量抽出をスキップしてエクスポート済みデータを使用
  )
  ```
* 読み込み時に `FeatureExporter.import_features(import_path)` で
  `features_df`（特徴量本体）, `loaded_labels`, `loaded_subjects`（メタ）を受け取っています。
* しかし、その直後に **現在の `labels`/`subjects` を `loaded_*` に置き換えていません**。コード上は長さ一致のみ確認しています（件数が同じなら通過）。
  これにより、**`features_df` の行順（エクスポート時の順序）と、`labels` の順序（`train_df["sequence_id"].unique()` の順序）が一致している保証がない**まま、`iloc` で分割し学習してしまいます。
* ラベルと特徴量の対応がズレると、学習は実質的に破綻し、**LBスコアが大きく低下**します（0.41という大きな差はこの症状と整合的です）。

### 該当箇所（要点）

```python
X_all, loaded_labels, loaded_subjects, extractor_state = FeatureExporter.import_features(import_path)
# ここで len だけチェックしており、labels/subjects を置き換えていない
if len(loaded_labels) != len(labels):
    use_precomputed = False
```

### 最小修正案（強く推奨）

**読み込み直後に `labels`/`subjects` を置き換える**だけで整合性が取れます。

```python
X_all, loaded_labels, loaded_subjects, extractor_state = FeatureExporter.import_features(import_path)
labels   = np.array(loaded_labels)      # ← 追加：順序をfeatures_dfに揃える
subjects = np.array(loaded_subjects)    # ← 追加
```

### 代替・補強策

* エクスポート時に **`sequence_id` を features.parquet に列として保存**し、インポート後に現行の `train_df` から得た順序に **明示的に並べ替え**る（`merge`/`sort_values`）。
  例：

  * エクスポート側：`features_df.insert(0, "sequence_id", sequence_ids)`
  * インポート側：`features_df = features_df.sort_values("sequence_id").reset_index(drop=True)`
    併せて `loaded_labels` / `loaded_subjects` も同順で並べ替える。
* 暫定回避として **`USE_EXPORTED_FEATURES = False`** にし、毎回生データから抽出（この場合は行順が一致します）。

---

## スコア差に寄与しうる副次的な要因

1. **`_extract_features_raw` の二重定義**
   `FeatureExtractor` 内で `_extract_features_raw` が2回定義されています（後者が前者を上書き）。動作上は後者が有効ですが、コメントと実装の齟齬は将来的にバグの温床です。**片方に統一**してください。

2. **学習済みモデルの強制ロード**
   既定で `USE_PRETRAINED_MODEL = True` となっており、`/kaggle/input/...` の外部モデルをロードします。
   もしそれが **IMU以外（ToF/サーマル込み）の特徴量で学習されたモデル**だと、今回のIMU-only特徴では **欠落カラムが0埋め**となり、性能が落ち得ます。Notebookとの厳密比較時は **`USE_PRETRAINED_MODEL = False`** にして、**同条件で学習**してください。

3. **ツリーモデルへのスケーリング**
   XGBoostなどの木ベースでは一般に標準化は不要で、ハイパラとの相互作用で不安定化する場合があります。まずは **スケーリング無し**で学習し、必要性を検証してください（連続値のみ・外れ値対策のみに限定する等）。

4. **`handedness` の型の一貫性**
   利き手反転処理で `handedness` を 0/1 前提に扱う箇所があります。データが `'L'/'R'` 等の文字列の場合は、学習前に以下のように数値化してください。

   ```python
   handed_map = {"L": 0, "R": 1, "Left": 0, "Right": 1}
   df["handedness"] = df["handedness"].map(handed_map).fillna(0).astype(int)
   ```

5. **フィルタ条件の差**
   スクリプトは `train_df = train_df[train_df["behavior"] == "Performs gesture"]` でフィルタしていますが、Notebook側では同等処理が見当たりません。**両者で条件を揃えて**比較してください（データ仕様次第では影響しませんが、整合性のため）。

6. **CV呼び出しの引数**
   `StratifiedGroupKFold` に `cv.split(labels, labels, subjects)` と渡しています。動作はしますが、可読性とヒューマンエラー防止の観点で **`cv.split(X_all, labels, subjects)`** の形に揃えると安全です。

---

## まずやるべき検証フロー（再現性の高い手順）

1. `feature_engineering_xgboost.py` に **最小修正**（`labels`/`subjects` の置換）を反映。
2. **`USE_EXPORTED_FEATURES = False`**、**`USE_PRETRAINED_MODEL = False`** に設定し、Notebookと同条件（同CV/同Seed）で学習→提出。
3. スコアが 0.71 近辺まで回復するかを確認。
4. 回復した場合、改めて `USE_EXPORTED_FEATURES = True` で再度試し、**順序不整合が解消されているか**（＝スコアが保てるか）を確認。
5. `_extract_features_raw` の二重定義を解消し、コードを一貫化。
