以下は **v2.0 の実装レビュー**です。前回の指摘に対して“解消できている点”と“まだ残っている／新たに入ったリスク”、さらに**具体的なパッチ**を提示します。要点から順にいきます。

---

## 総評（短評）

* **良くなった点**：ToF-PCA を fold 内で fit、Welch の動的 `nperseg`、品質(quality)特徴、同期系の安定化（`correlate` など）——方向性は良いです。
* **致命的な問題**：**Scaler を「PCAを含まない特徴」で fit している**ため、**transform 時に PCA 特徴が加わって次元不一致**になります（実行時エラー or 暗黙の破綻）。
* **列名の不一致**：データセット列名（`rot_*`/`thm_*`/`tof_1_...`）とコード側の想定（`quat_*`/`therm_*`/`tof_0_...`）が混在。**特徴が計算されない／辞書キー不一致**が発生します。
* **軽微〜中程度の問題**：マルチ解像度のフラグ名ミス、NaN 処理の後退（v1.1 では IMU を前後補完していた）、`print(f"...")` の文法バグ、推論時の列合わせ不足など。

このままだと **CV 実行でエラー**になる可能性が高いです（特に PCA×scaler）。以下のパッチを当てれば安定します。

---

## ✅ 解消できている点

* **ToF-PCA の fold 内学習**：全 train（fold 内）で PCA を fit→ val/test は transform のみ。◎
* **Welch の動的ウィンドウ**：短系列でも特長量がゼロ落ちしづらくなりました。
* **ToF 同期・ラグ推定の堅牢化**：`scipy.signal.correlate / correlation_lags` による安定化。
* **Quality features**：欠測・有効率など “学習に効くノイズ検出” が追加されたのは良いです。

---

## ⚠️ 未解消／退行している点

1. **Scaler を PCA なし特徴で fit → transform は PCA あり**

   * `FeatureExtractor.fit()` では `_extract_features_raw()` で **PCA を含まない**特徴を作り **scaler を fit**。
   * その後 `transform()` は `extract_features()` を呼び、**PCA 特徴を追加**してから `scaler.transform(X)`。
   * → **次元不一致**（列数が増える）でエラー。
   * **対策**：**PCA fit → PCA を使った “最終形の特徴” を作る → それに対して scaler を fit** に順序変更。
     また **列アライン**（不足列の 0 追加・余剰列の削除・並び揃え）も必須。

2. **列名の混在とセンサーIDのズレ**

   * クォータニオン：コードは `quat_w/x/y/z` を参照、データは多くの場合 `rot_w/x/y/z`。
   * サーマル：`therm_` と `thm_` が混在。
   * ToF センサー：`range(5)`（0–4）と `range(1, 6)`（1–5）が共存。同期・クロスモーダルで **辞書キーが一致しない**可能性。
   * **対策**：**列名の自動検出**と **ToF センサーIDの自動抽出** に統一。

3. **IMU の NaN 処理が後退**

   * v1.1 では `ffill→bfill→0`。v2.0 ではそのまま渡す箇所が増え、Welch 失敗→ゼロ返しが増える懸念。
   * **対策**：IMU（acc）にも **簡易補間**（ffill/bfill）を戻す。

4. **マルチ解像度のフラグ名ミス**

   * `if self.config.get("multi_resolution", False):` となっており、CONFIG のキーは `use_multi_resolution`。**常に無効**。

5. **推論・前計算特徴の整合**

   * `USE_EXPORTED_FEATURES=True` ルートで読み込むのは **既にスケール済み**の可能性が高いのに、fold 毎に再スケール。
   * さらに **その前計算特徴が v2.0 の列設計と一致する保証なし**。
   * **対策**：前計算は **“未スケールの生特徴”** を保存する設計にし、fold 内で scaler を fit。v2.0 と列が一致しない前計算は使わないのが安全（学習では `USE_EXPORTED_FEATURES=False` を推奨）。

6. **`subject` を特徴に入れている**

   * GroupKFold なのでリークは避けられているが、**未知 subject での一般化を阻害**しやすい（数値IDは順序/距離の意味を持たない）。
   * **対策**：`subject` は削除、もしくは **統計量の集約単位**としてのみ使う（特徴には入れない）。

7. **`compute_angular_velocity` の `dt` が固定**

   * サンプリングレートを `CONFIG["sampling_rate"]` に合わせるべき。

8. **小さなバグ**

   * `print(f"\n✓ Models saved to: {model_file}")` の改行位置が崩れていて **SyntaxError**。
   * `sequence_normalize` が **未使用**。
   * `extract_cross_modal_sync_features` は残っているが **どこからも呼ばれていない**（仕様なら OK、不要なら削除）。

---

## 重要パッチ（抜粋）

### 1) 列名・センサーIDの自動検出を導入

```python
def detect_quat_cols(df: pd.DataFrame):
    candidates = [
        ["rot_w","rot_x","rot_y","rot_z"],
        ["quat_w","quat_x","quat_y","quat_z"],
    ]
    for cols in candidates:
        if all(c in df.columns for c in cols):
            return cols
    return None  # なければ None

def detect_thermal_prefix(df: pd.DataFrame) -> str:
    if any(c.startswith("thm_") for c in df.columns): return "thm_"
    if any(c.startswith("therm_") for c in df.columns): return "therm_"
    return "thm_"  # デフォルト

def detect_tof_sensor_ids(df: pd.DataFrame) -> list[int]:
    # 例: 'tof_1_v0' → センサーID=1
    ids = set()
    for c in df.columns:
        if c.startswith("tof_") and "_v" in c:
            try:
                sid = int(c.split("_")[1])
                ids.add(sid)
            except: pass
    return sorted(ids)
```

利用側は **すべてこの検出結果に従う**よう統一（0–4 と 1–5 の混在を排除）。

---

### 2) IMU の NaN 前処理を戻す

```python
def fill_series_nan(x: np.ndarray):
    return (pd.Series(x).fillna(method="ffill")
                      .fillna(method="bfill")
                      .fillna(0).values)
# 使用例
if f"acc_{axis}" in df.columns:
    acc = fill_series_nan(df[f"acc_{axis}"].values)
```

---

### 3) PCA→最終特徴→scaler-fit の順に修正（fit の流れ）

```python
def fit(self, sequences, demographics):
    print("  Fitting transformers on training data...")
    # 1) ToF PCA用データ収集→fit
    tof_data_by_sensor = {}
    for seq_df, demo_df in zip(sequences, demographics):
        sensor_ids = detect_tof_sensor_ids(seq_df)
        handed = int(demo_df["handedness"].iloc[0]) if "handedness" in demo_df else 0
        for sid in sensor_ids:
            cols = [c for c in seq_df.columns if c.startswith(f"tof_{sid}_")]
            arr = seq_df[cols].values
            if self.config.get("tof_use_handedness_mirror", False):
                arr = np.stack([mirror_tof_by_handedness(a, handed) for a in arr], 0)
            arr = np.where(((arr>=0)&~np.isnan(arr)), arr, 0)
            tof_data_by_sensor.setdefault(sid, []).append(arr)
    if self.config.get("tof_use_pca", False):
        self.tof_pcas = {}
        for sid, chunks in tof_data_by_sensor.items():
            all_arr = np.vstack(chunks)
            n = min(all_arr.shape[0]-1, all_arr.shape[1], self.config["tof_pca_components"])
            if n >= 2:
                p = PCA(n_components=n).fit(all_arr)
                self.tof_pcas[sid] = p

    # 2) PCA を含む「最終形の特徴」を train に対して抽出
    feats = []
    for seq_df, demo_df in zip(sequences, demographics):
        feats.append(self.extract_features(seq_df, demo_df))  # ← ここで PCA transform が効く
    X_train_unscaled = pd.concat(feats, ignore_index=True)
    self.feature_names = list(X_train_unscaled.columns)

    # 3) scaler を「最終形の特徴」に対して fit
    self.scaler = RobustScaler() if self.config.get("robust_scaler", True) else StandardScaler()
    self.scaler.fit(X_train_unscaled[self.feature_names])
    self.is_fitted = True
    print(f"  ✓ Fitted on {len(sequences)} sequences with {len(self.feature_names)} features")
```

---

### 4) transform の列アライン（必須）

```python
def transform(self, sequences, demographics):
    if not self.is_fitted:
        raise ValueError("FeatureExtractor must be fitted before transform")
    feat_list = [self.extract_features(s, d) for s, d in zip(sequences, demographics)]
    X = pd.concat(feat_list, ignore_index=True)

    # 不足列は 0 で補い、余剰列は落として、並びを揃える
    for c in self.feature_names:
        if c not in X.columns: X[c] = 0
    X = X[self.feature_names]

    # スケール
    X_scaled = self.scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=self.feature_names)
```

---

### 5) マルチ解像度のフラグ修正

```python
# 誤: if self.config.get("multi_resolution", False):
# 正:
if self.config.get("use_multi_resolution", False):
    ...
```

---

### 6) 角速度の `dt`

```python
def compute_angular_velocity(rot: np.ndarray, dt: float = None) -> np.ndarray:
    if dt is None:
        dt = 1.0 / CONFIG.get("sampling_rate", 20)
    ...
```

---

### 7) `print` 文の SyntaxError 修正

```python
print(f"\n✓ Models saved to: {model_file}")
```

---

### 8) 推論・前計算特徴の扱い

* **学習時は** `USE_EXPORTED_FEATURES=False` を推奨（列不一致・二重スケーリング回避）。
* もし使うなら、**未スケールの生特徴**を保存し、fold ごとに scaler fit。
* `load_models()` は `model_path` を与えた場合でも `extractor_path` を決める分岐を追加してください。

---

## そのほかの改善提案（任意）

* **`sequence_normalize`** が未使用なら削除か、**シーケンス内 z-score** を IMU/ToF/thermal で選択的に適用できるように。
* **周波数特徴の安定化**：`band_power / (total_power + 1e-8)` など **ゼロ割回避**（すでに概ね実装済みですが、比の分母にも微小値を）。
* **閾値の fold 依存化**：`percentile_thresholds` を用意しているので、ToF の「近接」や thermal のイベント閾値を **fit 時に保存→transform で使用**すると安定します。
* **特徴の上限管理**：XGB に渡す前に **相関高すぎ（>0.995）／定数列**を落とすと収束が速くなります。

---

## 最低限のスモークチェック（テスト観点）

* 1 fold で回し、`X_train.shape == X_val.shape[1] == len(feature_names)` を assert。
* `transform()` 前後で **列名・順序が一致**しているかを assert。
* ランダムに数シーケンスで `extract_features` を 2 回呼んで\*\*再現性（同一入力→同一出力）\*\*を確認。
* 欠測だらけのシーケンスでも **例外にならず**、特徴が 0/低値で埋まることを確認。

---

## まとめ

* **最大の直し**は **fit の順序（PCA→最終特徴→scaler）と transform の列アライン**です。
* 次に **列名の自動検出**でデータ不一致を潰す。
* IMU の NaN 補間とマルチ解像度の有効化、細かなバグ修正で安定性が上がります。

ここまで直せば、v2.0 の設計意図（リーク対策・短系列対応・品質メトリクス強化）を**実運用レベル**で活かせるはずです。必要なら、上記パッチを適用した最小差分のファイルを用意します。
