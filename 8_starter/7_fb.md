

## 3) 潜在的な不具合・改善ポイント（具体的修正案つき）

以下は、**精度と再現性に直結し得る重要度順**に並べています。

### 3-1. 学習時の欠損処理が**推論と不一致**（NaN vs 0）

* **現状**
  各シーケンスごとに `result_df = result_df.fillna(0)` はしていますが、`pd.concat` 後に**列が存在しないサンプル**は**NaN**になります（例：ToFが存在しないサンプルは `tof_*` カラム自体が無いため、結合後にNaN）。
  一方、**推論時**は `align_features_for_inference` で **欠損列を0で追加**しています。
  ⇒ **学習はNaN、推論は0** という**分布差**が生じ、木のsplit（`isna`）が学習されると推論で不一致になります。
* **必須修正**（学習側で最終的に0埋めを徹底）

  ```python
  # X_train 構築後
  X_train = X_train.reindex(columns=sorted(X_train.columns)).fillna(0)
  ```

  さらに `validate_features(X_train)` の結果がFalseなら強制で `fillna(0)` を実施すると安全です。

### 3-2. 周波数特徴（短系列）の**キー不一致**

* **現状**
  `extract_frequency_features` の **短系列（len<32）** 分岐では
  `band_*_rel` と `band_*_log` を返していません（0のまま返すだけの想定キーに抜けがある）。
  一方、通常パス/exceptパスでは `*_rel` / `*_log` を埋めています。
  ⇒ **サンプルにより列集合が異なる** → 3-1のNaN起因にもつながる。
* **修正（常に同じキー集合を返す）**

  ```python
  if len(data) < 32:
      base = {
          "band_0.3_3":0, "band_3_8":0, "band_8_12":0,
          "band_0.3_3_rel":0, "band_3_8_rel":0, "band_8_12_rel":0,
          "band_0.3_3_log":0, "band_3_8_log":0, "band_8_12_log":0,
          "total_power":0, "spectral_centroid":0, "spectral_rolloff":0,
          "spectral_entropy":0, "dominant_freq":0, "zcr":0 if compute_zcr else 0
      }
      for k,v in base.items(): features[f"{prefix}_{k}"] = v
      return features
  ```

  これで**周波数特徴のカラムが常に揃う**ため、結合後のNaNを抑制できます。

### 3-3. 四元数の**NaN喪失**（build\_full\_quaternion）

* **現状**
  `build_full_quaternion` で既定を単位四元数（1,0,0,0）に置き、**列ごとに `fillna(0)` して上書き**しています。
  これだと本来「その時刻でrot\_wがNaN」のケースが**0に置換**され、後段の `handle_quaternion_missing_values` が**欠損を検知できない**（`NaN`前提のロジック）。
* **修正（NaNはNaNのまま流し、既定値は「列が欠損（存在しない）」時のみ使う）**

  ```python
  def build_full_quaternion(seq_df: pd.DataFrame, available_rot_cols: list) -> np.ndarray:
      n = len(seq_df)
      q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))  # 既定
      for c in available_rot_cols:
          if c in ROT_IDXS:
              vals = seq_df[c].to_numpy(dtype=float)
              # 有効値だけ上書き（NaNは既定値を残す）
              mask = np.isfinite(vals)
              q[mask, ROT_IDXS[c]] = vals[mask]
              # NaNのまま残すと handle_quaternion_missing_values で復元可能
              q[~mask, ROT_IDXS[c]] = np.nan
      return q
  ```

  ※ これで**部分NaN**を `handle_quaternion_missing_values` が正しく補完し、**角速度**の安定化にも寄与します。

### 3-4. Demographicsの**型安全性**

* **現状**
  `sex`/`adult_child` などが\*\*数値以外（文字列）\*\*の可能性があると、LightGBMがobject列を受けてエラー/暗黙変換のリスク。
* **対策（数値化）**

  ```python
  if len(demo_df) > 0:
      demo_row = demo_df.iloc[0]
      def _to_num(x, default=0):
          try:
              return float(x)
          except Exception:
              return default
      features["age"] = _to_num(demo_row.get("age", 0))
      features["adult_child"] = _to_num(demo_row.get("adult_child", 0))  # 0/1などに前処理しておく
      features["sex"] = _to_num(demo_row.get("sex", 0))                  # M/Fなら事前にmap
      features["handedness"] = _to_num(demo_row.get("handedness", 0))
      features["height_cm"] = _to_num(demo_row.get("height_cm", 0))
      features["shoulder_to_wrist_cm"] = _to_num(demo_row.get("shoulder_to_wrist_cm", 0))
      features["elbow_to_wrist_cm"] = _to_num(demo_row.get("elbow_to_wrist_cm", 0))
  ```

  ※ コンペ配布のdemographicsが既に数値なら現状でも動きますが、**将来のDL前処理でも安定**します。

### 3-5. ToFの「無効値」前提（`invalid_val=-1.0`）の**自動推定**

* **現状**
  ToFの無効値を `-1.0` と決め打ちしています。データセットや将来の拡張で異なる場合は漏れます。
* **改善**（自動検出のフォールバック）

  * 列ごとに**最頻値**や**明らかに非物理な値**（<0等）を候補として割合をみて**無効値を推定**。割合が高い場合その値を `invalid_val` として扱う。
  * または、ユーザ設定 `Config.TOF_INVALID_VALUE = None` を設け、`None` の時だけ推定する。

### 3-6. ToF 8×8マッピングの**列順依存**の明示化

* **現状**
  `build_tof_grid_index(tof_cols)` が渡された `tof_cols` 順で dict を作り、その **順序を前提**に `A = seq_df[tof_cols].values` と対応づけています。

  * Kaggleのカラム順は安定しているはずですが、**列順が変わる**と食い違いが起こる可能性。
* **対策**

  * `grid_map` は `{col: (r,c)}` に加え、**列名→インデックス**の写像も保持し、`A_t[j]` の `j` は `tof_cols.index(col)` で求めるようにする（オーダーに依存しない）。
  * あるいは `OrderedDict` で順序固定をもう少し明示。

### 3-7. 学習時の**最終バリデーションで自動修復**

* **現状**
  `validate_features` は警告のみ。
* **改善**

  * 3-1の0埋めを **常に**実行（`fillna(0)`）、かつ `Inf→0` へクリップ。
  * モダリティフラグと値の**整合性**（`mod_present_tof==0`なら`tof_*==0`）も自動で直すと再現性が上がります。

  ```python
  X_train.replace([np.inf, -np.inf], 0, inplace=True)
  X_train.fillna(0, inplace=True)
  ```

  * 不整合行はログに出す程度でOK（性能に影響しにくい）。

### 3-8. モダリティ・ドロップアウトの**確率と方針**

* **所感**
  `p=0.5` はやや強めです。ToF/THMの寄与が高い場合、落としすぎで学習が鈍ることがあるので、**0.2〜0.4** で試す価値があります（foldごとにseedを変えているのは良い）。
  また、**両方同時ドロップ**が多いとIMU-only学習に寄りすぎることがあるので、`both_dropped` の比率をモニタすると良いです。

---

## 4) 推論パスの安全性（IMUのみ/マルチモダリティ両対応）

* \*\*列合わせ（align\_features\_for\_inference）\*\*があるため、**推論は安全**です。
  ToF/THM未提供のテストでも、**学習時に存在した `tof_*`/`thm_*`/`xmod_*`** を **0で追加**してから予測するため、失敗しません。
* **クラス合わせ**（`model.classes_` → 18クラスへ埋め戻し）も正しいので、foldごとにクラスカバレッジが違っても整合します。
* **サンプリング周期推定**の誤認（たとえばタイムスタンプがミリ秒なのに秒として扱うなど）には、既に5–200 Hzチェックが入っており、**default fs=20** にフォールバックします。

---

## 5) すぐに入れてほしい**最小修正パッチ**（差分イメージ）

### (a) 学習後のNaN/Inf 0埋め（X\_train統一）

```python
# X_train 構築直後
X_train = X_train.reindex(columns=sorted(X_train.columns))
X_train.replace([np.inf, -np.inf], 0, inplace=True)
X_train.fillna(0, inplace=True)

# （任意）validateでNGなら例外ではなく自動修復済とログ
validate_features(X_train, verbose=True)
```

### (b) 周波数特徴のキー統一（短系列でも *\_rel/*\_log を返す）

```python
def extract_frequency_features(...):
    ...
    if len(data) < 32:
        keys = [
            "band_0.3_3","band_3_8","band_8_12",
            "band_0.3_3_rel","band_3_8_rel","band_8_12_rel",
            "band_0.3_3_log","band_3_8_log","band_8_12_log",
            "total_power","spectral_centroid","spectral_rolloff",
            "spectral_entropy","dominant_freq","zcr",
        ]
        for k in keys:
            features[f"{prefix}_{k}"] = 0
        return features
    ...
```

### (c) build\_full\_quaternion の NaN保持

```python
def build_full_quaternion(seq_df, available_rot_cols):
    n = len(seq_df)
    q = np.full((n,4), np.nan, dtype=float)  # まず全NaN
    q[:,0] = 1.0  # rot_wは既定1（単位四元数想定）
    for c in available_rot_cols:
        if c in ROT_IDXS:
            vals = seq_df[c].to_numpy(dtype=float)
            q[:, ROT_IDXS[c]] = vals  # NaNはNaNのまま
    return q
```

> これで `handle_quaternion_missing_values` が**真の欠損**を検知し、**単位長補正や1要素補完**を適切に実行できます。

### (d) Demographicsの数値化（安全側）

```python
def _to_num(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default
# features[...] = _to_num(...)
```

---

## 6) 今後のDLモデル/アンサンブルに向けた前処理の拡張提案

1. **FeatureStore的な列スキーマの固定化**

   * いまの `feature_names`（学習時の列順）を**YAML/JSON**で保存しておくと、他モデル（DL）にも**同じ順/同じ欠損埋め規約**でフィード可能。
2. **標準化/正規化の外だし**

   * 木系では不要ですが、DL併用時に**学習セット統計（mean/std）**を保存して**推論時再利用**できるようにクラス化しておくと便利。
3. **モダリティ毎のマスクベクトル**

   * `mod_present_*`に加えて**連続値の可用率**（例：`tof_seq_valid_ratio_mean` は既にある）を**DLのattentionマスク**として渡せるように。
4. \*\*ウィンドウ化（重複あり）\*\*のユーティリティ

   * 将来的なCNN/Transformer入力のために、**固定長ウィンドウ分割＆パディング**を共通化するモジュールを用意。

---

## 7) まとめ（IMUのみ/マルチの両ケースでの精度・推論可否）

* **推論は両ケースとも問題なし**：列アライン＋0埋めにより安全。
* **精度面**では、今回の**モダリティ・ドロップアウト**や**モダリティフラグ**により、**IMUのみ**のテストでも過度な劣化を抑える設計になっています。
* ただし、**学習時のNaNと推論時の0の不一致**は**必ず修正**してください（3-1, 3-2）。ここが解消されれば、**分布不一致による予期せぬ精度低下**を避けられます。
* 四元数NaN保持（3-3）は**角速度や世界座標化の安定化**に効くため推奨です。

---

必要であれば、上記パッチをあなたのノートブックにそのまま差し込みできる形でまとめてお渡しします。今回の修正で、**前処理の再現性が上がり、IMU-only/マルチ両対応の堅牢さ**が一段上がります。
