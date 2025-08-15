

## 2) 潜在的な問題・改善提案（重要度順）＋最小パッチ

### (A) 予期せぬ dtype 漏れ・メモリ使用の最適化

LightGBM は float32 で十分です。**学習・推論とも float32 化**しておくと、RAM削減・学習速度・予測速度の改善が見込めます。また、オブジェクト列の紛れ込み事故も防げます。

**パッチ例**

```python
# 学習直前（CV前）
X_train = X_train.astype(np.float32)

# CV 内（保険）
X_fold_train = X_fold_train.astype(np.float32)
X_fold_val   = X_fold_val.astype(np.float32)

# 推論の整列関数
def align_features_for_inference(result_df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    for col in feature_names:
        if col not in result_df.columns:
            result_df[col] = 0
    result_df = result_df[feature_names].fillna(0)
    return result_df.astype(np.float32)   # ← 追加
```

---

### (B) `apply_modality_dropout` の乱数シードがグローバルに影響

`np.random.seed(seed)` は**グローバルRNGをリセット**します。今のコードでも実害はほぼ出ませんが、将来の拡張（他のランダム処理）に備え、**局所RNG**に変更しておくのが安全です。

**パッチ例**

```python
def apply_modality_dropout(features_df: pd.DataFrame, dropout_prob: float = 0.5, seed: int = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    df_copy = features_df.copy()

    n_samples = len(df_copy)
    drop_tof = rng.random(n_samples) < dropout_prob
    drop_thm = rng.random(n_samples) < dropout_prob
    ...
```

---

### (C) `extract_features()` のデモグラ列が「デモなし」の時に生成されない

訓練時は concat 後に `fillna(0)` しているため問題は顕在化しませんが、**関数単体の出力安定性**の観点では、デモが無い場合も**同じキーを0で出力**しておく方が安全・明快です（将来、前処理のユニットテストや他モデルへの転用時に効きます）。

**パッチ例**

```python
# Demographics features with safe numeric conversion
if len(demo_df) > 0:
    ...
else:
    for k in ["age","adult_child","sex","handedness",
              "height_cm","shoulder_to_wrist_cm","elbow_to_wrist_cm"]:
        features[k] = 0.0
```

---

### (D) `detect_modalities()` の IMU 判定ロジックの表現ずれ

本実装では**回転が無い場合でも IMU 特徴（world/linear/omega=0系）が出る**よう設計されています。一方で `detect_modalities()` は「acc3軸＋rot4成分が揃って初めて imu=True」と判定しています。この戻り値は現状ほとんど使っていないため実害はありませんが、**読み手が混乱しないよう**、「IMU＝acc3軸があればTrue」とするか、この戻り値をそもそも使わないようにするのが無難です。

**パッチ案（任意）**

```python
present = {
    "imu": len(imu_acc) == 3,   # rotは不要（無ければidentityで処理）
    "tof": len(tof_cols) > 0,
    "thm": len(thm_cols) > 0,
}
```

---

### (E) ToF 空間特徴の重み付け（将来の安定化）

`tof_spatial_features_per_timestep()` は画素値を重みとして重心・モーメントを計算しています。データ特性次第では外れ画素の影響が強く出ます。\*\*オプションで「二値重み（有効画素=1）」や「分位数クリップ」\*\*を入れておくと、外乱に強くなります（性能チューニング枠・任意）。

**オプション差分（任意）**

```python
def tof_spatial_features_per_timestep(A_t, grid_map, invalid_val=-1.0, use_binary_weight=False, clip_q=None):
    ...
    W = np.where(V, M, np.nan)
    if clip_q is not None:
        lo, hi = np.nanquantile(W, [1-clip_q, clip_q])
        W = np.clip(W, lo, hi)
    if use_binary_weight:
        W = np.where(np.isfinite(W), 1.0, np.nan)
    W = np.nan_to_num(W, nan=0.0)
    ...
```

---

### (F) モデルバンドルのメタ情報の実態反映

`save_model_bundle()` の `"modalities": {"imu": True, "tof_possible": True, "thm_possible": True}` は**常に True**になっています。実データに ToF/THM が無かった場合は `*_possible=False` にしておくと、将来「テスト時にToFが来ても無視（推論整列でゼロ落ち）」という**挙動が説明しやすく**なります（任意）。

---

### (G) 速度最適化の余地

`summary_series_features()` で各系列に対して Welch を何度も回します。Kaggle環境なら問題ないはずですが、もし重いと感じたら：

* `nperseg` を固定小さめ（例えば 64）に
* 長さが同じ系列（例：`mean`/`std`/`valid_ratio`）は**PSDを一度だけ計算して再利用**
  などの高速化が可能です（パフォーマンスチューニング枠）。

---

## 3) 仕様の透明性：どの系統のどの特徴を使っているか

**メタ（列プレフィクス）**

* `sequence_length`, `age`, `adult_child`, `sex`, `handedness`, `height_cm`, `shoulder_to_wrist_cm`, `elbow_to_wrist_cm`
* モダリティフラグ: `mod_present_imu`, `mod_present_tof`, `mod_present_thm`

**IMU 由来（加速度/回転/派生）**

* 加速度（端末座標）：`acc_x|y|z_*`、大域座標：`world_acc_x|y|z_*`、重力除去：`linear_acc_x|y|z_*`
  → `*_mean, *_std, *_min, *_max, *_median, *_q25, *_q75, *_iqr, *_range, *_first, *_last, *_delta, *_skew, *_kurt, *_diff_mean, *_diff_std, *_n_changes, *_seg1|2|3_*`
* 角速度（クォータニオン差分）：`angular_vel_x|y|z_*`、角運動エネルギ：`angular_energy_*`
* クォータニオン成分：`rot_w|x|y|z_*`
* 大きさ系列：`acc_magnitude_*`, `world_acc_magnitude_*`, `linear_acc_magnitude_*`, `angular_vel_magnitude_*`
* ジャーク：`acc_jerk_*`, `world_acc_jerk_*`, `linear_acc_jerk_*`
* 相関：`corr_world_acc_xy|xz|yz`, `corr_linear_angular`
* ピーク：`*_peak_count|mean_height|mean_distance|mean_distance_sec`
* 自己相関：`*_autocorr_lag1|2|4|8`
* 勾配ヒスト：`*_grad_hist_bin0..9`
* 周波数（Welch）：`*_freq_band_0.3_3|3_8|8_12`, `_rel`, `_log`, `total_power`, `spectral_*`, `dominant_freq`
* 軸別スペクトル：`world_acc_{x|y|z}_freq_*`
* 姿勢不変系：`pose_vertical_horizontal_ratio`, `pose_tilt_angle_mean|std`
* ピラミッド統計：`*_pyramid_micro|short|medium_*`
* テール統計：`*_tail_mean|std|max|min`
* Euler角：`euler_roll|pitch|yaw_*` ＋ `*_mean_circ`, `*_R`

**ToF 由来**

* 存在フラグ：`mod_present_tof`
* 枠内要約系列：`tof_seq_(mean|std|min|max|valid_ratio|hotspot_ratio)_[統計/peak/ac/freq]`
* 空間系列（8x8マップ可のとき）：`tof_spatial_(cx|cy|mu20|mu02|mu11|ecc|lr_asym|ud_asym)_[統計/peak/ac/freq]`

**Thermal 由来**

* 存在フラグ：`mod_present_thm`
* 枠内要約系列：`thm_seq_(mean|std|min|max|valid_ratio|hotspot_ratio)_[統計/peak/ac/freq]`

**クロスモダリティ**

* `xmod_corr_linear_to_tofmin`, `xmod_corr_omega_to_tofvalid`, `xmod_corr_linear_to_thmmean`
  （Dropout時は該当モダリティ由来の xmod も 0 落とし）

---

## 4) 安全性チェック（要件との照合）

* **データが全部使われるか？**

  * `sequence_id` ごとに全行を使って系列要約を作っており、選択カラムは IMU +（存在すれば）ToF/THM。**漏れなし**。
* **推論時にカラム欠損でも動くか？**

  * `extract_features()` が無い列は自前で補う（IMU acc、rotはidentity補完）、ToF/THMは存在チェック、最終的に `align_features_for_inference()` が**学習時列に強制整列＋欠損0埋め** → **OK**。
* **実装の不具合や意図しない動作**

  * 重大なロジック不具合は見当たらず。上記 (B)(C)(D) は「より頑健にするための改善提案」。
* **データリーク**

  * `phase`/`gesture`/`subject` は**特徴量化していない**（OK）。`StratifiedGroupKFold` で被験者リークなし。
* **数値安定性**

  * NaN/Inf は抽出直後に0埋め、短系列の周波数特徴キーも完全化し、**列ごとに常に定義済み**。

---

## 5) ミニ・ユニットテスト（セルにそのまま貼れる想定）

```python
# 1) 最小系列・欠損多め
seq = pl.DataFrame({
    "acc_x":[0.0, 0.0],
    "acc_y":[np.nan, 0.0],
    "acc_z":[0.1, 0.2],
    "rot_x":[np.nan, np.nan],  # rot欠損
})
demo = pl.DataFrame({})
df = extract_features(seq, demo)
assert "mod_present_imu" in df.columns
assert not df.isna().any().any()

# 2) ToF/THM無し・IMUのみ → 0埋め整列で推論OK
X = align_features_for_inference(df, feature_names)
assert (X.columns == feature_names).all()

# 3) ToFだけ有る（ダミー）
seq2 = pl.DataFrame({
    "acc_x":np.random.randn(120),
    "acc_y":np.random.randn(120),
    "acc_z":np.random.randn(120),
    "tof_px0":np.random.rand(120)*1000,
    "tof_px1":np.random.rand(120)*1000,
})
df2 = extract_features(seq2, demo)
assert "mod_present_tof" in df2.columns
assert int(df2["mod_present_tof"][0]) == 1

# 4) Dropoutの整合性（ToF落としたら tof_* が0）
df3 = apply_modality_dropout(pd.concat([df2, df2], ignore_index=True), dropout_prob=1.0, seed=0)
tof_cols = [c for c in df3.columns if c.startswith("tof_") and c!="mod_present_tof"]
assert (df3.loc[0, tof_cols] == 0).all()
assert df3.loc[0, "mod_present_tof"] == 0
```

---

## 6) まとめ

* 現状の実装は、**IMU単独／IMU+ToF/THMの双方に対応**し、学習・推論の**特徴量整合も確実**です。
* 直ちに入れておくと良いのは **(A) float32 統一** と **(B) Dropout RNGの局所化**、そして\*\*(C) デモグラ無し時もキー作成\*\*です。
* そのほか (D)〜(G) は可読性や将来拡張のための改善案です。

必要なら、上記パッチを適用済みの差分版（抜粋）も作成できます。
