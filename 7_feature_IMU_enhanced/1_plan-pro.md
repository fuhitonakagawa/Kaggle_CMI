## 統合方針（前処理の原則）

1. **IMU コアは常に安定**して作る。非 IMU（ToF/サーマル）は\*\*“あれば使う”\*\*拡張として設計。
2. **欠損は NaN のまま保持**（XGBoost の missing 分岐を活用）。**0 埋めはしない**。
3. **スケーリングは原則オフ**（樹木系）。代わりに**相対・比率・順位・エネルギー**など**スケール頑健な特徴**を追加。
4. \*\*スマート窓（エネルギー最大窓）\*\*で S/M/L 代表区間を抽出。末尾固定から改善。
5. **品質・可用性フラグ**（has\_\*／valid_ratio 要約）を特徴に入れて、モダリティ有無をモデルに明示。
6. **列スキーマ固定**：学習時の列集合を真とし、推論時は不足列を**NaN 追加**で完全整列。
7. （学習用の前処理として）**モダリティ・ドロップアウト**を用意（ToF/サーマル特徴を確率的に NaN 化）→ テストの IMU-only に順応。

---

## 改修タスク（優先度順・具体）

### ✅ T1. NaN 保持 & スケーラ停止

**目的**：欠損を“値 0”と混同しない。
**変更**：

- `CONFIG` に追加

  ```python
  CONFIG.update({
      "preserve_nan_for_missing": True,
      "use_scaler_for_xgb": False,   # XGBoost時はスケーラ無効
  })
  ```

- `FeatureExtractor.transform()` の列アライメントを**NaN 補完**へ：

  ```python
  for col in self.feature_names:
      if col not in X.columns:
          X[col] = np.nan
  X = X[self.feature_names]

  if self.scaler is not None and self.config.get("use_scaler_for_xgb", True):
      X = pd.DataFrame(self.scaler.transform(X), columns=self.feature_names)
  ```

- `extract_features()` 末尾の**NaN/inf→0 一括置換を削除**（または設定で無効化）。

> メリット：IMU-only でも学習時と同じ欠損表現（NaN）が通り、分岐が安定。

---

### ✅ T2. 品質・可用性フラグの拡充（ゲーティング情報を特徴へ）

**目的**：モダリティ有無・品質をモデルに“見せる”。
**変更**：`extract_quality_features()` に以下を追加。

```python
features[f"{prefix}_has_imu"]     = int(all(c in sequence_df.columns for c in ["acc_x","acc_y","acc_z"]))
features[f"{prefix}_has_quat"]    = int(len(detect_quat_cols(sequence_df)) > 0)
features[f"{prefix}_has_tof"]     = int(any(c.startswith("tof_") for c in sequence_df.columns))
tp = detect_thermal_prefix(sequence_df)
features[f"{prefix}_has_thermal"] = int(any(c.startswith(tp) for c in sequence_df.columns))

# IMUの有効サンプル比（列単位→平均）
acc_valid = []
for axis in ["x","y","z"]:
    if f"acc_{axis}" in sequence_df.columns:
        v = sequence_df[f"acc_{axis}"].values
        acc_valid.append(1 - np.mean(np.isnan(v)))
features[f"{prefix}_imu_valid_ratio_mean"] = float(np.mean(acc_valid)) if acc_valid else 0.0

# ToF/サーマルは既存valid_ratioのセンサー別統計に加え、全体集約を追加
# 例: quality_tof_all_valid_ratio_mean / p25 / p75 / min
```

---

### ✅ T3. IMU コアのロバスト化（シーケンス内正規化＋重力除去フォールバック）

**目的**：装着差／被験者差／四元数欠落に頑健。
**変更**：

- **ロバスト正規化**（中央値/IQR）版の IMU 信号で、少数の基本統計＋周波数特徴を追加（生値系は従来通り残す）。

  ```python
  def robust_norm(x):
      med = np.nanmedian(x); iqr = np.nanpercentile(x,75) - np.nanpercentile(x,25)
      return (x - med) / (iqr + 1e-8)

  for axis in ["x","y","z"]:
      if f"acc_{axis}" in sequence_df.columns:
          acc = fill_series_nan(sequence_df[f"acc_{axis}"].values)
          acc_r = robust_norm(acc)
          features.update(extract_statistical_features(acc_r, f"accR_{axis}"))
          features.update(extract_frequency_features(acc_r, f"accR_{axis}"))
  ```

- **線形加速度のフォールバック**：四元数なし →`method="highpass"`で重力除去。

  ```python
  if quat_cols:
      linear_acc = compute_linear_acceleration(acc_raw, quaternions, method="subtract")
  else:
      linear_acc = compute_linear_acceleration(acc_raw, None, method="highpass")
  ```

---

### ✅ T4. “末尾固定”から\*\*エネルギー最大窓（スマート窓）\*\*へ

**目的**：S/M/L 各スケールで、動作が最も濃い区間を抽出（短系列の安定化）。
**変更**：`extract_multi_resolution_features()` をオプション化。

```python
CONFIG.update({"smart_windowing": True, "topk_windows": 1})

# acc_mag（無ければaccR合成）で移動RMSを作成→RMS最大の窓を代表採用
# 代表窓で mean/std/max/p10/p90 など基本量を少数抽出（特徴爆発は避ける）
```

---

### ✅ T5. 非 IMU の**計算スキップ**（低品質時）

**目的**：ToF/サーマルが“実質無い”ときに**高コスト抽出を回避**し、列を NaN に。
**変更**：`extract_features()` 冒頭で quality 集約を先に計算し、閾値（例：0.05）未満なら

- ToF フレーム走査／PCA／クラスタなど**重い処理をスキップ**
- **該当グループ列は生成はするが中身は NaN**（列スキーマ固定のため）

```python
q_tof_mean = features.get("quality_tof_all_valid_ratio_mean", 0.0)
HAS_TOF = (q_tof_mean is not None) and (q_tof_mean >= CONFIG["quality_thresholds"]["tof"])
if not HAS_TOF:
    # G_TOF列名のリストを用意して NaN で埋める（抽出ループは回さない）
```

---

### ✅ T6. データバリアントの恒常化（Full / IMU-only）

**目的**：同一 CV で**Full**と**IMU-only 模擬**を常時計測できるようにする。
**変更**：

- `build_dataset(variant="full"|"imu_only")` を用意。`"imu_only"`は**G_TOF/G_THM 列を一括 NaN 化**（G_QUAL は残す）。
- エクスポートは `.../features/{variant}/features.parquet` のように分けて保存。

---

### ✅ T7. 学習前処理：モダリティ・ドロップアウト（将来すぐ使える形で）

**目的**：IMU-only 多数なテスト分布への順応（データ拡張）。
**変更**：学習直前に適用できる関数を用意（保存も可能に）。

```python
CONFIG["modality_dropout_prob"] = 0.4  # 初期値の目安

def apply_modality_dropout(X: pd.DataFrame, p: float, seed: int=42):
    rng = np.random.RandomState(seed)
    rows = rng.rand(len(X)) < p
    drop_cols = [c for c in X.columns if c.startswith("tof_") or c.startswith(("thm_","therm_","thermal_"))]
    X.loc[rows, drop_cols] = np.nan
    return X
```

> **注意**：ここでは“用意”まで。適用の有無・p は CV で判断。

---

## 小さな仕様変更（安全性・効率のため）

- **周波数特徴の計算不能時**は**NaN 返却**に統一（現在は 0 を返す箇所が点在）。
  → `extract_frequency_features` 等にフラグ `return_nan_when_unavailable=True` を追加して分岐。
- **型統一**：出力は可能な限り `float32`。メモリ削減と I/O 高速化。
- **乱数源の一元化**：`FeatureExtractor` に `rng` を持たせ、スマート窓 Top-k 同点のブレやドロップアウトの再現性を確保。
- **feature_groups の明確化**：`G_IMU/G_TOF/G_THM/G_QUAL` の列名リストを JSON に保存（既存 Exporter に追記）。

---

## 受け入れ条件（Acceptance Criteria）

- [ ] **NaN 保持**：学習・推論とも**不足列は NaN 追加**で整列し、例外なく通る。
- [ ] **スケーリング無効時**でも既存学習・推論が動き、IMU-only の OOF が**悪化しない**。
- [ ] **品質フラグ**が特徴として出力され、FI 上位に少なくとも 1 本以上現れる。
- [ ] **スマート窓**有効時、短系列で PSD 失敗が NaN で扱われ、スコアが安定（分散 ↓）。
- [ ] **低品質 ToF/THM**で高コスト抽出がスキップされ、処理時間が短縮。
- [ ] **Full / IMU-only**の両データセットを同一 CV で比較できる（保存済み）。

---

## 変更パッチ（抜粋）

**CONFIG**

```python
CONFIG.update({
    "preserve_nan_for_missing": True,
    "use_scaler_for_xgb": False,
    "quality_thresholds": {"tof": 0.05, "thm": 0.05},
    "smart_windowing": True,
    "topk_windows": 1,
    "modality_dropout_prob": 0.4,
})
```

**transform（列アライメント＆スケーラ分岐）**

```python
for col in self.feature_names:
    if col not in X.columns:
        X[col] = np.nan
X = X[self.feature_names]

if self.scaler is not None and self.config.get("use_scaler_for_xgb", True):
    X = pd.DataFrame(self.scaler.transform(X), columns=self.feature_names)
```

**extract_features（NaN→0 一括置換の撤廃）**

```python
# 末尾の NaN/inf を 0 に置換する for-loop を削除
# （各 extract_* 内で“計算不能＝NaN”を返し、XGBoostに委ねる）
```

**quality 拡張（可用性フラグ）**

```python
features[f"quality_has_imu"]     = int(all(c in sequence_df.columns for c in ["acc_x","acc_y","acc_z"]))
features[f"quality_has_quat"]    = int(len(detect_quat_cols(sequence_df)) > 0)
features[f"quality_has_tof"]     = int(any(c.startswith("tof_") for c in sequence_df.columns))
tp = detect_thermal_prefix(sequence_df)
features[f"quality_has_thermal"] = int(any(c.startswith(tp) for c in sequence_df.columns))
```

**IMU ロバスト正規化＋重力除去フォールバック**（前掲の通り）

**スマート窓（例）**

```python
if config.get("smart_windowing", True):
    base = None
    if all(f"acc_{a}" in sequence_df.columns for a in ["x","y","z"]):
        base = np.sqrt(sequence_df["acc_x"]**2 + sequence_df["acc_y"]**2 + sequence_df["acc_z"]**2).values
    if base is not None:
        for win_name, (min_size, max_size) in config["window_sizes"].items():
            if len(base) < min_size: continue
            win = min(max_size, len(base))
            s = pd.Series(base)
            rms = s.rolling(win, min_periods=max(8, win//5)).apply(lambda v: np.sqrt(np.mean(v**2)))
            center = int(np.nanargmax(rms.values))
            start = max(0, min(center - win//2, len(base)-win))
            window_df = sequence_df.iloc[start:start+win]
            # ここから従来の統計・周波数など少数を生成
```

**モダリティ・ドロップアウト（準備）**（前掲の通り）
