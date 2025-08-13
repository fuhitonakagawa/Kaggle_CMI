# IMU のみでの頑健性を高めるための方針（XGBoost 前処理・特徴量専念）

テストの一部が**IMU（加速度＋クォータニオン）だけ**になる前提で、学習時から**IMU 限定の特徴空間**で統一し、欠損や長さ差に強い**ロバスト統計＋周波数・姿勢不変化**を中心に組み立てます。
以下は、あなたのスクリプトに対して**何を・どこに・どう変えるか**をタスク分解した実装ガイドです。

---

## 0) 結論（最小変更で通すなら）

- **ToF/サーマル/クロスモーダルの全特徴を抽出対象から外す**（学習・推論の両方）。
- **NaN は NaN のまま保持**して XGBoost の欠損分岐に任せる（ゼロ埋め禁止）。
- **スケーラーは使わない**（XGBoost はスケール不変＋ NaN 対応、scaler は NaN を扱えない）。
- **IMU 拡張特徴**（軸相関・姿勢傾き・エネルギー窓 など）を追加。
- **短系列でも壊れない PSD**は現状維持（動的 nperseg）。

---

## 1) 設計方針

1. **特徴空間の完全一致**：学習＝推論＝ IMU のみ。モダリティ混在学習をやめ、分布ズレを断つ。
2. **欠測は NaN のまま**：埋め値 0 は分岐を誤らせる。XGBoost の`missing=np.nan`に任せる。
3. **姿勢不変・スケール不変**：世界座標・線形加速度・傾き角・エネルギー比など、**重力や持ち方に不変**な量を増やす。
4. **ロバスト統計**：中央値・IQR・MAD・winsorize など**外れ値に強い**統計を多用。
5. **短系列対応**：動的 PSD ＋窓サイズの自動縮退、**上位エネルギー区間**からも統計を出す。

---

## 2) タスク分解（コード改修ポイント）

### A. コンフィグと「IMU のみ」ゲート

**目的**：ToF/サーマル/クロスモーダルを物理的に切る。返る列も IMU 限定にする。

- **追加（CONFIG）**

  ```python
  CONFIG.update({
      "modalities": {"imu": True, "tof": False, "thermal": False},
      "apply_scaler": False,       # XGBはスケーリング不要＋NaN保持のため
      "nan_policy": "keep",        # NaNを保持
      "imu_clip_g": 4.0,           # |acc|の物理クリップ(例: 4g)  ※任意
      "imu_energy_windows": [10, 20, 40],  # 0.5s/1.0s/2.0s @20Hz
      "imu_energy_topk": 2,        # エネルギー上位区間の数
  })
  ```

- **許可プレフィックス（最後にフィルター）**

  ```python
  ALLOWED_PREFIXES_IMU = [
      "age","handedness","quality_",      # qualityはIMU由来の列のみ後で残す
      "acc_", "acc_mag",
      "world_acc_", "world_acc_mag",
      "linear_acc_", "linear_acc_mag",
      "angular_vel_", "angular_vel_mag",
      "euler_", "pyramid_", "micro_", "short_", "medium_",
      "imu_corr_", "imu_cov_", "imu_eig_", "tilt_", "energywin_",
      # 周波数系:
      "acc_x_band", "acc_y_band", "acc_z_band", "acc_mag_band",
      "acc_x_spectral", "acc_y_spectral", "acc_z_spectral", "acc_mag_spectral",
  ]
  ```

- **新規ユーティリティ（抽出後に実行）**

  ```python
  def filter_feature_prefixes(df: pd.DataFrame, allowed_prefixes: list) -> pd.DataFrame:
      keep = []
      for c in df.columns:
          if any(c.startswith(p) for p in allowed_prefixes):
              keep.append(c)
      # quality_*のうちToF/thermal由来を除去
      drop_kw = ["tof_", "therm_", "thermal_"]
      keep = [c for c in keep if not any(k in c for k in drop_kw)]
      return df[keep]
  ```

**実装**：

- `FeatureExtractor.extract_features()` と `_extract_features_raw()` の **ToF/thermal/クロスモーダル**の大ブロックを

  ```python
  if self.config["modalities"]["tof"]:
      # ...（現行ToF特徴）
  if self.config["modalities"]["thermal"]:
      # ...（現行サーマル特徴）
  ```

  のガードで囲い、デフォルト False にする。

- `transform()`の最後で `X = filter_feature_prefixes(X, ALLOWED_PREFIXES_IMU)` を挿入。

---

### B. NaN 方針：ゼロ埋め禁止／scaler 停止

**目的**：XGBoost の`missing=np.nan`を最大活用。
**変更**：

- `extract_features()`末尾の

  ```python
  if np.isnan(features[key]) or np.isinf(features[key]): features[key] = 0
  ```

  を以下に置換：

  ```python
  if np.isinf(features[key]): features[key] = np.nan   # NaNは保持、infだけNaNへ
  ```

- `FeatureExtractor.fit()/transform()` で**scaler を使わない**分岐：

  ```python
  if not self.config.get("apply_scaler", False):
      self.scaler = None
      # 以降、transformでもスケーリングをスキップ
  ```

- 学習部（参照用）：`xgb.XGBClassifier(..., missing=np.nan, tree_method='hist')` を明示。
  _※「モデルは先にやる」の方針でも、**NaN を残せる前処理**が完成条件になります。_

---

### C. クォータニオンの連続性補正＋姿勢不変化

**目的**：`q`と`-q`の符号曖昧性を解消し、角速度・オイラーの跳びを抑える。**世界座標・線形加速度**で姿勢依存を減らす。

- **新規**：クォータニオン符号連続性

  ```python
  def enforce_quaternion_continuity(q: np.ndarray) -> np.ndarray:
      # q shape: (T,4) [w,x,y,z]
      out = q.copy()
      for t in range(1, len(q)):
          if np.dot(out[t-1], out[t]) < 0:
              out[t] = -out[t]
      return out
  ```

  呼び出し箇所：`quat_cols`が見つかった直後

  ```python
  quaternions = handle_quaternion_missing(quaternions)
  quaternions = enforce_quaternion_continuity(quaternions)
  ```

- **傾き（tilt）特徴**：世界座標 z（重力方向）からの傾き角を追加

  ```python
  def extract_tilt_features(world_acc: np.ndarray) -> dict:
      # world_acc: (T,3)
      g = np.linalg.norm(world_acc, axis=1) + 1e-8
      cos_theta = np.clip(world_acc[:,2] / g, -1, 1)   # z軸との角度
      theta = np.arccos(cos_theta)                     # [rad]
      feats = extract_statistical_features(theta, "tilt_theta")
      feats.update({"tilt_theta_iqr": np.percentile(theta,75)-np.percentile(theta,25)})
      return feats
  ```

  呼び出し：`world_acc`算出後に `features.update(extract_tilt_features(world_acc))`

---

### D. IMU 拡張特徴（相関・固有値・エネルギー窓）

**目的**：持ち方やスケールに左右されにくい**構造情報**を足す。

- **軸相関・共分散の固有値**

  ```python
  def extract_imu_correlation_features(ax, ay, az) -> dict:
      feats = {}
      # 相関
      for pair, (u,v) in {"xy":(ax,ay),"yz":(ay,az),"xz":(ax,az)}.items():
          if len(u)>1 and len(v)>1:
              feats[f"imu_corr_{pair}"] = np.corrcoef(u, v)[0,1]
          else:
              feats[f"imu_corr_{pair}"]=np.nan
      # 3軸共分散の固有値
      M = np.vstack([ax,ay,az])
      cov = np.cov(M)
      w, _ = np.linalg.eig(cov)
      w = np.sort(np.real(w))[::-1]
      for i,val in enumerate(w):
          feats[f"imu_eig_{i}"] = val
      feats["imu_eig_ratio"] = w[0]/(w[-1]+1e-8)
      return feats
  ```

  呼び出し：`acc_x,y,z`が揃ったら `features.update(extract_imu_correlation_features(...))`

- **エネルギー窓（上位区間）**

  ```python
  def extract_energy_window_features(x: np.ndarray, fs: float, wins: list, topk: int) -> dict:
      feats={}
      # 二乗移動平均（RMS相当）
      for W in wins:
          if len(x) < W: continue
          rms = pd.Series(x**2).rolling(W, min_periods=1).mean().values
          # 上位エネルギー区間（長さW）の開始位置をtop-k抽出
          seg_energy = pd.Series(rms).rolling(W, min_periods=1).mean().values
          idx = np.argsort(seg_energy)[::-1][:topk]
          for j, s in enumerate(idx):
              seg = x[max(0,s-W):s+1]
              pfx = f"energywin_W{W}_top{j+1}"
              feats.update(extract_statistical_features(seg, f"{pfx}_stat"))
              feats.update(extract_frequency_features(seg, f"{pfx}_spectral"))
      return {"energywin_count": 1, **feats}  # ダミー1列で存在を示す
  ```

  呼び出し：`acc_mag`算出後

  ```python
  feats_energy = extract_energy_window_features(acc_mag, fs=CONFIG["sampling_rate"],
                                                wins=CONFIG["imu_energy_windows"],
                                                topk=CONFIG["imu_energy_topk"])
  features.update(feats_energy)
  ```

---

### E. ロバスト前処理（任意だが効果大）

**目的**：外れ値とバイアスを緩和。

- **winsorize**（各軸の 1–99 パーセンタイルにクリップ）

  ```python
  def winsorize_series(x, p_low=1, p_high=99):
      lo, hi = np.nanpercentile(x, p_low), np.nanpercentile(x, p_high)
      return np.clip(x, lo, hi)
  ```

  呼び出し：`acc_x/y/z`入力直後に `x = winsorize_series(x)`
  ※ 物理クリップ（`imu_clip_g`）を使うなら `np.clip(x, -g*9.81, g*9.81)` を先に。

- **シーケンス内ロバスト正規化**（必要なら）

  ```python
  def robust_standardize(x):
      med = np.nanmedian(x); mad = np.nanmedian(np.abs(x-med)) + 1e-8
      return (x - med)/mad
  ```

  例：周波数特徴や相関は正規化版にも適用して二重化せず、**どちらかに統一**（推奨：未正規化はそのまま、周波数は未正規化のまま相対量で十分）。

---

### F. 短系列の安全装置（既存維持＋微修正）

- 既存の `extract_frequency_features()` は**動的 nperseg**で OK。
- \*\*超短系列（<32）\*\*のときゼロを返す仕様は維持。`zcr/relative power/log power`もすでに堅牢。

---

### G. Quality 特徴の IMU 限定化

**目的**：IMU 品質だけ残し、ToF/thermal 品質列は一切作らない。

- `extract_quality_features()` 内の **ToF/thermal 計算ブロックごと**を

  ```python
  if CONFIG["modalities"]["tof"]:
      # ToF品質...
  if CONFIG["modalities"]["thermal"]:
      # thermal品質...
  ```

  でガード（デフォルト False）。

- 返る列は `quality_acc_*` と `quality_quat_*` のみになる。

---

### H. 戻り値の列フィルタ（ダブルチェック）

- `FeatureExtractor.extract_features()` と `_extract_features_raw()` の**最後**で

  ```python
  df = pd.DataFrame([features])
  df = filter_feature_prefixes(df, ALLOWED_PREFIXES_IMU)
  return df
  ```

  を徹底。

- `transform()` の結合後にも同じフィルタを挿入（保険）。

---

### I.（参考）学習部に触れる最小変更

前処理の要請として**NaN 保持**を活かすための最小変更です。

- **スケーリング全停止**：`train_models()` の fold 内での `RobustScaler/StandardScaler` を削除し、`X_train_raw/X_val_raw` をそのまま使用。
- **XGBoost**：

  ```python
  xgb_params = CONFIG["xgb_params"].copy()
  xgb_params.update({"tree_method":"hist","device":"cpu","missing":np.nan})
  model = xgb.XGBClassifier(**xgb_params)
  ```

  _（GPU なら`gpu_hist`でも可。重要なのは `missing=np.nan`）_

- **列整合**：`final_extractor.feature_names` は**フィルタ後の IMU 列**で更新。

---

## 3) 「定義済みの関数への差分」サマリ

- `CONFIG`：A のキーを追加、`use_tof_spatial/use_thermal_trends/use_cross_modal` は今後参照しない（BOM を簡潔に）。
- `FeatureExtractor.*`：

  - ToF/thermal/クロスモーダルの各ブロックを `if self.config["modalities"]["..."]:` でガード。
  - NaN→0 変換の**撤廃**（inf→NaN のみ）。
  - クォータニオン連続性補正の**挿入**。
  - 世界座標 → 傾き特徴、軸相関・固有値、エネルギー窓の**追加**。
  - 返却直前・`transform()`直後の**列フィルタ**を挿入。

- `extract_quality_features()`：ToF/thermal 部をガードして**出さない**。
- 新規関数：`filter_feature_prefixes / enforce_quaternion_continuity / extract_tilt_features / extract_imu_correlation_features / extract_energy_window_features / winsorize_series`。

---

## 4) 完了条件（Definition of Done）

1. **train 時の全特徴列が ALLOWED_PREFIXES_IMU のみ**（`set(X.columns)-set(ALLOWED_PREFIXES_IMU)` が空）。
2. **NaN が残っている**（例：`X.isna().sum().sum()>0` でも学習が通る）。
3. **ToF/thermal 列が一切ない**（quality 列含め）。
4. **クォータニオン反転ノイズが消え、角速度・オイラーの連続性が改善**。
5. **短系列でも例外なく特徴抽出が完走**（長さ<32 でゼロ返しの箇所が動作）。
6. **CV のスコアが、IMU 専用化前と比較して IMU-only 検証（ToF/thermal 欠落の擬似条件）で改善**。

---

## 5) 追加メモ（任意の最適化）

- **サブモダリティ・ドロップアウト**（学習時のみ）：一定割合で**クォータニオン派生特徴を NaN 化**して学習させると、acc のみでも崩れにくくなります（実装は`train_idx`に対して列名`["euler_", "angular_vel_", "world_acc_", "linear_acc_"]`系を確率 p で NaN に）。
- **年齢/利き手**：未知被験者汎化に悪さをする場合は**切る**（列フィルタから除外すれば OK）。
- **特徴数の抑制**：学習後の`feature_importances_`で IMU 系トップ k だけ残す軽量パスも有効。

---

必要なら、上記のパッチをこちらで**最小差分コード**に起こします。まずは A〜D を入れて「IMU 限定＋ NaN 保持＋新規 IMU 特徴」で CV を回すのがおすすめです。
