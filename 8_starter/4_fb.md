
## 2) 実装レビュー（不具合/リスク/意図しない動作）

### 重要度：高（必ず直したい）

1. **周波数特徴で `fs` を渡していない**

   * `infer_dt_and_fs()` でサンプリング周期/周波数を推定しているのに、`extract_frequency_features(..., fs=...)` の呼び出しで **fs 引数を渡しておらずデフォルト 20Hz** になっています。
   * **影響**：タイムスタンプがあるデータやサンプリングがブレるケースで**周波数帯の境界がズレ**、スコア低下/ばらつき増大の原因。
   * **修正**：周波数特徴の全呼び出しに `fs=fs` を指定（下のパッチ A 参照）。

2. **四元数列が欠損/部分欠損のテスト入力に対する堅牢性不足**

   * 推論側 `extract_features()` では `available_rot_cols` を使いますが、そのまま `handle_quaternion_missing_values()` に渡すと **列数≠4 の場合に形状不一致代入が発生**（この関数は `missing_count>1` で `[1,0,0,0]` を代入するので、列数が 2 や 3 だと例外に）。
   * さらに `compute_angular_velocity()` は `rot[:, [1,2,3,0]]` を**無条件**で実行するため、列数<4 だと **IndexError** で落ちます。
   * **影響**：回転が全欠損 or rot\_w だけ欠損等の極端ケースで推論がクラッシュ。
   * **修正**：四元数を **常に Nx4 に拡張してから** クリーニング/派生計算する共通ユーティリティを追加（下のパッチ B）。`compute_angular_velocity()` も shape を見てゼロ返しのガードを入れる。

3. **`infer_dt_and_fs()` の単位補正ロジックが “小さいときのみ補正” で、ナノ秒やミリ秒の **大きい差分** に未対応**

   * 典型の「エポック ns」であれば `dt ≈ 2e7`（ns）になりますが、現行ロジックは `dt < 1e-6`/`<1e-3` のみを補正し、**大きすぎる dt を秒と誤認** → `fs ≪ 1` になる恐れ。
   * **修正**：**妥当な fs 帯（例 5–200 Hz）へ収まるよう** 1,000 倍/1,000,000 倍で調整する安全化（下のパッチ C）。

### 重要度：中（できれば直したい）

4. **LPF（`compute_linear_acceleration`）の `filtfilt` が短系列で失敗し得る**

   * padlen 以上の長さが無いと例外に。現状 try/except が無い。
   * **修正**：`try: filtfilt ... except: median フォールバック`（パッチ D）。

5. **fold weight の 0 除算の可能性**

   * `fold_weights = np.array(cv_scores) / np.sum(cv_scores)` で和が 0 の時に 0 除算。
   * **修正**：`denom = max(np.sum(cv_scores), 1e-12)` で防御（パッチ E）。

6. **ピーク間隔（mean\_distance）がサンプル数スケール**

   * 長さの違いで解釈しづらい。`/fs` で秒に変換した指標も持つと安定（`*_mean_distance_sec` を追加）。

7. **処理済み系列数の検証がない**

   * `len(train_features_list) == n_sequences` を assert/ログで確認しておくと安心。

---

## 3) 推論時の動作確認（ケース別）

### ケースA：**IMUのみ（ToF/THM 欠損）**

* **OK**：本コードは最初から IMU のみを前提に設計。ToF/THM はそもそも参照しないため**推論に影響なし**。
* **注意**：上記 **四元数が全部/一部欠損** の場合は、現状のままだと落ちる可能性（パッチ B で解消）。

### ケースB：**IMU + 他モダリティが同居（列として存在）**

* **OK**：`extract_features()` は IMU/デモグラ以外を**無視**します。`feature_names` に合わせて不要列は捨て、足りない列は 0 埋めするため、**推論は正常**に行われます。
* **ただし**：他モダリティを**活用はしていない**ため、IMUのみの精度と同等。ToF/THM を使った上積みは本実装にはありません。

---

## 4) すぐ直せるパッチ（安全性・頑健性を上げる最小修正）

### パッチA：周波数特徴の **fs 指定** を徹底

すべての `extract_frequency_features(...)` 呼び出しに `fs=fs` を渡してください（学習・推論とも）。例：

```python
# 例：学習/推論の extract_features 内
features.update(extract_frequency_features(acc_magnitude, fs=fs, prefix="acc_mag_freq", compute_zcr=False))
features.update(extract_frequency_features(world_acc_magnitude, fs=fs, prefix="world_acc_mag_freq", compute_zcr=False))
features.update(extract_frequency_features(linear_acc_magnitude, fs=fs, prefix="linear_acc_mag_freq", compute_zcr=False))
features.update(extract_frequency_features(angular_vel_magnitude, fs=fs, prefix="angular_vel_mag_freq", compute_zcr=False))
for i, axis in enumerate(["x", "y", "z"]):
    features.update(extract_frequency_features(world_acc_data[:, i], fs=fs, prefix=f"world_acc_{axis}_freq"))
```

### パッチB：四元数を **常に Nx4 に整形**（回転欠損に強くする）

```python
# 追加ユーティリティ
ROT_IDXS = {"rot_w": 0, "rot_x": 1, "rot_y": 2, "rot_z": 3}
def build_full_quaternion(seq_df: pd.DataFrame, available_rot_cols: list) -> np.ndarray:
    n = len(seq_df)
    q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))  # 既定は単位四元数
    for c in available_rot_cols:
        q[:, ROT_IDXS[c]] = seq_df[c].to_numpy()
    return q

# 推論/学習の extract_features で rot_data の直後に置換
available_rot_cols = [c for c in ROT_COLS if c in seq_df.columns]
rot_raw = build_full_quaternion(seq_df, available_rot_cols)
rot_data_clean = handle_quaternion_missing_values(rot_raw)
rot_data_clean = fix_quaternion_sign(rot_data_clean)
```

あわせて `compute_angular_velocity` を形状チェック：

```python
def compute_angular_velocity(rot: np.ndarray, dt: float = 1/20) -> np.ndarray:
    if rot.ndim != 2 or rot.shape[0] < 2 or rot.shape[1] != 4:
        return np.zeros((max(1, rot.shape[0] if rot.ndim==2 else 1), 3))
    # 以降は現行の処理
```

### パッチC：`infer_dt_and_fs()` の単位安全化

```python
def infer_dt_and_fs(seq_df: pd.DataFrame, default_fs: float = 20.0) -> tuple:
    time_cols = ["timestamp", "time", "elapsed_time", "seconds_elapsed"]
    time_col = next((c for c in time_cols if c in seq_df.columns), None)
    if time_col is None:
        return 1.0 / default_fs, default_fs
    t = np.asarray(seq_df[time_col], dtype=float)
    if t.size < 2:
        return 1.0 / default_fs, default_fs

    dt_raw = np.median(np.diff(t))
    if not np.isfinite(dt_raw) or dt_raw <= 0:
        return 1.0 / default_fs, default_fs

    # 候補（秒, ミリ秒→秒, ナノ秒→秒）から 5～200Hz に入るものを選ぶ
    candidates = [dt_raw,
                  dt_raw / 1e3,   # ms→s
                  dt_raw / 1e9]   # ns→s
    best = None
    for dt in candidates:
        fs = 1.0 / dt
        if 5.0 <= fs <= 200.0:
            best = (dt, fs); break
    if best is None:
        # 破綻時は既定
        return 1.0 / default_fs, default_fs
    return best
```

### パッチD：短系列での LPF フォールバック

```python
def compute_linear_acceleration(world_acc: np.ndarray, fs: float = 20.0, method: str = "lpf") -> np.ndarray:
    if method == "lpf":
        wc = 0.75 / (fs / 2.0)
        if wc >= 1.0 or world_acc.shape[0] < 9:
            gravity = np.median(world_acc, axis=0, keepdims=True)
        else:
            b, a = signal.butter(2, wc, btype="low")
            try:
                gravity = signal.filtfilt(b, a, world_acc, axis=0)
            except Exception:
                gravity = np.median(world_acc, axis=0, keepdims=True)
        return world_acc - gravity
    # 以降は現行通り（median/subtract/none）
```

### パッチE：fold weight の安定化

```python
denom = max(np.sum(cv_scores), 1e-12)
fold_weights = np.array(cv_scores) / denom
```

### 追加（任意だが有用）

* **ピーク距離の秒換算**：`features[f"{prefix}_mean_distance_sec"] = features[f"{prefix}_mean_distance"] / max(fs,1e-9)`
* **処理数検証**：学習前に `assert len(train_features_list) == n_sequences` or ログ出力

