## 1) 追加する特徴量と系統（どのモダリティ由来か）

### IMU（すでに実装済＋少し拡張）

- 既存：デバイス/ワールド/線形加速度、角速度、ジャーク、オイラー角（unwrap ＋円統計）、Welch-PSD、オートコリレーション、ピーク、勾配ヒストグラム、姿勢不変（垂直/水平比・傾き）等
- 追加（軽微）：

  - **モダリティ存在フラグ** `mod_present_imu=1`（下記「モダリティフラグ」参照）
  - **シーケンス健全性**（NaN 率・補間率）
    例：`imu_nan_ratio`, `imu_ffill_ratio`

### ToF（Time-of-Flight 8×8）

- **空間集約**：各時刻で 64 ピクセルの `mean/std/min/max/median/q25/q75/IQR/有効比率(valid!= -1)` → これを時系列化し、**系列統計**（平均/分散/分位/レンジ/ピーク数/オートコリレーション）と**周波数特徴**（Welch-PSD、支配周波数、帯域パワー、重心、エントロピー）を算出
  例：`tof_seq_mean_mean`, `tof_seq_min_p90`, `tof_seq_valid_ratio_mean`, `tof_seq_min_freq_dominant` など
- **空間モーメント/重心**（8×8 として行列に復元可能な場合）

  - 画素座標を (x,y) で与えて **重心 (cx, cy)**、**2 次中心モーメント (μ20, μ02, μ11)**、**偏心率**
  - **左右/上下の非対称**、**4 分割（象限）の差分**、**同心円/リング別平均**
    例：`tof_cx_mean`, `tof_eccentricity`, `tof_lr_asym`, `tof_ring1_minus_ring2`

- **イベント系列**：各時刻の**最小距離系列**（近接イベント）と**有効画素数系列**（欠測挙動）

  - これに対し**ピーク検出**、**持続長（連続閾値超え）**、**ZCR**、**変化点近似（CUSUM カウント）**

- **欠損指標**：`tof_invalid_ratio_mean/max`、`tof_row_dropout_rate` など
- **モダリティ存在フラグ**：`mod_present_tof`（列自体が無い or 全て -1 なら 0）

### Thermal（赤外/温度 N チャネル）

- **チャネル集約**：各時刻で N チャネルの `mean/std/min/max/median/q25/q75/IQR/有効比率` → 系列統計 & 周波数特徴
  例：`thm_seq_mean_mean`, `thm_seq_max_p95`, `thm_seq_mean_freq_centroid`
- **ホットスポット率**：しきい値（時刻ごとの平均＋ kσ）を超える比率の系列 → 統計/ピーク/持続長
  例：`thm_hotspot_ratio_mean`, `thm_hotspot_duration_p95`
- **欠損指標**：`thm_nan_ratio_mean` など
- **モダリティ存在フラグ**：`mod_present_thm`

### クロスモダリティ（軽量で頑健）

- **系列間の粗い相関**：

  - `corr_linearAccMag__tof_min`, `corr_angularVelMag__tof_validRatio`
  - `corr_linearAccMag__thm_mean`
    （相関は有限値が十分ある場合のみ。なければ 0）

- **時刻同期不要のペア特徴**：

  - 例：`ratio_tof_mean_over_linearAccMag_mean`（平均どうしの比）など。
    → タイムスタンプ不一致でも安全。

> **重要**：**「モダリティ存在フラグ」**と**「欠損/無効比率」**は、IMU のみテスト（ToF/THM 欠落）への**分布外一般化対策**として必須です。ToF/THM が丸ごと無いときは、該当特徴は 0 埋め＋`mod_present_* = 0` で LightGBM が状況を識別できます。

---

## 2) 実装方針（設計のキモ）

1. **モダリティ検出を自動化**
   列名をスキャンして `imu`, `tof`, `thm` の有無を判定。ToF/THM の列名は**前置詞リスト＋正規表現**で柔軟に検出（例：`tof_`, `tof_px`, `TimeOfFlight`, `thermal_`, `thm_`, `temp_`）。
2. **ユニオン特徴＋欠損安全**
   学習時に**全モダリティの特徴名の集合**を保存し、推論時は**無い特徴を 0 埋め**＋**モダリティ存在フラグ/欠損比率**を併せて投入。（現状の IMU 実装で既に「訓練の feature_names に合わせて 0 列追加」を実装済みなので、そのまま拡張）
3. **タイムベースの扱い**
   まずは**シーケンス全体の集約**中心（空間 × 時間を潰す）で整合性を確保。fs 推定は既存 `infer_dt_and_fs` を流用し、ToF/THM も**周波数特徴の上限帯域を Nyquist にクランプ**。
4. **名前空間の統一**
   すべての特徴名に接頭辞：`imu_`, `tof_`, `thm_`, `xmod_`。
   例：`tof_seq_min_mean`, `xmod_corr_linear_to_tofmin`。
5. **将来の DL/メタ学習のための保存物**
   `model_data` に **feature_names**, **modality_prefixes**, **学習時に観測された列パターン**, （任意で）**ToF の 8×8 座標マップ** を保存。
   PCA など学習を伴う前処理を入れる場合は**fit 済みオブジェクト**も保存。

---

## 3) 実装タスク（順序どおりに進めれば OK）

### タスク A. 列検出ユーティリティの実装

**目的**：モダリティ有無と列名セットを自動抽出
**実装**：

```python
# 1) 設定：列パターン
TOF_PREFIXES = ("tof", "time_of_flight", "tof_px", "tof_dist")
THM_PREFIXES = ("thermal", "thm", "temp", "ir")

def _cols_startswith(df_cols, prefixes):
    prefixes = tuple(p.lower() for p in prefixes)
    return [c for c in df_cols if c.lower().startswith(prefixes)]

def detect_modalities(seq_df: pd.DataFrame):
    cols = list(seq_df.columns)
    imu_acc = [c for c in cols if c in Config.ACC_COLS]
    imu_rot = [c for c in cols if c in Config.ROT_COLS]
    tof_cols = _cols_startswith(cols, TOF_PREFIXES)
    thm_cols = _cols_startswith(cols, THM_PREFIXES)
    present = {
        "imu": len(imu_acc) == 3 and len(imu_rot) == 4,
        "tof": len(tof_cols) > 0,
        "thm": len(thm_cols) > 0,
    }
    return present, {"tof": tof_cols, "thm": thm_cols}
```

**注意**：ToF の 8×8 復元が可能なら、次タスクで座標推定も実装。

---

### タスク B. ToF の空間レイアウト復元（可能なら）

**目的**：列名に `rX_cY`/`pxXX` 等が含まれる場合に 8×8 配列へ
**実装**：

```python
import re
def build_tof_grid_index(tof_cols):
    # 例: "tof_r3_c7" or "tof_px27" -> (row, col)
    grid_map = {}
    for c in tof_cols:
        m = re.search(r"r(\d+).*?c(\d+)", c, flags=re.IGNORECASE)
        if m:
            r, col = int(m.group(1)), int(m.group(2))
        else:
            m2 = re.search(r"px(\d+)", c, flags=re.IGNORECASE)
            if m2:
                k = int(m2.group(1))  # 0..63
                r, col = divmod(k, 8)
            else:
                r = col = None  # 不明→後段でフラット扱い
        grid_map[c] = (r, col)
    # 全て座標が取れた場合のみ 8×8 とみなす
    ok = all(v[0] is not None for v in grid_map.values())
    return grid_map if ok else None
```

**落とし穴**：一部しか座標が取れない場合は**フラット**運用にフォールバック。

---

### タスク C. ToF 基本時系列の構築

**目的**：各時刻での 64 画素集約（min/mean/std/有効比率等）の系列を作る
**実装**（フラットでも空間可能でも同じ集約）：

```python
def _safe_nan_to_num(a):
    a = np.asarray(a, float)
    a[~np.isfinite(a)] = np.nan
    return a

def tof_frame_aggregates(seq_df: pd.DataFrame, tof_cols: list, invalid_val=-1.0):
    if not tof_cols:
        return None  # モダリティなし
    A = _safe_nan_to_num(seq_df[tof_cols].values)  # (T, P)
    valid = (A != invalid_val) & np.isfinite(A)
    valid_count = valid.sum(axis=1).astype(float)
    # 有効値だけで統計
    def safe_stat(fn, fill=0.0):
        with np.errstate(all='ignore'):
            x = fn(np.where(valid, A, np.nan), axis=1)
        x = np.nan_to_num(x, nan=fill)
        return x
    agg = {
        "mean": safe_stat(np.nanmean),
        "std":  safe_stat(np.nanstd),
        "min":  safe_stat(np.nanmin, fill=np.inf),  # 後で np.isinf→0
        "max":  safe_stat(np.nanmax, fill=-np.inf),
        "valid_ratio": np.nan_to_num(valid_count / A.shape[1], nan=0.0),
    }
    agg["min"][~np.isfinite(agg["min"])] = 0.0
    agg["max"][~np.isfinite(agg["max"])] = 0.0
    return agg  # 各キーに (T,) ベクトル
```

**備考**：後段でこれらの系列に対して、既存の **統計/ピーク/周波数/オートコリ** 関数を適用。

---

### タスク D. ToF 空間特徴（8×8 復元可能時のみ）

**目的**：重心・モーメント・非対称・リング差分等
**実装**：

```python
def tof_spatial_features_per_timestep(A_t: np.ndarray, grid_map: dict, invalid_val=-1.0):
    # A_t: (P,) 単時刻、grid_map: {col: (r,c)}
    # 8×8 へ配置
    M = np.full((8,8), np.nan, float)
    for j,(r,c) in ((j,grid_map[col]) for j,col in enumerate(grid_map.keys())):
        if r is not None: M[r,c] = A_t[j]
    V = np.isfinite(M) & (M != invalid_val)
    if V.sum() == 0:
        return dict(cx=0, cy=0, mu20=0, mu02=0, mu11=0, ecc=0, lr_asym=0, ud_asym=0)
    # 重心
    yy, xx = np.indices(M.shape)
    W = np.where(V, M, np.nan)  # 適宜重み、または 1.0 でも可
    W = np.nan_to_num(W, nan=0.0)
    s = W.sum()
    cx = float((W*xx).sum() / (s+1e-9))
    cy = float((W*yy).sum() / (s+1e-9))
    # 中心モーメント
    dx, dy = xx-cx, yy-cy
    mu20 = float((W*dx*dx).sum()/(s+1e-9))
    mu02 = float((W*dy*dy).sum()/(s+1e-9))
    mu11 = float((W*dx*dy).sum()/(s+1e-9))
    # 偏心率（簡易）
    ecc = float(((mu20-mu02)**2 + 4*mu11**2)**0.5 / (mu20+mu02+1e-9))
    # 左右/上下非対称（平均差）
    left  = np.nanmean(np.where(V[:, :4],  M[:, :4], np.nan))
    right = np.nanmean(np.where(V[:, 4:],  M[:, 4:], np.nan))
    up    = np.nanmean(np.where(V[:4, :],  M[:4, :], np.nan))
    down  = np.nanmean(np.where(V[4:, :],  M[4:, :], np.nan))
    lr_asym = float(np.nan_to_num(left-right))
    ud_asym = float(np.nan_to_num(up-down))
    return dict(cx=cx, cy=cy, mu20=mu20, mu02=mu02, mu11=mu11, ecc=ecc,
                lr_asym=lr_asym, ud_asym=ud_asym)
```

**集約**：各時刻の空間特徴を**時系列化 → 統計/周波数**関数でまとめる（IMU と同じやり方）。

---

### タスク E. ToF 系列 → 特徴ベクトル化

**目的**：C/D で得た系列辞書を、既存の汎用関数でまとめて特徴化
**実装**（既存の `extract_statistical_features` / `extract_peak_features` / `extract_autocorrelation_features` / `extract_frequency_features` を流用）：

```python
def summarize_series_features(series_dict: dict, fs: float, prefix: str) -> dict:
    feats = {}
    for name, x in series_dict.items():
        feats.update(extract_statistical_features(x, f"{prefix}_{name}"))
        feats.update(extract_peak_features(x, f"{prefix}_{name}_peak", fs=fs))
        feats.update(extract_autocorrelation_features(x, prefix=f"{prefix}_{name}_ac"))
        feats.update(extract_frequency_features(x, fs=fs, prefix=f"{prefix}_{name}_freq", compute_zcr=False))
    return feats
```

---

### タスク F. Thermal 前処理

**目的**：ToF と同じ思想で N チャネルの時系列を集約
**実装**：

```python
def thermal_frame_aggregates(seq_df: pd.DataFrame, thm_cols: list):
    if not thm_cols:
        return None
    A = _safe_nan_to_num(seq_df[thm_cols].values)  # (T, C)
    valid = np.isfinite(A)
    valid_count = valid.sum(axis=1).astype(float)

    def safe_stat(fn, fill=0.0):
        with np.errstate(all='ignore'):
            x = fn(np.where(valid, A, np.nan), axis=1)
        return np.nan_to_num(x, nan=fill)

    agg = {
        "mean": safe_stat(np.nanmean),
        "std":  safe_stat(np.nanstd),
        "min":  safe_stat(np.nanmin, fill=np.inf),
        "max":  safe_stat(np.nanmax, fill=-np.inf),
        "valid_ratio": np.nan_to_num(valid_count / A.shape[1], nan=0.0),
    }
    agg["min"][~np.isfinite(agg["min"])] = 0.0
    agg["max"][~np.isfinite(agg["max"])] = 0.0

    # ホットスポット率（各時刻で mean+kσ 超え）
    k = 1.0
    thr = agg["mean"][:, None] + k*(agg["std"][:, None])
    hotspot = (A > np.nan_to_num(thr, nan=np.inf)).sum(axis=1)/(A.shape[1]+1e-9)
    agg["hotspot_ratio"] = np.nan_to_num(hotspot, nan=0.0)
    return agg
```

**→** `summarize_series_features(agg, fs, prefix="thm_seq")` で特徴化。

---

### タスク G. クロスモダリティ特徴

**目的**：同期不要の相関・比率
**実装**：

```python
def xmod_features(imu_series: dict, tof_series: dict|None, thm_series: dict|None):
    feats = {}
    # 例：線形加速度|角速度の大きさ（既存計算済みのベクトル）を想定
    lin = imu_series.get("linear_acc_mag")   # (T,)
    omg = imu_series.get("angular_vel_mag")  # (T,)
    if tof_series is not None:
        tmin = tof_series.get("min")             # (T,)
        tratio = tof_series.get("valid_ratio")   # (T,)
        if lin is not None and tmin is not None and len(lin)>1 and len(tmin)>1:
            feats["xmod_corr_linear_to_tofmin"] = float(np.corrcoef(lin, tmin)[0,1])
        if omg is not None and tratio is not None and len(omg)>1 and len(tratio)>1:
            feats["xmod_corr_omega_to_tofvalid"] = float(np.corrcoef(omg, tratio)[0,1])
    if thm_series is not None and lin is not None:
        tmean = thm_series.get("mean")
        if tmean is not None and len(lin)>1 and len(tmean)>1:
            feats["xmod_corr_linear_to_thmmean"] = float(np.corrcoef(lin, tmean)[0,1])
    # NaN防止
    for k,v in list(feats.items()):
        if not np.isfinite(v): feats[k]=0.0
    return feats
```

---

### タスク H. 「モダリティ存在フラグ」と欠損度合いの注入

**目的**：IMU のみテストでの劣化を抑える
**実装**：

- `mod_present_imu` は 1 固定（IMU 前提）。
- `mod_present_tof` / `mod_present_thm` は「列がそもそも無い」または「系列の有効比率平均がゼロに近い」なら 0。
- 欠損指標 `*_nan_ratio_mean`, `*_valid_ratio_mean` も併せて入れる（ToF/THM が存在しないときは 0）。

---

### タスク I. `extract_features()` の拡張（入口を守る）

**目的**：いまの IMU 実装に ToF/THM を**可搬的に合流**
**実装要点**：

1. `present, cols = detect_modalities(seq_df)` を最初に呼ぶ
2. 既存の IMU ブロックはそのまま（すでに堅牢）
3. `tof_frame_aggregates` → `summarize_series_features(..., prefix="tof_seq")`

   - 8×8 復元できたら、各時刻 `tof_spatial_features_per_timestep` をループ、出来た系列をまとめて `summarize_series_features(..., prefix="tof_spatial")`

4. Thermal も同様に `thermal_frame_aggregates` → `summarize_series_features(..., prefix="thm_seq")`
5. IMU のいくつかの**系列そのもの**を `imu_series` に保持（例：`linear_acc_mag`, `angular_vel_mag`）し、`xmod_features` を追加
6. **モダリティフラグ/欠損比**を最後に付与
7. `result_df = pd.DataFrame([features]).fillna(0)` を返す

> 既存の **“訓練時に保存した feature_names に合わせて、推論時は欠けている列を 0 で補う”** という安全策は、そのまま効きます。

---

### タスク J. 学習・保存メタデータを拡張

**目的**：将来の DL や追加前処理にも耐える
**実装**：`model_data` に下記を追加保存

```python
"modalities": {"imu": True, "tof_possible": True, "thm_possible": True},
"col_patterns": {"tof_prefixes": list(TOF_PREFIXES), "thm_prefixes": list(THM_PREFIXES)},
# 8×8 が復元できた場合のみ
"tof_grid_map": grid_map_example_or_None,
```

（PCA 等を導入する場合は `pca_state`: components\_ / mean\_ なども保存）

---

### タスク K. 推論ロジックの堅牢化（現構成でも OK だが補強）

**目的**：モダリティ欠落時のフェイルセーフ
**実装**：

- `extract_features` 内で `mod_present_*` を算出し、**ToF/THM 系が全欠測なら、そのブロックの特徴は生成せず**（計算負荷を避ける）
- 返却直前に**学習時の feature_names を満たすよう 0 列追加**（既に実装済）
- 将来「IMU-only モデル」と「Multi-modal モデル」を併存させる場合に備え、**`inference_policy`** を追加：

  - 例）`if feats['mod_present_tof']==0 and feats['mod_present_thm']==0: use imu_only_models else: use multimodal_models`

> いまは LightGBM 単系ですが、このポリシー分岐は将来のアンサンブルでそのまま利用できます。

---

### タスク L. 学習時の「モダリティ・ドロップアウト」（任意だが強く推奨）

**目的**：非公開テストの「IMU のみ半分」に合わせて一般化を強める
**実装**：

- 学習時に **一定確率 p（例 0.5）で ToF/THM 特徴を 0 にし、`mod_present_*` も 0** にする“データ拡張”をかけたコピーを混ぜて学習
- これにより、モデルは **ToF/THM が無い分布**も経験し、IMU-only テストでの落ち込みを抑制

---

### タスク M. ユニットチェック（データが全部使われているか／欠損時も安全か）

**実装**：

- 1 シーケンスに対して `assert` ベースの簡易検査を入れる

  - `present['imu'] is True`（IMU ないケースはこのコンペでは想定外なら early return）
  - ToF/THM が存在しない場合でも `extract_features` が**例外なく DataFrame を返す**こと
  - `result_df.columns` が `feature_names` を**完全包含**すること
  - 全列が有限値（±inf/NaN なし）
  - 消費されるサンプル数（グループ件数）が期待どおり

---

## 4) 既存コードへの差し込みポイント（要点のみ）

- `extract_features` 冒頭に **モダリティ検出**（タスク A）
- IMU ブロックはそのまま（すでに robust）
- **ToF/THM のブロック**を IMU と同じ関数パターンで追加：

  1. `*_frame_aggregates`（タスク C/F）
  2. `summarize_series_features(..., prefix="tof_seq"/"thm_seq")`
  3. （8×8 可なら）`tof_spatial_features_per_timestep` → `summarize_series_features(..., "tof_spatial")`
  4. `xmod_features`（タスク G）
  5. `mod_present_*` と `*_valid_ratio_mean` 等（タスク H）

- 返却前の **0 列補完**は既存どおり
- `model_data` へ **メタ保存**を拡張（タスク J）

---

## 5) 推論が両ケースで問題なく動くか

- **IMU のみ**：

  - ToF/THM 列が存在しない → 検出で `mod_present_tof/thm = 0`。ToF/THM の集約はスキップ。
  - 返却時に学習時の `feature_names` との差分列は 0 埋めされる（既存ロジック）。
  - **欠損フラグ**が 0 で入るため、LightGBM が「このサンプルは補助モダリティ無し」と認識でき、分布外をある程度回避。

- **補助モダリティあり**：

  - 検出が True → ToF/THM 集約が走り、特徴が追加。
  - 周波数系は `fs` の Nyquist チェック済み。
  - 空間 8×8 復元ができないときはフラット集約に**自動フォールバック**。

> すべての特徴は最終的に `feature_names` に合わせられ、**推論で列欠損があっても動作**します。

---

## 6) 実装の落とし穴（本件で起きやすいもの）

- **ToF の -1**：かならず invalid として扱い、**有効比率**を特徴にする（-1 をそのまま平均に入れない）。
- **空間 8×8 不確定**：座標復元できない場合、**空間特徴はスキップ**し、系列集約だけにすること。
- **周波数特徴**：`nperseg` を系列長と Nyquist に合わせてクリップ（既に IMU でやっているのと同様）。
- **相関・比率**：長さが合わない/有効長が 1 以下はスキップして 0。
- **推論の安定化**：必ず `mod_present_*` と `*_valid_ratio_mean` を一緒に入れておく。
- **計算負荷**：8×8 空間特徴は「1 時刻 → 空間 → 統計化」のループが重くなりがち。`np.nanmean` 等の **ベクトル化**で抑制（必要ならサンプリングダウンサンプリング）。

---

## 7) すぐ貼れるコード断片（抜粋）

> 下の 3 関数をあなたの notebook に追加し、`extract_features` 内から呼び出せば動きます（A〜H の要点を反映）。

```python
def extract_tof_features(seq_df: pd.DataFrame, fs: float) -> dict:
    present, cols = detect_modalities(seq_df)
    tof_cols = cols["tof"]
    feats = {}
    feats["mod_present_tof"] = 1 if (len(tof_cols) > 0) else 0
    if not tof_cols:
        return feats

    agg = tof_frame_aggregates(seq_df, tof_cols)
    if agg is None:
        return feats

    # 欠損度合い
    feats["tof_valid_ratio_mean"] = float(np.mean(agg["valid_ratio"]))
    feats.update(summarize_series_features(agg, fs, prefix="tof_seq"))

    # 8×8空間（可能なら）
    grid_map = build_tof_grid_index(tof_cols)
    if grid_map is not None:
        # 各時刻で空間特徴→系列化
        A = seq_df[tof_cols].values
        series_spatial = {k: [] for k in ["cx","cy","mu20","mu02","mu11","ecc","lr_asym","ud_asym"]}
        for t in range(len(seq_df)):
            f = tof_spatial_features_per_timestep(A[t], grid_map)
            for k in series_spatial: series_spatial[k].append(f[k])
        series_spatial = {k: np.asarray(v) for k,v in series_spatial.items()}
        feats.update(summarize_series_features(series_spatial, fs, prefix="tof_spatial"))
    return feats

def extract_thm_features(seq_df: pd.DataFrame, fs: float) -> dict:
    present, cols = detect_modalities(seq_df)
    thm_cols = cols["thm"]
    feats = {}
    feats["mod_present_thm"] = 1 if (len(thm_cols) > 0) else 0
    if not thm_cols:
        return feats
    agg = thermal_frame_aggregates(seq_df, thm_cols)
    if agg is None:
        return feats
    feats["thm_valid_ratio_mean"] = float(np.mean(agg["valid_ratio"]))
    feats.update(summarize_series_features(agg, fs, prefix="thm_seq"))
    return feats

def extract_xmod_features_for_union(imu_series: dict, tof_agg: dict|None, thm_agg: dict|None) -> dict:
    # imu_series はあなたの extract_features 内で linear_acc_mag, angular_vel_mag を保持して渡す
    return xmod_features(
        {"linear_acc_mag": imu_series.get("linear_acc_mag"),
         "angular_vel_mag": imu_series.get("angular_vel_mag")},
        tof_series=tof_agg, thm_series=thm_agg
    )
```

---

## 8) 検証チェックリスト（簡易）

- [ ] ToF/THM が**ある/ない**両方のサンプルで `extract_features` が走る
- [ ] 返却 DataFrame に **NaN/Inf が存在しない**
- [ ] 学習時に保存した `feature_names` を、推論時 `result_df` が**完全にカバー**
- [ ] `mod_present_tof/thm` が正しく 0/1 に（列が無い or 全 -1 → 0）
- [ ] ToF の `valid_ratio_mean` が 0/1 で極端なときもエラーなく動作
- [ ] 速度：1 シーケンスあたりの前処理時間が許容内
