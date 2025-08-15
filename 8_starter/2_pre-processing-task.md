了解です。3 本のノートブック（`imu-only-baseline-lgbm-using-worldacc.ipynb` / `cmi-bfrb-imu-inference-v10-frequency-features.ipynb` / `cmi-lightgbm-tof-pro-v1.ipynb`）と、いまの **IMU-only LightGBM** コード（CV=0.72）を読み、**特徴量エンジニアリングに限定**して「どこをどう拡張すべきか」を**実装タスク**として設計しました。
以下は、**いまの実装との差分**→**設計方針**→\*\*具体タスク（受け入れ基準/落とし穴込み）\*\*の順で整理しています。

---

## 0) いまの実装と参照ノートのギャップ（要点）

- いまの IMU-only コード

  - **できている**: 四元数欠損処理、**ワールド座標加速度**（quat→R.from_quat→apply）、軸別/合成の**時間領域の基本統計**。
  - **足りていない**:

    - **周波数領域**（Welch/バンドパワー/支配周波数/重心/ロールオフ/エントロピー/ZCR）
    - **ジャーク**、**角速度（四元数差分）**、**オイラー角**
    - **変化点系**（簡易 CUSUM, セグメント持続長など）
    - \*\*マルチスケール（micro/short/medium）\*\*の移動統計
    - **姿勢不変**な指標（垂直 vs 水平エネルギ比、傾き角の分布など）
    - **fold 加重の確率平均**（現状も確率平均はあり。ただし重み付け/後処理は未実装）

- 参照ノートのエッセンス（IMU 中心）

  - `imu-only-...-worldacc`: **ワールド化**と**時間領域の統計**を充実させて汎化。
  - `imu-inference-v10-frequency-features`: **Welch/FFT 系**の**周波数特徴**・**ジャーク**・**ピーク/変化点**を活用。**fold 加重平均**や**BFRB ブースト**等の後処理もあり。
  - `tof-pro-v1`: ToF/THM 側はここではスコープ外だが、**品質/欠損率のようなロバスト指標**は IMU にも有効（例: 欠損連続長、外れ値率等）。

---

## 1) 設計方針（IMU-only のまま、最小改修で最大効果）

1. **前処理の堅牢化**

   - 四元数の**符号連続性補正**（隣接クォータニオンの内積が負なら符号反転）
   - **線形加速度**（重力除去）と **世界座標**の併用
   - 軸別**ロバスト・クリップ（winsorize）**と**欠損/スパイク補正**（ffill→bfill→0）

2. **時間領域の拡張**

   - **ジャーク**（一次差分/|jerk|の統計）
   - **角速度**（quat 差分 → 回転ベクトル/dt）+ **回転エネルギ**（ω² の平均ほか）
   - **オイラー角**（roll/pitch/yaw）の統計・遅延相関
   - **ピーク/ゼロ交差/自己相関/勾配ヒストグラム**など軽量特徴

3. **周波数領域の導入（20 Hz 前提 / 動的 nperseg）**

   - Welch-PSD（短系列は `nperseg=min(128, len//4)` で自動調整）
   - **バンドパワー**（0.3–3, 3–8, 8–12 Hz の絶対/相対/対数）
   - **スペクトル重心/85%ロールオフ/エントロピー/支配周波数/ZCR**

4. **姿勢不変化の工夫**

   - **垂直 vs 水平**（world Z と XY）の**エネルギ比**・分散比
   - **傾き角**（acos(az/|a|)）の分布統計
   - **左/右利きの簡易ミラー**（必要なら world_x を符号反転）※ワールド変換後に最小限で

5. **マルチスケール（Temporal Pyramid）**

   - **micro (0.5s=10)** / **short (1s=20)** / **medium (2s=40)** 窓での移動平均/分散/分位
   - **末尾強調**（後段 1–2 秒の統計）— 予測安定化に寄与

6. **学習/推論の一貫性**

   - **単一の FE モジュール**に機能集約（関数名/Config/feature_names を共通化）
   - fold ごとのスケーラは **LGBM では不要**だが、**特徴の定義/順序の固定**は必須
   - **fold 加重確率平均**（重み=各 fold のバリデスコア）

---

## 2) 具体タスク設計（実装粒度・受け入れ基準つき）

> ★ は優先度高。**M1→M2→M3** の 3 マイルストーンで進めると手戻りが少ないです。

### M1. 前処理と時間領域の強化（IMU のみ）

**T1【★】四元数の符号連続性補正を追加**

- **What**: 連続する `q_t` と `q_{t+1}` の内積 < 0 のとき `q_{t+1} = -q_{t+1}` に反転。
- **Where**: `handle_quaternion_missing_values` の直後、`compute_world_acceleration` の前。
- **Acceptance**: 反転後に `||q||≈1` を維持、`world_acc` の連続性（大ジャンプの減少）を確認。
- **Pitfall**: `ffill/bfill` 後の NaN 列に注意、正規化を最後にもう一度。

**T2【★】線形加速度（重力除去）を導入**

- **What**: `world_acc - [0,0,9.81]`（or 低域除去 HPF）。
- **Output**: `linear_acc_{x,y,z}`, `linear_acc_mag` の統計。
- **Acceptance**: 静止区間で `linear_acc_mag` の平均が \~0 に近づく。
- **Pitfall**: サンプルレート 20 Hz 前提、HPF なら `butter(4, 2.0, highpass, fs=20)` 等で安定化。

**T3【★】ジャーク特徴**

- **What**: `jerk = diff(acc or world_acc or linear_acc)/dt`。
- **Stats**: mean(|jerk|), std, max, p90, L2。
- **Acceptance**: 0 長系列/NaN を 0 補完、1 サンプル系列で安全に 0 を返す。

**T4【★】角速度（四元数差分 → 回転ベクトル）**

- **What**: `ω_t = (R_{t+1} * R_t^{-1}).as_rotvec()/dt`、`ω_mag` の統計。
- **Plus**: **回転エネルギ** `mean(ω^2)`、ピーク数/間隔。
- **Acceptance**: 長さ合わせ（最後 1 サンプルのパディング）済、NaN 安全。

**T5** オイラー角（roll/pitch/yaw）統計

- **What**: `R.from_quat(...).as_euler("xyz")`。
- **Stats**: mean/std/min/max/q10/q90/seg(3) など。
- **Pitfall**: ラップアラウンド（±π）での不連続に注意（差分は使わないか、unwrap）。

**T6** 軸間/モード内相関

- **What**: `corr(world_acc_x, world_acc_y)` 等、`acc`/`world`/`linear`/`ω`で数個。
- **Limit**: 多すぎる相関は冗長なので 3–6 個に制限。

**T7** ピーク/自己相関/勾配ヒストグラム

- **What**: `find_peaks`（高さ=0.5σ）で `n_peaks/高さ/間隔`、`autocorr_lag{1,2,4,8}`、`grad_hist(10bin)`。
- **Acceptance**: len<2 は 0 埋め、例外なく動く。

---

### M2. 周波数領域 + 姿勢不変化 + マルチスケール

**T8【★】Welch-PSD + バンド/頻域特徴**

- **What**: `welch(data, fs=20, nperseg=min(128, len//4, len), noverlap=nperseg//2)`
- **Bands**: (0.3–3), (3–8), (8–12) の **abs/rel/log** パワー、**パワー比**（LF/HF）
- **Global**: **総パワー**、**重心**、**85%ロールオフ**、**エントロピー**、**支配周波数**、**ZCR**
- **Targets**: `acc_{x,y,z}`, `acc_mag`, `world_acc_{x,y,z}`, `world_acc_mag`, `linear_acc_mag`, `ω_mag`
- **Acceptance**: len<32 なら安全に 0 を返す実装、NaN/inf は 0 に置換。

**T9【★】垂直/水平エネルギ比 + 傾き角**

- **What**: `var(world_z)/var(sqrt(x^2+y^2))`、`mean/std(acos(world_z/|world|))`
- **Motivation**: 姿勢に左右されにくい動き量の尺度。
- **Acceptance**: 分母 0 なら小正則化（1e-8）。

**T10** マルチスケール（Temporal Pyramid）

- **What**: 窓 `micro=10, short=20, medium=40` サンプルで移動平均/分散/分位（p10/p90）
- **Where**: `world_acc_mag` と `linear_acc_mag` を主対象に、3–6 指標/窓。
- **Acceptance**: len\<window は `min_periods=1` で安全に。

**T11** 末尾強調（Tail window）

- **What**: 最後 `~1–2 秒` の統計（mean/std/max/min）。
- **Motivation**: 区間末付近の動作はクラス判別に寄与しやすい。
- **Acceptance**: len が短い場合は全体窓で代替。

---

### M3. 低リスクの後処理（学習/推論の一貫化）

**T12【★】特徴抽出器の単一点管理（Training/Inference 共有）**

- **What**: `extract_features()` を **共通モジュール**に切り出す（例 `imu_features.py`）。
- **Deliverables**: `Config(SAMPLING_RATE=20, FREQ_BANDS, NPERSEG_MAX=128)` を一元管理。
- **Acceptance**: 学習/推論で **同じ feature_names** をロードして使える状態（欠損列は 0 埋め）。

**T13【★】fold 加重の確率平均**

- **What**: `weights = score_fold / sum(score)` で fold 確率を加重平均。
- **Acceptance**: 学習後に `cv_scores` から自動的に重みを算出・保存・推論時に読込む。
- **Note**: 多数決は非推奨（確率情報を捨てるため）。

**T14** 簡易信頼度に基づく微調整（任意）

- **What**: `max_proba < τ` のとき **BFRB（0–7）を優遇**する small nudge（+α）
- **Acceptance**: しきい値と nudge 量は Config で調整、過学習しないようアブレーションで検証。

---

## 3) 実装インタフェース（関数仕様の提案）

- `fix_quaternion_sign(rot: np.ndarray) -> np.ndarray`
  連続内積<0 のとき後者を反転。**T1**

- `compute_linear_acceleration(acc: np.ndarray, rot: np.ndarray, method="subtract") -> np.ndarray`
  `world_acc - g` or HPF。**T2**

- `compute_angular_velocity(rot: np.ndarray, dt=1/20) -> np.ndarray`
  `as_rotvec()/dt`、末尾パディング。**T4**

- `extract_hjorth_parameters(x, prefix)` / `extract_peak_features(x, prefix)` / `extract_line_length(x, prefix)` / `extract_autocorrelation(x, prefix, lags=[1,2,4,8])` / `extract_gradient_histogram(x, prefix, n_bins=10)` / `extract_jerk_features(x, prefix, dt)`
  いずれも **NaN/短系列安全**。**T3/T6/T7**

- `extract_frequency_features(x, prefix, fs=20.0)`
  Welch + バンド/重心/ロールオフ/エントロピー/支配/ZCR。**T8**

- `extract_temporal_pyramid(x, prefix, windows={"micro":10,"short":20,"medium":40})`
  移動平均/分散/分位（p10/p90）。**T10**

- `extract_pose_invariant_features(world_acc, prefix)`
  垂直/水平エネルギ比、傾き角分布。**T9**

> これらは、あなたの「高度 FE v2 スクリプト」にある実装を**縮約/移植**できます（Welch・ジャーク・角速度・Hjorth・ピークなどはすでに品質良く実装済み）。

---

## 4) 変更箇所（ファイル/関数レベル）

- `imu_features.py`（新規）

  - 上記 **T1–T11** の機能を実装。`Config` と `FEATURE_NAMES` を持つ。

- 学習ノート（IMU-only LGBM）

  - 既存の `extract_features()` を **imu_features.py** を呼ぶ形に置換。
  - 学習後に `{"feature_names": [...], "fold_scores": [...]}` を `joblib` 保存（**T12/T13**）。

- 推論ノート

  - **同一の imu_features** を import。
  - `models` 読込み時に `fold_weights` を計算して **確率の加重平均**（**T13**）。
  - （任意）低信頼度時の微調整（**T14**）。

---

## 5) 受け入れ基準（E2E）

1. すべての系列で `extract_features()` が **例外なく** DataFrame（1 行 ×N 列）を返却（N は訓練時と一致）。
2. 学習/推論の **feature_names が完全一致**（欠損列は 0 埋め）。
3. **短系列（<32 サンプル）でも落ちない**（Welch 等は 0 を返す）。
4. `cv_scores` に基づく **fold 加重平均**で推論できる。
5. 変更を段階適用（M1→M2→M3）して**逐次アブレーション**（5-fold CV）を回せる。

---

## 6) アブレーション計画（推奨）

- **Step0**: 現状（0.72）を再現。
- **+M1**: T1–T7 を導入 → 期待：**+0.01〜0.03** 程度（目安）。
- **+M2**: T8–T11 を追加 → 期待：**+0.01〜0.03**。
- **+M3**: T12–T14（fold 加重/微調整）→ 期待：**安定化**（分散減）。

> ※数値は目安。実スコアは fold 依存なので、**CV の再現性**を優先。

---

## 7) タスクリスト（チェックリスト）

### M1（優先）

- [ ] **T1** `fix_quaternion_sign` を実装し `extract_features` 内で適用
- [ ] **T2** `compute_linear_acceleration` 実装 & 統計追加（axis/mag）
- [ ] **T3** `extract_jerk_features` を acc/world/linear に適用
- [ ] **T4** `compute_angular_velocity` 実装 & 統計（mag/energy/peaks）
- [ ] **T5** Euler 角の統計（roll/pitch/yaw）
- [ ] **T6** 軸間相関（3〜6 指標）
- [ ] **T7** ピーク/自己相関/勾配ヒストグラム（短系列安全）

### M2

- [ ] **T8** Welch-PSD とバンド/重心/ロールオフ/エントロピー/支配/ZCR
- [ ] **T9** 垂直/水平比と傾き角分布
- [ ] **T10** Temporal Pyramid（micro/short/medium）
- [ ] **T11** 末尾強調統計（1–2 秒）

### M3

- [ ] **T12** FE 共有モジュール化 + feature_names 固定 + Config 一元化
- [ ] **T13** fold **加重**確率平均（重み=fold CV スコア）
- [ ] **T14** 低信頼度時の微調整（任意、Config 駆動）

---

## 8) 実装の注意点（落とし穴）

- **リーク回避**: 特徴抽出は**各系列内完結**。学習時のスケーリングは原則不要（LGBM）。
- **短系列**: Welch/差分系は **len チェック**で 0 埋めの安全設計。
- **NaN/inf**: 最終 DataFrame 直前に **一括ゼロ埋め**。
- **時間計算**: `dt=1/20` を定数化、関数内で繰り返し定義しない。
- **利き手**: ワールド化後の**最小限のミラー**（必要に応じ `world_x*=-1`）は Config で ON/OFF。
- **機能爆発**: 相関やヒストは**数を絞る**。まずは重要ブロック（T3/T4/T8/T9/T10）。

---

このプランは、**参照ノートの強み（Welch/周波数/ジャーク/角速度/確率加重）**を、あなたの IMU-only LightGBM に**最小のリファクタ**で取り込む構成です。
まず **M1** を入れて CV を回し、効果を確認しつつ **M2**→**M3** と段階導入するのがおすすめです。必要なら、このタスクに沿った **関数スケルトン（最小コード）** もすぐお出しします。
