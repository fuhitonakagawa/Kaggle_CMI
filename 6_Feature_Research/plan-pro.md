以下は、**CMI - Detect Behavior with Sensor Data** における \*\*データ前処理／特徴量作成の詳細計画（実装前仕様）\*\*です。
目的は「**BFRB（8 クラス）内の識別力**を上げるための“堅牢で再現性のある”特徴量パイプライン」を作り込み、CV での **Macro F1（BFRB 内）** を押し上げることです。公開ノート／議論の方針（例：**World 座標化・線形加速度化・周波数特徴・TOF 空間構造の活用**）を取り入れています。([Kaggle][1])

---

## 1) データ概観と前提

- **センサ構成とタスク**
  Helios リストバンドからの **IMU（加速度＋四元数）**, **ToF（5 基の 8×8 距離グリッド）**, **Thermal（5ch）** を用い、**BFRB 8**＋**非 BFRB 10**の計 18 クラスを識別。学習は主に `behavior == "Performs gesture"` のシーケンスを対象。サンプリングは 20 Hz。([Kaggle][1])
- **ハード前提（ToF の 8×8）**
  一般的な 8×8 マルチゾーン ToF（例：ST VL53L5CX 系）想定の空間解像度・60 Hz までの更新能力に近い特性。**各フレーム＝距離画像(8×8)×5 面**の空間パターンを持つ。([STMicroelectronics][2])
- **実務的示唆（公開ノートより）**
  IMU では **World Acc（世界座標加速度）** や **線形加速度（重力除去）** が底上げに有効、周波数特徴の追加も常套。([Kaggle][3])
- **利き手の影響**
  センサ配置が手首周りで左右対称ではないため、**handedness** によって TOF/IMU の見え方が変わりうる（左右で最初に視界に入るセンサーが異なる等）。([Kaggle][4])

---

## 2) 前処理パイプライン（順序厳守）

### 2.1 データ読み込みとフィルタ

1. `train.csv` を読み込み → `behavior == "Performs gesture"` のみ抽出。
2. 列の基本整合性チェック：

   - IMU: `acc_{x,y,z}`, `rot_{w,x,y,z}`
   - ToF: `tof_{1..5}_v{0..63}`（-1 は欠測扱い）
   - Thermal: `thm_{1..5}`
   - メタ: `sequence_id`, `subject`, `gesture`, `handedness`, ほか。([Kaggle][1])

### 2.2 時間軸の整備（20 Hz 基準）

1. シーケンス内の重複タイムスタンプ除去・整列。
2. サンプリング間隔の乱れを**線形補間**で補正（端点は最近傍値保持）。
3. ToF/Thermal/IMU の**行同期**（行数差が出た場合は短い系列に合わせて切り落とし or 補間）。

   - **注意**：補間は\*\*回転（四元数）\*\*に直接適用しない（→ 2.3 で処理）。

### 2.3 IMU：四元数の健全化と重力除去・座標変換

1. **四元数の NaN 補完 → 正規化**：`ffill→bfill→(1,0,0,0) で補完` → L2 正規化。
2. **世界座標化**：四元数 $q_t$ を用いて、ローカル加速度 $a_t$ を**ワールド系**に回転。
3. **重力除去（線形加速度）**：

   - 方法 A：ワールド系で $g=[0,0,9.81]$ を逆回ししてセンサ座標へ投影 → 差し引き。
   - 方法 B：ローパス（例：2–3 Hz）で重力成分を推定し、高域側を**線形加速度**とみなす。

4. **角速度の近似**：四元数差分からの回転ベクトル $\omega_t \approx \log(q_{t}^{-1} q_{t+1}) / \Delta t$。
   ※ WorldAcc／Linear Acc を使う手法は IMU ベースのベースラインで頻用。([Kaggle][3])

### 2.4 ToF：欠測・外れ値・空間圧縮

1. **欠測処理**：`-1` → `NaN`。各フレームにつき

   - 有効画素率（valid_ratio）、最小・最大・中央値、分位（p10/p50/p90）、**最小距離（min_dist）**。

2. **外れ値処理**：各フレームの有効画素を**分位クリップ（p1–p99）**。
3. **空間圧縮（学習時に fit→ 推論時 transform）**：

   - **PCA（各 ToF 64→8 成分）**：全フレームの(8×8)をベクトル化 → 標準化 →PCA。
   - 代替：**2D モーメント（重心・2 次モーメント・偏心率）**や**Laplacian/勾配和**。

4. **左右差対策**：`handedness` で**センサー番号の左右反転**を試す（マッピング策定が難しい場合は、「各フレームで**センサー間 min/mean**のグローバル特徴＋各センサー個別特徴」を両方持つ）。

   - 利き手で見え方が変わる点は公開 EDA でも指摘あり。([Kaggle][4])

### 2.5 Thermal：平滑化と差分

1. **平滑化**：移動平均（例：5〜9 サンプル）。
2. **一次／二次差分**：上昇・下降のスロープを抽出。
3. **温度勾配の指標**：max–min、分位幅（p90–p10）、**温度イベント率**（|差分|>閾値）。

### 2.6 正規化の粒度（リーク防止）

- **Sequence 内 Z-score**（各チャンネル）→ **Fold 内 Train で fit した RobustScaler/StandardScaler** を順次適用。

  - 目的：被験者間差を平準化しつつ、CV リークを防止。

- **PCA・分位基準値**も**fold 内 train で fit**し、推論は transform のみ。

---

## 3) 特徴量ファミリ（作成仕様）

> ここからは「**シーケンス全体**」に対して集約して作る**sequence-level 特徴**を中心に定義します（Kaggle 評価サーバは 1 シーケンス →1 予測を想定）。IMU の **WorldAcc／Linear Acc**、周波数特性、ToF の**空間 × 時間**、Thermal の**遅い傾き**を取りこぼさないのが肝です。([Kaggle][1])

### 3.1 IMU（加速度・線形加速度・角速度）

**入力系列**：

- `acc_world_{x,y,z}`, `acc_world_mag`
- `linacc_{x,y,z}`, `linacc_mag`
- 角速度 `omega_{x,y,z}`, `omega_mag`（四元数差分から）
- オリエンテーション `roll/pitch/yaw`（四元数 → オイラー）

**時間領域（各系列）**

- 基本統計：mean/std/min/max/median/p10/p90/IQR/range/CV
- 形状：skew/kurtosis、**line length**（|差分|和）、**Hjorth**（activity, mobility, complexity）
- 変化率：一階差分（jerk/角速度差分）の mean/std/max/分位
- 区間統計：\*\*3 分割（early/mid/late）\*\*の mean/std/max、変化量（last–first）
- ピーク：`find_peaks` によるピーク数、平均間隔、最大振幅

**周波数領域（各系列）**

- **Welch PSD** → バンドパワー：**0.3–3 Hz**, **3–8 Hz**, **8–12 Hz**
- スペクトル重心・**roll-off(85%)**・**エントロピー**・最大ピーク周波数＆パワー
- **ZCR**（符号反転率、正規化）

> WorldAcc／Linear Acc の活用・周波数特徴の有効性は公開ノートで繰り返し言及。([Kaggle][3])

**相関／結合**

- 軸間相関（x–y, y–z, z–x; world & linear）
- `linacc_mag` と `omega_mag` の相関（「速い動き＋ひねり」）
- roll/pitch/yaw のレンジ、標準偏差、方向転換回数

### 3.2 ToF（5 基 ×8×8） ※各センサーごと＋センサー横断

**フレーム内（空間）**

- 統計：valid_ratio, mean/std/min/max, p10/p50/p90, **min_dist**
- 形状：**重心 (cx, cy)**、二次モーメント（xx, yy, xy）、**偏心率**、主成分角度
- テクスチャ：Sobel/Laplacian 勾配和、**エッジ率**、**局所コントラスト**
- クラスタ：閾値（近距離）で 2 値化 → 最大連結成分の **面積/周囲長/円形度**
- PCA 係数（64→8）と再構成誤差（空間圧縮後の残差）

**時間方向（時系列）**

- `min_dist`／`mean` の速度（一次差分）・加速度（二次差分）
- 低周波変動：移動平均の変化量、**区間統計（3 分割）**
- **到達イベント率**：近距離閾値を下回る割合、下回りの連続長（触れにいく動き）
- **センサー跨ぎの同期**：5 基 `min_dist` の同時低下率・時差（face/neck への接近の手掛かり）

**センサー横断**

- その時点での **min across sensors**（「どれかが近い」）の時系列 → 統計／頻度
- **handedness** でセンサー番号を左右反転した派生系列（A/B 両方作る）

> ToF は**8×8 の“距離画像”**であり、単なる mean/std のみでは空間パターンが失われるため、**重心・モーメント・PCA**など**空間形状**を残す。8×8 の前提は一般的マルチゾーン ToF 資料にも沿う。([STMicroelectronics][2])

### 3.3 Thermal（5ch）

- 統計：mean/std/min/max/p10/p90/IQR
- 傾き：一次／二次差分の mean/std/max、**トレンド相関（時刻 vs 値）**
- 変化イベント率：|一階差分|>閾値 の割合
- ToF との結合：`min_dist` 低下時の Thermal の変化量（接近 → 温度上昇の連動）

### 3.4 クロスモーダル特徴（IMU×ToF×Thermal）

- **ミクロ同期**：`linacc_mag` のピーク近傍（±0.5 s）で `min_dist` がどれくらい下がるか（平均/最小）
- **整合度**：IMU の進行方向（world 座標の符号）と、どの ToF 面の `min_dist` が先に下がるかの一致率
- **トリプレット**：(`min_dist` 低下 → Thermal 上昇 → `linacc` ピーク) の順序一致度

---

## 4) 正規化・安定化・品質指標

- **二段正規化**：

  1. **Sequence 内 Z-score**（欠測除外）
  2. **Fold 内 Train で fit した Scaler**（Robust/Standard）

- **外れ値クリップ**：各系列で **p1–p99** へクリップ（学習 fold 内で閾値推定）。
- **品質メトリクス**：

  - 欠測率（IMU/ToF/Thermal 別）
  - ToF の valid_ratio の平均と分散、valid 連続 0 区間の最大長
  - シーケンス長、**有効長/パディング比**
  - これらを**説明変数としても保持**（“悪いデータ”の影響を緩和）

---

## 5) 時系列の切り方と集約戦略

- **パディング長**：長さ分布の **95–98 パーセンタイル** を CV ごとに採用（無理に切り捨てない）。
- **セグメント集約**：**early/mid/late（3 分割）**の統計を**全体統計と併置**（“立ち上がり／離脱”を捉える）。
- **多解像度集約**：0.5 s／1 s／2 s の**窓平均を畳み**、その最終統計（mean/std/p10/p90）も追加（“Temporal Pyramid”の簡易版）。

---

## 6) CV・特徴評価（実装時の検証設計）

- **分割**：`StratifiedGroupKFold(n_splits=5, group=subject)`（層＝ BFRB/非 BFRB ＋ BFRB 内頻度）。
- **評価**：Fold ごとに

  - **Binary F1**（BFRB vs 非 BFRB）
  - **Macro F1（BFRB 8 内）**
  - \*_Combined = 0.5_(Binary F1 + Macro F1)\*\*

- **スクリーニング**：

  - LightGBM/XGBoost の **OOF 重要度**（Gain）
  - **Permutation Importance** でリークや冗長を除去
  - **相関し過ぎた特徴**の枝刈り（|ρ|>0.95）

- **アブレーション順**：

  1. IMU: Raw → WorldAcc → LinearAcc → 周波数追加
  2. ToF: mean/std のみ → +min/valid_ratio → +PCA/重心/モーメント
  3. Thermal: 統計のみ → +傾き/差分 → +ToF 連動
  4. クロスモーダル（ピーク同期 等）

> ここまでの CV・評価方針はコンペの評価形に準拠。([Kaggle][1])

---

## 7) 推論時の再現性・実装勘所

- **同一処理の厳密再現**：PCA/Scaler/分位閾値は**学習 fold の fit 資産**をロードして**transform のみ**。
- **計算量制約**：Kaggle の推論環境で**1 シーケンスあたり数十 ms〜数百 ms**目安。

  - ToF の**空間畳み込みは避け**、**事前 fit 済み PCA**＋軽量モーメントで済ます。
  - Welch は\*\*窓長を共通（例：128, overlap 50%）\*\*に固定。

- **テストの健全性**：

  - 入力 NaN 大量時の**フォールバック（既定値）**
  - Sequence が極端に短い場合（＜ 10 サンプル）は**簡易統計のみ**で埋める

---

## 8) 具体的な“特徴量リスト”（抜粋・拡張可）

> **1 シーケンスあたり**で計算・格納（数百〜千次元程度を想定）

**IMU（各系列：acc_world_x/y/z, acc_world_mag, linacc_x/y/z, linacc_mag, ω_x/y/z, ω_mag, roll/pitch/yaw）**

- 基本統計（10 個前後）× 系列数
- 差分（jerk/角速度差分）の統計（5 個前後）× 系列数
- Welch バンドパワー（3 帯域）＋最大ピーク周波数/電力＋重心/エントロピー/roll-off（6〜8 個）× 系列数
- 区間統計（3 分割 ×2 個）× 系列数
- 軸間相関（3 組）×（world/linear など対象群）

**ToF（各センサー × フレーム → 集約）**

- フレーム統計（valid_ratio, min/mean/std/分位、min_dist）→**時間集約（mean/std/p10/p90/max/イベント率）**
- 空間モーメント（重心・2 次モーメント・偏心）→**時間集約**
- PCA 係数（8 次元）→**時間集約（mean/std/max/分位）**
- 近接イベント（閾値下回り率、最長連続長、回数）
- 5 センサー横断：時点ごとの **min across sensors** シリーズ → 同様に時間集約

**Thermal（5ch）**

- 基本統計＋差分統計＋トレンド（相関係数）
- ToF の近接イベント窓内での温度上昇量（平均/最大）

**クロスモーダル**

- `linacc_mag` ピーク近傍の `min_dist` 低下量（mean/min）
- `omega_mag` ピーク時の ToF 有効画素率の変化
- roll/pitch/yaw の変化方向と、どの ToF 面の `min_dist` が先落ちするかの一致率

---

## 9) 品質管理・ログ

- **特異シーケンスの自動マーキング**：

  - 欠測率>30%、ToF 全センサー valid_ratio<0.2 が長く続く 等

- **分布監視**：学習 fold 間で主特徴（例：`linacc_mag` mean, ToF `min_dist` p10）の分布差を可視化
- **再現性**：乱数 SEED 固定、バージョン付け（PCA/Scaler の `fit_version`）

---

## 10) 参考（公開情報の方向性）

- **データと目的**：CMI 公式ページ（Helios、BFRB/非 BFRB、センサ構成の概説）。([Kaggle][1])
- **IMU 系の強化**：World Acc / Linear Acc を使う IMU ベースライン、BiLSTM+GRU の線形加速度活用。([Kaggle][3])
- **周波数特徴**：IMU に周波数ドメイン特徴を足した推論ノート。([Kaggle][5])
- **IMU+ToF/THM の併用**：2 ブランチ型 DL での取り組み（構成例）。([Kaggle][6])
- **ToF の 8×8 背景**：一般的 ToF の 8×8 仕様（VL53L5CX 等）。([STMicroelectronics][2])
- **利き手の影響**：EDA での指摘（センサー配置 ×handedness）。([Kaggle][4])

---

### 実装時の ToDo（チェックリスト）

- [ ] **CV 基盤**：`StratifiedGroupKFold(subject)` の共通化（層は BFRB/非 BFRB ＋ BFRB 内分布）
- [ ] **四元数系**：NaN 補完 → 正規化 →World 座標 → 重力除去（A/B 切替可能）
- [ ] **角速度**：四元数差分からの $\omega$ 導出関数
- [ ] **ToF**：-1→NaN、分位クリップ、**PCA(64→8)** の fit/transform 資産化、重心・モーメント
- [ ] **Thermal**：平滑化・差分・傾き
- [ ] **二段正規化**：sequence 内 Z-score → fold-scaler（fit/transform 分離）
- [ ] **特徴生成**：IMU 時間/周波数、ToF 空間/時系列、Thermal、クロスモーダル
- [ ] **品質特徴**：valid 率・欠測率・連続欠測長・長さなど
- [ ] **速度最適化**：Polars/NumPy ベースの**ベクトル化**、Welch 窓固定、PCA バッチ変換
- [ ] **記録**：fit 済み資産（Scaler/PCA/分位）を pickle 化、バージョン管理

---

必要であれば、この計画をそのまま**関数スケルトン＋型**まで展開します（`extract_features(sequence_df, demo_df, assets)->feature_row` のかたち）。まずはこの仕様で不足がないかご確認ください。

[1]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/data?utm_source=chatgpt.com "Detect Behavior with Sensor Data - CMI"
[2]: https://www.st.com/en/imaging-and-photonics-solutions/vl53l5cx.html?utm_source=chatgpt.com "VL53L5CX - Time-of-Flight (ToF) 8x8 multizone ranging ..."
[3]: https://www.kaggle.com/code/ryenhails/imu-only-baseline-lgbm-using-worldacc?utm_source=chatgpt.com "[IMU-only baseline] LGBM using WorldAcc"
[4]: https://www.kaggle.com/code/tarundirector/sensor-pulse-viz-eda-for-bfrb-detection?utm_source=chatgpt.com "📉Sensor Pulse🧠| Viz & EDA for BFRB Detection"
[5]: https://www.kaggle.com/code/satoshissss/cmi-bfrb-imu-inference-v10-frequency-features/input?scriptVersionId=244926159&utm_source=chatgpt.com "CMI-BFRB IMU Inference v10 - Frequency Features"
[6]: https://www.kaggle.com/code/hideyukizushi/cmi25-imu-thm-tof-tf-bilstm-gru-attention-lb-75?utm_source=chatgpt.com "CMI25 | IMU+THM/TOF |TF BiLSTM+GRU+Attention|LB.75"
