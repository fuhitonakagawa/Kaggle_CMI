# CMI – Detect Behavior with Sensor Data

## 特徴量エンジニアリングの詳細計画（前処理を“完璧”にするための版）

> 目的：**指標（Binary F1 と BFRB 内 Macro F1 の等重み）に直結**する、堅牢な前処理・特徴量作成パイプラインを、**IMU-only と Full（IMU+TOF/THM）双方で破綻なく**動く形で定義します。公開ディスカッション/ノートの示唆（世界座標化、TOF 8×8、IMU-only が公開テストに多い等）を反映済みです。 ([Kaggle][1])

---

## 0. 全体方針（評価仕様と公開知見の取り込み）

* **評価最適化**：BFRB 検出（2値）と BFRB 内 8 クラス識別の両立が必要。**末尾窓の重視**（ラベルは動作区間の後半で安定）と**BFRB 側の分離能向上**に効く特徴を優先。 ([Kaggle][2])
* **モダリティ差**：**テストの約半分が IMU-only**のため、**TOF/THM が無い場合にも安定**に働く特徴セット（IMUから取れる不変量・周波数特性）を核にする。TOF/THM は**あれば加点**する設計に。 ([Kaggle][3])
* **幾何と物理の素直な活用**：**クォータニオン由来の世界座標化/重力除去**は公開議論の王道。これを**時間・周波数・統計の3面**から多解像度に集約する。 ([Kaggle][4])
* **TOF 8×8 グリッド & THM 温度**：**5基×(8×8 距離配列)**のTOFと**5素子の熱**を\*\*“顔近接/向き”の指標**に落とす。左/右利きでの**ミラー配置\*\*考慮を前処理で吸収。 ([Kaggle][5])
* **推論 I/F**：`predict(sequence, demographics)->gesture(str)` の**評価サーバ仕様**に合わせ、**1シーケンス内で全処理を完結**できるよう関数粒度で設計。 ([Kaggle][6])

---

## 1. データ読み込みと基本クレンジング

**入力**：`train.csv`（連続系列）、`train_demographics.csv`、各列（例：`acc_*`, `rot_*`, `tof_[1-5]_v[0-63]`, `thm_[1-5]`, `gesture`, `behavior`, `subject`, `sequence_id` …）

1. **時系列整列**

   * 同一 `sequence_id` 内をタイムスタンプ昇順でソート。重複行はユニーク化。
   * 異常時刻（逆行/重複）は近傍線形補間か除外（割合閾値 0.5%）。

2. **欠損値・番兵値処理**

   * **IMU**：`acc_*` NaN は 0 補、`rot_*` は**先行→後行補間＋単位四元数フォールバック**（`w=1,x=y=z=0`）。
   * **TOF**：`-1` は未計測として NaN 扱い、**後続の集約時に無視**。
   * **THM**：NaN は系列内中央値で補完（外れ温度の突発を抑制）。

3. **フェーズ抽出（学習時）**

   * `behavior == "Performs gesture"` を基本集合（開始/終了遷移のノイズを避ける）。推論は全文脈から後述の**末尾窓**を使う。

4. **サンプリング整合**

   * 公称 **20 Hz** 前提を確認。微小ズレはリサンプリング（線形補間）で整数ステップ化。 ([Kaggle][1])

---

## 2. 姿勢処理（クォータニオン）と IMU 基本派生量

> ここが精度差の源泉。**世界座標化＋重力除去**を丁寧に。 ([Kaggle][4])

1. **四元数の正規化**

   * 各時刻で `q /= ||q||`。連続欠損が長い区間は**SLERP**による滑らかな補間（短区間は線形で可）。

2. **世界座標化 / 重力除去**

   * **重力ベクトル `g_world=(0,0,9.81)`** をセンサ座標へ回転し、`acc_linear = acc_raw - R(q)^(-1) g_world`。
   * 併せて `acc_world = R(q) acc_raw` を計算（**垂直/水平**分解が可能）。

3. **角運動**

   * 隣接四元数から `ΔR = R_t^(-1) R_{t+1}` → **回転ベクトル/角速度** `ω = rotvec(ΔR)/Δt`（3軸＋ノルム）。

4. **フィルタリング**

   * IMU 系列は **0.2–8 Hz バンドパス**（BFRB 周期域の仮定）＋移動平均（w=3–5）。※カットオフは後でCV掃引。

5. **正規化**

   * **系列内 z-score**（ドリフト耐性）＋ **データセット大域 z-score**（外れ被験者の影響緩和）を**両用**。

---

## 3. ウィンドウ設計（学習/推論共通）

* **多解像度**：

  * **S**：1.0–1.5 s（20–30 step）
  * **M**：3–4 s（60–80 step）
  * **L**：10–12.8 s（200–256 step）
* **学習**：オーバーラップ窓（例：stride=1/4）で**多数サンプル化**。
* **推論**：**末尾中心**で S/M/L を**TTA 平均**（“末尾のみでまず試せ”の公開示唆を一般化）。 ([Kaggle][2])

---

## 4. IMU 特徴量（“不変量＋時間/周波数/統計”の三本柱）

### 4.1 時間領域（各窓×各チャネル）

* 基本統計：mean, std, median, min, max, range, IQR, **RMS**, **CV**, MAD
* 形状：**skew**, **kurtosis**, **Hjorth（activity/mobility/complexity）**
* ダイナミクス：

  * **jerk**（一次差分）統計（mean/std/max/`L2`）
  * **零交差率**、**peak 数/間隔/高さ**、**勾配ヒストグラム**
* ベクトル不変量：

  * `|acc_linear|`, `|acc_world|`, `|ω|`（ノルム）
  * **水平/垂直成分**（`acc_world` を z と xy に分け、各統計）
* 相関：

  * **軸間相関**（3×3 共分散の固有値・条件数）
  * **自己相関**（ラグ 1, 2, 4, 8）

### 4.2 周波数領域（Welch/FFT）

* **Welch PSD**：窓内でハミング、セグメント 4、50% オーバーラップ
* 帯域パワー＆比：

  * `P(0.2–1)`, `P(1–3)`, `P(3–8)`、**比率** `P(1–3)/P(0.2–1)` 等
* スペクトル要約：**dominant freq / power**, **centroid**, **rolloff(85%)**, **entropy**, **flux**
* 調和：1st/2nd/3rd の**ピーク比**（周期動作の同定）

> 世界座標化/重力除去と組み合わせることで、**姿勢に依らない周期性**を拾いやすくなります（公開議論ベース）。 ([Kaggle][4])

---

## 5. TOF（8×8×5）特徴量

> **距離の2次元場**を\*\*“顔近接・向き・遮蔽”**の要約へ落とす。左/右の**ミラー配置\*\*は前処理で統一。 ([Kaggle][5])

1. **ミラー正規化**

   * `handedness` が右のとき、**各 TOF の 8×8 を左右反転**（列 `v0..v63` を 8×8 に reshape→flip→flatten 戻し）。

2. **無効値処理**

   * `-1`→NaN。**分位点ベース**でしきい値を決め、**絶対距離に依らない**近接度を使う（装着個体差対策）。

3. **空間要約（各センサ）**

   * **全体統計**：mean/median/std/min/max/IQR/欠損率
   * **領域別**：中心3×3、内環、外環、左右/上下半分、4象限の**平均/面積比**
   * **近接度指標**：

     * `near_frac(q)`：**距離の下位 q 分位**未満のセル比率（q=10, 20）
     * **COM（重心）**：`1/d` を重みとした**重心 (x̄,ȳ)**、**分散/偏心率**
     * **アナイソトロピー**：主成分の**固有値比**（方向性）

4. **時間要約（窓集約）**

   * 近接度/COM/各領域均値の**Δ（一次差分）統計**、**ピーク頻度**、**持続時間**（連続近接の長さ）
   * **近接イベント率**（近い状態のフレーム比）

5. **センサ間整合**

   * 5基の**相互差**（左右/上下の非対称）や**最大/最小センサの ID**、**max-min**、**max/mean**

---

## 6. THM（5素子）特徴量

* **系列内ベースライン差**：`thm_i - median(thm_i)`（個体差を除去）
* 要約：mean/std/min/max/IQR/欠損率、**上位分位–下位分位の spread**
* 勾配：一次差分統計（温度変化の速度）
* センサ間差：`max(thm) - min(thm)`、左右/上下の**不均衡**（装着向きに応じた対を定義）
* **TOF 近接と共起**：近接イベント中の `thm` 上昇量（顔に近い＋温度上昇）

> 競技ページの可視化ツール言及：**TOF 8×8 と Thermopile 温度のライブ可視化**が示されている（存在確認の根拠）。 ([Kaggle][5])

---

## 7. 末尾重視の窓選択と集約（推論互換）

* **学習**：S/M/L 全窓を使うが、**末尾 1/3 に重み 2×**（BFRB の確度が高い）。
* **推論**：各 S/M/L で**末尾中心の複数窓**を切り出し、**確率または特徴の平均**（TTA）。
* **アブレーション**：**末尾のみ** vs **全体**で OOF を比較（公開示唆の再現確認）。 ([Kaggle][2])

---

## 8. デモグラ・メタ特徴

* `age, sex, handedness, height_cm, shoulder_to_wrist_cm, elbow_to_wrist_cm, adult_child` をそのまま／離散化（bin）
* **モダリティフラグ**：`has_tof`, `has_thm`（**IMU-only との整合**に必須）
* **利き手**：TOF/THM のミラー正規化後も**元の handedness**を特徴として残す（残差の寄与を拾う）

> IMU-only が多いことを踏まえ、**フラグでの分岐**と**ミラー正規化**が上位の共通戦略。 ([Kaggle][3])

---

## 9. 特徴の健全性チェック & 漏洩防止

1. **主観リーク回避**：**StratifiedGroupKFold(group=subject)** で一貫。OOF を指標最適化の土台に。 ([Kaggle][7])
2. **分布監視**：学習 vs OOF vs テスト（ローカルゲートウェイ）で、**各特徴の平均/分散/欠損率**をドリフト監視。
3. **定数/低分散削除 & 相関カット**：相関>0.98 を片方削除、VIF>20 を削減。
4. **単調性/物理整合**：

   * 近接度↑で**距離平均↓**、COM がセンサ配置と整合するか
   * 世界座標 `acc_world_z` の平均が\*\*±9.81 を跨がない\*\*こと
5. **スケールの独立性**：分位/比率/正規化を多用（被験者差・装着差に頑健）

---

## 10. 実装仕様（後で `unified_cmi.py` に統合）

### 10.1 前処理関数（疑似インタフェース）

* `clean_quat(df) -> df`：正規化＋補間（slerp）
* `acc_linear_world(df, g=(0,0,9.81)) -> df`：重力除去 & 世界座標化
* `imu_filters(df, band=(0.2,8.0), fs=20) -> df`：BPF＋移動平均
* `make_windows(arr, sizes=[30,80,256], strides=[8,16,64]) -> List[Win]`
* `feat_imu_time(win)`, `feat_imu_freq(win, fs=20)`
* `reshape_tof(row) -> (5,8,8)`／`mirror_tof(grid, handedness)`
* `feat_tof_spatial(grid)`, `feat_tof_temporal(wseq)`
* `feat_thm_time(wseq)`, `feat_thm_cross(wseq)`
* `aggregate_SML(features_S, features_M, features_L, mode="mean|weighted")`

### 10.2 学習時／推論時の差分

* **学習**：全窓抽出＋`behavior=="Performs gesture"`優先、**末尾窓に重み**
* **推論**：**sequence→TTA 窓抽出→特徴作成→モデル**（評価サーバ互換、1コール完結） ([Kaggle][6])

---

## 11. 優先実装順（短期ロードマップ）

1. **IMU 基本核**：世界座標化＋重力除去 → **不変量/周波数**（S/M/L）
2. **TOF 空間要約**：ミラー→近接度/COM/領域指標→時間差分
3. **THM**：ベースライン差と変化速度＋センサ間差
4. **健全性バッテリー**：定数/相関カット、OOF レポート自動出力
5. **末尾TTA**：推論 `predict()` に S/M/L 末尾窓平均を実装
6. **CV安定化**：SGKF の固定と OOF ベースのしきい値・温度最適化（※実装は次段）

---

## 12. 参考（根拠となる公開情報）

* **競技ページ/データ**（センサ構成と評価の前提）：IMU/TOF/THM の統合課題。 ([Kaggle][1])
* **世界座標化/重力除去の議論・可視化**：四元数→世界座標、加速度の扱い。 ([Kaggle][4])
* **TOF は 8×8×5、可視化ツール**：8×8 グリッドとサーモの可視化。 ([Kaggle][5])
* **左/右でミラーの話題**：ハード配置の鏡像を考慮。 ([Kaggle][8])
* **IMU-only がテストで多い旨の言及**：IMU-only を意識した戦略が必要。 ([Kaggle][3])
* **末尾窓をまず試す**という連続データ特有のテクニック。 ([Kaggle][2])
* **提出 I/F（`predict`→`submission.parquet`）**。 ([Kaggle][6])

---

### 補足

* 数式レベルの詳細（Hjorth・PSD・COM 等）は実装段で**ユニットテスト**化し、**OOF レポート**に「どの特徴群が Binary/Macro に効いたか」を自動要約します。
* 本計画は**モデル非依存**で、LightGBM/TabNet/1D-CNN いずれにも接続可能です。モデル工夫は**この前処理が固まってから**着手します。

[1]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/data?utm_source=chatgpt.com "Detect Behavior with Sensor Data - CMI"
[2]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/582344?utm_source=chatgpt.com "CMI - Detect Behavior with Sensor Data ..."
[3]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/589961?utm_source=chatgpt.com "CMI - Detect Behavior with Sensor Data ..."
[4]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/583080?utm_source=chatgpt.com "CMI - Detect Behavior with Sensor Data ..."
[5]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/583118?utm_source=chatgpt.com "CMI - Detect Behavior with Sensor Data ..."
[6]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/589266?utm_source=chatgpt.com "CMI - Detect Behavior with Sensor Data ..."
[7]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/596787?utm_source=chatgpt.com "CMI - Detect Behavior with Sensor Data ..."
[8]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/583452?utm_source=chatgpt.com "CMI - Detect Behavior with Sensor Data ..."
