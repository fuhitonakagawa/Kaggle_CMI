# 概要

**手首装着型デバイスのセンサーデータから、毛抜きなどのBFRB（Body-Focused Repetitive Behaviors：身体集中反復行動）を検知・分類する**課題です。

# データ

* **手首装着デバイス由来の多変量時系列**。IMU（加速度・ジャイロ）に加え、温度・近接等のチャンネルが含まれます。コミュニティ知見では**隠しテストの一部が IMU のみ**である点が示唆されており、チャネル欠損への頑健性が有用です。

# 評価指標

* **カスタム指標（階層型 macro-F1）**。ざっくり言うと

  1. 「BFRB か否か」の**二値 F1** と、
  2. **BFRB 対象クラス間の macro-F1**
     を**同等に重み付け**して総合スコアを算出します。公式メトリクス実装ノートブックが公開されています。([Kaggle][4])

# 提出形式

* **コード提出型（Code Competition）**。ノートブック／モデルを用いた再現可能な推論が求められます（ディスカッション／ベースラインでもコード前提で議論）。


---

# まず押さえる前提（Discussionより）

* **評価指標**：本コンペの評価は**2成分の合成（変種 Macro-F1）**で、①「対象BFRBか否か」の**Binary F1**と、②「対象BFRB内のクラスごとの**Macro-F1**」を等重みで平均する方式がアナウンスされています。したがって**二段階（バイナリ→多クラス）**や**クラス再重み**が効きやすい設計です。([Kaggle][1])

* **センサー事情**：Discussionでは、\*\*IMU（加速度 acc\_x/acc\_y/acc\_z、ジャイロ/磁気）\*\*などの構成が整理されており、\*\*世界座標系への変換（World Acceleration）\*\*を使った特徴抽出が頻出です。([Kaggle][2])

* **テストの半分はIMUのみ**：**公開テストの約50%がIMUのみ**（他センサー欠損）で構成される旨が共有され、**IMU専用モデル**や**IMU/フルセンサー別モデルのアンサンブル**が推奨されています。([Kaggle][3])

* **推論APIの制約**：提出用`predict`は**1シーケンス（＝1 subject相当）単位**で呼ばれる想定がDiscussionで明示。**逐次推論**前提での前後文脈依存やバッチングの工夫が必要です。([Kaggle][4])

* **データの内訳**：データ説明では、**BFRB様8ジェスチャ＋非BFRB様10ジェスチャ**の分類タスクである旨が示されています（全18クラス）。([Kaggle][5])

---

# 手法別：実用“ベースライン”と公開Notebook

以下は**Discussionで議論が活発**、かつ**Notebookで実装が公開**されている“足場”です。まずは①→②の順で着手すると安定してLBに乗りやすいです。

### 1) IMU特化：統計＋勾配ブースティング（LightGBM/XGBoost系）

* **WorldAcc変換＋統計量→LightGBM**
  *ポイント*：端末座標→\*\*世界座標（WorldAcc）\*\*で重力を扱いやすくし、窓統計・ピーク・分散・相関などを積んだライト級の強学習器。**IMU-onlyテスト対策**としても堅い初手。
  代表Notebook：**IMU-only baseline (WorldAcc) + LGBM**、学習/推論分離ノートあり。([Kaggle][6])

* **LightGBM + ToF/他センサー（フル入力）**
  *ポイント*：利用可能なセンサーを広く入れ、IMU-only版と**二本立て**して**重み付け平均**でLBを底上げ。
  代表Notebook：**LightGBM + TOF PRO v1**。([Kaggle][7])

* **周波数領域特徴（FFT/Welch等） + LGBM**
  *ポイント*：**スペクトル特徴**はジェスチャ周期性の識別に効きやすい。IMU-onlyに容易に追加可能。
  代表Notebook：**Frequency Features 版**、**Spectrum EDA**。([Kaggle][8])

### 2) 1D-CNN / CNN-GRU 系（深層学習の軽量ベースライン）

* **IMU-only 1D-CNN**
  *ポイント*：生波形を**1D-CNN**で畳み込み。軽量・高速で、IMU限定の汎用ベースラインとして有用。
  代表Notebook：**1D CNN IMU-Only Baseline**。([Kaggle][9])

* **IMU＋Thermopile等の** **CNN-GRU**
  *ポイント*：センサー混在時に\*\*畳み込み（局所パターン）＋GRU（時間依存）\*\*のハイブリッド。IMU-onlyセットと組み合わせると堅い。
  代表Notebook：**CNN-GRU ベースライン**, **BFRB Detection (IMU + Thermopile CNN-GRU)**。([Kaggle][10])

* **PyTorch “All-in-one” 学習パイプライン**
  *ポイント*：学習・検証・推論を一体化したテンプレ。自分流に差し替えやすい。
  代表Notebook：**\[CMI] Pytorch All in One Baseline \[Train]**。([Kaggle][11])

### 3) アンサンブル／ブースティングの強化

* **IMU-only と フルセンサーを**別学習→**重み付き平均**
  *ポイント*：IMU-only比率が高いテスト構成に合致。**CV性能に応じた重み付け**が推奨されています。([Kaggle][12])

* **勾配ブースティング強化版**
  *ポイント*：IMU-onlyベースに**特徴拡張＋学習器の見直し**で底上げ。
  代表Notebook：**Boosting Baseline for BFRB Gestures**。([Kaggle][13])

* **公開ベースラインの改良例**
  *ポイント*：オーソドックスな“公開Baseline”を**特徴/折り方/TTA**で改善。LB \~0.79 例の共有もあり（数値は時期で変動）。([Kaggle][14])

---

# CV・評価・実装の勘所（Discussionからの示唆）

* \*\*CVは StratifiedGroupKFold（subjectをグループ）\*\*が定番。**被験者リーク**を避けつつクラス分布を保つ分割が推奨。([Kaggle][15])

* **k-foldの汎用的注意**（同一HPでの建設的議論あり）：**同ハイパラでkモデル学習→全域評価**という原則を徹底。([Kaggle][16])

* **データの相（Gesture/Transition）比率や外れ系列**の分析スレッドもあり。閾値設計や損失重みの調整の参考に。([Kaggle][2])

* **LeaderboardとCVの乖離**に関する注意喚起も出ています。過度な公開LB追随は非推奨。([Kaggle][3])

* **コード実装**：**シーケンス単位推論**（subject単位）前提のDiscussionを踏まえ、**逐次処理の前後状態の扱い**や**窓ずらし/TTA**の設計が鍵。([Kaggle][4])

---

# 最短で“走る”ための推奨ルート

1. **IMU-only LGBM（WorldAcc）**を即日で再現（学習/推論ノートあり）
   → **IMU-only 1D-CNN**を追加して**平均/重み付き平均**でLB安定化。([Kaggle][6])
2. 余力があれば、**周波数特徴**や\*\*CNN-GRU（フルセンサー）\*\*を差し込み、**IMU-only系と二本立てのアンサンブル**へ。([Kaggle][8])
3. \*\*CVはStratifiedGroupKFold（subject）\*\*で固定し、**評価指標の二段階性**（Binary＋Macro）を意識した損失/閾値調整。([Kaggle][15])

---

## 参考リンク（抜粋）

* コンペ概要/データ：概要・データ・LB。([Kaggle][17])
* 評価指標の説明スレ：**Binary F1 + Macro-F1 の等重み**。([Kaggle][1])
* IMU-only & WorldAcc系：**LGBMベースライン**、学習/推論ノート。([Kaggle][6])
* 1D-CNN / CNN-GRU：**IMU-only 1D-CNN**、**CNN-GRU**。([Kaggle][9])
* フルセンサーLGBM：**LightGBM + TOF PRO**。([Kaggle][7])
* スペクトル特徴：**Frequency Features**、**Spectrum EDA**。([Kaggle][8])
* アンサンブル論点/注意喚起：**重み付き平均**提案、**IMU-only比率**情報。([Kaggle][12])
* CV設計/推論API：**StratifiedGroupKFold（subject）**、**逐次推論**。([Kaggle][15])

---

必要でしたら、上記のうち**どれをベースにするか**を決めて、\*\*1〜2週間の短期実装計画（再現→CV固定→特徴拡張→アンサンブル→閾値調整）\*\*まで落とし込みます。どの路線（LGBM主体 / 1D-CNN主体 / ハイブリッド）で進めましょうか。

[1]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/582658?utm_source=chatgpt.com "CMI - Detect Behavior with Sensor Data | Kaggle"
[2]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/593675?utm_source=chatgpt.com "CMI - Detect Behavior with Sensor Data | Kaggle"
[3]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/589961?utm_source=chatgpt.com "CMI - Detect Behavior with Sensor Data | Kaggle"
[4]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/583504?utm_source=chatgpt.com "CMI - Detect Behavior with Sensor Data | Kaggle"
[5]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/data?utm_source=chatgpt.com "Detect Behavior with Sensor Data - CMI - Kaggle"
[6]: https://www.kaggle.com/code/ryenhails/imu-only-baseline-lgbm-using-worldacc?utm_source=chatgpt.com "[IMU-only baseline] LGBM using WorldAcc - Kaggle"
[7]: https://www.kaggle.com/code/feliperodriguez95/cmi-lightgbm-tof-pro-v1?utm_source=chatgpt.com "CMI - LightGBM + TOF PRO [v1]"
[8]: https://www.kaggle.com/code/satoshissss/cmi-bfrb-imu-inference-v10-frequency-features/output?scriptVersionId=244926159&utm_source=chatgpt.com "CMI-BFRB IMU Inference v10 - Frequency Features - Kaggle"
[9]: https://www.kaggle.com/code/richolson/cmi-2025-1d-cnn-imu-only-baseline?utm_source=chatgpt.com "CMI 2025: 1D CNN IMU-Only Baseline - Kaggle"
[10]: https://www.kaggle.com/code/kakuteki/cmi-baseline-cnn-gru/input?utm_source=chatgpt.com "CMI_Baseline CNN-GRU - Kaggle"
[11]: https://www.kaggle.com/code/takanashihumbert/cmi-pytorch-all-in-one-baseline-train/output?utm_source=chatgpt.com "[CMI] Pytorch All in One Baseline [Train] - Kaggle"
[12]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/586562?utm_source=chatgpt.com "CMI - Detect Behavior with Sensor Data | Kaggle"
[13]: https://www.kaggle.com/code/sayedathar11/imu-only-boosting-baseline-for-bfrb-gestures?utm_source=chatgpt.com "IMU Only: Boosting Baseline for BFRB Gestures - Kaggle"
[14]: https://www.kaggle.com/code/garyzhao13/cmi-baseline-model-add-aug-folds-tta-lb-0-79/comments?utm_source=chatgpt.com "CMI Baseline+model_add, Aug, folds, TTA--LB-0.79 - Kaggle"
[15]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/583023?utm_source=chatgpt.com "CMI - Detect Behavior with Sensor Data | Kaggle"
[16]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/596787?utm_source=chatgpt.com "CMI - Detect Behavior with Sensor Data | Kaggle"
[17]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data?utm_source=chatgpt.com "CMI - Detect Behavior with Sensor Data - Kaggle"


---

CMIコンペの**サブミッション形式**は通常のCSVではなく、**評価API（inference server）で生成されるParquet**を提出します。要点だけ端的にまとめます。

# サマリ

* **提出は評価API必須**：主催が用意したPython評価APIで、**テストを1シーケンスずつ**モデルに投げて推論します。([Kaggle][1])
* **提出ファイル**：推論実行後に\*\*`submission.parquet`\*\*が自動生成され、**そのParquetをアップロード**します。([Kaggle][2])

# ファイル仕様（最低限）

* **ファイル名**：`submission.parquet`（単一ファイル） 。([Kaggle][3])
* **カラム**：

  * `sequence_id`（文字列）
  * `gesture`（文字列；**クラス名**）
    代表的な公開ノートでは`submission.parquet`を読み込むと\*\*`sequence_id` / `gesture`\*\*の2列で構成されています。([Kaggle][4])
* **行の対応**：**テストの各sequence\_idに対して1行**（重複なし）。不足や余分行があると失敗/減点の原因になります（評価APIが全シーケンスを流して作る前提）。([Kaggle][1])
* **クラス名**：`gesture`の値は、コンペページに記載の**全18クラスの正式名称**のいずれか（**大文字小文字・スペースまで完全一致**させるのが安全）。公式ページに**クラス一覧表と動画**があります。([Kaggle][5])

# 評価APIまわり（実装の肝）

* **`predict()` のI/F**：評価APIが**1つのシーケンス（＋メタ情報）**を渡すので、`predict()`は**そのシーケンスの予測クラス名（文字列）を返す**実装にします。([Kaggle][6])
* **生成フロー**：Demo Submissionに沿って`predict()`を実装→評価APIで隠しテストが逐次推論→**`submission.parquet`が出力**→それを「Submit Predictions」から提出。([Kaggle][2])

# ありがちな落とし穴（チェックリスト）

* `gesture`の**表記ゆれ**（公式名と完全一致していない）。([Kaggle][5])
* **行数不一致**（テスト全sequence\_idを網羅していない／重複している）。([Kaggle][1])
* **形式違い**（CSVで出力してしまう等）。**Parquet必須**です。([Kaggle][2])

# 最小サンプル（自前で整形する場合）

```python
import pandas as pd
df = pd.DataFrame({
    "sequence_id": ["SEQ_000001", "SEQ_000002"],
    "gesture": ["Above ear - pull hair", "Wave hello"]  # ← 公式の18クラス名から選択
})
df.to_parquet("submission.parquet", index=False)
```

（公開ノートでも`to_parquet("submission.parquet")`で作っています。）([Kaggle][3])

---

必要でしたら、**提出直前のバリデーション用スクリプト**（重複・欠落・未知ラベル検査）をお作りします。クラス名リストを公式表から機械化して組み込み、**完全一致のみ許容**するチェックにしてお渡し可能です。

[1]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/data?utm_source=chatgpt.com "Detect Behavior with Sensor Data - CMI"
[2]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/589266?utm_source=chatgpt.com "CMI - Detect Behavior with Sensor Data ..."
[3]: https://www.kaggle.com/code/saikoh/notebook7f51e7bf0c?utm_source=chatgpt.com "notebook7f51e7bf0c"
[4]: https://www.kaggle.com/code/nina2025/cmi-ensemble-of-3-solutions-without-random/notebook?scriptVersionId=254971684&utm_source=chatgpt.com "CMI | Ensemble of 3 solutions, withOUT RANDOM"
[5]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data?utm_source=chatgpt.com "CMI - Detect Behavior with Sensor Data"
[6]: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/583504?utm_source=chatgpt.com "CMI - Detect Behavior with Sensor Data ..."
