以下は、いただいた実装（**v1.1 Advanced Feature Engineering with XGBoost**）を、先に共有した計画（plan-pro.md 相当）と照合してレビューしたものです。結論から言うと、**カバー範囲は広く、方向性は良い**です。特に IMU（World/Linear）、周波数、Hjorth、ToF の空間モーメント、到達イベント、クロスモーダル同期など、**上位解法で効く要素は一通り入っています**。
一方で、**CV リーク（PCA/Scaler）**、**ToF-PCA の fit タイミングとロジック**、**短系列の周波数特徴**、**左右反転（handedness）の扱い**、**一部バグ/堅牢化**など、**早急に直したいポイント**が複数あります。以下、**優先度順**に具体的に指摘します。

---

## A. Must-fix（精度と再現性に直結）

### A-1) CV リーク：Scaler（および PCA）を CV の外で fit している

- 現状：`FeatureExtractor.fit_transform()` 内で **全訓練シーケンスを一括抽出 → Scaler.fit_transform** しています。その後で `StratifiedGroupKFold` を回すため、**fold の検証分布まで見たスケールで学習**しており、CV が楽観的になります。
- 対応：

  - **各 fold 内で Scaler/PCA を fit** してください。設計は以下のいずれか：

    1. **fold ごとに extractor を新規作成** → `extractor.fit(train_sequences)` → `extractor.transform(train/val_sequences)`
    2. あるいは\*\*特徴抽出自体は“非学習系”（Z-score 除く）\*\*にし、**学習が必要な変換（PCA/Scaler）は fold 内で fit**。

  - ついでに、**分位クリップのしきい値、近接判定の percentile 等**も fold の train から推定（val を含めない）。

### A-2) ToF-PCA の fit ロジックが不整合（ほぼ per-sequence fit になっている）

- 現状：`extract_features()` 内で `self.is_fitted` が **True になるのは `fit_transform()` 終了時**です。よって `fit_transform()` 実行中は **各シーケンスごとに PCA を fit** し、`self.tof_pcas[...]` を**上書き**してしまいます。最後に処理されたシーケンスの PCA が残るだけ、という不整合が発生。
- バグもあり：`tof_pca_features is None` でも `.shape` を呼ぶ箇所や、`reconstructed = pca.inverse_transform(...)` が `pca` 未定義になり得ます。
- 対応（推奨フロー）：

  - **二段パス**に分ける：

    1. **fit 段**：fold の**train シーケンスの全フレーム**から、**各 ToF センサーごと**に PCA を fit（必要なら `IncrementalPCA`）。
    2. **transform 段**：各シーケンスの全フレームを **fit 済み PCA で transform** → 時系列統計に集約。

  - 実装上は、fold ループの中で `extractor.fit_pca(train_sequences)` を呼び、その後に `extractor.extract(sequence)` が PCA を **transform 専用**で使う形が安全。

### A-3) 周波数特徴：短系列でゼロ埋めになっている

- 現状：`welch_nperseg=128` 固定のため、**長さ<128**の系列は**全部ゼロ**に。短い動作ほど特徴が死にます。
- 対応：

  - `nperseg = min(len(data), 128)`、`noverlap = nperseg//2` など **動的に縮める**。最小は 32 などを閾に。
  - 併せて **「相対パワー」**（各バンド/総パワー）を追加（スケール頑健）。対数化（`log1p`）も有効。

### A-4) 二段正規化（設計で推した項目）が未実装

- 現状：**sequence 内 Z-score** → **fold 内 train-fit Scaler** の二段を設計に入れていましたが、実装では **後段の Scaler のみ**。
- 対応：

  - **時系列から特徴を作る前に“sequence 内 Z-score or robust 標準化”** を入れてください（IMU/ToF/THM に適用、ただし ToF は空間 PCA の前は値域の解釈が必要なので注意）。
  - PSD や ZCR など振幅依存の指標には **正規化/相対化**で頑健化。

---

## B. Correctness / Bug

### B-1) ToF 同期特徴で `padded_data` を使っていない

- `extract_tof_sensor_sync_features()`：最初に `padded_data` を作っているのに **相関・相互相関で `sensor_data` を使い続ける**ため、**長さ不一致**に引っかかります。
- 対応：**相関/相互相関は `padded_data`** をソースにする。相互相関は `scipy.signal.correlate`＋`correlation_lags` を推奨。

### B-2) `tof_use_pca` 部分の None ハンドリング

- `tof_pca_features is None` のときに `.shape` アクセス／`pca` 未定義の可能性があります。
- 対応：**早期`continue`** か **if tof_pca_features is not None: ...** のガードを追加。

### B-3) ToF 内部領域マスクのロジック

- `extract_tof_region_features()` の **center/inner/outer** マスクは可読性が低く、意図どおりになっていない可能性があります（`inner_mask` 初期化 → 範囲 True/False の切替が分かりづらい）。
- 対応：狙いを **明示的に三層（中心 3×3 / 5×5 リング / 周縁リング）** へ分けるコードに修正（オフバイワン注意）。

### B-4) invalid に `np.inf` を入れてから勾配/縁検出

- `edge_sum`/`gradient_sum` を計算する前に invalid を 0 に置換していますが、**0 埋めは人工的なエッジを作る**可能性。
- 対応：無効画素は **NaN のまま**、`np.nan*` 系で集計（もしくは**有効領域のみに限定**して差分・勾配を取る）。

---

## C. 方針との差分（入っている/抜けている）

### C-1) ⭕ 良い点（方針反映）

- **World/Linear Acc、角速度、Hjorth、ZCR、周波数、3 分割セグメント、ピーク、ライン長**：互換性あり・網羅的。
- **ToF 空間モーメント／近接イベント／クラスタリング**：空間情報の保持ができている。
- **クロスモーダル**：IMU ピークと ToF 近接・Thermal 上昇の**トリプレット整合**は計画通り。
- **Multi-resolution（Temporal Pyramid）**：短中窓の併用が入っている（ただし IMU の一部列のみ）。

### C-2) ❗ 不足/改善推奨

1. **相対パワー**（バンド/総パワー）と **パワー比**（例：低周波/高周波）
2. **短窓 PSD**の**時間集約**（移動窓でバンドパワー →mean/var）。今は全区間 Welch 一発で、変化点を逃しやすい。
3. **ToF の min across sensors** を**時系列として**扱い、その**到達イベント統計**（連続長、初回到達までの時間）。現状は集約（平均）中心。
4. **Thermal×ToF の遅延相関**（ToF 近接 →Thermal 上昇の最尤遅れ）。
5. **Quality 特徴**の明示（最長連続欠測、ToF valid_ratio の最小値/分位、系列有効長/パディング比）。一部は入っていますが、**連続欠測長**は有用。
6. **Multi-resolution の適用範囲拡張**：`world_acc_mag`, `linear_acc_mag`, `tof_min_dist_global`（「5 基の最小」）にも適用。
7. **左右反転（handedness）**は**要検証**。現状 `tof_use_handedness_mirror=True` は**正方向が未確認**のまま適用されている恐れ。CV で**on/off 比較**して裏取りしてからデフォルト化を。

---

## D. 安定性・堅牢化

- **ピーク検出**：`height=np.std*0.5` はデータ依存度が高い。`prominence` を併用し、**移動平均で平滑後に検出**すると安定します。
- **相関・自己相関**：定数系列で NaN が出やすいので、**分母ゼロ回避**（既に z-score していますが、さらに `np.nan_to_num`）。
- **NaN/Inf 処理**：最後に 0 埋めしていますが、**途中計算で派生値が増える**ので、**各関数冒頭で NaN を除外/補完**する方が安全。
- **速度**：ToF の**フレームごとに多数の空間指標**を出すのはコスト大。

  - 代替：**フレーム間サンプリング**（等間引き）、**イベント近傍のみ**詳細計算、もしくは**低次元（PCA8）→ 時系列統計**へ寄せて計算量を抑える。

---

## E. 学習・推論の運用（副次的だが重要）

- **`predict()` 内で `train_models()` を呼ぶ設計**は、**評価サーバの初回呼び出しで学習が走る**ため**タイムアウト/資源競合のリスク**。

  - 推奨：**main で学習 → モデル/変換器を保存 →predict はロードのみ**。Kaggle `input` に置く運用に。

- **XGBoost GPU**：`device="cuda:0"` は **XGBoost のバージョン依存**。評価環境で 1.x 系だと動かない場合あり。`tree_method="hist"` フォールバックを用意。

---

## F. 優先修正の具体的提案（実装指針）

1. **CV 内 fit へのリファクタ**

   - CV ループ外の `fit_transform()` をやめ、**fold ごとに**：

     - `extractor = FeatureExtractor(CONFIG)`
     - `extractor.fit(train_sequences, train_demos)` ← **ここで**

       - **sequence 内正規化設定**を反映
       - **ToF-PCA（各センサー）を fit**（全 train フレームをストリーミングで `IncrementalPCA` へ）
       - **Scaler.fit(train_features_raw)**

     - `X_train = extractor.transform(train_sequences, ...)`
     - `X_val   = extractor.transform(val_sequences, ...)`

   - こうすれば **Scaler/PCA/分位基準**の**fold leakage を遮断**できます。

2. **Welch の動的 nperseg ＋相対バンドパワー**

   - `nperseg = max(32, min(128, len(data)))`、`noverlap = nperseg // 2`
   - 各バンド `power_band / power_total` を追加、`log1p` も別特徴に。

3. **ToF-PCA の二段化と None ガード**

   - `extract_features()` 側では **transform 専用**にして、`if tof_pca_features is None: continue`。
   - 再構成誤差は `pca.inverse_transform` が失敗しないようガード。

4. **handedness の検証フラグ**

   - `tof_use_handedness_mirror=False` をデフォルトにし、**アブレーションで on/off** を比較。
   - 必要なら**センサー ID の実配置に基づく反転**（単純左右反転ではなく、実ハード配置に合わせたマッピング）を後日導入。

5. **ToF 同期特徴の修正**

   - 相関・相互相関は **`padded_data`** を使用。
   - 相互相関は `scipy.signal.correlate`＋`correlation_lags` で**正しい遅れ**を算出。

6. **Multi-resolution の適用拡張**

   - `world_acc_mag`, `linear_acc_mag`, **`min_dist_global(t) = min_sensors min_dist_s(t)`** に対しても micro/short/medium の移動統計を付与。

7. **Quality 指標の追加**

   - ToF **連続欠測長の最大値**、**valid_ratio の p5/p50/p95**、**系列長/有効長** を明示的に特徴量へ。

---

## G. 期待インパクト（優先度 × 効果感）

| 修正項目                         | 期待効果                                                                      |
| -------------------------------- | ----------------------------------------------------------------------------- |
| **CV 内 fit（Scaler/PCA/分位）** | **高**：リーク解消で真の汎化性能に整合。Macro F1 の改善方向性の判断も正確に。 |
| **ToF-PCA 二段化＋ None ガード** | **高**：ToF 特徴の安定化・再現性向上。                                        |
| **Welch 動的化＋相対パワー**     | **中〜高**：短系列の情報を活かし、被験者間スケール差を緩和。                  |
| **handedness の検証**            | **中**：誤った反転が入っている場合の悪化を防止。正なら軽い改善。              |
| **同期特徴のバグ修正**           | **中**：クロスモーダルの寄与を素直に回収。                                    |
| **Multi-resolution 適用拡張**    | **中**：立ち上がり/離脱の時間局所性をより捉える。                             |

---

## H. 最小修正チェックリスト

- [ ] **fold ごと**に `extractor.fit(train)` → `transform(train/val)` に変更（Scaler/PCA/分位閾値も fold 内 fit）。
- [ ] ToF-PCA：**fit 段で集約**、`extract_features()` は **transform 専用**。`None` ガード追加。
- [ ] Welch：**nperseg 動的化**、**相対バンドパワー/対数化**を追加。
- [ ] handedness ミラー：**デフォルト OFF**、アブレーションで検証。
- [ ] ToF 同期：**padded_data を使用**、`signal.correlate`＋`correlation_lags`。
- [ ] sequence 内 Z-score を**前処理段**に導入（少なくとも IMU/THM、ToF は注意深く）。
- [ ] Quality 特徴（連続欠測長、有効長比）を追加。
- [ ] `predict()`内学習をやめ、**ロード運用**へ（評価安定性のため）。

---

上記を反映すれば、**二値検出（Binary F1）を維持したまま、BFRB 内 Macro F1 の底上げ**が十分に狙えます。
次ステップとして、必要であれば **fold 内 fit 対応の抽出器（`fit_pca`, `fit_scaler`, `extract/transform`）に分解した最小差分パッチ**をお渡しします。
