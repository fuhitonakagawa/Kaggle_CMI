了解！新しく共有いただいた 3 本の Notebook をざっと“中身まで”確認して、要点を一枚にまとめました。
（対象）

- `cmi-2025-1d-cnn-imu-only-baseline.ipynb`（Keras）
- `cmi-baseline-cnn-gru.ipynb`（Keras）
- `cmi-pytorch-all-in-one-baseline-train.ipynb`（PyTorch）

# サマリー（ざっくり比較）

| 項目                    | 1D-CNN IMU-only (Keras)                                                   | CNN+GRU IMU-only (Keras)                                                                      | All-in-one Baseline (PyTorch)                                                                                                    |
| ----------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 入力センサー            | IMU のみ（`acc_*`, `rot_*`）※`thm_`, `tof_`は明示的に除外                 | IMU のみ（`acc_*`, `rot_*`）※`thm_`, `tof_`は除外                                             | IMU + Thermal（`acc_*`, `rot_*`, `thm_*`）※ToF は未使用                                                                          |
| 正規化                  | `StandardScaler`（\[samples×time, feat]→ 全体正規化）                     | 同上                                                                                          | Fold ごとの平均/分散を事前計算して正規化（fold leakage を避ける設計）                                                            |
| 欠損/クォータニオン処理 | NaN 埋め+正規化あり                                                       | 同様                                                                                          | あり（IMU 差分の追加もあり）                                                                                                     |
| 長さ揃え                | `pad_sequences`（固定長、P95 付近）                                       | 同上                                                                                          | 独自`pad_sequence`（固定長=約 188、切り詰め/ゼロ埋め）                                                                           |
| 追加特徴                | なし（生波形）                                                            | なし（生波形）                                                                                | 追加：IMU 軸間の差分（例：x−y）を複数ペアで生成                                                                                  |
| モデル                  | Conv1D 多数（フィルタ 512→2048） + BN + ReLU + MP + Dropout → GAP → Dense | Conv1D× 数層 + BN + ReLU + MP + Dropout → **BiGRU**(return_sequences) → Conv 系 → GAP → Dense | Conv1d ブロック ×3（7/5/3 カーネル, BN, GELU, MaxPool, Dropout）→ Flatten → 全結合(512→256) → 出力                               |
| RNN/注意                | なし                                                                      | **BiGRU あり**（Attention は未実装）                                                          | なし（全て CNN ベース）                                                                                                          |
| ロス/最適化             | CE + label smoothing(0.1) / Adam / ES & ReduceLROnPlateau                 | CE +（概ね同様）/ Adam / ES & ReduceLROnPlateau                                               | CrossEntropy / **AdamW** + **OneCycleLR**（cos anneal）, AMP（GradScaler）                                                       |
| 検証方法                | **Train/Val 固定分割**（Stratified はインポートされるが未使用）           | **Train/Val 固定分割**                                                                        | **StratifiedGroupKFold×4**（subject & layer 相当で層化）                                                                         |
| 保存/推論               | `*.h5` + `pad_len.npy` 保存。`CMIInferenceServer`対応                     | 同上                                                                                          | **fold ごと**に`*.pt`保存。IMU-only モデル（`cmi_model_foldX.pt`）と IMU+THM モデル（`cmithm_model_foldX.pt`）の**2 系統**を保存 |
| アンサンブル            | なし                                                                      | なし                                                                                          | **Folds 間**で OOF/推論が前提（最終平均）                                                                                        |

---

## 各 Notebook のポイントと所感

### 1) 1D-CNN IMU-only（Keras）

- **強み**: 大容量 CNN で局所時系列パターンを強く捉えられる。実装がシンプルで高速学習。
- **注意**:

  - Thermal/ToF を使っていないため、**接近/接触系の手掛かり**が弱い。
  - 固定 split のみで、**被験者リークの偏り**が残る懸念（SGKF 推奨）。
  - フィルタ本数が大きく**過学習**しやすい（L2 や Stochastic Depth 的な正則化・Data Aug が有効）。

### 2) CNN+GRU IMU-only（Keras）

- **強み**: CNN で局所 →**BiGRU で中長期依存**を捉える構成。IMU だけでも「挙動の持続性」を活かせる。
- **注意**:

  - Thermal/ToF 未使用。**接近や温度上昇の前兆**を使えていない。
  - こちらも固定 split。SGKF/KFold 化で頑健性を上げたい。

### 3) All-in-one Baseline（PyTorch）

- **強み**:

  - **IMU+Thermal**を入力に取り込み、fold ごとの**統計で正規化**→ リーク対策が明確。
  - **OneCycleLR + AdamW + AMP**で学習安定&高速。
  - 追加の\*\*差分特徴（IMU 軸間）\*\*で装着方向/傾き由来の相対変化を強化。
  - **4-fold SGKF**で被験者間一般化を意識した評価・保存ができる。

- **注意**:

  - ToF（8×8 距離）は未使用。**空間的近接の情報**を加えられる余地あり。
  - 現状は**CNN のみ**。BiLSTM/Attention で時間配置の分解能を上げる余地あり。
  - 2 本（IMU-only と IMU+THM）を fold ごとに保存しているが、**最終アンサンブルの手順**（重み/平均）がノート内で明示されていないので追記推奨。

---

## 今後の使い分け（実戦投入の観点）

- **まずは PyTorch 版を主軸**に：

  - 正規化/SGKF/スケジューラが整っており、**CV→Fold 平均**の流れが作りやすい。
  - Thermal を使って\*\*BFRB と非 BFRB の切り分け（Binary F1）\*\*を底上げしやすい。

- **Keras の 2 本は補完**：

  - IMU-only の軽量器として**判別境界が違う**ので、**ソフト投票**で補完的に効くことが多い。
  - ただし Keras 側も**SGKF 化**して、fold 平均の logits/probs を出せる形にすると、最終ブレンドが安定。

---

## 具体的な改良提案（すぐ効く順）

1. **検証の統一**

   - 3 本すべてを\*\*StratifiedGroupKFold（subject 単位）\*\*に揃える。
   - これで OOF が比較可能になり、アンサンブルの\*\*重み最適化（LogLoss 最小化や凸最適化）\*\*ができる。

2. **前処理の統一**

   - 正規化は**学習 fold 統計のみ**を使用（PyTorch と同様のリーク回避）。
   - パディング長は**P95**などで統一し、**前段でクリッピング**して外れ値の影響を低減。

3. **Thermal の導入（Keras 側にも）**

   - 温度の**移動平均/一次差分/二次差分/イベント率**（↑ はあなたの XGBoost 版で実装済み）を**DL のチャネル**としてそのまま入れる。
   - まずは**単純チャネル追加**→ 精度が伸びるなら、\*\*枝分かれ 2-branch（IMU branch + THM branch）\*\*へ。

4. **ToF の簡易導入（まずは時系列 Scalar として）**

   - 各フレームの**min 距離**（invalid は補間）だけでも 1 チャネル追加。
   - 効けば**PCA 3–8 成分**や**中心/半径方向のモーメント**など、あなたの XGB 特徴を**1D 化して追加**。

5. **軽い Attention**

   - Keras の CNN+GRU の**BiGRU 出力上**に簡易 Attention を 1 層（Dense(1)→softmax→ 重み和）で追加。

6. **学習テク**

   - Keras 側：`CosineAnnealing` or `OneCycle`風のスケジューラ/`Lookahead`/`EMA`の導入。
   - PyTorch 側：**CutMix/MixUp（時系列版）**、SpecAugment 的**マスク**（時間やチャネル drop）を小さく入れる。

---

## 最終アンサンブルの形（当面のゴール）

- **fold 平均**（各モデル内）→**モデル間ソフト投票**（重み付き）

  - 例）`w1 * PyTorch(IMU+THM) + w2 * PyTorch(IMU-only) + w3 * Keras(CNN+GRU) + w4 * Keras(Deep-CNN)`
  - 初期値は `w1=0.5, w2=0.2, w3=0.2, w4=0.1` からスタート。
  - OOF で**重み最適化**（Macro F1 寄与>Binary F1 寄与の比重を少し上げ、BFRB 内の混同改善を狙う）。

---

## まとめ（結論）

- **現状ベース**としては、**PyTorch の All-in-one**が最も大会仕様（CV/正規化/学習スケジュール）に近く、主軸に据えるのが良いです。
- **Keras の 2 本**は IMU-only の**異なる判別境界**を提供する“サブモデル”としてアンサンブルで活きます。
- Thermal（→ToF の最小距離）を**まずは 1 チャネル**から導入し、効果が出れば 2-branch 化や Attention へ拡張。
- すべて**SGKF に統一**して OOF を比較可能にし、**重みブレンド**で最終性能を押し上げる、が次の一手です。

必要なら、この 3 本を**SGKF・共通前処理・共通推論 API**で回せるテンプレ骨組み（推論時も同じ前処理＆平均）をまとめて用意します。
