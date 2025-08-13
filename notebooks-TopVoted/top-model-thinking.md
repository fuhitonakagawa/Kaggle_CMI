いいね、しっかり読みました。追加でいただいた 4 本（LB0.82 ブレンド、重みだけ変えた版、LB0.77 線形加速度+BiLSTM/GRU/Attention、EDA+PyTorch）を“特徴量・モデル・推論/アンサンブル・実装 Tips”の観点でまとめます。前回までの要約に継ぎ足しやすい形にしてあります。

# 追加ノートブックの要約

### 1) `cmi25-imu-thm-tof-tf-blendingmodel-lb-82.ipynb`

**モダリティ/特徴量（時系列フレームごと）**

- IMU 系

  - 線形加速度（重力除去）：四元数 → ワールド座標に回し `[0,0,9.81]` を減算して `linear_acc_{x,y,z}` を作成
  - |linear|の大きさ `linear_acc_mag` とそのジャーク `linear_acc_mag_jerk`
  - 四元数から角速度（隣接クォータニオン差分 → 回転ベクトル/Δt）：`omega_{x,y,z}`, `omega_mag`
  - 角距離 `angular_distance = 2*arccos(rot_w)`（※命名は「angular_distance」）

- ToF/サーマル（各フレーム 8×8 の ToF を集約）

  - 各 ToF センサ(1…5)について、**1 フレーム内**の 8×8 画素を `mean/std/min/max` に圧縮（`tof_i_mean/std/min/max`）
  - Thermal は明示の高次特徴は少なめ（本ノートでは ToF 集約が中心）

- 前処理

  - `final_feature_cols` を固定し、`StandardScaler` で**各時刻の特徴ベクトルをスケール**
  - シーケンスは `pad_len` で後詰めパディング（不足は 0 埋め）

**モデル（TensorFlow/Keras）**

- 2 ブランチ構成：

  - IMU ブランチ：Residual SE-CNN（64→128, ReLU, BN, MaxPool, Dropout）
  - ToF/THM ブランチ：軽量 CNN（64→128, ReLU, BN, Pool, Dropout）

- 結合後：BiLSTM(128) ＋ BiGRU(128) ＋（並列）GaussianNoise→Dense(16, ELU) を**Concat**
- Attention で時系列重み付け → Dense(256→128, BN+ReLU+Dropout) → Softmax(18)
- 学習設定：EarlyStopping/Checkpoint あり（詳細は省略、既存重みのロード中心）

**推論/アンサンブル**

- **多数の Keras モデル（フォルダを横断して読み込み）を同一前処理で推論し、確率の単純平均**
- 最終は `argmax` でクラス決定（**重み付けなし**の平均）

**実装 Tips**

- ToF は**フレーム内統計**に留め、チャンネル数を増やし過ぎない方針 → DL 側に時系列を任せる
- 特徴列が固定（`final_feature_cols`）。スケーラーも学習時のものを再利用（整合性 ◎）

---

### 2) `just-changed-the-ensemble-weights.ipynb`

**モダリティ/特徴量**

- パイプライン 1：Keras 系（上とほぼ同系）→ 上記と同等の IMU+ToF/THM 時刻特徴＋スケール＋パディング
- パイプライン 2：PyTorch モデル（後述の EDA ノートに近い 2 ブランチ CNN+RNN+Attention 系）

**モデル**

- **Keras 系の平均予測**（複数モデル平均） → `predict1_prob`
- **PyTorch 系の平均予測**（複数モデル平均） → `predict2_prob`

**推論/アンサンブル**

- **最終ブレンド：0.4 \* predict1 + 0.6 \* predict2**（重みを“変えた”だけのノート）
- クラスはブレンド確率の `argmax`

**実装 Tips**

- **二系統（TF+Torch）で平均 → 重み付き平均**の二段ブレンド
- 片系統の劣化時も他系統で緩和できる（ロバストさ ◎）

---

### 3) `lb-0-77-linear-accel-tf-bilstm-gru-attention.ipynb`

**モダリティ/特徴量**

- IMU メイン。**重力除去 → 線形加速度**（四元数からワールドへ → 重力差分）
- 派生：`acc_mag`, `linear_acc_mag`, それぞれの微分/ジャーク、四元数由来の角速度、`rot_angle` とその差分など
- ToF/THM があれば 2 ブランチに拡張可（実装は二択構成）

**モデル（TensorFlow/Keras）**

- IMU のみ：Residual SE-CNN(64→128→256) → BiLSTM(128)＋ BiGRU(128) → Attention → Dense(256→128) → Softmax
- IMU+ToF/THM：1)と同様に**2 ブランチ**（IMU 深め/ToF 軽め）

**推論/学習**

- シーケンス長は**上位パーセンタイル**で `pad_len` を決定（例：P95）
- `StandardScaler` でフレームごと標準化 → パディング → 推論
- **単一モデル**での評価（K-Fold 平均ではない）
- ノート名通り **LB ≈ 0.77** 相当の構成

**実装 Tips**

- **線形加速度（重力除去）を“時刻特徴”として直接食わせる**発想が主
- ToF/THM は“軽め”の分岐として追加（本体は IMU）

---

### 4) `sensor-pulse-viz-eda-for-bfrb-detection.ipynb`

**モダリティ/EDA**

- EDA 中心。シーケンス長別に IMU ジャーク、Gesture 区間の回転レンジ、サーモのピーク差分などを可視化
- “センサ脈動（pulse）”という表現で**ピークや変化率**を重視する視点（ジャーク、多段差分）

**PyTorch モデルの素案**

- **2 ブランチ Conv1D**（IMU ブランチ/ToF+THM ブランチ）→**Concat**→ **BiLSTM** → **GRU** → **Attention** → 全結合
- 損失：**Soft Cross Entropy**（ラベルスムージング寄り）
- 学習：標準的な DataLoader/評価ループ（コード有）

**実装 Tips**

- **Conv→RNN→Attention の王道**を PyTorch で実装。Keras 系との**系統差**が出せる
- 解析で得た“ピーク・勾配・区間レンジ”の有効性を**手工特徴でも DL でも**取り込める

---

# 全体の整理（今回分の共通点・相違点）

- **共通の前処理哲学**

  - **重力除去 → 線形加速度**、**角速度**、**角距離**はほぼ共通。
  - ToF は\*\*フレーム内統計（mean/std/min/max）\*\*で圧縮し、DL 側で時系列学習。
  - **スケーリングは“時刻特徴ベクトル”単位**で実施 → **pad_len でパディング** → シーケンスモデルへ。

- **違い（モデル/アンサンブル）**

  - Keras 系は**大量モデルの単純平均**（LB0.82 のブレンド）。
  - “重みだけ変えた版”は **Keras 平均** と **PyTorch 平均** を **0.4/0.6** ブレンド。
  - LB0.77 版は**単体モデル**での仕上げ（IMU 主体、ToF/THM は軽接続）。
  - EDA は**PyTorch 2 ブランチ + RNN + Attention** のベース実装＆指標群の着想源。

---

# あなたの現在実装（XGBoost 大規模手工特徴）との接続ポイント

- **そのまま取り込める/取り込みたい要素**

  - 線形加速度/角速度/角距離の\*\*“時刻ベース”\*\*特徴は既に取り込めていますが、
    \*\*ジャーク、区間内ピーク統計、Hjorth、自己相関、スペクトル（Welch）\*\*などは XGB 側でさらに厚い（既に対応済み 👍）。
  - **ToF のフレーム内統計（mean/std/min/max, 左右/上下非対称、中心/リング、クラスタ）** → 既に網羅的に実装済みで、
    **LB0.82 系の考え方と合致**（“DL には時系列、集約は前処理”）。

- **将来の DL 併用に向けて**

  - **“時刻ベースの固定列”**（`final_feature_cols` 的な）を 1 セット定義 → Keras/PyTorch の**両方で完全一致**の前処理・スケーラーを使う。
  - K-Fold で **TF 系数十本 + Torch 系数十本** の**平均**（まずは等重み）、
    そこに **XGB（手工特徴アグリゲーション）** を**確率平均で浅くブレンド**（0.1–0.2）するのが安全。
  - その後、ブレンド重み最適化（OOF で**単純 2 変数の凸最適化**）→ “just-changed-the-ensemble-weights” の思想をより体系化。

---

# すぐ使える“まとめ表”

| ノート                                           | 特徴量の主眼                                                 | モデル                                                      | アンサンブル                              | 備考                                               |
| ------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------------------- | ----------------------------------------- | -------------------------------------------------- |
| **cmi25-imu-thm-tof-tf-blendingmodel-lb-82**     | 線形加速度・角速度・角距離（時刻ベース）+ ToF フレーム内統計 | Keras 2 ブランチ（IMU 深め+ToF 軽め）→ BiLSTM+GRU+Attention | **多数 Keras モデルの単純平均**           | LB≈0.82 系。`final_feature_cols`固定＆スケーラ必須 |
| **just-changed-the-ensemble-weights**            | 上と同等 + PyTorch 系                                        | **Keras 集合** と **PyTorch 集合**                          | **0.4 (TF) + 0.6 (Torch)** の重み付き平均 | 重みをいじるだけの検証用                           |
| **lb-0-77-linear-accel-tf-bilstm-gru-attention** | **線形加速度中心** + 角速度/角距離（必要に応じ ToF 軽接続）  | Keras（IMU のみ or 2 ブランチ）→ BiLSTM+GRU+Attention       | 単体モデル                                | LB≈0.77。pad_len=高パーセンタイル                  |
| **sensor-pulse-viz-eda-for-bfrb-detection**      | EDA：**ジャーク/ピーク/レンジ**等“パルス”指標                | **PyTorch** 2 ブランチ →BiLSTM→GRU→Attention                | 平均ベースの雛形あり                      | Soft-CE 的損失。Keras 系と系統分離に最適           |

---

# この要約にもとづく次アクション（提案）

1. **XGB の土台を維持**（すでに強い）

   - あなたの手工特徴群は**LB 上位の発想を包含**。XGB-L2 正則で過学習を抑えつつ、**OOF で Δ 重要度**を確認。

2. **DL で“時刻ベース固定列+スケーラ共通化”**

   - `final_feature_cols` を**プロジェクト共通**にして、Keras/PyTorch の両方が**同じテンソル**を受け取るよう整理。

3. **まずは“等重み平均”→“2 重み最適化”**

   - Keras 集合平均 + PyTorch 集合平均 → **0.5/0.5**で始め、OOF で**2 変数の重み最適化**。
   - その上に **XGB** を **0.1〜0.2** 程度で浅めにブレンドして頑健化。

4. **ToF の空間情報は“フレーム内統計”重視**

   - LB0.82 系のように**ディープ前段のチャネル爆発を避ける**方針を踏襲。

5. **評価運用**

   - まずは**K-Fold 平均のままで OK**。最終一括再学習は**推論コードや pad_len/スケーラ不整合のリスク**があるため、
     安定後（最終提出直前）に“保存済み最良重みでの通し再学習モデル”を追加で*比較*するくらいが安全です。

---

必要なら、上のまとめを `/mnt/data/Competition_Overview.md` に追記する体裁で整えます。次は「共通 `final_feature_cols` の確定」と「Keras/Torch の**同一前処理関数**」を作っておくと、後のブレンド最適化が一気に進みます。
