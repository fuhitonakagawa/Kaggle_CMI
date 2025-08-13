結論から先に──**今の構成のままだと、IMUのみのテスト半分に対しては“安全ではありません”**。理由は主に2つです。

1. **推論時のカラム不一致（落ちる／壊れる）**

* `FeatureExtractor.fit()` は *PCAなしの生特徴* で `self.scaler` を学習し、`self.feature_names` を保存。
* しかし `transform()` では **PCA特徴を追加** してからスケーリングしています。→ **fit時と列数がズレて `scaler.transform()` が形状不一致**になります。
* さらに **ToF/サーマルが丸ごと欠損するシーケンスでは、その由来の特徴列自体が生成されない**ため、やはり **fit時の列集合と不一致**になり得ます。

2. **分布ギャップ（精度が落ちる）**

* 学習は“フルモダリティ（IMU+ToF+Thermal）が前提”の特徴を大量に含みます。テストの半分がIMUのみなら、**学習分布と推論分布が乖離**。
* これに対しては **モダリティDropout/知識蒸留/ルーティング** といった「欠損モダリティ耐性」の仕掛けが必要です（後述）。こうした考え方はマルチモーダルの定番で、**Modality Dropout（ModDrop）**、**EmbraceNet**、**HeMIS** などで広く使われています。 ([ResearchGate][1])

---

## まず直すべき最小修正（これを入れないと推論が安定しません）

**A. どのモダリティでも同じ“固定スキーマ”で整形する**

* **fit時**に、**（PCA適用後の）完全な列集合**を確定し、その順序を `self.feature_names` に保存。
* **transform時**は、抽出した特徴を **`reindex(columns=self.feature_names, fill_value=0.0)`** でゼロ埋め整列してからスケーリング。

**変更イメージ（要点のみ）**：

```python
# --- FeatureExtractor.fit() の最後 ---
# 1) 先にToF PCAをfit済み → self.is_fitted = True にしておく
self.is_fitted = True

# 2) trainシーケンスを extract_features()（PCA込み）で一度通し、
#    列のユニオンを確定
train_feature_dfs = []
all_cols = set()
for seq_df, demo_df in sequences:
    f = self.extract_features(seq_df, demo_df)   # ここはPCA込み
    train_feature_dfs.append(f)
    all_cols |= set(f.columns)
self.feature_names = sorted(all_cols)

# 3) ゼロ埋めで整形してからスケーラfit
X_raw = pd.concat(
    [df.reindex(columns=self.feature_names, fill_value=0.0) for df in train_feature_dfs],
    ignore_index=True
)
self.scaler = RobustScaler() if self.config["robust_scaler"] else StandardScaler()
self.scaler.fit(X_raw)
```

```python
# --- FeatureExtractor.transform() ---
X = pd.concat(feature_dfs, ignore_index=True)
X = X.reindex(columns=self.feature_names, fill_value=0.0)  # ★ここが肝
if self.scaler is not None:
    X[:] = self.scaler.transform(X.values)  # 形状は常に一致
return X
```

**B. “モダリティ有無フラグ”を必ず入れる**
`extract_features()` の冒頭で、以下のバイナリ特徴を追加してください。ゼロ埋めだけだと「本当にゼロなのか、欠損でゼロにしたのか」をモデルが判別できません。

```python
features["has_imu"]    = int(all(c in sequence_df.columns for c in ["acc_x","acc_y","acc_z"]))
features["has_tof"]    = int(any(c.startswith("tof_")   for c in sequence_df.columns))
features["has_thermal"]= int(any(c.startswith("therm_") for c in sequence_df.columns))
```

**C. 明確なバグ修正**

* `multi_resolution` → 設定名は `use_multi_resolution`。ブロック内のキー参照を修正。
* `compute_angular_velocity()` の `dt` は `1.0 / CONFIG["sampling_rate"]` を渡す。
* `extract_cross_modal_sync_features()` が ToFセンサーを **1..5** でループしていますが、他の箇所は **0..4**。インデックスを統一。
* `subject` を学習特徴に入れるのはNG（汎化を阻害）。`age`/`handedness` はOK、`subject` は除外を推奨。
* `FeatureExporter` でエクスポートしているのが **“スケーリング済みX”** になっている点も整理を。将来の再学習用に **未スケールの生特徴** を保存し、fold内で再スケールが原則。

---

## IMUだけでも戦える設計（推奨ロードマップ）

テスト半分がIMUのみでも崩れないために、以下のどれか（できれば複数）を入れてください。

### 1) 2本立てモデル + ルーター（**実装が最も簡単で堅い**）

* **IMU-onlyモデル**（列名で `acc_`, `quat_`, `world_acc_`, `linear_acc_`, `omega/jerk/freq` 系のみ）
* **Multi-modalモデル**（IMU + ToF + Thermal すべて）
* `predict()` で `has_tof/has_thermal` を見て **自動でモデルを切り替え**。

  * 片方しか無い（例：IMU+ToFのみ）ケースがあるなら、必要に応じて **IMU+ToF 専用モデル** を追加。
* この方式は分布ギャップに自然に対処でき、保守もしやすい。

### 2) 単一モデルだが **Modality Dropout（学習時にToF/Thermalを確率的にゼロ化）**

* 学習の各foldで、一定確率 `p` で **ToF/サーマル由来の列（\`"tof\_" / "therm\_" で始まる全列）を丸ごとゼロ化** して学習データを増強。
* 同時に `has_tof/has_thermal` を0にしておく。
* マルチモーダルの汎用的な耐性付与手法で、古典的には **ModDrop** と呼ばれる考え方に相当します。

### 3) 教師-生徒（知識蒸留）

* **教師：マルチモーダル**（IMU+ToF+Thermal）
* **生徒：IMU-only**
* 教師のソフトラベルで生徒を学習させ、**IMUだけでも教師の識別境界を模倣**します。欠損に強い定番アプローチです（EmbraceNet/HeMIS系の思想や蒸留は多数の文献で有効性が示されています）。 ([arXiv][2])

> ※ 2) と 3) は単一モデル運用をしたいときに特に有効。まずは 1) を入れて、その後 2)/3) を追加すると安全です。

---

## 検証の仕方（テスト条件を再現）

* 各foldの **validationの半分を人工的にIMU-only化**（ToF/サーマル列をゼロ化し、フラグを0）してメトリクスを出す。
* `Score_full`（フル）と `Score_imu_only`（IMU-only）の両方を毎foldで記録。
* そうすれば「本番分布（半分IMU-only）」に近い **合成スコア** をCVで見積もれます（例：`0.5*Score_full + 0.5*Score_imu_only`）。
