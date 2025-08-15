以下は、**いまの単一スクリプト**に「Keras 系モデル（2 ブランチ CNN→BiRNN→Attention）」を追加して、**LGBM + Torch + Keras の3系統アンサンブル**を回すための、**実装レベルの追加・改修タスクリスト**です。
（Keras の2ブランチ+BiLSTM/GRU+Attention、多数モデル平均→Torch と重み付きブレンド、という流れは、共有ノートに沿った方針です。 ）

---

## 0) 追加ディレクトリ & 出力物（統一）

* 追加ディレクトリ

  * `keras_models/` … Keras fold 重み・bundle・状態（JSON）
  * `oof/` … 各系統の OOF 予測（.npy/.pkl）と y\_true
* 主要出力

  * `keras_models/keras_bundle.pkl` … **pad\_len / feature\_order / 各foldスケーラ統計 & 重みパス**
  * `oof/oof_keras_proba.npy`, `oof/oof_torch_proba.npy`, `oof/oof_lgbm_proba.npy`, `oof/y_true.npy`
  * （任意）`ensemble/weights.json` … OOF から最適化した最終ブレンド重み

---

## 1) 依存ライブラリの条件付き import を追加（先頭の import 節）

```python
# === Keras/TensorFlow imports (conditional) ===
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    KERAS_AVAILABLE = True
    print("✓ TensorFlow/Keras available")
except Exception as e:
    KERAS_AVAILABLE = False
    print(f"⚠️ TensorFlow/Keras not available: {e} — Keras pipeline disabled")
```

> Keras が無い場合も全体は動き続ける（LGBM/Torch のみ）ようにします。

---

## 2) 設定クラスの追加・拡張

### 2.1 Keras 学習用設定

```python
class KerasConfig:
    OUT_DIR = os.path.join(Config.OUTPUT_PATH, "keras_models")
    N_FOLDS = Config.N_FOLDS
    SEED = Config.SEED

    # 前処理（Torch と共通関数を使用）
    PAD_LEN_PERCENTILE = 95
    FIXED_PAD_LEN = None  # 明示長がある場合は整数

    # 学習設定
    MAX_EPOCHS = int(os.getenv("KERAS_MAX_EPOCHS", "40"))
    BATCH_SIZE = int(os.getenv("KERAS_BATCH_SIZE", "64"))
    LR = float(os.getenv("KERAS_LR", "1e-3"))
    DROPOUT = float(os.getenv("KERAS_DROPOUT", "0.2"))
    LABEL_SMOOTHING = float(os.getenv("KERAS_LABEL_SMOOTHING", "0.1"))
    EARLY_STOPPING_PATIENCE = int(os.getenv("KERAS_ES_PATIENCE", "8"))
    REDUCE_LR_PATIENCE = int(os.getenv("KERAS_RLR_PATIENCE", "4"))

    # 保存ファイル名
    BUNDLE_NAME = "keras_bundle.pkl"
    WEIGHT_TMPL = "fold{:02d}.keras"  # KerasModel 保存
    STATE_JSON = os.path.join(OUT_DIR, "keras_state.json")
```

### 2.2 アンサンブル設定の拡張（Keras 重み）

```python
class EnsembleConfig:
    W_LGBM = float(os.getenv("ENSEMBLE_W_LGBM", "0.20"))
    W_TORCH = float(os.getenv("ENSEMBLE_W_TORCH", "0.45"))
    W_KERAS = float(os.getenv("ENSEMBLE_W_KERAS", "0.35"))

    # 既存…
    TORCH_BUNDLE_PATH = os.getenv("TORCH_BUNDLE_PATH", os.path.join(DLConfig.TORCH_OUT_DIR, DLConfig.BUNDLE_NAME))
    LOAD_TORCH_FOLDS_IN_MEMORY = bool(int(os.getenv("LOAD_TORCH_FOLDS_IN_MEMORY", "1")))
    FAIL_IF_TORCH_MISSING = bool(int(os.getenv("FAIL_IF_TORCH_MISSING", "0")))

    # Keras バンドル
    KERAS_BUNDLE_PATH = os.getenv("KERAS_BUNDLE_PATH", os.path.join(KerasConfig.OUT_DIR, KerasConfig.BUNDLE_NAME))
    LOAD_KERAS_FOLDS_IN_MEMORY = bool(int(os.getenv("LOAD_KERAS_FOLDS_IN_MEMORY", "1")))
```

> 0.35/0.45/0.20 は初期値。OOF から自動最適化できるフックも後述。
> Keras/Torch の**二段平均→最終重み付き平均**は、共有ノートの実装思想に一致します。&#x20;

---

## 3) 前処理は **既存の DL 用共通関数**を Keras でも利用

* そのまま `build_frame_features` / `compute_scaler_stats` / `apply_standardize` / `decide_pad_len` / `pad_and_mask` を流用。
* **注意**：Keras でも fold ごとの scaler 統計（μ, σ）を使う（リーク回避のため Torch と同じ運用）。

---

## 4) Keras データセット生成（tf.data）

### 4.1 逐次→テンソル化ユーティリティ

```python
def make_keras_tensor(sequence_pl: pl.DataFrame, stats: dict, pad_len: int, feat_order: list[str]) -> tuple[np.ndarray, np.ndarray]:
    frame_df = build_frame_features(sequence_pl).reindex(columns=feat_order).fillna(0)
    x_std = apply_standardize(frame_df, stats).to_numpy(np.float32)  # (T, C)
    x_pad, m_pad = pad_and_mask(x_std, pad_len)  # (L, C), (L,)
    return x_pad, m_pad
```

### 4.2 Dataset 構築（fold 内）

* `tf.data.Dataset.from_generator` で `(x_pad, m_pad), y` を生成。
* あるいは、**事前に numpy に落としてから** `tf.data.Dataset.from_tensor_slices`（学習が安定）。

---

## 5) Keras モデル定義（2 ブランチ／IMU-only）

> 2 ブランチ CNN → BiLSTM + BiGRU → Attention → 全結合、という構成を入れます（ノートの系統に準拠）。

### 5.1 軽量 Attention レイヤ

```python
class KerasTemporalAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(1)

    def call(self, h, mask):
        # h: (B, T, D), mask: (B, T)
        logit = self.dense(h)[:, :, 0]        # (B, T)
        logit = tf.where(tf.equal(mask, 1.0), logit, tf.fill(tf.shape(logit), tf.constant(-1e9, logit.dtype)))
        w = tf.nn.softmax(logit, axis=1)      # (B, T)
        pooled = tf.matmul(w[:, None, :], h)  # (B, 1, D)
        return tf.squeeze(pooled, axis=1)     # (B, D)
```

### 5.2 2 ブランチモデル（IMU 深め＋ToF/THM 軽め）

```python
def build_keras_two_branch(input_shape: tuple[int, int], n_classes: int, dropout: float = 0.2) -> keras.Model:
    x_in = layers.Input(shape=input_shape, name="x")     # (L, C)
    m_in = layers.Input(shape=(input_shape[0],), name="mask")  # (L,)

    # IMU/ToF/THM チャネルを分けたければここで split（まずは単一枝でOK→2枝拡張）
    x = x_in

    # Residual SE-CNN stack（簡易版）
    def conv_block(x, ch, k):
        y = layers.Conv1D(ch, k, padding="same")(x)
        y = layers.BatchNormalization()(y); y = layers.ReLU()(y)
        y = layers.Conv1D(ch, k, padding="same")(y)
        y = layers.BatchNormalization()(y)
        # Squeeze-Excitation (簡易)
        se = layers.GlobalAveragePooling1D()(y)
        se = layers.Dense(ch//4, activation="relu")(se)
        se = layers.Dense(ch, activation="sigmoid")(se)
        y = layers.Multiply()([y, layers.Reshape((1, ch))(se)])
        # Residual
        if x.shape[-1] != ch:
            x = layers.Conv1D(ch, 1, padding="same")(x)
        y = layers.Add()([x, y]); y = layers.ReLU()(y)
        y = layers.MaxPooling1D(2)(y)
        y = layers.Dropout(dropout)(y)
        return y

    h = conv_block(x, 128, 7)
    h = conv_block(h, 256, 5)
    h = conv_block(h, 256, 3)

    # RNN
    h = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(h)
    h = layers.Bidirectional(layers.GRU(128, return_sequences=True))(h)

    # マスクを downsample（MaxPool を3回通したぶん）
    m = m_in
    for _ in range(3):
        m = tf.nn.max_pool1d(m[:, :, None], ksize=2, strides=2, padding="VALID")[:, :, 0]
        # 長さが合わない場合のガード
        m = m[:, :tf.shape(h)[1]]

    # Attention
    pooled = KerasTemporalAttention()(h, m)

    # Head
    z = layers.Dense(256, activation="relu")(pooled)
    z = layers.Dropout(dropout)(z)
    z = layers.Dense(128, activation="relu")(z)
    y = layers.Dense(n_classes, activation="softmax")(z)

    model = keras.Model(inputs=[x_in, m_in], outputs=y)
    opt = keras.optimizers.Adam(learning_rate=KerasConfig.LR)
    model.compile(
        optimizer=opt,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=KerasConfig.LABEL_SMOOTHING),
        metrics=[keras.metrics.CategoricalAccuracy(name="acc")]
    )
    return model
```

> **IMU-only 軽量 CNN(+GRU)** も 1 本用意（2–3 Conv → BiGRU → GAP → Dense）。サブモデルの**境界差**をアンサンブルで活かす想定です。

---

## 6) Keras 学習ルーチン（SGKF + OOF 生成）

### 6.1 入口（Torch と同じく sequence リストを作成）

```python
def train_keras_models(train_df: pl.DataFrame, train_demographics: pl.DataFrame):
    if not KERAS_AVAILABLE:
        print("⚠️ Keras not available. Skip Keras training.")
        return

    os.makedirs(KerasConfig.OUT_DIR, exist_ok=True)

    base_cols = ["sequence_id", "subject", "phase", "gesture"]
    all_cols = train_df.columns
    sensor_cols = (
        [c for c in all_cols if c in Config.ACC_COLS + Config.ROT_COLS]
        + _cols_startswith(all_cols, TOF_PREFIXES)
        + _cols_startswith(all_cols, THM_PREFIXES)
    )
    cols_to_select = base_cols + sensor_cols
    grouped = train_df.select(pl.col(cols_to_select)).group_by("sequence_id", maintain_order=True)

    seq_list, y_list, subj_list, lengths = [], [], [], []
    for _, seq in grouped:
        seq_list.append(seq)
        y_list.append(GESTURE_MAPPER[seq["gesture"][0]])
        subj_list.append(seq["subject"][0])
        lengths.append(len(seq))

    pad_len = decide_pad_len(lengths, KerasConfig.FIXED_PAD_LEN, KerasConfig.PAD_LEN_PERCENTILE)
```

### 6.2 SGKF 分割 → fold ごとに scaler を fit → tf.data で学習

* fold ごとに `tr_frames = [build_frame_features(...) for i in tr_idx]` → `scaler_stats = compute_scaler_stats(tr_frames)`。
* **OOF 確率**を `oof_keras[val_idx, :]` に書き戻す（後の重み最適化に必須）。

擬似コード（要点）：

```python
    n_classes = len(GESTURE_MAPPER)
    cv = StratifiedGroupKFold(n_splits=KerasConfig.N_FOLDS, shuffle=True, random_state=KerasConfig.SEED)
    oof_proba = np.zeros((len(seq_list), n_classes), dtype=np.float32)

    feat_order = list(build_frame_features(seq_list[0]).columns)  # 列順固定

    for fold, (tr_idx, va_idx) in enumerate(cv.split(seq_list, np.array(y_list), np.array(subj_list))):
        print(f"\n--- Keras Fold {fold+1}/{KerasConfig.N_FOLDS} ---")
        # scaler fit
        tr_frames = [build_frame_features(seq_list[i]) for i in tr_idx]
        scaler_stats = compute_scaler_stats(tr_frames)

        # numpy へ事前変換（高速）
        def to_xy(idxs):
            X, M, Y = [], [], []
            for i in idxs:
                x, m = make_keras_tensor(seq_list[i], scaler_stats, pad_len, feat_order)
                X.append(x); M.append(m); Y.append(y_list[i])
            X = np.stack(X).astype(np.float32)       # (N, L, C)
            M = np.stack(M).astype(np.float32)       # (N, L)
            Y = keras.utils.to_categorical(np.array(Y), num_classes=n_classes)
            return X, M, Y

        Xtr, Mtr, Ytr = to_xy(tr_idx)
        Xva, Mva, Yva = to_xy(va_idx)

        # モデル（2ブランチ）
        model = build_keras_two_branch(input_shape=(pad_len, Xtr.shape[-1]), n_classes=n_classes, dropout=KerasConfig.DROPOUT)

        cbs = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=KerasConfig.EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=KerasConfig.REDUCE_LR_PATIENCE, min_lr=1e-5, verbose=1),
            keras.callbacks.ModelCheckpoint(os.path.join(KerasConfig.OUT_DIR, KerasConfig.WEIGHT_TMPL.format(fold)), monitor="val_loss", save_best_only=True, verbose=1),
        ]

        hist = model.fit(
            x={"x": Xtr, "mask": Mtr}, y=Ytr,
            validation_data=({"x": Xva, "mask": Mva}, Yva),
            epochs=KerasConfig.MAX_EPOCHS,
            batch_size=KerasConfig.BATCH_SIZE,
            verbose=2,
            callbacks=cbs,
        )

        # OOF proba
        proba_va = model.predict({"x": Xva, "mask": Mva}, batch_size=KerasConfig.BATCH_SIZE, verbose=0)
        oof_proba[va_idx] = proba_va

        # fold メタ保存
        fold_path = os.path.join(KerasConfig.OUT_DIR, KerasConfig.WEIGHT_TMPL.format(fold))
        state = _load_json(KerasConfig.STATE_JSON, default={})
        best_scores = state.get("best_scores", {})
        best_scores[str(fold)] = float(np.max(hist.history.get("val_acc", [0.0])))
        state["best_scores"] = best_scores
        state["pad_len"] = pad_len
        _save_json(KerasConfig.STATE_JSON, state)

        # fold ごとの bundle entry
        # （最後にまとめて保存するのでリスト化しておく）

    # fold 重み（バリデーション acc などから）を正規化
    best_scores = _load_json(KerasConfig.STATE_JSON, default={}).get("best_scores", {})
    fold_scores = [float(best_scores.get(str(i), 1.0)) for i in range(KerasConfig.N_FOLDS)]
    fold_weights = (np.array(fold_scores) / max(np.sum(fold_scores), 1e-12)).tolist()

    # bundle 保存
    bundle = {
        "pad_len": pad_len,
        "feature_order": feat_order,
        "folds": [
            {"weight": fold_weights[i],
             "scaler_stats": compute_scaler_stats([build_frame_features(seq_list[j]) for j in tr_idx]),  # ここは fold 内で計算済みの scaler_stats を保存するようローカル変数で保持する
             "weight_path": os.path.join(KerasConfig.OUT_DIR, KerasConfig.WEIGHT_TMPL.format(i))}
            for i, (tr_idx, _) in enumerate(cv.split(seq_list, np.array(y_list), np.array(subj_list)))
        ],
        "gesture_mapper": GESTURE_MAPPER,
        "reverse_gesture_mapper": REVERSE_GESTURE_MAPPER,
    }
    joblib.dump(bundle, os.path.join(KerasConfig.OUT_DIR, KerasConfig.BUNDLE_NAME))
    np.save(os.path.join(Config.OUTPUT_PATH, "oof", "oof_keras_proba.npy"), oof_proba)
```

> 実装では `scaler_stats` を fold ループ内で保持しておき、bundle の `folds[i]["scaler_stats"]` に対応させてください（上の擬似では簡略化のため再計算の体にしています）。

---

## 7) Keras 推論関数の追加

### 7.1 ランタイムキャッシュ

```python
_KERAS_RUNTIME = {"bundle": None, "fold_models": []}

def _load_keras_bundle_and_models():
    if _KERAS_RUNTIME["bundle"] is not None:
        return
    bp = EnsembleConfig.KERAS_BUNDLE_PATH
    if not (KERAS_AVAILABLE and os.path.exists(bp)):
        print(f"ℹ️ Keras bundle not found or Keras unavailable: {bp}")
        _KERAS_RUNTIME["bundle"] = None
        return
    bundle = joblib.load(bp)
    _KERAS_RUNTIME["bundle"] = bundle

    if EnsembleConfig.LOAD_KERAS_FOLDS_IN_MEMORY and KERAS_AVAILABLE:
        _KERAS_RUNTIME["fold_models"] = []
        for f in bundle["folds"]:
            model = keras.models.load_model(f["weight_path"], custom_objects={"KerasTemporalAttention": KerasTemporalAttention})
            _KERAS_RUNTIME["fold_models"].append(model)
        print(f"✓ Loaded {len(_KERAS_RUNTIME['fold_models'])} Keras models into memory")
```

### 7.2 予測（確率）

```python
def predict_keras_proba(sequence: pl.DataFrame, demographics: pl.DataFrame) -> np.ndarray | None:
    _load_keras_bundle_and_models()
    bundle = _KERAS_RUNTIME["bundle"]
    if bundle is None: return None

    pad_len = bundle["pad_len"]
    feat_order = bundle["feature_order"]
    frame_df = build_frame_features(sequence).reindex(columns=feat_order).fillna(0)
    n_classes = len(bundle["reverse_gesture_mapper"])
    proba_accum = np.zeros(n_classes, dtype=np.float64)

    for i, f in enumerate(bundle["folds"]):
        stats = f["scaler_stats"]
        x_std = apply_standardize(frame_df, stats).to_numpy(np.float32)
        x_pad, m_pad = pad_and_mask(x_std, pad_len)
        X = {"x": x_pad[None, ...], "mask": m_pad[None, ...]}

        if EnsembleConfig.LOAD_KERAS_FOLDS_IN_MEMORY and _KERAS_RUNTIME["fold_models"]:
            model = _KERAS_RUNTIME["fold_models"][i]
        else:
            model = keras.models.load_model(f["weight_path"], custom_objects={"KerasTemporalAttention": KerasTemporalAttention})
        proba = model.predict(X, verbose=0)[0]
        proba_accum += float(f["weight"]) * proba

    s = proba_accum.sum()
    return proba_accum / s if s > 0 else proba_accum
```

---

## 8) 既存 **predict()** を 3 系統ブレンドに拡張

```python
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    # LGBM
    try: proba_lgbm = predict_lgbm_proba(sequence, demographics)
    except Exception as e:
        print(f"⚠️ LGBM prediction error: {e}"); proba_lgbm = None

    # Torch
    proba_torch = None
    if TORCH_AVAILABLE:
        try: proba_torch = predict_torch_proba(sequence, demographics)
        except Exception as e:
            print(f"⚠️ Torch prediction error: {e}")

    # Keras
    proba_keras = None
    if KERAS_AVAILABLE:
        try: proba_keras = predict_keras_proba(sequence, demographics)
        except Exception as e:
            print(f"⚠️ Keras prediction error: {e}")

    # 合成
    w_l, w_t, w_k = EnsembleConfig.W_LGBM, EnsembleConfig.W_TORCH, EnsembleConfig.W_KERAS
    weights = []
    probas = []
    if proba_lgbm is not None: weights.append(w_l); probas.append(proba_lgbm)
    if proba_torch is not None: weights.append(w_t); probas.append(proba_torch)
    if proba_keras is not None: weights.append(w_k); probas.append(proba_keras)

    if not probas:
        return "Text on phone"  # フォールバック
    W = np.array(weights, dtype=np.float64)
    W = W / max(W.sum(), 1e-12)
    final_proba = np.sum([W[i] * probas[i] for i in range(len(probas))], axis=0)
    final_class = int(np.argmax(final_proba))
    return reverse_gesture_mapper[final_class]
```

---

## 9) OOF の保存と**ブレンド重み最適化**（任意だが強推奨）

### 9.1 LGBM 側：OOF 確率を保存（fold ループで）

* いまは `val_preds = model.predict(X_fold_val)` のみ → `predict_proba` を使って `oof_lgbm[val_idx] = proba` を保存。

```python
oof_lgbm = np.zeros((len(X_train), len(GESTURE_MAPPER)), dtype=np.float32)
# fold 内:
val_proba = model.predict_proba(X_fold_val)  # (n_val, n_classes_local)
# クラス id を global に戻す（既存 predict_lgbm_proba と同様の埋め戻し）
val_proba_full = np.zeros((len(X_fold_val), len(GESTURE_MAPPER)), dtype=np.float32)
for local_j, cls_id in enumerate(model.classes_):
    val_proba_full[:, int(cls_id)] = val_proba[:, local_j]
oof_lgbm[val_idx] = val_proba_full
# 最後に保存
os.makedirs(os.path.join(Config.OUTPUT_PATH, "oof"), exist_ok=True)
np.save(os.path.join(Config.OUTPUT_PATH, "oof", "oof_lgbm_proba.npy"), oof_lgbm)
np.save(os.path.join(Config.OUTPUT_PATH, "oof", "y_true.npy"), y_train)
```

### 9.2 Torch 側：OOF 確率を保存（検証ループで `softmax` を保持）

* 既存バリデーションで `pred` のみ → `prob` を `oof_torch[val_idx]` へ。

### 9.3 Keras 側：前述の `oof_keras` 保存ですでに対応。

### 9.4 重み最適化ユーティリティ（MacroF1/BinaryF1 合成）

```python
def optimize_ensemble_weights(oof_list: list[np.ndarray], y_true: np.ndarray, trials=4096, seed=42):
    rng = np.random.default_rng(seed)
    def score_w(w):
        w = np.maximum(w, 0); w = w / max(w.sum(), 1e-12)
        p = sum(w[i] * oof_list[i] for i in range(len(oof_list)))
        y_pred = np.argmax(p, axis=1)
        bin_f1 = f1_score((y_true <= 7).astype(int), (y_pred <= 7).astype(int), zero_division=0.0)
        macro_f1 = f1_score(np.where(y_true <= 7, y_true, 99), np.where(y_pred <= 7, y_pred, 99), average="macro", zero_division=0.0)
        return 0.5 * (bin_f1 + macro_f1)

    best = (-1, None)
    for _ in range(trials):
        w = rng.random(len(oof_list))
        s = score_w(w)
        if s > best[0]: best = (s, w / w.sum())
    return best  # (score, weights)
```

* トレーニング完了後に `oof_lgbm/torch/keras` を読み込み → `optimize_ensemble_weights([oof_lgbm, oof_torch, oof_keras], y_true)` →
  `EnsembleConfig.W_*` に反映し、`ensemble/weights.json` に保存するフローを追加。

---

## 10) チェックポイント／再開の整備

* `CheckpointConfig` に Keras 用の項目を追加：

```python
class CheckpointConfig:
    # …既存…
    KERAS_STATE_JSON = os.path.join(KerasConfig.OUT_DIR, "keras_state.json")
    KERAS_EARLY_EXIT_IF_BEST_EXISTS = True
```

* 既に最良重みが存在する fold は**学習スキップ**できるよう分岐（Torch と同等）。

---

## 11) トレーニング実行ブロックの改修点

* 既存の LGBM → Torch 学習のあとに **Keras 学習**を追加（条件付き）。
  「Torch → Keras の順」は GPU 共有の都合でどちらでも構いませんが、**メモリ解放**を挟む場合は `tf.keras.backend.clear_session()` を折に触れて呼ぶ。

```python
# ==== Keras training (Always try) ====
if KERAS_AVAILABLE:
    print("\nStarting Keras training...")
    train_keras_models(train_df, train_demographics)
    print("✓ Keras training complete")
else:
    print("ℹ️ Keras not available. Skipping Keras training.")
```

---

## 12) 推論前初期化の拡張

* モデルバンドル読み込み部のログに **Keras bundle** の存在確認を追加。
* `predict()` 内はすでに 3 系統ブレンドに対応済み。

---

## 13) 失敗時フォールバック & ロバスト性

* いずれかの系統が無い/失敗 → **残りの平均**で動作（既に `predict()` にフォールバックあり）。
* 0 除算防止の**正規化**は全箇所で統一（`max(sum, 1e-12)`）。

---

## 14) 追加の小さな実装 Tips

* **乱数固定**：`tf.random.set_seed(Config.SEED)` を Keras 学習前に。
* **GPU メモリ**：`tf.config.experimental.set_memory_growth(gpu, True)` を初期化時に（任意）。
* **Modality Dropout / Time Masking**：Keras では学習時に `tf.numpy_function` などで `(x, mask)` に対してチャネル/時間のゼロ化を入れても良い（まずは無しで OK）。
* **カスタムオブジェクト保存**：`KerasTemporalAttention` を `custom_objects` でロード（上記コード済み）。

---

## 15) 動作確認（最小）

* 既存の**ダミー予測**に Keras を混ぜた動作確認を追加：

  * Keras バンドルが無い場合はログ表示のみで通過。
  * ある場合は `predict_keras_proba` を通して shape/正規化/argmax の安定を確認。

---

## 16) まとめ（変更一覧：実装順チェックリスト）

1. **import**：Keras 条件付き import を追加。
2. **Config**：`KerasConfig`, `EnsembleConfig` に `W_KERAS`, `KERAS_BUNDLE_PATH` 等を追加。
3. **前処理**：既存の DL 共通関数を Keras でも利用（新規関数は `make_keras_tensor` 程度）。
4. **モデル**：`KerasTemporalAttention` と `build_keras_two_branch()`（＋IMU-only 1 本）を追加。
5. **学習**：`train_keras_models()` を実装（SGKF / fold scaler / pad\_len / OOF proba 保存 / bundle 保存）。
6. **推論**：`_KERAS_RUNTIME` / `_load_keras_bundle_and_models()` / `predict_keras_proba()` を追加。
7. **アンサンブル**：`predict()` を 3 系統ブレンドへ拡張（W の正規化・フォールバック）。
8. **OOF**：LGBM/Torch/Keras で OOF 確率を保存する処理を追加。
9. **重み最適化**：`optimize_ensemble_weights()` を追加（任意）→ `weights.json` に反映。
10. **実行ブロック**：Keras 学習の呼び出しを追加。
11. **ログ/再開**：Keras 用 state JSON・学習スキップの分岐を実装。

---

## 設計とのトレース

* **Keras 2ブランチ + BiLSTM/GRU + Attention**、**Keras 多数平均 → Torch と重み付きブレンド**という上位戦略は、共有ノートに一致しています（「LB0.82 ブレンド」「重みだけ変えた版」などの実装思想の一般化）。&#x20;
* **fold ごとの scaler/pad\_len を bundle に保存→推論で完全再現**は、既存 Torch 実装と同じ哲学で、**リーク防止と再現性**を担保します。

---

このタスクリストに沿って最小実装から入れれば、**現行の学習・推論スクリプト一式のまま**、Keras パイプラインを安全に差し込み、**LGBM / Torch / Keras の3系統ブレンド**が動作します。重み最適化まで入れると、提出前の最終伸びしろを取り切りやすくなります。
