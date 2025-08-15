以下に **コード監査の要点 / 精度を落とさずに高速化できる施策 / Torch・Keras の詳細進捗ログ / フレーム特徴のメモリキャッシュ** の実装詳細と、すぐ着手できる **タスクリスト** をまとめました。
（パッチは丸ごと貼り付け可能な形で分割しています）

---

## 1) 監査サマリ（潜在的な問題・改善余地）

**潜在的な不具合/リスク**

- **DL のフレーム特徴を毎バッチ再計算**
  `TorchDataset.__getitem__` で `build_frame_features` を都度呼び出し → 回転/重力除去/フィルタが高コスト。学習の大半が前処理に費やされます（Keras 学習では fold ごとに一括生成しているので比較的良いが、推論時は各 fold で再標準化/パディングを繰り返す）。
- **Torch のマスク処理**
  MaxPool 後も float のまま `mask == 0` 比較しています。数値誤差は小さいが、**bool に明示変換**しておくと堅牢。
- **Torch DataLoader**
  `persistent_workers`/`prefetch_factor` 未設定。逐次ワーカー起動コストが残る。
- **Keras 推論**
  `predict_keras_proba` が fold ごとにモデルロード/標準化/パディングを繰り返し。**フレーム特徴キャッシュ**と**fold モデルの常駐化**で短縮可能（後者は既に `LOAD_KERAS_FOLDS_IN_MEMORY=1` に対応済みだが、フレーム特徴は未キャッシュ）。
- **Torch 学習ログ**
  進捗が epoch 粒度。**バッチ粒度の損失/LR/スループット/GPU メモリ**を出すと不調の早期検知に有効。
- **Keras 学習ログ**
  `verbose=2` のみ。**CSVLogger/JSONL の進捗**や **LR ログ**があると再現と比較が容易。

**精度を落とさずに高速化できる主な施策**

1. **フレーム特徴のメモリ LRU キャッシュ**
   1 回の計算結果（`build_frame_features(sequence_pl)`）を **sequence_id キー**で保持。Torch/Keras/推論の全経路で再利用。
   ※標準化・パディングは軽いので、**「フレーム特徴（未標準化 DataFrame）」をキャッシュ**するのがコスパ良。
2. **Torch DataLoader の効率化**
   `persistent_workers=True`, `prefetch_factor=2〜4` を有効化。
   追加で `pin_memory=True`（既にあり）と組合せ。
3. **Torch 進捗ログ（tqdm + JSONL）**
   ステップごとの loss/LR/スループット/（あれば）GPU メモリを表示・記録。
4. **Keras 進捗ログ**
   `CSVLogger` と軽量な `ProgressCallback` を追加。fold/epoch/val_loss/acc/LR/所要時間を JSONL に追記。
   （任意）**mixed precision** を `KERAS_MP=1` で切替。最終 Dense を `float32` 出力にして数値安定性を確保（精度影響ほぼ無しが、オフにするトグルも用意）。
5. **Keras/Torch 推論のキャッシュ利用**
   `predict_keras_proba` / `predict_torch_proba` で **フレーム特徴キャッシュ**を必ず通す。

---

## 2) 追加実装（コピペ用パッチ）

> **使い方**：下の「PATCH 1→8」を順に貼り込み。既存行の置換はコメントに明記しています。

### PATCH 1: 新しい設定（ログ頻度/キャッシュ制限/MixPrecision トグル）

```python
# === Logging & Cache config (NEW) ===
class LogConfig:
    LOG_EVERY_STEPS = int(os.getenv("LOG_EVERY_STEPS", "50"))  # Torch batch log interval
    SAVE_JSONL = bool(int(os.getenv("SAVE_JSONL", "1")))       # 進捗を JSON Lines でも保存
    OUT_DIR = os.path.join(Config.OUTPUT_PATH, "logs")
    os.makedirs(OUT_DIR, exist_ok=True)

class FrameCacheConfig:
    ENABLE = bool(int(os.getenv("FRAME_CACHE", "1")))
    MAX_ITEMS = int(os.getenv("FRAME_CACHE_MAX_ITEMS", "400"))     # LRU 上限個数
    MAX_BYTES = int(float(os.getenv("FRAME_CACHE_MAX_MB", "800")) * 1024 * 1024)  # ~800MB
    STATS_KEY_MODE = os.getenv("FRAME_CACHE_STATS_KEY", "id")      # "id" or "hash"

class KerasSpeedConfig:
    MIXED_PRECISION = bool(int(os.getenv("KERAS_MP", "0")))  # 1 で有効化（任意）
```

### PATCH 2: フレーム特徴のメモリ LRU キャッシュ

```python
# === Frame Feature LRU Cache (NEW) ===
from collections import OrderedDict
import threading
import time
try:
    from joblib import hash as joblib_hash
except Exception:
    joblib_hash = None

class _FrameFeatureCache:
    def __init__(self, max_items=400, max_bytes=800*1024*1024):
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.size_bytes = 0
        self._cache = OrderedDict()  # key -> (df, bytes)
        self._lock = threading.Lock()

    def _key(self, seq_pl):
        try:
            sid = seq_pl["sequence_id"][0]
            return f"seq:{sid}"
        except Exception:
            # fallback to object id
            return f"obj:{id(seq_pl)}"

    def get(self, seq_pl):
        if not FrameCacheConfig.ENABLE:
            # Bypass cache
            df = build_frame_features(seq_pl)
            df.replace([np.inf, -np.inf], 0, inplace=True)
            df.fillna(0, inplace=True)
            return df

        k = self._key(seq_pl)
        with self._lock:
            if k in self._cache:
                df, b = self._cache.pop(k)
                self._cache[k] = (df, b)  # move to tail (MRU)
                return df

        # miss → 計算
        df = build_frame_features(seq_pl)
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)
        b = int(df.memory_usage(index=True, deep=True).sum())

        with self._lock:
            # evict if needed
            while (len(self._cache) >= self.max_items) or (self.size_bytes + b > self.max_bytes and len(self._cache) > 0):
                _, (ev_df, ev_b) = self._cache.popitem(last=False)
                self.size_bytes -= ev_b
            self._cache[k] = (df, b)
            self.size_bytes += b
        return df

FRAME_CACHE = _FrameFeatureCache(
    max_items=FrameCacheConfig.MAX_ITEMS,
    max_bytes=FrameCacheConfig.MAX_BYTES
)
```

### PATCH 3: TorchDataset でキャッシュを使う & マスクの bool 化

> **置換**: `TorchDataset.__getitem__` 内の最初の 2 行を差し替え、マスクを bool に。

```python
# inside TorchDataset.__getitem__
frame_df = FRAME_CACHE.get(self.sequences[idx])  # ← 置換: build_frame_featuresを直接呼ばない
frame_df = apply_standardize(frame_df, self.scaler_stats)
x = frame_df.to_numpy(dtype=np.float32)
x, m = pad_and_mask(x, self.pad_len)
m = (m > 0.5).astype(np.float32)  # ← 追加: 明示的に0/1へ
y = -1 if self.labels is None else int(self.labels[idx])
return x, m, y
```

### PATCH 4: Torch DataLoader のスループット改善

> **置換**: `train_torch_models` 内 DataLoader 作成部。

```python
dl_tr = torch.utils.data.DataLoader(
    ds_tr,
    batch_size=DLConfig.BATCH_SIZE,
    shuffle=True,
    num_workers=DLConfig.NUM_WORKERS,
    collate_fn=collate_batch,
    pin_memory=True,
    persistent_workers=True,   # ← 追加
    prefetch_factor=2          # ← 追加（2〜4で調整可）
)
dl_va = torch.utils.data.DataLoader(
    ds_va,
    batch_size=DLConfig.BATCH_SIZE,
    shuffle=False,
    num_workers=DLConfig.NUM_WORKERS,
    collate_fn=collate_batch,
    pin_memory=True,
    persistent_workers=True,   # ← 追加
    prefetch_factor=2          # ← 追加
)
```

### PATCH 5: Torch 学習の詳細進捗ログ（tqdm + JSONL）

```python
# === Torch training loop: detailed progress (NEW) ===
from tqdm import tqdm

def _gpu_mem_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0

def _log_jsonl(path, obj):
    if not LogConfig.SAVE_JSONL:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ... train_torch_models 内 epoch ループを差し替え（概ね同じ構造のまま） ...
for epoch in range(epoch_start, DLConfig.MAX_EPOCHS):
    t0 = time.time()
    model.train()
    running = 0.0
    nstep = 0
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    pbar = tqdm(dl_tr, total=len(dl_tr), desc=f"[Torch] fold {fold} epoch {epoch+1}", leave=False)
    for xb, mb, yb in pbar:
        xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=DLConfig.AMP):
            logits = model(xb, mb)
            loss = soft_ce_loss(logits, yb, smoothing=DLConfig.LABEL_SMOOTHING, n_classes=n_classes)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running += float(loss.detach().item())
        nstep += 1
        if nstep % LogConfig.LOG_EVERY_STEPS == 0:
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=f"{running/nstep:.4f}", lr=f"{lr:.2e}")

    # --- valid
    model.eval()
    all_pred, all_true, all_prob = [], [], []
    with torch.no_grad():
        for xb, mb, yb in dl_va:
            xb, mb = xb.to(device), mb.to(device)
            with torch.cuda.amp.autocast(enabled=DLConfig.AMP):
                logits = model(xb, mb)
                prob = torch.softmax(logits, dim=1)
            pred = prob.argmax(dim=1).cpu().numpy()
            all_pred.append(pred); all_true.append(yb.numpy()); all_prob.append(prob.cpu().numpy())
    y_true = np.concatenate(all_true); y_pred = np.concatenate(all_pred); y_prob = np.concatenate(all_prob)
    bF1, mF1, score = compute_torch_metrics(y_true, y_pred)
    epoch_time = time.time() - t0
    max_mem = _gpu_mem_gb()

    print(f"  [fold {fold}] epoch {epoch+1}/{DLConfig.MAX_EPOCHS} "
          f"| loss={running/max(nstep,1):.4f} | score={score:.4f} "
          f"(BinF1={bF1:.4f}, MacroF1={mF1:.4f}) | time={epoch_time:.1f}s | max_mem={max_mem:.2f}GB")

    _log_jsonl(os.path.join(LogConfig.OUT_DIR, "torch_progress.jsonl"), {
        "fold": fold, "epoch": epoch, "train_loss": running/max(nstep,1),
        "bin_f1": bF1, "macro_f1": mF1, "score": score,
        "secs": epoch_time, "max_mem_gb": max_mem,
        "lr": scheduler.get_last_lr()[0]
    })

    # 以降の best 保存 / エポック ckpt 保存は既存と同じ（そのまま）
```

### PATCH 6: Keras 進捗ログ & Mixed Precision（任意）

```python
# === Keras logging and optional mixed precision (NEW) ===
if KERAS_AVAILABLE and KerasSpeedConfig.MIXED_PRECISION:
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("✓ Keras mixed precision enabled (float16 compute / float32 vars)")
    except Exception as e:
        print(f"⚠️ Failed to enable mixed precision: {e}")

class ProgressCallback(keras.callbacks.Callback):
    def __init__(self, fold, out_jsonl):
        super().__init__()
        self.fold = fold
        self.t0 = None
        self.out_jsonl = out_jsonl

    def on_train_begin(self, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        rec = {
            "fold": self.fold,
            "epoch": int(epoch),
            "time_from_start_sec": float(time.time() - self.t0),
            "loss": float(logs.get("loss", 0)),
            "acc": float(logs.get("acc", 0)),
            "val_loss": float(logs.get("val_loss", 0)),
            "val_acc": float(logs.get("val_acc", 0)),
            "lr": float(getattr(self.model.optimizer, "lr", 0.0).numpy() if hasattr(self.model.optimizer, "lr") else 0.0),
        }
        _log_jsonl(self.out_jsonl, rec)

# train_keras_models 内 callbacks に追加
cbs = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=KerasConfig.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=KerasConfig.REDUCE_LR_PATIENCE,
        min_lr=1e-5, verbose=1),
    keras.callbacks.ModelCheckpoint(weight_path, monitor="val_loss", save_best_only=True, verbose=1),
    keras.callbacks.CSVLogger(os.path.join(KerasConfig.OUT_DIR, f"fold{fold:02d}_train.csv"), append=True),
    ProgressCallback(fold, os.path.join(LogConfig.OUT_DIR, "keras_progress.jsonl"))
]
```

> **Mixed Precision 有効時の最終出力** > `build_keras_two_branch` の最後の出力 Dense を **float32** に固定：

```python
# Head 部分の最後だけ dtype="float32" を指定（MP時でも出力をFP32へ）
y = layers.Dense(n_classes, activation="softmax", dtype="float32")(z)
```

### PATCH 7: Keras/Torch 推論でキャッシュ利用

> **置換**: `predict_keras_proba` / `predict_torch_proba` のフレーム特徴生成箇所

```python
# predict_keras_proba 内
frame_df = FRAME_CACHE.get(sequence).reindex(columns=feat_order).fillna(0)  # ← 置換

# predict_torch_proba 内
frame_df = FRAME_CACHE.get(sequence).reindex(columns=feat_order).fillna(0)  # ← 置換
```

### PATCH 8: Keras モデル構築時（MP を使う場合の注意）

> 既に上のパッチで `y = Dense(..., dtype='float32')` に変更済み。
> 追加で学習開始前に以下のログを出すと切り替え確認が容易です。

```python
if KERAS_AVAILABLE:
    print("Keras global dtype policy:", getattr(tf.keras.mixed_precision, "global_policy", lambda: "n/a")())
```

---

## 3) 速度面の期待効果（目安）

- **フレーム特徴 LRU**：各エポックでの **前処理の再計算をほぼゼロ**に。バッチ当たりの前処理時間 → ほぼメモリアクセスのみ。
- **DataLoader 最適化**：ミニバッチ供給の待ち時間を削減（CPU ボトルネックの緩和）。
- **Keras MP**（任意）：GPU が A100/T4 等なら **計算時間短縮**が期待（出力 FP32 化で精度影響を抑制）。

※ メモリ使用量はフレームキャッシュに依存します。デフォルト上限（\~800MB/400 本）内で LRU 退避します。必要なら `FRAME_CACHE_MAX_MB` と `FRAME_CACHE_MAX_ITEMS` を調整してください。

---

## 4) テスト観点（精度を落とさないための確認）

- **再現性**：キャッシュは **未標準化のフレーム特徴**のみを保持。標準化（fold 固有の統計）とパディングは従来どおり実行 → 数値一致するはずです。
- **マスク**：Torch 側で bool 化（`m>0.5`）しただけで意味論は不変。
- **Keras MP**：出力を FP32 に固定、損失/メトリクスも FP32 → 収束・精度は従来と同等を想定。スイッチでいつでも OFF 可能。

---

## 5) 具体的ログ例

**Torch（JSONL 一行）**

```json
{
  "fold": 0,
  "epoch": 3,
  "train_loss": 0.8421,
  "bin_f1": 0.7812,
  "macro_f1": 0.7455,
  "score": 0.7634,
  "secs": 57.2,
  "max_mem_gb": 2.31,
  "lr": 0.00073
}
```

**Keras（JSONL 一行）**

```json
{
  "fold": 1,
  "epoch": 7,
  "time_from_start_sec": 132.5,
  "loss": 1.045,
  "acc": 0.67,
  "val_loss": 0.982,
  "val_acc": 0.69,
  "lr": 0.0005
}
```

---

## 6) タスクリスト（着手順・完了条件つき）

**A. フレーム特徴キャッシュ**

- [ ] `PATCH 1,2` を追加（設定・キャッシュ実装）。
      **完了条件**: 初回学習で `logs/` 未生成でもエラーにならない。
- [ ] `PATCH 3,7` を適用（TorchDataset/Keras 推論でキャッシュ使用）。
      **完了条件**: 2 エポック目以降のバッチ前処理時間が目に見えて短縮（tqdm 体感/プロファイル）。

**B. Torch 学習ログ・データローダ**

- [ ] `PATCH 4` で `persistent_workers` と `prefetch_factor` 有効化。
      **完了条件**: エポックまたぎのワーカー再起動が発生しない（起動ログ/時間短縮）。
- [ ] `PATCH 5` で tqdm & JSONL ロガー導入。
      **完了条件**: `logs/torch_progress.jsonl` に fold/epoch ごとの記録が増えていく。

**C. Keras 学習ログ・高速化**

- [ ] `PATCH 6` を追加（`CSVLogger` + `ProgressCallback`）。
      **完了条件**: `keras_models/foldXX_train.csv` と `logs/keras_progress.jsonl` が作成される。
- [ ] （任意）`KERAS_MP=1` で Mixed Precision を有効化、`build_keras_two_branch` の最終 Dense を FP32 出力へ。
      **完了条件**: 収束挙動が従来と同等、学習時間が短縮（環境に依存）。

**D. 精度・再現性チェック**

- [ ] 変更前後で **OOF スコア**（Binary F1 / Macro F1 / 合成スコア）を比較。
      **完了条件**: 変動が ±0.001 以内、または改善。
- [ ] 乱数 SEED 固定で再学習 → 同一スコア ± わずかな差分を確認。
      **完了条件**: ばらつきが従来範囲内。

**E. 付随メンテ**

- [ ] `README/Competition_Overview.md` へ新しい環境変数とログ/キャッシュの使い方を追記。
      **完了条件**: 実行パラメータ（`FRAME_CACHE_MAX_MB` など）の説明がある。

---

### 付録：環境変数スイッチ一覧（提案）

- `FRAME_CACHE=1`（既定 1）: フレーム特徴 LRU キャッシュ ON/OFF
- `FRAME_CACHE_MAX_MB=800` / `FRAME_CACHE_MAX_ITEMS=400`
- `LOG_EVERY_STEPS=50` / `SAVE_JSONL=1`
- `KERAS_MP=0|1` : Mixed Precision の切替

---

必要に応じて、**キャッシュを標準化後・パディング後のテンソルに拡張**（fold 固有 `stats` をキーに含めた二段キャッシュ）も可能です。ただしメモリ使用量が跳ね上がるため、まずは **未標準化フレーム特徴の LRU** に留めるのがバランス良いです。

このパッチで「進捗ログの可視性」と「前処理の重複計算の削減」による学習速度向上が狙えます。もし「さらに縮めたい」場合は、（環境が許せば）**Keras/Torch のバッチサイズ最適化**や **LightGBM の `force_col_wise=True`**（列方向学習を強制）も検討可能です。
