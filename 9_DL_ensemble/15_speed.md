## 0) ゴール（効果のねらい）

- **時間短縮**：`pad_len=90`、Conv の Pooling+1 段、バッチサイズ拡大、Torch/Keras の EarlyStopping、FrameFeature LRU 強化、前処理の事前計算で **手計測での 1–数倍 短縮狙い**。
- **安定化**：混合精度（Keras）で速度/メモリ両取り、Torch は EarlyStopping と OOM フォールバックでこけにくく。

---

## 1) 設定変更（コンフィグ類）

### 1-1. pad_len を固定 90

**理由**: 時間軸を詰めることで畳み込み/RNN 計算量を直接削減。
**修正**（差分イメージ）:

```python
# === Torch training/inference config ===
class DLConfig:
-    PAD_LEN_PERCENTILE = 95
-    FIXED_PAD_LEN = None
+    PAD_LEN_PERCENTILE = 95
+    FIXED_PAD_LEN = 90  # ← 固定長

# === Keras training/inference config ===
class KerasConfig:
-    FIXED_PAD_LEN = None
+    FIXED_PAD_LEN = 90  # ← 固定長
```

### 1-2. FrameFeature LRU キャッシュ拡大

**理由**: frame-level 特徴（DL 用）の再計算を減らして学習/予測の壁時間を下げる。
**修正**:

```python
class FrameCacheConfig:
-    MAX_ITEMS = int(os.getenv("FRAME_CACHE_MAX_ITEMS", "400"))
-    MAX_BYTES = int(float(os.getenv("FRAME_CACHE_MAX_MB", "800")) * 1024 * 1024)
+    MAX_ITEMS = int(os.getenv("FRAME_CACHE_MAX_ITEMS", "1600"))      # 4x
+    MAX_BYTES = int(float(os.getenv("FRAME_CACHE_MAX_MB", "2048")) * 1024 * 1024)  # 2GB
```

> ※ Kaggle 環境の空きメモリを見ながら `FRAME_CACHE_MAX_MB` は環境変数で調整可能。

### 1-3. バッチサイズ 512（自動フォールバック付き）

**理由**: ステップ回数削減。OOM 時は段階的に下げる。
**修正**:

```python
class DLConfig:
-    BATCH_SIZE = 64
+    BATCH_SIZE = 512
+    ACCUM_STEPS = 1
+    NUM_WORKERS = int(os.getenv("DL_NUM_WORKERS", "2"))  # 0→2 に
```

> Torch 側に OOM フォールバック (後述 3-3) を追加。Keras も以下で 512 に（混合精度前提）。

```python
class KerasConfig:
-    BATCH_SIZE = int(os.getenv("KERAS_BATCH_SIZE", "64"))
+    BATCH_SIZE = int(os.getenv("KERAS_BATCH_SIZE", "512"))
```

### 1-4. Keras 混合精度と積極的 EarlyStopping

**理由**: 速度・メモリ効率向上、無駄 epoch を早めに切る。
**修正**:

```python
class KerasSpeedConfig:
-    MIXED_PRECISION = bool(int(os.getenv("KERAS_MP", "0")))
+    MIXED_PRECISION = bool(int(os.getenv("KERAS_MP", "1")))  # 既定ON

class KerasConfig:
-    EARLY_STOPPING_PATIENCE = int(os.getenv("KERAS_ES_PATIENCE", "8"))
-    REDUCE_LR_PATIENCE = int(os.getenv("KERAS_RLR_PATIENCE", "4"))
+    EARLY_STOPPING_PATIENCE = int(os.getenv("KERAS_ES_PATIENCE", "3"))  # 早めに
+    REDUCE_LR_PATIENCE = int(os.getenv("KERAS_RLR_PATIENCE", "2"))
```

---

## 2) モデル・前処理の構造変更

### 2-1. Torch: Conv の Pooling を +1 段（2→3 段）

**理由**: 時間長をさらに 1/2 倍圧縮し、下流 RNN の計算量とメモリを削減。`pad_len=90` なら 3 回プーリングで T’ ≈ 90/8 ≈ 11。
**修正**（`TimeSeriesNet` 定義に 3 ブロック目を追加、Attention マスクの downsample 回数も 3 回に調整）:

```python
class TimeSeriesNet(nn.Module):
    def __init__(...):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),     # ← 1
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2),     # ← 2
+           nn.Conv1d(256, 256, kernel_size=3, padding=1),
+           nn.BatchNorm1d(256),
+           nn.GELU(),
+           nn.MaxPool1d(2),     # ← 3 (追加)
            nn.Dropout(dropout),
        )
        ...

    def forward(self, x, mask):
        x = x.transpose(1, 2)
        x = self.conv(x)             # (B,256,T')
        x = x.transpose(1, 2)        # (B,T',256)

        m = mask
-       for _ in range(2):
+       for _ in range(3):            # ← ConvのPool回数に合わせて3回downsample
            m = F.max_pool1d(m.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        m = m[:, : x.size(1)]
        ...
```

> **注意**: Pool を増やした分だけマスクのダウンサンプル回数も必ず一致させる（既に実装済だった 2→**3** に変更）。

> （任意）**Keras** 側は既に `conv_block`×3 + MaxPool で 3 段です。さらに+1 段したい場合は `conv_block(h, 256, 3)` をもう 1 発追加し、マスクの downsample ループを **4 回** に変更してください（`pad_len=90` だと T’≈5 になり LSTM/GRU は動きますが情報量・安定性のバランスは要確認）。

### 2-2. Torch: かんたん EarlyStopping を導入

**理由**: ログ上、fold0 は epoch 20–27 あたりで頭打ち。そこで**学習を切り上げ**る。
**実装**（ユーティリティ + ループに組込み）:

```python
# どこか Torch ヘルパの近くに
class EarlyStopper:
    def __init__(self, patience=6, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.bad = 0
    def step(self, metric: float) -> bool:
        if metric > self.best + self.min_delta:
            self.best = metric
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience
```

```python
# train_torch_models() 内、foldごとの学習直前
early = EarlyStopper(
    patience=int(os.getenv("TORCH_ES_PATIENCE", "6")),
    min_delta=float(os.getenv("TORCH_ES_MIN_DELTA", "1e-4")),
)

# 各epochのvalid後
if score > best_score:
    ... # 既存の best 更新処理

if early.step(score):
    print(f"  ↯ Early stopping at epoch {epoch + 1} (best={early.best:.4f})")
    break
```

### 2-3. Torch: 勾配蓄積と OOM フォールバック（**バッチ 512**を守るため）

**理由**: 512 がギリギリな環境での安定性向上。OOM 時に自動で縮退。
**実装の要点**:

- `DLConfig.ACCUM_STEPS` を使って **勾配蓄積** (`loss/ACCUM_STEPS` で backward、`step()` は nstep%ACCUM_STEPS==0 の時のみ)。
- DataLoader 生成を関数化し、**OOM 例外を拾って batch を 512→256→128→64 と段階的に下げ**てリトライ。

**差分例（ポイントのみ）**:

```python
def make_loader(dataset, batch_size):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=DLConfig.NUM_WORKERS,
        collate_fn=collate_batch,
        pin_memory=True,
        persistent_workers=(DLConfig.NUM_WORKERS > 0),
    )
    # prefetch_factor は num_workers>0 の時だけ渡す（Noneは渡さない）
    if DLConfig.NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = 2
    return torch.utils.data.DataLoader(dataset, **kwargs)

# 作成時
bs = DLConfig.BATCH_SIZE
for attempt in [bs, bs//2, bs//4, bs//8]:
    try:
        dl_tr = make_loader(ds_tr, attempt)
        dl_va = make_loader(ds_va, attempt)
        print(f"✓ Using batch_size={attempt}")
        break
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) and attempt >= 64:
            print(f"OOM at batch={attempt}, retrying with {attempt//2}...")
            continue
        else:
            raise
```

> 既存の `prefetch_factor=2 if ... else None` は **PyTorch のバージョンによっては `None` で型エラー**になるため、**引数自体を渡さない**ように修正（潜在バグの部でも触れます）。

---

## 3) Keras: 混合精度・早期打ち切り・Attention マスクの fp16 安全化

### 3-1. 混合精度と ES 設定（上記 1-4 で既に変更）

- `KERAS_MP=1` を既定 ON、ES/ReduceLROnPlateau の patience を詰める。
- さらに `model.compile(..., jit_compile=False)`（Keras3 の XLA は環境で速度が揺れるため既定は **OFF** 推奨。高速化する環境なら ON も検討可）。

### 3-2. Attention のマスク値を half 安全値に

**理由**: 現状 `KerasTemporalAttention` は `-1e9` を埋めており、**fp16 で -inf に飽和 →NaN** のリスク。Torch 側は `-65504`（fp16 最小）に統一している。
**修正**:

```python
class KerasTemporalAttention(layers.Layer):
    def call(self, h, mask):
        logit = self.dense(h)[:, :, 0]
-       logit = tf.where(tf.equal(mask, 1.0),
-                        logit,
-                        tf.fill(tf.shape(logit), tf.constant(-1e9, logit.dtype)))
+       min_val = tf.cast(-65504.0, logit.dtype)  # fp16でも安全な最小域
+       logit = tf.where(tf.equal(mask, 1.0), logit, tf.fill(tf.shape(logit), min_val))
        w = tf.nn.softmax(logit, axis=1)
        ...
```

---

## 4) 前処理の**事前計算**（DL 用 frame 特徴の永続化）

**目的**: 各 fold/epoch で同じシーケンスの frame 特徴を都度計算しているため無駄が多い → **1 シーケンス=1 ファイル**で Parquet 永続化して再利用。

### 4-1. ディレクトリ型ストアを用意

```python
class DLConfig:
-    FRAME_FEATURE_CACHE = os.path.join(Config.OUTPUT_PATH, "frame_features.parquet")
+    FRAME_FEATURE_DIR = os.path.join(Config.OUTPUT_PATH, "frame_features")  # 1seq 1file
```

### 4-2. ビルド関数（学習前に一度だけ）

```python
def build_frame_feature_store(train_df: pl.DataFrame, cols_to_select: list[str]):
    os.makedirs(DLConfig.FRAME_FEATURE_DIR, exist_ok=True)
    grouped = train_df.select(pl.col(cols_to_select)).group_by("sequence_id", maintain_order=True)
    for _, seq in grouped:
        sid = int(seq["sequence_id"][0])
        out = os.path.join(DLConfig.FRAME_FEATURE_DIR, f"{sid}.parquet")
        if os.path.exists(out):
            continue
        df = build_frame_features(seq)
        df.to_parquet(out, index=False)  # pandas の to_parquet
```

> 既存の features.cache（LGBM 用の「集約特徴」）とは別物です。これは **DL の frame-level** 用。

### 4-3. LRU キャッシュに「ディスク → メモリ」ロード経路を追加

```python
class _FrameFeatureCache:
    def _key(self, seq_pl):
        sid = int(seq_pl["sequence_id"][0]) if "sequence_id" in seq_pl.columns else id(seq_pl)
        return f"seq:{sid}"

    def get(self, seq_pl):
        # LRU 命中
        ...
        # miss → まずディスクを探す
        k = self._key(seq_pl)
        try:
            sid = int(seq_pl["sequence_id"][0])
            path = os.path.join(DLConfig.FRAME_FEATURE_DIR, f"{sid}.parquet")
            if os.path.exists(path):
                df = pd.read_parquet(path)
                df.replace([np.inf, -np.inf], 0, inplace=True)
                df.fillna(0, inplace=True)
                ... # LRU に put して返す
                return df
        except Exception:
            pass
        # それでも無ければ計算
        df = build_frame_features(seq_pl)
        ...
        return df
```

### 4-4. いつ作る？

- `main()` の **学習データ読み込み直後**に、DL を回す場合のみ以下を追加：

```python
if RUN_TORCH_TRAINING or RUN_KERAS_TRAINING:
    print("Building frame feature store (if missing)...")
    build_frame_feature_store(train_df, cols_to_select)
```

---

## 5) 実装後の**確認手順（必須）**

- **形状**: Torch/Keras ともに `pad_len=90`、**Conv の Pool 回数とマスクの downsample 回数が一致**していること（Torch=3, Keras=3 のまま/4 にした場合は 4）。
- **時間短縮の主要因**が効いているか：

  - DataLoader の batch 実効値（OOM フォールバックで落ちていないかログに出す）
  - Torch の EarlyStopping が発火しているか（fold ごとに `↯ Early stopping` ログ）
  - FrameFeature LRU のヒット率（必要なら `get()` 内で簡易カウンタ出力）

- **数値**: スコア変動（BinF1/MacroF1）を fold ごとに監視。`score` のピークエポックを記録。
- **Keras**: 混合精度で NaN が出ない（Attention マスク修正の効果）。`val_loss` が素直に落ちるか。
- **OOM**: 学習開始直後に OOM が起きない（起きたらフォールバックで batch=256/128/64 のいずれかで進む）。

---

## 6) 追加で直しておくと良い**潜在的な問題 / 改善点**

1. **DataLoader の `prefetch_factor` を `None` で渡す**件

   - 一部 PyTorch では `num_workers==0` のとき `prefetch_factor` を渡すとエラー。**上の make_loader() のように、`num_workers>0` の時だけ引数に含める**。

2. **Keras TemporalAttention の -1e9 マスク**

   - 混合精度(fp16)で **-inf/NaN の誘発**要因。\*\*上記修正（-65504.0）\*\*を必ず入れる。

3. **Torch 勾配蓄積の未使用**

   - `DLConfig.ACCUM_STEPS` が定義だけされている。上記 2-3 のように **実際の backward/step に反映**させると大バッチ時の安定性が上がる。

4. **Torch の OneCycleLR と EarlyStopping の整合**

   - EarlyStop で終了しても問題はないが、**最後に最良重みは既に保存**済みであることを確認（このコードは best 毎に保存しているので OK）。

5. **Keras `jit_compile`**

   - XLA は環境依存で速くも遅くもなるため、**デフォルトは OFF**。速いとわかったら `model.compile(..., jit_compile=True)` を ON。

6. **最適アンサンブルの試行回数が多い**

   - `optimize_ensemble_weights(trials=4096)` は学習直後にやると時間を食う。**学習中は 1024 程度**に下げるか、後回しに。

7. **OOM 時のログ**

   - OOM は静かに落ちるケースもあるので、**例外メッセージを明示的に表示**（上のリトライログで可視化）。

8. **Keras の `ModelCheckpoint` 保存形式**

   - 既に `.keras` で OK。**Torch との混同に注意**（`.pt`）。

9. **`predict()` のフェイルセーフ**

   - 3 モデル全部失敗時に `"Text on phone"` を返す仕様。必要なら **ログに警告を出す**と追跡しやすい。
