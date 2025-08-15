了解です。**既存の LGBM を壊さず**に、まずは **PyTorch モデルを追加**するための実装タスクリストを、設計の要点と**差し込みコード**（そのままコピペしやすい最小骨子）付きでまとめました。
※フレーム単位（時刻ベース）の ToF/THM は**フレーム内統計（mean/std/min/max/valid_ratio）だけに圧縮**し、**時系列学習は DL 側で**、という方針を採用します（上位ノートの設計に準拠）。DL 側の基本骨子は **Conv → BiLSTM → GRU → Attention** の王道構成です。&#x20;

---

## 0) ゴールと前提

- **ゴール**：既存ノートに **PyTorch 学習・推論・保存/読込**を追加。まずは **LGBM と独立稼働**（競合回避）、のちにアンサンブルへ拡張できる土台。
- **ハード**：Kaggle GPU（T4×2 または P100）。**AMP**、**AdamW + OneCycleLR**、\*\*DataParallel（任意）\*\*に対応。
- **データ設計**：

  - LGBM ＝「**シーケンス集約特徴（1 行/sequence）**」
  - Torch ＝「**フレーム単位特徴（T×C）** → パディング → 時系列モデル」
    同じ生センサから**別系統の表現**を使うため、LGBM と干渉しません。

- **リーク回避**：標準化は **fold の学習側統計のみ**を使用（OOF と整合）。

---

## 1) 追加パラメータ（Config 拡張）

**タスク 1-1**：`Config` に DL 用設定を追加（既存値は変更しない）

```python
# === NEW: Torch training/inference config ===
class DLConfig:
    USE_TORCH = bool(int(os.getenv("USE_TORCH", "0")))  # 0: LGBMのみ, 1: Torch使用
    TORCH_OUT_DIR = os.path.join(Config.OUTPUT_PATH, "torch_models")
    N_FOLDS = Config.N_FOLDS
    SEED = Config.SEED

    # frame-level features
    PAD_LEN_PERCENTILE = 95      # P95 を既定
    FIXED_PAD_LEN = None         # 明示指定したい場合は整数、未指定なら上記PCTLから決める
    FRAME_FEATURE_CACHE = os.path.join(Config.OUTPUT_PATH, "frame_features.parquet")

    # training
    MAX_EPOCHS = 30
    BATCH_SIZE = 64              # OOM時は 32/16 に
    ACCUM_STEPS = 1              # 勾配蓄積で実効バッチ拡張
    LR = 1e-3
    WEIGHT_DECAY = 1e-2
    DROPOUT = 0.2
    LABEL_SMOOTHING = 0.05

    # runtime
    AMP = True
    NUM_WORKERS = 2

    # file names
    BUNDLE_NAME = "torch_bundle.pkl"      # メタ（列順、スケーラ、pad_len等）
    WEIGHT_TMPL = "fold{:02d}.pt"         # 各foldの重み
```

> **設計メモ**
>
> - Torch の保存物は **メタ（バンドル）** と **fold ごとの重み**を分離。
> - これにより、**学習再実行なしで推論**でき、LGBM バンドルと衝突しません。

---

## 2) フレーム単位の特徴テンソル生成（DL 向け）

既存コードの関数を**最大限再利用**して、**時刻 t の特徴ベクトル**を作ります。最小構成（推奨）：

- **IMU**：`linear_acc_{x,y,z}`, `omega_{x,y,z}`, `linear_acc_mag`, `omega_mag`
- **ToF**：`mean/std/min/max/valid_ratio`（1 タイムステップ＝ 5 スカラー）
- **THM**：`mean/std/min/max/valid_ratio`（同上）

**タスク 2-1**：フレーム特徴抽出器を追加

```python
def build_frame_features(sequence: pl.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame of shape (T, C) for DL, with fixed column order.
    """
    seq_df = sequence.to_pandas()
    dt, fs = infer_dt_and_fs(seq_df)

    # IMU 必須列の存在保証
    for c in Config.ACC_COLS:
        if c not in seq_df.columns:
            seq_df[c] = 0.0
    available_rot_cols = [c for c in Config.ROT_COLS if c in seq_df.columns]
    if available_rot_cols:
        rot = build_full_quaternion(seq_df, available_rot_cols)
        rot = handle_quaternion_missing_values(rot)
        rot = fix_quaternion_sign(rot)
    else:
        rot = np.tile(np.array([1.,0.,0.,0.]), (len(seq_df),1))

    acc = seq_df[Config.ACC_COLS].ffill().bfill().to_numpy(dtype=float)
    world_acc = compute_world_acceleration(acc, rot)
    linear_acc = compute_linear_acceleration(world_acc, fs=fs)
    omega = compute_angular_velocity(rot, dt=dt)

    # magnitudes
    lin_mag = np.linalg.norm(linear_acc, axis=1, keepdims=True)
    omg_mag = np.linalg.norm(omega, axis=1, keepdims=True)

    # ToF/THM frame aggregates
    _, mod_cols = detect_modalities(seq_df)
    tof_agg = tof_frame_aggregates(seq_df, mod_cols["tof"]) if mod_cols["tof"] else None
    thm_agg = thermal_frame_aggregates(seq_df, mod_cols["thm"]) if mod_cols["thm"] else None

    feats = {}
    feats["linear_acc_x"] = linear_acc[:,0]
    feats["linear_acc_y"] = linear_acc[:,1]
    feats["linear_acc_z"] = linear_acc[:,2]
    feats["omega_x"] = omega[:,0]
    feats["omega_y"] = omega[:,1]
    feats["omega_z"] = omega[:,2]
    feats["linear_acc_mag"] = lin_mag[:,0]
    feats["omega_mag"] = omg_mag[:,0]

    # ToF
    if tof_agg is not None:
        feats["tof_mean"] = tof_agg["mean"]
        feats["tof_std"]  = tof_agg["std"]
        feats["tof_min"]  = tof_agg["min"]
        feats["tof_max"]  = tof_agg["max"]
        feats["tof_valid_ratio"] = tof_agg["valid_ratio"]
    else:
        for k in ["mean","std","min","max","valid_ratio"]:
            feats[f"tof_{k}"] = np.zeros(len(seq_df))

    # THM
    if thm_agg is not None:
        feats["thm_mean"] = thm_agg["mean"]
        feats["thm_std"]  = thm_agg["std"]
        feats["thm_min"]  = thm_agg["min"]
        feats["thm_max"]  = thm_agg["max"]
        feats["thm_valid_ratio"] = thm_agg["valid_ratio"]
    else:
        for k in ["mean","std","min","max","valid_ratio"]:
            feats[f"thm_{k}"] = np.zeros(len(seq_df))

    frame_df = pd.DataFrame(feats)
    frame_df.replace([np.inf, -np.inf], 0, inplace=True)
    frame_df.fillna(0, inplace=True)
    return frame_df
```

> **根拠**：ToF/THM は**フレーム内統計に圧縮**し、時系列は DL で学習する方針。IMU は**重力除去線形加速度 + 角速度**が主軸。

---

## 3) 前処理の統一（標準化・パディング）

**タスク 3-1**：**fold 内学習サンプル全フレーム**を縦結合して `mu, sigma` を計算（リーク回避）

```python
def compute_scaler_stats(frame_dfs: list[pd.DataFrame]) -> dict[str, tuple[float,float]]:
    # 入力: 学習fold内の全 sequence の frame_df リスト
    concat = pd.concat(frame_dfs, axis=0, ignore_index=True)
    stats = {}
    for c in concat.columns:
        x = concat[c].values.astype(np.float64)
        mu = float(np.mean(x))
        sd = float(np.std(x) + 1e-8)
        stats[c] = (mu, sd)
    return stats

def apply_standardize(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = df.copy()
    for c, (mu, sd) in stats.items():
        if c in out.columns:
            out[c] = (out[c].astype(np.float32) - mu) / sd
    return out
```

**タスク 3-2**：`pad_len` を決める（P95）し、**後詰めパディング**＋**mask**を作成

```python
def decide_pad_len(lengths: list[int], fixed: int|None, pctl:int=95) -> int:
    if fixed is not None: return int(fixed)
    return int(np.percentile(lengths, pctl))

def pad_and_mask(x: np.ndarray, pad_len: int) -> tuple[np.ndarray, np.ndarray]:
    # x: (T,C) -> (pad_len, C), mask: (pad_len,) 1=valid, 0=pad
    T, C = x.shape
    out = np.zeros((pad_len, C), dtype=np.float32)
    msk = np.zeros((pad_len,), dtype=np.float32)
    t = min(T, pad_len)
    out[:t] = x[:t]
    msk[:t] = 1.0
    return out, msk
```

> **注意**：**スケーラは fold ごと**に保存（メタバンドルへ）。**pad_len** も保存して推論で再利用。

---

## 4) Torch データセット/コラテ関数

**タスク 4-1**：`TorchDataset` と `collate_fn` を追加

```python
class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, sequences: list[pl.DataFrame], labels: np.ndarray|None,
                 scaler_stats: dict, pad_len: int):
        self.sequences = sequences
        self.labels = labels
        self.scaler_stats = scaler_stats
        self.pad_len = pad_len

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        frame_df = build_frame_features(self.sequences[idx])
        frame_df = apply_standardize(frame_df, self.scaler_stats)
        x = frame_df.to_numpy(dtype=np.float32)
        x, m = pad_and_mask(x, self.pad_len)
        y = -1 if self.labels is None else int(self.labels[idx])
        return x, m, y

def collate_batch(batch):
    xs, ms, ys = zip(*batch)
    x = torch.from_numpy(np.stack(xs,0))   # (B, T, C)
    m = torch.from_numpy(np.stack(ms,0))   # (B, T)
    y = torch.tensor(ys, dtype=torch.long)
    return x, m, y
```

---

## 5) モデル本体（Conv → BiLSTM → GRU → Attention）

**タスク 5-1**：軽量な汎用モデルを実装（Attention は mask 対応）

```python
import torch, torch.nn as nn, torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, h, mask):
        # h: (B,T,D), mask: (B,T) 1=valid
        logit = self.proj(h).squeeze(-1)            # (B,T)
        logit = logit.masked_fill(mask==0, -1e9)    # padを弾く
        w = torch.softmax(logit, dim=1)             # (B,T)
        pooled = torch.bmm(w.unsqueeze(1), h).squeeze(1)  # (B,D)
        return pooled, w

class TimeSeriesNet(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, hidden: int=128, dropout: float=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(256, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.gru    = nn.GRU(2*hidden, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.attn   = TemporalAttention(2*hidden)
        self.head   = nn.Sequential(
            nn.Linear(2*hidden, 256),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, mask):
        # x: (B,T,C) -> Conv1dは (B,C,T)
        x = x.transpose(1,2)
        x = self.conv(x)           # (B, 256, T')
        x = x.transpose(1,2)       # (B, T', 256)
        # マスクもT'に合わせて間引き（2回MaxPoolのため /4）
        m = mask[:, :x.size(1)]
        h,_ = self.bilstm(x)
        h,_ = self.gru(h)
        pooled, w = self.attn(h, m)
        logits = self.head(pooled)
        return logits
```

> **補足**：Conv 段で 2 回プーリングしているため、mask を**長さに合わせて切るだけ**で十分（pad 部分は 0 なので影響が薄い）。Attention で**明示マスク**して pad の寄与を排除。
> **設計根拠**：Conv→RNN→Attention は上位解や EDA ノートの素案と一致。&#x20;

---

## 6) 学習ループ（SGKF、AMP、OneCycle）

**タスク 6-1**：学習ユーティリティ

```python
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

def soft_ce_loss(logits, targets, smoothing=0.05, n_classes=18):
    with torch.no_grad():
        true_dist = torch.zeros_like(logits).fill_(smoothing/(n_classes-1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    log_prob = F.log_softmax(logits, dim=1)
    return -(true_dist * log_prob).sum(dim=1).mean()

def compute_metrics(y_true, y_pred):
    # Binary: 0-7 vs 8-17（既存と揃える）
    bin_true = (y_true <= 7).astype(int)
    bin_pred = (y_pred <= 7).astype(int)
    binary_f1 = f1_score(bin_true, bin_pred, zero_division=0.0)
    # Macro F1（BFRB内：0..7 のみ評価する現行式に揃えるなら調整可）
    macro_f1 = f1_score(np.where(y_true<=7, y_true, 99),
                        np.where(y_pred<=7, y_pred, 99),
                        average="macro", zero_division=0.0)
    return binary_f1, macro_f1, 0.5*(binary_f1+macro_f1)
```

**タスク 6-2**：Fold 学習（学習・評価・保存）

```python
def train_torch_models(train_df: pl.DataFrame, demo_df: pl.DataFrame, labels: np.ndarray, subjects: np.ndarray):
    # 1) sequence ごとの生データを準備
    base_cols = ["sequence_id","subject","phase","gesture"]
    all_cols = train_df.columns
    sensor_cols = [c for c in all_cols if c in Config.ACC_COLS + Config.ROT_COLS] \
                + _cols_startswith(all_cols, TOF_PREFIXES) \
                + _cols_startswith(all_cols, THM_PREFIXES)
    cols_to_select = base_cols + sensor_cols
    grouped = train_df.select(pl.col(cols_to_select)).group_by("sequence_id", maintain_order=True)

    seq_list, y_list, subj_list, lengths = [], [], [], []
    for seq_id, seq in grouped:
        seq_list.append(seq)
        y_list.append(GESTURE_MAPPER[seq["gesture"][0]])
        subj_list.append(seq["subject"][0])
        lengths.append(len(seq))

    # 2) pad_len 決定
    pad_len = decide_pad_len(lengths, DLConfig.FIXED_PAD_LEN, DLConfig.PAD_LEN_PERCENTILE)
    os.makedirs(DLConfig.TORCH_OUT_DIR, exist_ok=True)

    # 3) SGKF
    cv = StratifiedGroupKFold(n_splits=DLConfig.N_FOLDS, shuffle=True, random_state=DLConfig.SEED)
    fold_weights, models_meta = [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold, (tr_idx, va_idx) in enumerate(cv.split(seq_list, np.array(y_list), np.array(subj_list))):
        # 3.1) スケーラ統計（学習foldのみ）
        tr_frames = [build_frame_features(seq_list[i]) for i in tr_idx]
        scaler_stats = compute_scaler_stats(tr_frames)

        # 3.2) Dataset/DataLoader
        ds_tr = TorchDataset([seq_list[i] for i in tr_idx], np.array(y_list)[tr_idx], scaler_stats, pad_len)
        ds_va = TorchDataset([seq_list[i] for i in va_idx], np.array(y_list)[va_idx], scaler_stats, pad_len)
        dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=DLConfig.BATCH_SIZE, shuffle=True,
                                            num_workers=DLConfig.NUM_WORKERS, collate_fn=collate_batch, pin_memory=True)
        dl_va = torch.utils.data.DataLoader(ds_va, batch_size=DLConfig.BATCH_SIZE, shuffle=False,
                                            num_workers=DLConfig.NUM_WORKERS, collate_fn=collate_batch, pin_memory=True)

        # 3.3) モデル/最適化
        n_classes = len(GESTURE_MAPPER)
        in_ch = tr_frames[0].shape[1]
        model = TimeSeriesNet(in_ch, n_classes, hidden=128, dropout=DLConfig.DROPOUT)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=DLConfig.LR, weight_decay=DLConfig.WEIGHT_DECAY)
        total_steps = DLConfig.MAX_EPOCHS * max(1, len(dl_tr))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=DLConfig.LR,
                                                        total_steps=total_steps, pct_start=0.1, anneal_strategy="cos")
        scaler = torch.cuda.amp.GradScaler(enabled=DLConfig.AMP)

        best_score, best_path = -1.0, os.path.join(DLConfig.TORCH_OUT_DIR, DLConfig.WEIGHT_TMPL.format(fold))
        for epoch in range(DLConfig.MAX_EPOCHS):
            # --- train ---
            model.train()
            for xb, mb, yb in dl_tr:
                xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=DLConfig.AMP):
                    logits = model(xb, mb)
                    loss = soft_ce_loss(logits, yb, smoothing=DLConfig.LABEL_SMOOTHING, n_classes=n_classes)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                scheduler.step()

            # --- valid ---
            model.eval()
            all_pred, all_true = [], []
            with torch.no_grad():
                for xb, mb, yb in dl_va:
                    xb, mb = xb.to(device), mb.to(device)
                    with torch.cuda.amp.autocast(enabled=DLConfig.AMP):
                        logits = model(xb, mb); prob = torch.softmax(logits, dim=1)
                    pred = prob.argmax(dim=1).cpu().numpy()
                    all_pred.append(pred); all_true.append(yb.numpy())
            y_true = np.concatenate(all_true); y_pred = np.concatenate(all_pred)
            bF1, mF1, score = compute_metrics(y_true, y_pred)
            if score > best_score:
                best_score = score
                torch.save({"state_dict": model.state_dict(),
                            "in_ch": in_ch, "n_classes": n_classes}, best_path)

        fold_weights.append(best_score)
        # fold メタ（scaler, pad_len）を保存用に保持
        models_meta.append({"scaler_stats": scaler_stats, "weight_path": best_path})

    # 4) メタバンドルを保存
    denom = max(float(np.sum(fold_weights)), 1e-12)
    fold_w = (np.array(fold_weights) / denom).tolist()
    bundle = {
        "pad_len": pad_len,
        "feature_order": list(tr_frames[0].columns),
        "folds": [{"weight": fold_w[i], **models_meta[i]} for i in range(DLConfig.N_FOLDS)],
        "gesture_mapper": GESTURE_MAPPER,
        "reverse_gesture_mapper": REVERSE_GESTURE_MAPPER,
    }
    joblib.dump(bundle, os.path.join(DLConfig.TORCH_OUT_DIR, DLConfig.BUNDLE_NAME))
    print("✓ Torch training done. Saved to:", DLConfig.TORCH_OUT_DIR)
```

> **実装 Tips（Kaggle GPU）**
>
> - T4×2 の場合は `DataParallel` が有効になる設計。P100 でも AMP は効きます。
> - OOM 時は **BATCH_SIZE を下げる / ACCUM_STEPS を上げる** / Conv のチャネルを削る。
> - `num_workers=2` 程度が安定（I/O 過多を避ける）。

---

## 7) 推論（Torch 単独）

**タスク 7-1**：**Torch 用の predict 関数**を追加（LGBM の `predict` とは別名）

```python
def predict_torch(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    bundle_path = os.path.join(DLConfig.TORCH_OUT_DIR, DLConfig.BUNDLE_NAME)
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Torch bundle not found: {bundle_path}")
    bundle = joblib.load(bundle_path)
    pad_len = bundle["pad_len"]; feat_order = bundle["feature_order"]

    # frame features -> standardize -> pad
    frame_df = build_frame_features(sequence)
    # 列順を合わせる（訓練時の順）
    frame_df = frame_df.reindex(columns=feat_order).fillna(0)
    # fold ごとに scaler が違う点に注意
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_classes = len(bundle["reverse_gesture_mapper"])
    proba_accum = np.zeros(n_classes, dtype=np.float64)

    for f in bundle["folds"]:
        stats = f["scaler_stats"]
        x_std = apply_standardize(frame_df, stats).to_numpy(np.float32)
        x_pad, m_pad = pad_and_mask(x_std, pad_len)
        xb = torch.from_numpy(x_pad[None, ...]).to(device)
        mb = torch.from_numpy(m_pad[None, ...]).to(device)

        ckpt = torch.load(f["weight_path"], map_location=device)
        model = TimeSeriesNet(in_ch=ckpt["in_ch"], num_classes=ckpt["n_classes"], hidden=128, dropout=DLConfig.DROPOUT)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model = model.to(device); model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=DLConfig.AMP):
            prob = torch.softmax(model(xb, mb), dim=1).cpu().numpy()[0]
        proba_accum += f["weight"] * prob

    final_cls = int(np.argmax(proba_accum))
    return bundle["reverse_gesture_mapper"][final_cls]
```

> **ポイント**
>
> - **fold 個別スケーラ**を都度適用してから推論（訓練時と完全一致）。
> - **列順**・**pad_len**もバンドルから復元。

---

## 8) 既存コードとの「非競合」統合

**タスク 8-1**：`predict` の切替は **環境変数**で制御（デフォルトは LGBM のまま）

```python
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    if DLConfig.USE_TORCH:
        return predict_torch(sequence, demographics)
    # 既存の LGBM 推論（そのまま）
    try:
        raw_features = extract_features(sequence, demographics)
        X = align_features_for_inference(raw_features, feature_names)
        n_classes_global = len(reverse_gesture_mapper)
        proba_accum = np.zeros(n_classes_global, dtype=float)
        for i, model in enumerate(models):
            proba = model.predict_proba(X)[0]
            proba_full = np.zeros(n_classes_global, dtype=float)
            for local_idx, cls_id in enumerate(model.classes_):
                proba_full[int(cls_id)] = proba[local_idx]
            proba_accum += proba_full * float(fold_weights[i])
        final_class = int(np.argmax(proba_accum))
        return reverse_gesture_mapper[final_class]
    except Exception as e:
        print(f"LGBM Prediction error: {e}")
        return "Text on phone"
```

> **効果**：**デフォルト挙動は不変（LGBM）**。Torch を使いたい時だけ `USE_TORCH=1` をセット。既存の学習・推論・サーバ初期化フローに**一切の破壊的変更がありません**。

---

## 9) 学習ブロックの挿入位置

**タスク 9-1**：LGBM 学習の後段、あるいは `RUN_TRAINING` ブロックの直後に「Torch 学習（任意）」を追加

```python
# ==== (optional) Torch training ====
if os.getenv("TORCH_TRAIN", "0") == "1":
    print("Starting Torch training...")
    # 既存の train_df, train_demographics から labels/subjects を流用
    # y_train / subjects は LGBM ブロックで定義済みならそれを使う
    train_torch_models(train_df, train_demographics, y_train, subjects)
    print("✓ Torch training complete")
```

> **注意**：Torch 学習は**スイッチ式**。Kaggle で両方回す／片方だけ回す、を柔軟に選べます。

---

## 10) 追加保存物とディレクトリ

- `/<OUTPUT>/torch_models/torch_bundle.pkl`（**必須メタ**：`pad_len`, `feature_order`, 各 fold の `scaler_stats` & `weight_path`, `gesture_mapper`）
- `/<OUTPUT>/torch_models/fold00.pt` … `fold{K-1}.pt`（**各 fold 重み**）

> 既存の `imu_lgbm_model.pkl`（LGBM バンドル）と**別名ディレクトリ**に分離 → **競合なし**。

---

## 11) 追加の運用 Tips（T4×2/P100 向け）

- **メモリ**：OOM 時は

  1. `BATCH_SIZE` を下げる
  2. `ACCUM_STEPS` を 2–4 に
  3. Conv チャネルを 128→96 に縮小

- **高速化**：`pin_memory=True`、`non_blocking=True`（必要に応じて）で転送効率を改善。
- **汎化**：軽い **タイムマスク（ランダムに数十フレーム 0）**、**チャネル Dropout** を DataLoader 内で確率適用するとロバスト化。
- **評価**：LGBM と同じ **Binary F1 と Macro F1** を毎 epoch 検証して保存ルールを統一 → 後のアンサンブル重み学習が容易。

---

## 12) 最後に（次の一手）

- まずは **Torch 単独推論**が安定することを確認（`USE_TORCH=1`）。
- 安定後、**Torch（確率）と LGBM（確率）を重み平均**する `predict_ensemble` を 1 関数追加（`ENSEMBLE_WEIGHTS="lgbm:0.4,torch:0.6"` のような env で可変）。
- ToF/THM を**2 ブランチ化**（Conv を IMU/ToF+THM で分ける）や、**SpecAug 風の時間/チャネルマスク**、**EMA** 追加は、上記土台で容易に拡張可能です。

---

### チェックリスト（実装順）

1. `DLConfig` 追加（§1）
2. `build_frame_features` 追加（§2）
3. 標準化 & パディング関数（§3）
4. `TorchDataset` と `collate_fn`（§4）
5. `TimeSeriesNet`（§5）
6. 学習ループ `train_torch_models`（§6）
7. `predict_torch`（§7）
8. 既存 `predict` を env 切替対応（§8）
9. Torch 学習ブロックを**スイッチ式**で追加（§9）
10. 保存/読込のディレクトリ設計（§10）

このタスクリストとスケルトンをそのまま差し込めば、**LGBM を維持したまま** PyTorch モデルの学習〜推論が追加できます。必要なら、この骨子をあなたのノートに**差分パッチ形式**で書き起こします。
