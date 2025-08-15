以下の変更で **学習の途中再開（resume）** を LightGBM（fold 単位）と PyTorch（epoch ＋ fold 単位）に実装しました。加えて、**特徴量のキャッシュ**、**トーチの Attention マスクの不整合**なども直しています。
ブレンド比（LGBM\:Torch=0.4:0.6）は、これまで共有いただいたノートの方針に合わせています。 また、Torch 側は SGKF・OneCycleLR・AdamW を前提にした設計で、こちらも共有ノートの流儀に沿っています。

---

## 変更点（要約）

1. **チェックポイント & 再開（LGBM / Torch）**

- LGBM: **fold ごとにモデルを保存**し、途中終了しても**次回は未完 fold から再開**します（scikit の `LGBMClassifier` は fold 内での中断再開が難しいため、fold 単位の再開にしています）。
- Torch: **各 fold・各 epoch の “last” チェックポイント**（モデル・最適化器・スケジューラ・AMP ステート）を保存。**同じ fold/epoch から再開**できます。すでに最良重み（`foldXX.pt`）が存在する fold は**スキップ**も可能。

2. **特徴量のキャッシュ**

- 学習前処理で作る `X_train, y_train, subjects, tof_grid_map` を **joblib 1 ファイル**に保存・再利用。長い特徴量抽出を何度もやらずに済みます。

3. **Attention マスクの形状不整合を修正（Torch）**

- Conv→MaxPool(×2)で時系列長が 1/4 になるのに、元の `mask` をただスライスしていた問題を**MaxPool1d でダウンサンプリング**して整合をとるよう修正。

4. **付随の堅牢化**

- JSON ヘルパ／ディレクトリ作成の徹底、例外時のフェイルセーフ、出力ログの明確化など。

---

## 使い方（環境変数）

- `CKPT_DIR`（省略時 `/kaggle/working/checkpoints`） … LGBM 用チェックポイント保存先
- `RESUME`（既定 `1`） … チェックポイントがあれば**再開**
- `USE_TORCH`（既定 `0`） … 1 にすると Torch 学習も走ります
- `RESUME_TORCH`（既定 `1`） … Torch の epoch 再開を有効化
- `MODEL_PATH` … ここが **指定されていれば学習をスキップ**し、そのモデル束を使用（既存仕様のまま）

> ※ LGBM 学習が**すべての fold を完了**すると、従来どおり最終 `pkl` バンドルを生成します（途中状態では生成しません）。
> ※ Torch は fold ごとの最良重み（`foldXX.pt`）を保存し、最後にバンドル（`torch_bundle.pkl`）を組み立てます。既存の最良重みがあれば fold をスキップしても可。

---

## 追加／修正コード（そのまま貼り付けて OK）

### 1) 【追加】Checkpoint 設定とユーティリティ

**（`EnsembleConfig` の下あたりに追加）**

```python
# === NEW: Checkpoint/Resume configuration & helpers ===
import json

class CheckpointConfig:
    CKPT_DIR = os.path.join(Config.OUTPUT_PATH, os.getenv("CKPT_DIR", "checkpoints"))
    RESUME = bool(int(os.getenv("RESUME", "1")))   # 1: 再開する
    FEATURES_CACHE = os.path.join(CKPT_DIR, "train_features.joblib")
    LGBM_FOLD_TMPL = "lgbm_fold{:02d}.pkl"
    LGBM_STATE_JSON = os.path.join(CKPT_DIR, "lgbm_state.json")
    TORCH_EPOCH_CKPT_TMPL = os.path.join(DLConfig.TORCH_OUT_DIR, "fold{:02d}_last.pth")
    TORCH_STATE_JSON = os.path.join(DLConfig.TORCH_OUT_DIR, "torch_state.json")
    SKIP_TORCH_FOLD_IF_BEST_EXISTS = True  # 既に最良重みがあれば fold をスキップ

def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)

def _load_json(path, default=None):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {} if default is None else default

os.makedirs(CheckpointConfig.CKPT_DIR, exist_ok=True)
```

---

### 2) 【差し替え（小修正）】Torch Attention マスクの整合

**（`TimeSeriesNet.forward` のマスク処理部分を以下に差し替え）**

```python
        def forward(self, x, mask):
            # x: (B,T,C) -> Conv1dは (B,C,T)
            x = x.transpose(1, 2)
            x = self.conv(x)  # (B, 256, T')
            x = x.transpose(1, 2)  # (B, T', 256)

            # === FIX: マスクも MaxPool と同等に2回ダウンサンプリングして T' に整合させる ===
            m = mask  # (B, T)
            for _ in range(2):  # conv 内の MaxPool1d を2回適用しているため
                m = F.max_pool1d(m.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
            m = m[:, : x.size(1)]  # 念のため長さを一致させる

            h, _ = self.bilstm(x)
            h, _ = self.gru(h)
            pooled, w = self.attn(h, m)
            logits = self.head(pooled)
            return logits
```

---

### 3) 【差し替え】Torch 学習（fold/epoch チェックポイント＋再開対応）

**（`if TORCH_AVAILABLE:` ブロック内の `train_torch_models` を丸ごと置換）**

```python
    def train_torch_models(train_df: pl.DataFrame, train_demographics: pl.DataFrame):
        base_cols = ["sequence_id", "subject", "phase", "gesture"]
        all_cols = train_df.columns
        sensor_cols = (
            [c for c in all_cols if c in Config.ACC_COLS + Config.ROT_COLS]
            + _cols_startswith(all_cols, TOF_PREFIXES)
            + _cols_startswith(all_cols, THM_PREFIXES)
        )
        cols_to_select = base_cols + sensor_cols
        grouped = train_df.select(pl.col(cols_to_select)).group_by(
            "sequence_id", maintain_order=True
        )

        seq_list, y_list, subj_list, lengths = [], [], [], []
        for _, seq in grouped:
            seq_list.append(seq)
            y_list.append(GESTURE_MAPPER[seq["gesture"][0]])
            subj_list.append(seq["subject"][0])
            lengths.append(len(seq))

        pad_len = decide_pad_len(lengths, DLConfig.FIXED_PAD_LEN, DLConfig.PAD_LEN_PERCENTILE)
        os.makedirs(DLConfig.TORCH_OUT_DIR, exist_ok=True)

        cv = StratifiedGroupKFold(
            n_splits=DLConfig.N_FOLDS, shuffle=True, random_state=DLConfig.SEED
        )
        fold_weights, models_meta = [], []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_state = _load_json(CheckpointConfig.TORCH_STATE_JSON, default={})

        for fold, (tr_idx, va_idx) in enumerate(
            cv.split(seq_list, np.array(y_list), np.array(subj_list))
        ):
            print(f"\n--- Torch Fold {fold + 1}/{DLConfig.N_FOLDS} ---")
            best_path = os.path.join(DLConfig.TORCH_OUT_DIR, DLConfig.WEIGHT_TMPL.format(fold))

            # 既に最良重みがあれば学習をスキップ（任意）
            if (CheckpointConfig.SKIP_TORCH_FOLD_IF_BEST_EXISTS
                and os.path.exists(best_path)):
                print(f"✓ Found existing best weights for fold {fold} at {best_path} — skip training")
                best_score = float(_load_json(CheckpointConfig.TORCH_STATE_JSON, {}).get("best_scores", {}).get(str(fold), -1.0))
                if best_score < 0:
                    # スコアが未記録でも重みは利用可能。暫定で 1.0 を採用
                    best_score = 1.0
                fold_weights.append(best_score)
                # スケーラ統計・メタは再構築して保存に必要
                tr_frames = [build_frame_features(seq_list[i]) for i in tr_idx]
                scaler_stats = compute_scaler_stats(tr_frames)
                models_meta.append({"scaler_stats": scaler_stats, "weight_path": best_path})
                continue

            # === fold固有のスケーラ統計
            tr_frames = [build_frame_features(seq_list[i]) for i in tr_idx]
            scaler_stats = compute_scaler_stats(tr_frames)

            ds_tr = TorchDataset(
                [seq_list[i] for i in tr_idx], np.array(y_list)[tr_idx], scaler_stats, pad_len
            )
            ds_va = TorchDataset(
                [seq_list[i] for i in va_idx], np.array(y_list)[va_idx], scaler_stats, pad_len
            )
            dl_tr = torch.utils.data.DataLoader(
                ds_tr, batch_size=DLConfig.BATCH_SIZE, shuffle=True,
                num_workers=DLConfig.NUM_WORKERS, collate_fn=collate_batch, pin_memory=True
            )
            dl_va = torch.utils.data.DataLoader(
                ds_va, batch_size=DLConfig.BATCH_SIZE, shuffle=False,
                num_workers=DLConfig.NUM_WORKERS, collate_fn=collate_batch, pin_memory=True
            )

            n_classes = len(GESTURE_MAPPER)
            in_ch = tr_frames[0].shape[1]
            model = TimeSeriesNet(in_ch, n_classes, hidden=128, dropout=DLConfig.DROPOUT)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=DLConfig.LR, weight_decay=DLConfig.WEIGHT_DECAY
            )
            total_steps = DLConfig.MAX_EPOCHS * max(1, len(dl_tr))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=DLConfig.LR, total_steps=total_steps,
                pct_start=0.1, anneal_strategy="cos",
            )
            scaler = torch.cuda.amp.GradScaler(enabled=DLConfig.AMP)

            # === Resume (epoch 再開)
            epoch_start, best_score = 0, -1.0
            last_ckpt_path = CheckpointConfig.TORCH_EPOCH_CKPT_TMPL.format(fold)
            if bool(int(os.getenv("RESUME_TORCH", "1"))) and os.path.exists(last_ckpt_path):
                print(f"↻ Resuming fold {fold} from epoch checkpoint: {last_ckpt_path}")
                ckpt = torch.load(last_ckpt_path, map_location=device)
                state = ckpt["model_state"]
                if any(k.startswith("module.") for k in state.keys()):
                    state = {k.replace("module.", "", 1): v for k, v in state.items()}
                model.load_state_dict(state, strict=True)
                optimizer.load_state_dict(ckpt["optim_state"])
                scheduler.load_state_dict(ckpt["sched_state"])
                scaler.load_state_dict(ckpt["scaler_state"])
                epoch_start = int(ckpt["epoch"]) + 1
                best_score = float(ckpt.get("best_score", -1.0))

            # === Train
            for epoch in range(epoch_start, DLConfig.MAX_EPOCHS):
                model.train()
                for xb, mb, yb in dl_tr:
                    xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=DLConfig.AMP):
                        logits = model(xb, mb)
                        loss = soft_ce_loss(logits, yb, smoothing=DLConfig.LABEL_SMOOTHING, n_classes=n_classes)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                # --- valid
                model.eval()
                all_pred, all_true = [], []
                with torch.no_grad():
                    for xb, mb, yb in dl_va:
                        xb, mb = xb.to(device), mb.to(device)
                        with torch.cuda.amp.autocast(enabled=DLConfig.AMP):
                            logits = model(xb, mb)
                            prob = torch.softmax(logits, dim=1)
                        pred = prob.argmax(dim=1).cpu().numpy()
                        all_pred.append(pred)
                        all_true.append(yb.numpy())
                y_true = np.concatenate(all_true)
                y_pred = np.concatenate(all_pred)
                bF1, mF1, score = compute_torch_metrics(y_true, y_pred)
                print(f"  [fold {fold}] epoch {epoch+1}/{DLConfig.MAX_EPOCHS} | score={score:.4f} (BinF1={bF1:.4f}, MacroF1={mF1:.4f})")

                # Save best
                if score > best_score:
                    best_score = score
                    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                    torch.save({"state_dict": state, "in_ch": in_ch, "n_classes": n_classes}, best_path)
                    print(f"  ↳ New best! saved: {best_path}")

                # Save epoch checkpoint
                torch.save({
                    "epoch": epoch,
                    "model_state": (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()),
                    "optim_state": optimizer.state_dict(),
                    "sched_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "best_score": best_score,
                    "best_path": best_path,
                }, last_ckpt_path)

            fold_weights.append(best_score)
            models_meta.append({"scaler_stats": scaler_stats, "weight_path": best_path})

            # Torch state json 更新
            ts = _load_json(CheckpointConfig.TORCH_STATE_JSON, default={})
            bs = ts.get("best_scores", {})
            bs[str(fold)] = float(best_score)
            ts["best_scores"] = bs
            ts["pad_len"] = pad_len
            _save_json(CheckpointConfig.TORCH_STATE_JSON, ts)

        # 重み正規化＋バンドル保存
        denom = max(float(np.sum(fold_weights)), 1e-12)
        fold_w = (np.array(fold_weights) / denom).tolist()
        bundle = {
            "pad_len": pad_len,
            "feature_order": list(tr_frames[0].columns),
            "folds": [{"weight": fold_w[i], **models_meta[i]} for i in range(len(models_meta))],
            "gesture_mapper": GESTURE_MAPPER,
            "reverse_gesture_mapper": REVERSE_GESTURE_MAPPER,
        }
        joblib.dump(bundle, os.path.join(DLConfig.TORCH_OUT_DIR, DLConfig.BUNDLE_NAME))
        print("✓ Torch training done. Saved to:", DLConfig.TORCH_OUT_DIR)
```

---

### 4) 【差し替え】特徴量のキャッシュ & LGBM CV チェックポイント

**（`if RUN_TRAINING:` ブロック内の “FEATURE EXTRACTION ～ CV TRAINING ～ SAVE MODEL BUNDLE” 部分を下記に置き換え）**

```python
    # ------------------ FEATURE EXTRACTION with CACHE ------------------
    print("Extracting features for training sequences (with cache)...")

    features_cache_path = CheckpointConfig.FEATURES_CACHE
    cached = CheckpointConfig.RESUME and os.path.exists(features_cache_path)

    if cached:
        cache = joblib.load(features_cache_path)
        X_train = cache["X_train"]
        y_train = cache["y_train"]
        subjects = cache["subjects"]
        tof_grid_map = cache.get("tof_grid_map", None)
        print(f"✓ Loaded cached features: {X_train.shape} from {features_cache_path}")
    else:
        train_features_list = []
        train_labels = []
        train_subjects = []

        unique_sequences = train_df["sequence_id"].unique()
        n_sequences = len(unique_sequences)
        print(f"Total sequences to process: {n_sequences}")

        train_sequences = train_df.select(pl.col(cols_to_select)).group_by(
            "sequence_id", maintain_order=True
        )

        for i, (sequence_id, sequence_data) in enumerate(train_sequences):
            if i % 1000 == 0:
                print(f"Processing sequence {i + 1}/{n_sequences}")

            subject_id = sequence_data["subject"][0]
            subject_demographics = train_demographics.filter(pl.col("subject") == subject_id)

            features = extract_features(sequence_data, subject_demographics)
            train_features_list.append(features)

            gesture = sequence_data["gesture"][0]
            label = GESTURE_MAPPER[gesture]
            train_labels.append(label)
            train_subjects.append(subject_id)

        assert len(train_features_list) == n_sequences, (
            f"Feature extraction failed: {len(train_features_list)} != {n_sequences}"
        )
        print(f"✓ Successfully processed all {n_sequences} sequences")

        X_train = pd.concat(train_features_list, ignore_index=True)
        y_train = np.array(train_labels)
        subjects = np.array(train_subjects)

        print(f"✓ Features extracted: {X_train.shape}")
        print(f"✓ Number of classes: {len(np.unique(y_train))}")

        # Cleaning / standardize dtype
        print("Cleaning and standardizing features...")
        X_train = X_train.reindex(columns=sorted(X_train.columns))
        X_train.replace([np.inf, -np.inf], 0, inplace=True)
        X_train.fillna(0, inplace=True)
        X_train = X_train.astype(np.float32)

        # Validate features
        print("Validating extracted features...")
        validate_features(X_train, verbose=True)

        joblib.dump(
            {"X_train": X_train, "y_train": y_train, "subjects": subjects, "tof_grid_map": tof_grid_map},
            features_cache_path
        )
        print(f"✓ Saved features cache to {features_cache_path}")

    # ------------------ CV TRAINING with CHECKPOINT ------------------
    print("Training LightGBM models with cross-validation (with checkpoint)...")

    cv = StratifiedGroupKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    models, cv_scores = [], []

    # 既存状態の読み込み
    lgbm_state = _load_json(CheckpointConfig.LGBM_STATE_JSON, default={
        "model_paths": {}, "cv_scores": {}, "completed_folds": []
    })
    completed = set(int(k) for k in lgbm_state.get("model_paths", {}).keys())

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train, subjects)):
        model_path = os.path.join(CheckpointConfig.CKPT_DIR, CheckpointConfig.LGBM_FOLD_TMPL.format(fold))

        if CheckpointConfig.RESUME and (fold in completed) and os.path.exists(model_path):
            print(f"↻ Resuming: loading fold {fold} model from {model_path}")
            model = joblib.load(model_path)
            models.append(model)
            cv_scores.append(float(lgbm_state["cv_scores"].get(str(fold), 0.0)))
            continue

        print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")
        X_fold_train = X_train.iloc[train_idx].reset_index(drop=True).astype(np.float32)
        X_fold_val   = X_train.iloc[val_idx].reset_index(drop=True).astype(np.float32)
        y_fold_train = y_train[train_idx]
        y_fold_val   = y_train[val_idx]

        if Config.USE_MODALITY_DROPOUT:
            print(f"Applying modality dropout with p={Config.MODALITY_DROPOUT_PROB}")
            X_fold_train = apply_modality_dropout(
                X_fold_train, dropout_prob=Config.MODALITY_DROPOUT_PROB, seed=Config.SEED + fold
            )

        print(f"Train size: {len(X_fold_train)}, Val size: {len(X_fold_val)}")
        model = LGBMClassifier(**Config.LGBM_PARAMS)

        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)], eval_names=["valid"], eval_metric="multi_logloss",
            callbacks=[log_evaluation(period=50),
                       early_stopping(stopping_rounds=100, verbose=True)]
        )

        # Fold 保存（checkpoint）
        joblib.dump(model, model_path)
        print(f"✓ Saved fold {fold} model to {model_path}")

        models.append(model)
        val_preds = model.predict(X_fold_val)

        binary_f1 = f1_score(np.where(y_fold_val <= 7, 1, 0),
                             np.where(val_preds <= 7, 1, 0), zero_division=0.0)
        macro_f1 = f1_score(np.where(y_fold_val <= 7, y_fold_val, 99),
                            np.where(val_preds <= 7, val_preds, 99),
                            average="macro", zero_division=0.0)
        score = 0.5 * (binary_f1 + macro_f1)
        cv_scores.append(score)
        print(f"Fold {fold + 1} Score: {score:.4f} (Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f})")

        # JSON 状態更新
        lgbm_state["model_paths"][str(fold)] = model_path
        lgbm_state["cv_scores"][str(fold)] = float(score)
        lgbm_state["completed_folds"] = sorted(list(set(lgbm_state["model_paths"].keys())), key=int)
        _save_json(CheckpointConfig.LGBM_STATE_JSON, lgbm_state)

    print("\n✓ Cross-validation complete!")
    print(f"Overall CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # ------------------ SAVE MODEL BUNDLE ------------------
    RUNTIME_MODEL_PATH = save_model_bundle(
        models=models,
        X_train=X_train,
        cv_scores=cv_scores,
        output_dir=Config.OUTPUT_PATH,
        filename=Config.MODEL_FILENAME,
        tof_grid_map=tof_grid_map,
    )

    # 特徴量重要度の保存（従来どおり）
    feature_importance = pd.DataFrame(
        {"feature": X_train.columns,
         "importance": np.mean([m.feature_importances_ for m in models], axis=0)}
    ).sort_values("importance", ascending=False)
    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20))
    feature_importance.to_csv(os.path.join(Config.OUTPUT_PATH, "feature_importance.csv"), index=False)
    print("\n✓ LGBM Training complete!")

    # ==== Torch training ====
    if TORCH_AVAILABLE:
        print("\nStarting Torch training...")
        train_torch_models(train_df, train_demographics)
        print("✓ Torch training complete")
    else:
        msg = "PyTorch is not available. Skipping Torch training."
        if EnsembleConfig.FAIL_IF_TORCH_MISSING:
            raise RuntimeError(msg)
        print("⚠️ " + msg)
```

---

## 動作の流れ（再開時）

- **1 回目**: 特徴量を作成 → fold0 の LGBM を学習 → 保存 → fold1…
  途中で中断しても OK。
- **2 回目以降**: 特徴量は **キャッシュからロード** → 既に保存済みの fold モデルは **読み込み＆スキップ** → 未完 fold から続きます。
  Torch は、各 fold の `foldXX_last.pth` があれば **epoch 単位**で続行、`foldXX.pt`（最良）があれば fold 自体をスキップ可能。

---

## 包括チェック（破綻・改善ポイント）

**✅ 再現性/再開**

- LGBM は fold 単位での再開にしています（sklearn ラッパーでは `init_model` 再学習が扱いづらいため）。
- Torch は **optimizer / scheduler / scaler を含めて epoch 再開**。OneCycleLR の `total_steps` はデータ長が一致していれば正しく復元されます。

**✅ マスク形状の整合（重大度: 中）**

- 2 回の MaxPool を経た T' に対して、元の `mask (T)` をただスライスしていたため、**無効領域の位置がズレる**可能性がありました。**MaxPool1d によるダウンサンプリング**で整合させました（上記差し替え）。

**✅ 特徴量の一貫性**

- 学習時に `X_train` の列を `sorted()` に固定 → モデル束に列順を保存 → 推論時は `align_features_for_inference()` で同順に整列。
- **キャッシュは列順/内容そのまま**保存するため、**再開・再実行でも一致**します。

**✅ 数値安定性**

- 全域で `np.inf/-np.inf → 0`、`fillna(0)` を徹底。フィルタや Welch の境界条件エラーも try/fallback 済み。
- 四元数のノルムが極小の場合の処理、`sqrt(max(0, ...))` 等も既に安全化されています。

**✅ 既存仕様との整合**

- 既存の **Ensemble（0.4/0.6）**・推論 API・CMI サーバ連携は変更していません（Torch の学習が無い場合は LGBM のみで動作）。この重み配分は共有ノートの“2 段ブレンド”に沿っています。

---

## 補足（運用のコツ）

- Kaggle 環境では**時間切れ**や**メモリ制約**が起きやすいので、まず LGBM の fold を**1 ～ 2 個**だけ回し、途中で止めて**再開の挙動**を確認するのがおすすめです。
- Torch 側は `RESUME_TORCH=1` のまま、`foldXX_last.pth` が残っていれば**途中から**再開します。**最良重み（`foldXX.pt`）が残っている fold は学習をスキップ**しても最終バンドルに組み込まれます。
