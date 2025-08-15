## 0) 目的の再整理（今回の完成形）

- **学習**：同一の CV（StratifiedGroupKFold with subject grouping）で
  **① LGBM（手工特徴）** → **② PyTorch（フレーム特徴）** を**必ず**順番に学習し、成果物を保存。
- **推論**：LGBM と PyTorch の**確率**を**重み付きで平均**して最終クラスを決定（デフォルトは **Torch:0.6, LGBM:0.4** で開始）。
  ※この重みは、頂いた Top ノートの方針（TF 系と Torch 系の二段ブレンド、0.4/0.6）を踏襲しています。
- **提出（inference-only）**：環境変数で
  **LGBM バンドル（\*.pkl）** と \*_Torch バンドル（torch_bundle.pkl ＋ fold 重み _.pt）** の**パス指定**を受け取り、**CMIInferenceServer\*\* を起動。
  ※Kaggle の T4x2 / P100 でも動く前提。

---

## 1) 設計の骨子（追加・変更する構成要素）

1. **設定追加（EnsembleConfig）**

   - `ENSEMBLE_W_LGBM` / `ENSEMBLE_W_TORCH`（env で上書き可。デフォルト 0.4 / 0.6）
   - `TORCH_BUNDLE_PATH`（env で指定。未指定なら `/kaggle/working/torch_models/torch_bundle.pkl` を既定）
   - `LOAD_TORCH_FOLDS_IN_MEMORY`: True（推論高速化。False でオンデマンド読み込み）
   - `FAIL_IF_TORCH_MISSING`: False（True にすると Torch 不在時にエラー、False なら LGBM 単独で継続）

2. **学習オーケストレーション**

   - 既存の LGBM 学習完了後に **必ず** `train_torch_models(...)` を実行（GPU が無い場合は AMP off/CPU fallback）。
   - 学習完了時、**LGBM バンドル**と**Torch バンドル**の**保存先パスをログ出力**。

3. **確率推論 API の整理**

   - `predict_lgbm_proba(sequence, demographics) -> np.ndarray[n_classes]`
   - `predict_torch_proba(sequence, demographics) -> np.ndarray[n_classes]`
   - `predict_ensemble(sequence, demographics) -> str`（上 2 つの確率を重み和 →argmax）

4. **事前学習済みの読込**

   - `MODEL_PATH` が与えられると LGBM をロード（現行通り）
   - `TORCH_BUNDLE_PATH` が与えられると Torch バンドルをロード
   - **どちらも必須**ではなく、足りない方は**学習 or フォールバック**（競合環境の都合に合わせて柔軟さを残す）

5. **CMI サーバへの接続**

   - `predict` は **常にアンサンブル**を実行（Torch 失敗時だけ LGBM にフォールバック）

> ※ ブレンドの考え方は「TF 系多数平均 ↔ Torch 系多数平均 → 重み付き平均」という LB0.82 系の流儀（0.4/0.6）に一致します。PyTorch 単系統 vs LGBM の二段ブレンドも同様にシンプルな**確率重み平均**が安定です。&#x20;

---

## 2) 具体タスク（実装指示）

### A. 設定の追加（Config/DLConfig の直下に新規クラスを追加）

**Task A-1：Ensemble 設定を追加**

```python
class EnsembleConfig:
    # weights for final soft-voting
    W_LGBM = float(os.getenv("ENSEMBLE_W_LGBM", "0.4"))
    W_TORCH = float(os.getenv("ENSEMBLE_W_TORCH", "0.6"))
    # torch bundle path (for inference-only)
    TORCH_BUNDLE_PATH = os.getenv("TORCH_BUNDLE_PATH",
        os.path.join(DLConfig.TORCH_OUT_DIR, DLConfig.BUNDLE_NAME))
    LOAD_TORCH_FOLDS_IN_MEMORY = bool(int(os.getenv("LOAD_TORCH_FOLDS_IN_MEMORY", "1")))
    FAIL_IF_TORCH_MISSING = bool(int(os.getenv("FAIL_IF_TORCH_MISSING", "0")))
```

- **チェック**：`W_LGBM + W_TORCH > 0` を学習・推論開始時に assert（0 除算/NaN 予防）。

---

### B. 学習フローの常時二本化

**Task B-1：学習ブロックで Torch 学習を「必ず」呼ぶ**

現状：

```python
# ==== (optional) Torch training ====
if os.getenv("TORCH_TRAIN", "0") == "1" and TORCH_AVAILABLE:
    ...
```

修正：

```python
# ==== Torch training (Always try) ====
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

- **注**：GPU 有無で AMP/num_workers/バッチを落とす場合は `DLConfig` を環境で切替（T4x2/P100 を想定）。

---

### C. 保存と読込（Torch の堅牢化）

**Task C-1：DataParallel 保存バグを回避**

> 現行は `nn.DataParallel` で学習した場合に **`module.` 接頭辞付き state_dict** が保存され、単機推論の `TimeSeriesNet` へ `strict=True` で読み込むとキー不整合で落ちます（推論関数に try/except は無い）。

**修正（保存側）**：学習時の保存を以下へ変更

```python
state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
torch.save({"state_dict": state, "in_ch": in_ch, "n_classes": n_classes}, best_path)
```

**修正（読込側）**：推論ロード時に安全化

```python
ckpt = torch.load(f["weight_path"], map_location=device)
state = ckpt["state_dict"]
# remove 'module.' prefix if present
if any(k.startswith("module.") for k in state.keys()):
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
model.load_state_dict(state, strict=True)
```

---

### D. 確率 API の提供とアンサンブル

**Task D-1：LGBM の確率推論関数**

```python
def predict_lgbm_proba(sequence: pl.DataFrame, demographics: pl.DataFrame) -> np.ndarray:
    raw_features = extract_features(sequence, demographics)
    X = align_features_for_inference(raw_features, feature_names)
    n_classes_global = len(reverse_gesture_mapper)
    proba_accum = np.zeros(n_classes_global, dtype=np.float64)
    for i, model in enumerate(models):
        proba = model.predict_proba(X)[0]  # (local_n_classes,)
        proba_full = np.zeros(n_classes_global, dtype=np.float64)
        for local_idx, cls_id in enumerate(model.classes_):
            proba_full[int(cls_id)] = proba[local_idx]
        proba_accum += proba_full * float(fold_weights[i])
    # 正規化（浮動誤差防止）
    s = proba_accum.sum()
    return proba_accum / s if s > 0 else proba_accum
```

**Task D-2：Torch の確率推論関数（再利用＆キャッシュ）**

- 既存 `predict_torch` はラベル返却のため、**確率版**を追加し、**バンドルと fold モデルをグローバルにキャッシュ**。

```python
_TORCH_RUNTIME = {"bundle": None, "fold_models": []}  # global cache

def _load_torch_bundle_and_models():
    if _TORCH_RUNTIME["bundle"] is not None:
        return
    bundle_path = EnsembleConfig.TORCH_BUNDLE_PATH
    if not (TORCH_AVAILABLE and os.path.exists(bundle_path)):
        msg = f"Torch bundle not found or torch unavailable: {bundle_path}"
        if EnsembleConfig.FAIL_IF_TORCH_MISSING:
            raise FileNotFoundError(msg)
        print("⚠️ " + msg)
        _TORCH_RUNTIME["bundle"] = None
        return
    bundle = joblib.load(bundle_path)
    _TORCH_RUNTIME["bundle"] = bundle
    if EnsembleConfig.LOAD_TORCH_FOLDS_IN_MEMORY:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _TORCH_RUNTIME["fold_models"] = []
        for f in bundle["folds"]:
            ckpt = torch.load(f["weight_path"], map_location=device)
            state = ckpt["state_dict"]
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            model = TimeSeriesNet(in_ch=ckpt["in_ch"], num_classes=ckpt["n_classes"], hidden=128, dropout=DLConfig.DROPOUT)
            model.load_state_dict(state, strict=True)
            model = model.to(device).eval()
            _TORCH_RUNTIME["fold_models"].append(model)

def predict_torch_proba(sequence: pl.DataFrame, demographics: pl.DataFrame) -> np.ndarray:
    _load_torch_bundle_and_models()
    bundle = _TORCH_RUNTIME["bundle"]
    if bundle is None:
        return None  # フォールバック用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pad_len = bundle["pad_len"]; feat_order = bundle["feature_order"]
    frame_df = build_frame_features(sequence).reindex(columns=feat_order).fillna(0)
    n_classes = len(bundle["reverse_gesture_mapper"])
    proba_accum = np.zeros(n_classes, dtype=np.float64)
    for i, f in enumerate(bundle["folds"]):
        stats = f["scaler_stats"]
        x_std = apply_standardize(frame_df, stats).to_numpy(np.float32)
        x_pad, m_pad = pad_and_mask(x_std, pad_len)
        xb = torch.from_numpy(x_pad[None, ...]).to(device)
        mb = torch.from_numpy(m_pad[None, ...]).to(device)
        if EnsembleConfig.LOAD_TORCH_FOLDS_IN_MEMORY and _TORCH_RUNTIME["fold_models"]:
            model = _TORCH_RUNTIME["fold_models"][i]
        else:
            ckpt = torch.load(f["weight_path"], map_location=device)
            state = ckpt["state_dict"]
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            model = TimeSeriesNet(in_ch=ckpt["in_ch"], num_classes=ckpt["n_classes"], hidden=128, dropout=DLConfig.DROPOUT)
            model.load_state_dict(state, strict=True)
            model = model.to(device).eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=DLConfig.AMP):
            prob = torch.softmax(model(xb, mb), dim=1).cpu().numpy()[0]
        proba_accum += float(f["weight"]) * prob
    s = proba_accum.sum()
    return proba_accum / s if s > 0 else proba_accum
```

**Task D-3：最終 `predict` をアンサンブル専用に差し替え**

現行は「Torch 使えたら Torch、ダメなら LGBM」。
→ **必ず両方の確率**を取りに行き、**重み平均**するように変更。

```python
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    # LGBM
    proba_lgbm = predict_lgbm_proba(sequence, demographics)
    # Torch
    proba_torch = None
    if TORCH_AVAILABLE:
        try:
            proba_torch = predict_torch_proba(sequence, demographics)
        except Exception as e:
            print(f"⚠️ Torch prediction error: {e}")
    # 合成
    w_l, w_t = EnsembleConfig.W_LGBM, EnsembleConfig.W_TORCH
    if proba_torch is None:  # Torch 不可時は LGBM 単独
        final_proba = proba_lgbm
    else:
        final_proba = (w_l * proba_lgbm + w_t * proba_torch)
        s = final_proba.sum()
        if s > 0: final_proba /= s
    final_class = int(np.argmax(final_proba))
    return reverse_gesture_mapper[final_class]
```

---

### E. 推論オンリー・提出パスの整備

**Task E-1：Skip training ブロックで Torch も検証**

```python
# SKIP TRAINING（LGBM は現行通り）
# 追加で Torch バンドルの存在確認（推論前にログを出すだけでもOK）
tbp = EnsembleConfig.TORCH_BUNDLE_PATH
if TORCH_AVAILABLE and os.path.exists(tbp):
    print(f"✓ Using Torch bundle at: {tbp}")
else:
    print(f"ℹ️ Torch bundle not found at: {tbp} (will fall back to LGBM-only if needed)")
```

---

### F. ロギングと安全装置

**Task F-1：学習・推論の先頭でチェック**

- `assert EnsembleConfig.W_LGBM + EnsembleConfig.W_TORCH > 0`
- Torch 不在時の扱い（`FAIL_IF_TORCH_MISSING`）

**Task F-2：速度・メモリ**

- `LOAD_TORCH_FOLDS_IN_MEMORY=1` が既定（T4x2/P100 なら fold=5 でも常駐可能）
- OOM 時は `DLConfig.BATCH_SIZE` や `AMP=False` に落とし、max_len を P90 にする等の**退避ガイド**をログ表示

---

## 3) 破綻・不整合の包括チェック（指摘と修正案）

1. **DataParallel の state_dict 問題**（重大）
   → **Task C-1** で保存・読込の双方に対処。これをしないと「複数 GPU で学習 → 単 GPU で推論」時に**必ず壊れます**。

2. **Torch 推論の遅延ロード**
   現行は「各 fold の ckpt を**毎回**ロード」する設計（`predict_torch` 内）。
   → **Task D-2** で**グローバルキャッシュ**を導入（大幅に高速化）。

3. **推論が “片方のみ”**
   現行 `predict` は「Torch が有効なら Torch だけ」。
   → **Task D-3** で**常に確率アンサンブル**へ統一。

4. **Torch バンドルの場所**
   現行は学習時の `/kaggle/working/torch_models/` 前提。提出フェーズでは `/kaggle/input/...` に置くため**環境変数で上書き**が必要。
   → **Task A-1 / E-1** で `TORCH_BUNDLE_PATH` を導入し、ログにも出す。

5. **マスクの時間解像度（Conv の Pooling 後）**
   `TimeSeriesNet.forward` の `mask` は単純に `mask[:, :x.size(1)]` 切り詰め。
   → できれば `F.interpolate(mask[:,None,:], size=x.size(1), mode="nearest").squeeze(1)` に変更（任意改善）。

6. **乱数・再現性**
   PyTorch の seed / cudnn 設定が未統合。
   → 学習開始前に
   `torch.manual_seed(DLConfig.SEED); torch.cuda.manual_seed_all(DLConfig.SEED)`、
   `torch.backends.cudnn.benchmark = True`（速度優先）を推奨。完全再現が必要なら `deterministic=True` に。

7. **評価指標の揃え**
   LGBM と Torch で Binary/Macro F1 の定義を合わせ済み（OK）。OOF で**重み最適化**を入れると更に安定（後述）。

8. **型/欠損の整合**

- `apply_standardize` は存在列のみを標準化 → OK
- `build_frame_features` の ToF/THM 無し時は 0 埋め → OK
- `infer_dt_and_fs` → サンプリング周波数の異常値に対して堅牢化済み → OK

9. **ファイルサイズ/IO**
   Torch の fold \*.pt を 5 本保存。予測前に常駐すると IO が減る（**Task D-2**）。
   LGBM バンドルは既存の `joblib.dump` で OK。

---

## 4) （任意強化）OOF を使った**重み最適化**の導入

- 今回はデフォルト **Torch 0.6 / LGBM 0.4**（LB で実績のある比率）で開始。
- 将来は各 fold の OOF 確率 `P_torch_oof`, `P_lgbm_oof` を保存して、
  `w` の**1 変数探索**（0〜1 の線形探索で Macro/Binary F1 の加重平均を最大化）を実施。
- OOF 保存先は `/kaggle/working/oof/` に `parquet` で良い（sequence_id, y_true, p\_\* ）。

---

## 5) 動作確認チェックリスト

- [ ] `MODEL_PATH` **あり**＋`TORCH_BUNDLE_PATH` **あり** → **両方ロード**し、サーバ起動。
- [ ] `MODEL_PATH` **あり**＋`TORCH_BUNDLE_PATH` **なし** → Torch を**フォールバック**（LGBM 単独）。
- [ ] 学習モード → **LGBM 学習 → Torch 学習 → 2 つの出力のパス表示**。
- [ ] サニティテスト：ダミーシーケンスで `predict()` が**必ずアンサンブル**を通ること。
- [ ] 複数 GPU（T4x2）での学習後、単 GPU（採点環境）で**推論が通る**（DataParallel 対応の確認）。

---

## 6) 参考（今回の設計が合致する根拠）

- **フレーム時刻特徴**を**標準化＋パディング**して**CNN→RNN→Attention**で時系列学習、
  **多数モデルの平均**や\*\*二段ブレンド（TF 系 ×Torch 系）**で最終を安定化する方針は、
  **LB 0.82 クラスのブレンド**実装や**重みだけ変えた版（0.4/0.6）\*\*の流儀と一致。
- 今回は TF ではなく LGBM を片翼に置いたブレンドですが、**異系統の決定境界を重ねる**という狙いは同じで、
  **確率の重み平均**がもっとも安全に伸びます。
