- あるべきチェックポイント（features cache／fold 別モデル／epoch ckpt／bundle）が**実在するなら再利用／再開**、
- **無ければ学習**、
- そして**必ず開始前に「何をどうするか」を要約表示**
  という流れにすると分かりやすくなります。

下のコードをそのまま挿入すれば動きます（**差し込み位置**は `main()` の先頭あたり、既存の `RUN_LGBM_TRAINING` 等を決めている行の**代わり**として使ってください）。
また、features cache はユーザーが言及した **`train_feature.joblib`（単数形）** と、現行コードの **`train_features.joblib`（複数形）** の**両方を自動検出**し、存在する方を使います。

---

## 1) 追加：CheckPointPaths と実行プランビルダ

```python
# ==== NEW: CheckPointPaths & Run Plan =========================================
class CheckPointPaths:
    # 既定は CheckpointConfig を踏襲。環境変数で上書き可能。
    BASE_DIR = os.getenv("CKPT_BASE_DIR", CheckpointConfig.CKPT_DIR)

    # features cache: 公式名 + 旧名（単数形）の両方をサポート
    FEATURES_CACHE_MAIN = os.getenv(
        "FEATURES_CACHE",
        os.path.join(BASE_DIR, "train_features.joblib")
    )
    FEATURES_CACHE_LEGACY = os.getenv(
        "FEATURES_CACHE_LEGACY",
        os.path.join(BASE_DIR, "train_feature.joblib")  # ← 旧ファイル名を自動検知
    )

    # LGBM fold/状態ファイル（既存テンプレートを尊重／上書き可）
    LGBM_FOLD_TMPL = os.getenv("LGBM_FOLD_TMPL", CheckpointConfig.LGBM_FOLD_TMPL)
    LGBM_STATE_JSON = os.getenv("LGBM_STATE_JSON", CheckpointConfig.LGBM_STATE_JSON)

    # Torch/Keras の epoch/状態ファイル
    TORCH_EPOCH_TMPL = os.getenv("TORCH_EPOCH_TMPL", CheckpointConfig.TORCH_EPOCH_CKPT_TMPL)
    KERAS_STATE_JSON = os.getenv("KERAS_STATE_JSON", CheckpointConfig.KERAS_STATE_JSON)


def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def _bool_env(name, default=0):
    try:
        return bool(int(os.getenv(name, str(default))))
    except Exception:
        return bool(default)

def build_run_plan(n_folds=Config.N_FOLDS):
    """再利用／再開／学習の方針を一括決定してログに使える dict を返す。"""
    plan = {}

    # 1) features cache
    feat_path = _first_existing(CheckPointPaths.FEATURES_CACHE_MAIN,
                                CheckPointPaths.FEATURES_CACHE_LEGACY)
    plan["features"] = {
        "path": feat_path or CheckPointPaths.FEATURES_CACHE_MAIN,
        "exists": feat_path is not None,
        "action": "reuse" if feat_path is not None else "build",
        "reason": "cache found" if feat_path else "no cache"
    }

    # 2) LGBM（pretrained bundle / fold再開 / 学習）
    lgbm_bundle = Config.MODEL_PATH  # ModelPaths から引き継がれている想定
    lgbm_bundle_exists = bool(lgbm_bundle) and os.path.exists(lgbm_bundle)
    fold_paths = [
        os.path.join(CheckpointConfig.CKPT_DIR,
                     CheckpointConfig.LGBM_FOLD_TMPL.format(i))
        for i in range(n_folds)
    ]
    some_fold_exists = any(os.path.exists(p) for p in fold_paths)
    if _bool_env("FORCE_TRAIN_LGBM", 0):
        lgbm_action, reason = "train", "FORCE_TRAIN_LGBM=1"
    elif lgbm_bundle_exists:
        lgbm_action, reason = "reuse", "pretrained bundle found"
    elif some_fold_exists:
        lgbm_action, reason = "resume", "found fold checkpoints"
    else:
        lgbm_action, reason = "train", "no bundle/checkpoints"
    plan["lgbm"] = {
        "bundle_path": lgbm_bundle,
        "bundle_exists": lgbm_bundle_exists,
        "fold_paths": fold_paths,
        "has_fold_ckpt": some_fold_exists,
        "action": lgbm_action,
        "reason": reason,
    }

    # 3) Torch
    torch_bundle = EnsembleConfig.TORCH_BUNDLE_PATH
    torch_bundle_exists = bool(torch_bundle) and os.path.exists(torch_bundle)
    torch_best_paths = [
        os.path.join(DLConfig.TORCH_OUT_DIR, DLConfig.WEIGHT_TMPL.format(i))
        for i in range(DLConfig.N_FOLDS)
    ]
    torch_has_best = any(os.path.exists(p) for p in torch_best_paths)
    if (not TORCH_AVAILABLE) or (not Config.USE_TORCH):
        torch_action, reason = "skip", "torch unavailable or disabled"
    elif _bool_env("FORCE_TRAIN_TORCH", 0):
        torch_action, reason = "train", "FORCE_TRAIN_TORCH=1"
    elif torch_bundle_exists:
        torch_action, reason = "reuse", "bundle found"
    elif torch_has_best:
        torch_action, reason = "resume", "found per-fold best weights"
    else:
        torch_action, reason = "train", "no bundle/weights"
    plan["torch"] = {
        "bundle_path": torch_bundle,
        "bundle_exists": torch_bundle_exists,
        "has_best": torch_has_best,
        "action": torch_action,
        "reason": reason,
    }

    # 4) Keras
    keras_bundle = EnsembleConfig.KERAS_BUNDLE_PATH
    keras_bundle_exists = bool(keras_bundle) and os.path.exists(keras_bundle)
    keras_best_paths = [
        os.path.join(KerasConfig.OUT_DIR, KerasConfig.WEIGHT_TMPL.format(i))
        for i in range(KerasConfig.N_FOLDS)
    ]
    keras_has_best = any(os.path.exists(p) for p in keras_best_paths)
    if (not KERAS_AVAILABLE) or (not Config.USE_KERAS):
        keras_action, reason = "skip", "keras unavailable or disabled"
    elif _bool_env("FORCE_TRAIN_KERAS", 0):
        keras_action, reason = "train", "FORCE_TRAIN_KERAS=1"
    elif keras_bundle_exists:
        keras_action, reason = "reuse", "bundle found"
    elif keras_has_best:
        keras_action, reason = "resume", "found per-fold best weights"
    else:
        keras_action, reason = "train", "no bundle/weights"
    plan["keras"] = {
        "bundle_path": keras_bundle,
        "bundle_exists": keras_bundle_exists,
        "has_best": keras_has_best,
        "action": keras_action,
        "reason": reason,
    }
    return plan

def print_plan(plan):
    def _line(name, entry):
        icon = {"reuse": "✓", "resume": "↻", "train": "🛠", "build": "🛠", "skip": "⏭"}.get(entry["action"], "?")
        print(f"[PLAN] {name:7s}: {icon} {entry['action']:6s} — {entry.get('reason','')}")
        if "path" in entry:
            print(f"        path   : {entry['path']} (exists={entry.get('exists', False)})")
        if "bundle_path" in entry:
            print(f"        bundle : {entry['bundle_path']} (exists={entry.get('bundle_exists', False)})")
    print("========== RUN PLAN ==========")
    _line("features", plan["features"])
    _line("lgbm",     plan["lgbm"])
    _line("torch",    plan["torch"])
    _line("keras",    plan["keras"])
    print("==============================")
# ============================================================================

```

---

## 2) 置き換え：`main()` 冒頭の判定ロジック

既存の

```python
# Training flags for each model type
RUN_LGBM_TRAINING = Config.MODEL_PATH is None
RUN_TORCH_TRAINING = (
    TORCH_AVAILABLE
    and Config.USE_TORCH
    and not os.path.exists(EnsembleConfig.TORCH_BUNDLE_PATH)
)
RUN_KERAS_TRAINING = (
    KERAS_AVAILABLE
    and Config.USE_KERAS
    and not os.path.exists(EnsembleConfig.KERAS_BUNDLE_PATH)
)
RUN_ANY_TRAINING = RUN_LGBM_TRAINING or RUN_TORCH_TRAINING or RUN_KERAS_TRAINING
```

を **下記に置き換え**：

```python
    # === NEW: CheckPointPaths を使って実行プランを固める ===
    plan = build_run_plan(n_folds=Config.N_FOLDS)
    print_plan(plan)

    # features cache の最終採用パスを全体設定へ反映（以降の処理はこのパスを見る）
    CheckpointConfig.FEATURES_CACHE = plan["features"]["path"]

    # これ以降のフラグは plan に準拠（resume も学習扱いでOK：内部で fold毎にスキップ済）
    RUN_LGBM_TRAINING  = plan["lgbm"]["action"]  in ("train", "resume")
    RUN_TORCH_TRAINING = plan["torch"]["action"] in ("train", "resume")
    RUN_KERAS_TRAINING = plan["keras"]["action"] in ("train", "resume")
    RUN_ANY_TRAINING = RUN_LGBM_TRAINING or RUN_TORCH_TRAINING or RUN_KERAS_TRAINING
```

> ✅ これで **「デフォルトの空パスなのに『再利用』と見えるログが出る」問題は発生しません**。
> plan は存在確認済みのときだけ `reuse` を出し、無ければ `train` / `resume` を出します。

---

## 3) 置き換え：分岐ごとのログを「プラン準拠」に

`main()` 内のそれぞれのモデルのログも、**plan を根拠に出す**ように軽く修正するとより明確です。

### LGBM（冒頭の「学習スキップ」ログ）

このブロック：

```python
else:
    print("✓ Skipping LGBM training. Using pretrained model at:", Config.MODEL_PATH)
    RUNTIME_MODEL_PATH = Config.MODEL_PATH
```

を次のように：

```python
else:
    if plan["lgbm"]["action"] == "reuse":
        print("✓ Skipping LGBM training. Using pretrained model at:", Config.MODEL_PATH)
        RUNTIME_MODEL_PATH = Config.MODEL_PATH
    else:
        # plan 上ここに来ない想定だが、保険で明示エラーにする
        raise FileNotFoundError(
            f"No LGBM bundle found. Plan decided '{plan['lgbm']['action']}'. "
            "Set Config.MODEL_PATH=None to train or provide a valid bundle path."
        )
```

### Torch

```python
if RUN_TORCH_TRAINING:
    print("\nStarting Torch training..." if plan["torch"]["action"] == "train" else "\nResuming Torch training...")
    train_torch_models(train_df, train_demographics)
    print("✓ Torch training complete")
elif TORCH_AVAILABLE and Config.USE_TORCH:
    if plan["torch"]["action"] == "reuse":
        print(f"✓ Skipping Torch training. Using pretrained bundle at: {EnsembleConfig.TORCH_BUNDLE_PATH}")
    elif plan["torch"]["action"] == "skip":
        print("⏭ Torch disabled/unavailable. Skipping.")
    else:
        print(f"⚠️ Torch bundle not found at: {EnsembleConfig.TORCH_BUNDLE_PATH}")
```

### Keras も同様

```python
if RUN_KERAS_TRAINING:
    print("\nStarting Keras training..." if plan["keras"]["action"] == "train" else "\nResuming Keras training...")
    train_keras_models(train_df, train_demographics)
    print("✓ Keras training complete")
elif KERAS_AVAILABLE and Config.USE_KERAS:
    if plan["keras"]["action"] == "reuse":
        print(f"✓ Skipping Keras training. Using pretrained bundle at: {EnsembleConfig.KERAS_BUNDLE_PATH}")
    elif plan["keras"]["action"] == "skip":
        print("⏭ Keras disabled/unavailable. Skipping.")
    else:
        print(f"⚠️ Keras bundle not found at: {EnsembleConfig.KERAS_BUNDLE_PATH}")
else:
    print("ℹ️ Keras not available or disabled. Skipping Keras training.")
```

---

## 4) features cache の再利用（`train_feature.joblib` 対応）

上の「プラン」挿入で、**`CheckpointConfig.FEATURES_CACHE` が既存ファイルに差し替わる**ため、
既存のこの分岐はそのままで動きます：

```python
features_cache_path = CheckpointConfig.FEATURES_CACHE
cached = CheckpointConfig.RESUME and os.path.exists(features_cache_path)
if cached:
    cache = joblib.load(features_cache_path)
    ...
else:
    # 抽出して保存
    joblib.dump({...}, features_cache_path)
```

> 🔁 これで **前回の `train_feature.joblib`（単数形）** があれば自動で拾い、
> 無ければ **`train_features.joblib`（複数形）** に出力します。
> （必要なら `FEATURES_CACHE` / `FEATURES_CACHE_LEGACY` 環境変数で任意のパスに固定できます）

---

## 5) 使い方（運用チートシート）

- **何もしなくても**起動時に次のような**プラン要約**が出ます（例）：

  ```
  ========== RUN PLAN ==========
  [PLAN] features: ✓ reuse  — cache found
          path   : /kaggle/working/checkpoints/train_feature.joblib (exists=True)
  [PLAN] lgbm   : 🛠 train  — no bundle/checkpoints
          bundle : /kaggle/input/cmi-bfrb-v9-lightgbm/.../imu_lgbm_model-4.pkl (exists=False)
  [PLAN] torch  : ↻ resume — found per-fold best weights
          bundle : /kaggle/working/torch_models/torch_bundle.pkl (exists=False)
  [PLAN] keras  : ⏭ skip   — keras unavailable or disabled
  ==============================
  ```

- **環境変数で強制制御**：

  - `FORCE_TRAIN_LGBM=1` / `FORCE_TRAIN_TORCH=1` / `FORCE_TRAIN_KERAS=1`
    → 既存アーティファクトがあっても**必ず学習**。
  - `FEATURES_CACHE=/kaggle/working/checkpoints/train_feature.joblib`
    → cache パスを**明示**。
  - `CKPT_BASE_DIR=/kaggle/working/my_ckpts`
    → すべての checkpoint 既定ベースディレクトリを切替え。

- **Model bundle の再利用を明示したい**ときは従来どおり：

  - `ModelPaths.LGBM_BUNDLE_PATH` に Kaggle Dataset の**絶対パス**を置く
  - `TORCH_BUNDLE_PATH` / `KERAS_BUNDLE_PATH` を **存在するファイル**に設定
    → Plan が `reuse` と表示し、学習はスキップされます。

---

## 6) なぜこの設計が効くか

- **単一の「真実の源」(CheckPointPaths + Plan)** で開始前に**存在確認 → 方針決定 → ログ明示**を行うため、
  _「デフォルトの空パスなのに再利用っぽく見える」_ ログは出ません。
- `resume` の場合でも、既存の

  - LGBM: fold ごとの checkpoint／state json、
  - Torch: fold best があればスキップ、epoch ckpt から再開、
  - Keras: best 重みがあれば fold スキップ
    という**再開挙動をそのまま活かす**構造になっています。

- `train_feature(s).joblib` の**両対応**で、ファイル名の揺れにも強いです。
