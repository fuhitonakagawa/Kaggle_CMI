「**モデルのパス**（推論用バンドル）と **チェックポイントのパス**（学習の再開・中間生成物）を**1か所**で定義し、そこから全体に反映 & 実行方針（再利用/再開/学習）をはっきりログに出す」という方針で、**最小差分**で済む編集案をまとめました。
以下をそのまま貼り替え/追記すれば動きます。

---

## 1) パス定義を1か所に集約（`Paths` クラス）

`ModelPaths` と `CheckPointPaths` を統合し、**1クラス**で宣言→**resolve() で存在確認と反映**まで行うようにします。既存の環境変数もすべて尊重します（上書きしません）。

> 置き場所：`# Configuration` 群の直後（`EnsembleConfig` の後）に **置き換え**／**新規追加**。

```python
# === Single source of truth for all paths ====================================
from dataclasses import dataclass

@dataclass
class Paths:
    # ---- bases ----
    OUTPUT: str = Config.OUTPUT_PATH
    CKPT_BASE: str = os.getenv("CKPT_BASE_DIR", os.path.join(Config.OUTPUT_PATH, "checkpoints"))
    LOG_BASE:  str = os.path.join(Config.OUTPUT_PATH, "logs")

    # ---- bundles (pretrained for inference) ----
    # LGBM は MODEL_PATH と LGBM_BUNDLE_PATH の両方を受け付け、前者を後方互換として採用
    LGBM_BUNDLE: str | None = os.getenv("LGBM_BUNDLE_PATH", os.getenv("MODEL_PATH", None))
    TORCH_BUNDLE: str = os.getenv("TORCH_BUNDLE_PATH", os.path.join(DLConfig.TORCH_OUT_DIR, DLConfig.BUNDLE_NAME))
    KERAS_BUNDLE: str = os.getenv("KERAS_BUNDLE_PATH", os.path.join(KerasConfig.OUT_DIR,  KerasConfig.BUNDLE_NAME))

    # ---- features cache (legacy nameも拾う) ----
    FEATURES_CACHE_MAIN:   str = os.path.join(os.getenv("CKPT_BASE_DIR", os.path.join(Config.OUTPUT_PATH, "checkpoints")), "train_features.joblib")
    FEATURES_CACHE_LEGACY: str = os.path.join(os.getenv("CKPT_BASE_DIR", os.path.join(Config.OUTPUT_PATH, "checkpoints")), "train_feature.joblib")  # 旧名

    # ---- templates / states ----
    LGBM_FOLD_TMPL:  str = os.path.join(os.getenv("CKPT_BASE_DIR", os.path.join(Config.OUTPUT_PATH, "checkpoints")), "lgbm_fold{:02d}.pkl")
    LGBM_STATE_JSON: str = os.path.join(os.getenv("CKPT_BASE_DIR", os.path.join(Config.OUTPUT_PATH, "checkpoints")), "lgbm_state.json")

    TORCH_EPOCH_TMPL: str = os.path.join(DLConfig.TORCH_OUT_DIR, "fold{:02d}_last.pth")
    TORCH_STATE_JSON: str = os.path.join(DLConfig.TORCH_OUT_DIR, "torch_state.json")

    KERAS_STATE_JSON: str = os.path.join(KerasConfig.OUT_DIR, "keras_state.json")

    # ---- resolved (後で埋める) ----
    FEATURES_CACHE: str | None = None

    def resolve(self):
        # ディレクトリの確保
        for d in [self.CKPT_BASE, DLConfig.TORCH_OUT_DIR, KerasConfig.OUT_DIR, self.LOG_BASE]:
            os.makedirs(d, exist_ok=True)

        # features cache はメイン or レガシーのうち "最初に存在するもの" を優先
        self.FEATURES_CACHE = _first_existing(self.FEATURES_CACHE_MAIN, self.FEATURES_CACHE_LEGACY) or self.FEATURES_CACHE_MAIN

        # 既存の設定へ**一括反映**（以降は従来参照でも同じものが見える）
        Config.MODEL_PATH                   = self.LGBM_BUNDLE
        EnsembleConfig.TORCH_BUNDLE_PATH    = self.TORCH_BUNDLE
        EnsembleConfig.KERAS_BUNDLE_PATH    = self.KERAS_BUNDLE

        # CheckpointConfig へも集約結果を反映
        CheckpointConfig.CKPT_DIR           = self.CKPT_BASE
        CheckpointConfig.LGBM_STATE_JSON    = self.LGBM_STATE_JSON
        CheckpointConfig.LGBM_FOLD_TMPL     = os.path.basename(self.LGBM_FOLD_TMPL) if os.path.isabs(self.LGBM_FOLD_TMPL) else self.LGBM_FOLD_TMPL
        CheckpointConfig.TORCH_EPOCH_CKPT_TMPL = self.TORCH_EPOCH_TMPL
        CheckpointConfig.TORCH_STATE_JSON   = self.TORCH_STATE_JSON
        CheckpointConfig.KERAS_STATE_JSON   = self.KERAS_STATE_JSON

    def print_summary(self):
        def _flag(p):  # 存在表示
            return f"{p}  [{'✓' if p and os.path.exists(p) else '×'}]"
        print("========== PATHS ==========")
        print(" LGBM  :", _flag(self.LGBM_BUNDLE or "(train)"))
        print(" Torch :", _flag(self.TORCH_BUNDLE))
        print(" Keras :", _flag(self.KERAS_BUNDLE))
        print(" CKPT  :", self.CKPT_BASE)
        print(" Cache :", _flag(self.FEATURES_CACHE))
        print(" Logs  :", self.LOG_BASE)
        print("===========================")

# ここで即 resolve
PATHS = Paths()
PATHS.resolve()
PATHS.print_summary()
# ============================================================================
```

> **ポイント**
>
> * これで**このクラスだけ**直せば、モデルバンドル／チェックポイント／キャッシュ／ログの**全パス**が一元管理になります。
> * 旧 `ModelPaths` / `CheckPointPaths` は**不要**（削除OK）。以降の参照は `Config` / `EnsembleConfig` / `CheckpointConfig` に**反映済み**なので既存コードへの影響は最小です。

---

## 2) 実行プラン（reuse/resume/train）の参照元を `PATHS` に変更

`build_run_plan()` 内の features キャッシュ参照を `PATHS.FEATURES_CACHE` に寄せ、LGBM/Torch/Keras の bundle も `PATHS` 経由で見ます。**分岐ロジックはそのまま**使えます。

```diff
-def build_run_plan(n_folds=Config.N_FOLDS):
+def build_run_plan(n_folds=Config.N_FOLDS):
     plan = {}

-    feat_path = _first_existing(CheckPointPaths.FEATURES_CACHE_MAIN,
-                                CheckPointPaths.FEATURES_CACHE_LEGACY)
+    feat_path = PATHS.FEATURES_CACHE
     plan["features"] = {
         "path": feat_path,
-        "exists": feat_path is not None and os.path.exists(feat_path),
+        "exists": feat_path is not None and os.path.exists(feat_path),
         "action": "reuse" if (feat_path and os.path.exists(feat_path)) else "build",
         "reason": "cache found" if (feat_path and os.path.exists(feat_path)) else "no cache"
     }

-    lgbm_bundle = Config.MODEL_PATH
+    lgbm_bundle = PATHS.LGBM_BUNDLE  # 一元化
     lgbm_bundle_exists = bool(lgbm_bundle) and os.path.exists(lgbm_bundle)
     ...
-    torch_bundle = EnsembleConfig.TORCH_BUNDLE_PATH
+    torch_bundle = PATHS.TORCH_BUNDLE
     ...
-    keras_bundle = EnsembleConfig.KERAS_BUNDLE_PATH
+    keras_bundle = PATHS.KERAS_BUNDLE
     ...
```

> **ログの見え方**はこれまで通り `print_plan(plan)` で、**存在有無**と **採用方針**（reuse/resume/train/skip）が1行で分かります。
> 既定パスが表示されるだけで「再利用しそうに見える」問題を避けるため、**exists フラグ**を必ず出しています。

---

## 3) `main()` の最初で **features cache のパス作成**と **RUNTIME\_MODEL\_PATH** の初期化

`RUNTIME_MODEL_PATH` が `None` のままになる角ケース（Torch/Keras を学習するが LGBM は reuse 等）を潰します。

```diff
 def main():
     global RUNTIME_MODEL_PATH
-    plan = build_run_plan(n_folds=Config.N_FOLDS)
+    plan = build_run_plan(n_folds=Config.N_FOLDS)
     print_plan(plan)

-    CheckpointConfig.FEATURES_CACHE = plan["features"]["path"]
+    CheckpointConfig.FEATURES_CACHE = plan["features"]["path"]
+    os.makedirs(os.path.dirname(CheckpointConfig.FEATURES_CACHE), exist_ok=True)

     RUN_LGBM_TRAINING  = plan["lgbm"]["action"]  in ("train", "resume")
     RUN_TORCH_TRAINING = plan["torch"]["action"] in ("train", "resume")
     RUN_KERAS_TRAINING = plan["keras"]["action"] in ("train", "resume")
     RUN_ANY_TRAINING = RUN_LGBM_TRAINING or RUN_TORCH_TRAINING or RUN_KERAS_TRAINING

-    RUNTIME_MODEL_PATH = None
+    # ここで先に決め打ち（後で学習したら上書き）
+    if plan["lgbm"]["action"] == "reuse":
+        RUNTIME_MODEL_PATH = plan["lgbm"]["bundle_path"]
+    else:
+        RUNTIME_MODEL_PATH = os.path.join(Config.OUTPUT_PATH, Config.MODEL_FILENAME)
```

---

## 4) **TorchDataset でのモダリティ Dropout の型バグ修正（重要）**

`FRAME_CACHE.get()` は **pandas.DataFrame** を返しますが、`with_columns(pl.lit(...))` を呼んでおり、**Polars API を Pandas に対して呼んでいる**ためエラーになります。
**Pandas で列代入**に直します。

```diff
 class TorchDataset(torch.utils.data.Dataset):
     ...
     def __getitem__(self, idx):
-        frame_df = FRAME_CACHE.get(self.sequences[idx])  # pandas.DataFrame
-        frame_df = apply_standardize(frame_df, self.scaler_stats)
-        
-        # Apply modality dropout during training (ToF and THM columns)
-        if self.train_mode and np.random.random() < self.modality_dropout_prob:
-            # Randomly drop ToF or THM modality
-            if np.random.random() < 0.5:
-                # Drop ToF columns
-                tof_cols = [col for col in frame_df.columns if col.startswith('tof_')]
-                if tof_cols:
-                    frame_df = frame_df.with_columns([
-                        pl.lit(0.0).alias(col) for col in tof_cols
-                    ])
-            else:
-                # Drop THM columns
-                thm_cols = [col for col in frame_df.columns if col.startswith('thm_')]
-                if thm_cols:
-                    frame_df = frame_df.with_columns([
-                        pl.lit(0.0).alias(col) for col in thm_cols
-                    ])
+        frame_df = FRAME_CACHE.get(self.sequences[idx])            # pandas.DataFrame
+        frame_df = apply_standardize(frame_df, self.scaler_stats)  # pandas.DataFrame
+
+        # Apply modality dropout during training (ToF / THM) - use **pandas** assignment
+        if self.train_mode and np.random.random() < self.modality_dropout_prob:
+            if np.random.random() < 0.5:
+                cols = [c for c in frame_df.columns if c.startswith("tof_")]
+            else:
+                cols = [c for c in frame_df.columns if c.startswith("thm_")]
+            if cols:
+                frame_df.loc[:, cols] = 0.0  # pandas 列代入

         x = frame_df.to_numpy(dtype=np.float32)
         x, m = pad_and_mask(x, self.pad_len)
         ...
```

---

## 5) Keras の LR ログ取得をより堅牢に

`optimizer.lr` が無い環境があるため、`learning_rate` フォールバックを追加しておくと安全です。

```diff
 class ProgressCallback(keras.callbacks.Callback):
     ...
     def on_epoch_end(self, epoch, logs=None):
         ...
-        rec = {
+        lr_attr = None
+        if hasattr(self.model.optimizer, "lr"):
+            try: lr_attr = float(self.model.optimizer.lr.numpy())
+            except: pass
+        if lr_attr is None and hasattr(self.model.optimizer, "learning_rate"):
+            try: lr_attr = float(self.model.optimizer.learning_rate.numpy())
+            except: lr_attr = 0.0
+        rec = {
             "fold": self.fold,
             "epoch": int(epoch),
             ...
-            "lr": float(
-                getattr(self.model.optimizer, "lr", 0.0).numpy()
-                if hasattr(self.model.optimizer, "lr")
-                else 0.0
-            ),
+            "lr": lr_attr if lr_attr is not None else 0.0,
         }
```

---

## 6) 使い方（環境変数）

* **バンドル再利用（学習スキップ）**

  * LGBM: `LGBM_BUNDLE_PATH=/kaggle/input/.../imu_lgbm_model.pkl`
  * Torch: `TORCH_BUNDLE_PATH=/kaggle/input/.../torch_bundle.pkl`
  * Keras: `KERAS_BUNDLE_PATH=/kaggle/input/.../keras_bundle.pkl`
* **再開用の保存先**

  * `CKPT_BASE_DIR=/kaggle/working/checkpoints`（既定）
* **方針の強制**

  * `FORCE_TRAIN_LGBM=1` / `FORCE_TRAIN_TORCH=1` / `FORCE_TRAIN_KERAS=1`
* **途中再開の有効化**

  * `RESUME=1`（既定）
  * `train_feature.joblib`（旧名）または `train_features.joblib`（新名）があれば自動再利用

実行時、最初に

```
========== PATHS ==========
 LGBM  : /.../imu_lgbm_model.pkl  [✓]
 Torch : /.../torch_bundle.pkl    [×]
 Keras : /.../keras_bundle.pkl    [×]
 CKPT  : /kaggle/working/checkpoints
 Cache : /kaggle/working/checkpoints/train_features.joblib  [✓]
 Logs  : /kaggle/working/logs
===========================
```

のように**存在（✓/×）が明示**され、続く `RUN PLAN` で **reuse/resume/train** が一目瞭然になります。

---

## 7) そのほか潜在的な問題点と改善提案

1. **`RUNTIME_MODEL_PATH` が未設定のケース**

   * 上記 #3 の修正で、*LGBMを学習しないが Torch/Keras は学習*というケースでも `predict()` の LGBM ローダが落ちません。

2. **features cache 保存先ディレクトリ**

   * 環境変数で `CKPT_BASE_DIR` を変えると未作成の可能性があるので、上記 #3 で `os.makedirs(dirname, exist_ok=True)` を追加しています。

3. **OOF の重み最適化**

   * `oof_dir` が存在しても LGBM しかファイルがない場合は**警告してスキップ**。現状の実装でOKですが、必要なら「1つでもあれば単独出力」などの分岐を追加しても良いです。

4. **Keras の class\_weight と one-hot**

   * Keras は one-hot ターゲットでも `class_weight` を解釈しますが、挙動がバージョンで微妙に違います。万一効いていない場合は `sample_weight` を自前で作って渡すと確実です（`np.take(class_weight, y_int)` で系列ごとに重みベクトルを作り、各サンプルに同じ重みを適用）。

5. **ログのノイズ**

   * 既に `print_plan()` で十分明快ですが、さらに静かにしたい場合は LightGBM の `verbosity: -1` に加えて `log_evaluation(period=0)` にし、代わりに fold 結果だけを明示出力する方法もあります。

6. **型ヒント（Python 3.8 未満互換）**

   * Kaggle は 3.10 なので `list[pd.DataFrame]` などの新構文はOK。外部で実行するなら `from __future__ import annotations` を先頭に置いておくと安心です。

---

以上です。
このパッチで **「パスは `Paths` だけ見ればOK」** になり、**再利用／再開／学習の方針**も**誤解のないログ**で確認できます。必要なら、同じ `Paths` に**提出物（`submission.parquet`）の出力先**や**OOF格納先**も寄せてしまうとさらに迷いが減ります。
