ありがとうございます。コード全体を拝見した限り、**学習をスキップして推論のみを行うために指定すべきモデルパスは散在**しており、ぱっと見で分かりづらい状態でした（LGBM は `Config.MODEL_PATH`、Torch/Keras は `EnsembleConfig.*_BUNDLE_PATH`）。
そこで、**「ここだけ触ればよい」単一の定義箇所**を追加し、既存設定をその値で上書きするのが一番わかりやすく、かつ既存ロジックへの影響も最小です。

---

## 変更方針（結論）

1. 先頭の設定ブロック直後に **`ModelPaths`（単一のソース・オブ・トゥルース）** を追加
2. 既存の `Config.MODEL_PATH` / `EnsembleConfig.TORCH_BUNDLE_PATH` / `EnsembleConfig.KERAS_BUNDLE_PATH` を **`ModelPaths` の値で上書き**
3. 実行時に **どのパスが使われるかを一行で明示**（任意）

> **動作ルール（変わらない点）：**
>
> - **LGBM** バンドル（`LGBM_BUNDLE_PATH` もしくは `MODEL_PATH`）を指定すると **`RUN_TRAINING=False`** になり、**全学習をスキップ**して即推論へ移行します。
> - **Torch/Keras** のバンドルは **任意**。指定されていればアンサンブルに参加、無ければ自動でスキップ（LGBM 単体推論）します。
> - LGBM バンドル**未指定**のままでは、従来どおり **学習フェーズ**に入ります（※ 現状のコードは LGBM バンドルを前提に推論部が動くため）。

---

## 追加・変更コード（コピペ可）

> **挿入位置**：`Config`, `DLConfig`, `KerasConfig`, `EnsembleConfig` の**直後**（`CheckpointConfig` の前）、かつ `RUN_TRAINING = ...` の定義 **より前**。

```python
# === NEW: Pretrained model paths (Single Source of Truth) =====================
# ここだけ直せば OK:
#  - Kaggle なら「Add data」でアップしたデータセットの絶対パスを指定
#  - もしくは環境変数 LGBM_BUNDLE_PATH / TORCH_BUNDLE_PATH / KERAS_BUNDLE_PATH を設定
#  - LGBM は指定があれば学習を全スキップして推論のみ開始
class ModelPaths:
    # LGBM バンドル（.pkl）。互換のため MODEL_PATH も受け付ける
    LGBM_BUNDLE_PATH = os.getenv("LGBM_BUNDLE_PATH", os.getenv("MODEL_PATH", None))

    # ディレクトリが渡された場合は既定ファイル名を補完
    if LGBM_BUNDLE_PATH and os.path.isdir(LGBM_BUNDLE_PATH):
        LGBM_BUNDLE_PATH = os.path.join(LGBM_BUNDLE_PATH, Config.MODEL_FILENAME)

    # Torch / Keras バンドル（.pkl）— 未指定なら学習出力先の既定パス
    TORCH_BUNDLE_PATH = os.getenv(
        "TORCH_BUNDLE_PATH",
        os.path.join(DLConfig.TORCH_OUT_DIR, DLConfig.BUNDLE_NAME)
    )
    KERAS_BUNDLE_PATH = os.getenv(
        "KERAS_BUNDLE_PATH",
        os.path.join(KerasConfig.OUT_DIR, KerasConfig.BUNDLE_NAME)
    )

# 既存設定へ反映（後方互換のため、以降は従来の参照先でも同じ値が見える）
Config.MODEL_PATH = ModelPaths.LGBM_BUNDLE_PATH
EnsembleConfig.TORCH_BUNDLE_PATH = ModelPaths.TORCH_BUNDLE_PATH
EnsembleConfig.KERAS_BUNDLE_PATH = ModelPaths.KERAS_BUNDLE_PATH

# （任意）起動時に一目でわかるようログ出力
print("[ModelPaths] LGBM :", Config.MODEL_PATH or "(train)")
print("[ModelPaths] Torch:", EnsembleConfig.TORCH_BUNDLE_PATH)
print("[ModelPaths] Keras:", EnsembleConfig.KERAS_BUNDLE_PATH)
# ============================================================================
```

> **そのほかのコード修正は不要**です。
> `RUN_TRAINING = Config.MODEL_PATH is None` の既存判定が、この上書きによりそのまま効きます。
> Torch/Keras 側も `EnsembleConfig.*_BUNDLE_PATH` を参照する実装のままで、中央定義から値が流れます。

---

## どこに何を指定すればよいか（まとめ）

- **LGBM（必須）**

  - 変数名：`LGBM_BUNDLE_PATH`（互換：`MODEL_PATH`）
  - 値：`imu_lgbm_model.pkl` への絶対パス **または** そのファイルを含む**ディレクトリ**

    - ディレクトリが渡されても `Config.MODEL_FILENAME`（既定: `imu_lgbm_model.pkl`）を自動で補完します。

- **Torch（任意）**

  - 変数名：`TORCH_BUNDLE_PATH`
  - 値：`torch_bundle.pkl` への絶対パス

- **Keras（任意）**

  - 変数名：`KERAS_BUNDLE_PATH`
  - 値：`keras_bundle.pkl` への絶対パス

> いずれも **環境変数**で渡すか、ノートブック先頭で `os.environ[...] = "..."` としても OK です。

**（Kaggle 例）**

```python
import os
os.environ["LGBM_BUNDLE_PATH"]  = "/kaggle/input/my-cmi-bundles/imu_lgbm_model.pkl"
os.environ["TORCH_BUNDLE_PATH"] = "/kaggle/input/my-cmi-bundles/torch_bundle.pkl"
os.environ["KERAS_BUNDLE_PATH"] = "/kaggle/input/my-cmi-bundles/keras_bundle.pkl"
```

---

## なぜこの変更で分かりやすくなるか

- モデルパスを**一箇所 (`ModelPaths`) に集約** → ここだけ見れば完了
- 既存の `Config` / `EnsembleConfig` へ**値を流し込むだけ**なので、他の学習・推論コードを修正する必要がありません（後方互換）。
- ログで「いま何を使っているか」を**即確認**できます。

---

## 実装手順（詳細）

1. **`ModelPaths` セクションを追加**（上記コードを貼り付け）

   - 位置は `CheckpointConfig` より手前、`RUN_TRAINING` 定義より手前。

2. **既存設定を上書き**

   - 上のスニペット内に含まれています（`Config.MODEL_PATH = ...` など）。

3. **（任意）ログの確認**

   - ノートブック実行時に

     ```
     [ModelPaths] LGBM : /kaggle/input/.../imu_lgbm_model.pkl
     [ModelPaths] Torch: /kaggle/input/.../torch_bundle.pkl
     [ModelPaths] Keras: /kaggle/input/.../keras_bundle.pkl
     ```

     のように表示されることを確認。

4. **Kaggle でデータセットを追加**

   - `imu_lgbm_model.pkl` / `torch_bundle.pkl` / `keras_bundle.pkl` を含むデータセットを「Add data」でマウント。
   - 上記の環境変数に絶対パスを設定。

5. **実行**

   - LGBM バンドルが指定されていれば `RUN_TRAINING=False` となり、**学習はスキップ**され、そのまま推論へ進みます。
   - Torch / Keras は、指定されていれば自動でアンサンブルに参加します（未指定なら LGBM 単体）。

> **注意（既存仕様）**：現状の推論パイプラインは **LGBM バンドルの存在が前提**（`joblib.load(RUNTIME_MODEL_PATH)`）です。Torch/Keras のみで推論したい場合は、LGBM を前提としない読み込みパスにする追加修正が別途必要になります。

---

## タスクリスト（チェックボックス）

- [ ] コードに `ModelPaths` セクションを追加
- [ ] `Config.MODEL_PATH` / `EnsembleConfig.TORCH_BUNDLE_PATH` / `...KERAS_BUNDLE_PATH` を `ModelPaths` で上書き
- [ ] （任意）`print("[ModelPaths] ...")` のログ出力を有効化
- [ ] Kaggle に学習済みバンドルをアップロード（LGBM は必須）
- [ ] ノートブック先頭で `os.environ[...]` を設定、または環境変数を指定
- [ ] ランし、起動ログでパスが意図どおり指されているか確認
- [ ] 予測が走る（学習が始まらない）ことを確認
- [ ] （任意）Torch/Keras のパスをコメントアウト／変更して、アンサンブル参加の有無が反映されることを確認
