# 📚 使用ガイド - CMI BFRB Detection v5.0

## 🚀 クイックスタート

### 1. 環境構築

```bash
# uvを使用してパッケージをインストール
uv add tensorflow lightgbm xgboost catboost scipy scikit-learn pandas polars joblib

# または pip を使用（非推奨）
pip install tensorflow lightgbm xgboost catboost scipy scikit-learn pandas polars joblib
```

### 2. データの準備

データをダウンロードして以下の構造で配置：

```
Kaggle_CMI/
├── cmi-detect-behavior-with-sensor-data/
│   ├── train.csv
│   ├── test.csv
│   ├── train_demographics.csv
│   └── test_demographics.csv
└── 5_CMI_comprehensive/
    └── comprehensive_solution.py
```

## 🔧 実行方法

### ローカル環境での訓練

#### 方法1: コマンドライン実行

```bash
# フル訓練実行
python 5_CMI_comprehensive/comprehensive_solution.py --mode train

# クイックテスト実行
python 5_CMI_comprehensive/quick_test.py
```

#### 方法2: シェルスクリプト実行

```bash
# 実行権限を付与
chmod +x 5_CMI_comprehensive/run_training.sh

# 訓練実行
./5_CMI_comprehensive/run_training.sh
```

### Kaggleでの実行

#### 方法1: ノートブックで実行

```python
# Kaggleノートブックのセルで実行

# 1. comprehensive_solution.pyをアップロード
# 2. 以下を実行

import sys
sys.path.append('/kaggle/working')

# モデル訓練
from comprehensive_solution import train_full_pipeline
ensemble, processor = train_full_pipeline()

# 推論モード
from kaggle_submission import predict
import kaggle_evaluation.cmi_inference_server

inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
inference_server.serve()
```

#### 方法2: Kaggle用スクリプト実行

```python
# kaggle_submission.py を直接実行
!python kaggle_submission.py
```

## 📊 期待される結果

### 訓練完了時の出力例

```
====================================================================
CROSS-VALIDATION RESULTS
====================================================================
binary_f1      : 0.9514 (+/- 0.0123)
macro_f1       : 0.7386 (+/- 0.0254)
combined_score : 0.8450 (+/- 0.0188)
====================================================================

✓ Training complete! Models saved to: 5_CMI_comprehensive/models/
✓ Final CV Score: 0.8450
```

### 生成されるファイル

```
5_CMI_comprehensive/
├── models/
│   ├── processor.pkl           # データ処理パイプライン
│   ├── feature_columns.pkl     # 特徴量リスト
│   └── final_ensemble.pkl      # 学習済みアンサンブルモデル
└── outputs/
    └── training_results.json    # 訓練結果
```

## 🎯 主要機能

### 1. 階層的分類
- **Binary分類**: BFRB (0-7) vs Non-BFRB (8-17)
- **BFRB分類**: 8クラスのBFRBサブタイプ
- **Full分類**: 18クラスの完全分類

### 2. 包括的特徴量
- **IMU特徴量**: 重力除去、角速度、角距離
- **FFT特徴量**: 周波数領域の7特徴量
- **統計特徴量**: 12種類の統計量
- **TOF/Thermal**: 空間パターンと時系列特徴

### 3. アンサンブル
- **階層的分類器** (weight: 2.0)
- **LightGBM** (weight: 1.5)
- **XGBoost** (weight: 1.0)
- **CatBoost** (weight: 1.0, オプション)

## ⚡ パフォーマンス最適化

### メモリ不足の場合

```python
# バッチサイズを減らす
Config.BATCH_SIZE = 32  # デフォルト: 64

# 特徴量を削減
processor.feature_columns = processor.feature_columns[:200]
```

### 訓練時間を短縮

```python
# エポック数を減らす
Config.EPOCHS = 50  # デフォルト: 150

# モデル数を減らす
Config.N_FOLDS = 3  # デフォルト: 5
```

## 🐛 トラブルシューティング

### よくある問題と解決策

#### 1. ImportError: No module named 'catboost'

```bash
# CatBoostのインストール
uv add catboost
# またはCatBoostなしで実行（自動的にスキップされます）
```

#### 2. FileNotFoundError: train.csv not found

```bash
# データパスを確認
ls cmi-detect-behavior-with-sensor-data/

# パスを修正
Config.DATA_PATH = Path("your/data/path/")
```

#### 3. GPU not detected

```python
# CPUモードで実行
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

## 📈 結果の解釈

### メトリクス

- **Binary F1**: BFRBの検出精度（目標: 0.95+）
- **Macro F1**: BFRBタイプの分類精度（目標: 0.75+）
- **Combined Score**: 競技スコア = (Binary F1 + Macro F1) / 2

### ログファイル

訓練結果は `outputs/training_results.json` に保存：

```json
{
  "cv_scores": {
    "binary_f1": [0.951, 0.948, 0.955, 0.950, 0.953],
    "macro_f1": [0.732, 0.745, 0.738, 0.741, 0.737],
    "combined_score": [0.841, 0.846, 0.847, 0.845, 0.845]
  },
  "mean_binary_f1": 0.9514,
  "mean_macro_f1": 0.7386,
  "mean_combined_score": 0.8450,
  "timestamp": "2025-08-13T12:34:56"
}
```

## 💡 Tips & Best Practices

1. **最初にクイックテストを実行**
   ```bash
   python quick_test.py
   ```

2. **段階的に訓練**
   - まず小さなデータセットでテスト
   - 問題がなければフルデータで訓練

3. **モデルの保存と再利用**
   - 訓練済みモデルは `models/` に自動保存
   - 再実行時は自動的にロード

4. **Kaggle提出前の確認**
   - ローカルでCV scoreが0.80以上を確認
   - submission.parquetが生成されることを確認

## 📞 サポート

問題が解決しない場合：

1. エラーメッセージ全文を確認
2. `quick_test.py` でコンポーネントを個別テスト
3. Kaggleディスカッションフォーラムで質問

---

**Created**: 2025年8月13日  
**Version**: 5.0  
**Status**: Production Ready