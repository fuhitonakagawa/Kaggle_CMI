"""
モデル訓練スクリプト
LightGBMとXGBoostのアンサンブルモデルを訓練
"""

import os
import sys
import yaml
import json
import pickle
import joblib
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import lightgbm as lgb
import xgboost as xgb

# Custom modules
from feature_engineering import extract_comprehensive_features
from postprocessing import apply_postprocessing

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ジェスチャーマッピング
GESTURE_MAPPER = {
    "Above ear - pull hair": 0,
    "Cheek - pinch skin": 1,
    "Eyebrow - pull hair": 2,
    "Eyelash - pull hair": 3,
    "Forehead - pull hairline": 4,
    "Forehead - scratch": 5,
    "Neck - pinch skin": 6,
    "Neck - scratch": 7,
    "Drink from bottle/cup": 8,
    "Feel around in tray and pull out an object": 9,
    "Glasses on/off": 10,
    "Pinch knee/leg skin": 11,
    "Pull air toward your face": 12,
    "Scratch knee/leg skin": 13,
    "Text on phone": 14,
    "Wave hello": 15,
    "Write name in air": 16,
    "Write name on leg": 17,
}

REVERSE_GESTURE_MAPPER = {v: k for k, v in GESTURE_MAPPER.items()}


def load_config(config_path: str) -> dict:
    """設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    コンペティションの評価指標を計算
    (Binary F1 + Macro F1) / 2
    """
    # Binary F1: BFRB vs non-BFRB
    binary_f1 = f1_score(
        np.where(y_true <= 7, 1, 0),
        np.where(y_pred <= 7, 1, 0),
        zero_division=0.0,
    )
    
    # Macro F1: BFRB行動内での分類
    # 非BFRBを99にマップして除外
    macro_f1 = f1_score(
        np.where(y_true <= 7, y_true, 99),
        np.where(y_pred <= 7, y_pred, 99),
        average="macro",
        zero_division=0.0,
    )
    
    # 最終スコア
    final_score = 0.5 * (binary_f1 + macro_f1)
    
    return final_score, binary_f1, macro_f1


def load_and_prepare_data(config: dict) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    データを読み込んで前処理を行う
    
    Returns:
        X_train: 特徴量データフレーム
        y_train: ラベル配列
        subjects: 被験者ID配列
        feature_cols: 特徴量カラム名リスト
    """
    print("データを読み込み中...")
    
    # Polarsで高速読み込み
    train_df = pl.read_csv(config['data']['train_path'])
    train_demographics = pl.read_csv(config['data']['train_demographics_path'])
    
    # IMUカラムのみを抽出（テストデータとの共通カラムを使用）
    imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
    meta_cols = ['sequence_id', 'subject', 'gesture', 'behavior']
    
    # 'Performs gesture'フェーズのみを使用
    train_filtered = train_df.filter(pl.col('behavior') == 'Performs gesture')
    
    print(f"訓練シーケンス数: {train_filtered['sequence_id'].n_unique()}")
    
    # シーケンスごとに特徴量を抽出
    print("特徴量を抽出中...")
    features_list = []
    labels = []
    subjects = []
    
    # グループごとに処理
    sequences = list(train_filtered.group_by('sequence_id', maintain_order=True))
    
    total_sequences = len(sequences)
    for idx, (seq_id, seq_data) in enumerate(sequences):
        if idx % 500 == 0:
            print(f"  処理中: {idx}/{total_sequences} シーケンス")
        
        # シーケンスIDを取得
        sequence_id = seq_id[0] if isinstance(seq_id, tuple) else seq_id
        
        # 被験者IDを取得
        subject_id = seq_data['subject'][0]
        
        # デモグラフィックデータを取得
        subject_demographics = train_demographics.filter(pl.col('subject') == subject_id)
        
        # Pandasに変換して特徴量抽出
        seq_df = seq_data.to_pandas()
        demo_df = subject_demographics.to_pandas() if not subject_demographics.is_empty() else pd.DataFrame()
        
        # 特徴量抽出
        features = extract_comprehensive_features(seq_df, demo_df, config)
        features_list.append(features)
        
        # ラベルを取得
        gesture = seq_data['gesture'][0]
        label = GESTURE_MAPPER[gesture]
        labels.append(label)
        
        # 被験者IDを保存
        subjects.append(subject_id)
    
    # 特徴量を結合
    X_train = pd.concat(features_list, ignore_index=True)
    y_train = np.array(labels)
    subjects = np.array(subjects)
    
    # 特徴量カラムを取得（sequence_idを除く）
    feature_cols = [col for col in X_train.columns if col not in ['sequence_id']]
    
    print(f"特徴量数: {len(feature_cols)}")
    print(f"訓練サンプル数: {len(X_train)}")
    print(f"クラス分布:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        gesture_name = REVERSE_GESTURE_MAPPER[cls]
        print(f"  {gesture_name}: {count} ({count/len(y_train)*100:.1f}%)")
    
    return X_train, y_train, subjects, feature_cols


def train_lightgbm_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    subjects: np.ndarray,
    feature_cols: List[str],
    config: dict
) -> Tuple[List, np.ndarray, List[float]]:
    """LightGBMモデルを訓練"""
    
    print("\nLightGBMモデルを訓練中...")
    
    n_folds = config['general']['n_folds']
    lgb_params = config['lightgbm'].copy()
    
    # 不要なパラメータを削除
    lgb_params.pop('early_stopping_rounds', None)
    
    # クロスバリデーション設定
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=config['general']['seed'])
    
    models = []
    oof_predictions = np.zeros((len(y_train), 18))  # 18クラス
    cv_scores = []
    
    X_features = X_train[feature_cols]
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_features, y_train, subjects)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        # データ分割
        X_fold_train = X_features.iloc[train_idx]
        X_fold_val = X_features.iloc[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]
        
        print(f"訓練サイズ: {len(X_fold_train)}, 検証サイズ: {len(X_fold_val)}")
        
        # モデル訓練
        model = lgb.LGBMClassifier(**lgb_params)
        
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            eval_names=['valid'],
            eval_metric=['multi_logloss'],
            callbacks=[
                lgb.log_evaluation(period=config['training']['log_interval']),
                lgb.early_stopping(stopping_rounds=config['lightgbm']['early_stopping_rounds'])
            ]
        )
        
        # 予測
        val_preds = model.predict_proba(X_fold_val)
        oof_predictions[val_idx] = val_preds
        
        # 評価
        val_pred_classes = np.argmax(val_preds, axis=1)
        score, binary_f1, macro_f1 = competition_metric(y_fold_val, val_pred_classes)
        cv_scores.append(score)
        
        print(f"Fold {fold + 1} - スコア: {score:.4f} (Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f})")
        
        models.append(model)
    
    # 全体のCV性能
    oof_pred_classes = np.argmax(oof_predictions, axis=1)
    overall_score, overall_binary_f1, overall_macro_f1 = competition_metric(y_train, oof_pred_classes)
    
    print(f"\n{'='*60}")
    print("LightGBM クロスバリデーション結果")
    print(f"{'='*60}")
    print(f"全体スコア: {overall_score:.4f} ± {np.std(cv_scores):.4f}")
    print(f"Binary F1: {overall_binary_f1:.4f}")
    print(f"Macro F1: {overall_macro_f1:.4f}")
    print(f"各Foldスコア: {[f'{s:.4f}' for s in cv_scores]}")
    
    return models, oof_predictions, cv_scores


def train_xgboost_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    subjects: np.ndarray,
    feature_cols: List[str],
    config: dict
) -> Tuple[List, np.ndarray, List[float]]:
    """XGBoostモデルを訓練"""
    
    print("\nXGBoostモデルを訓練中...")
    
    n_folds = config['general']['n_folds']
    xgb_params = config['xgboost'].copy()
    
    # 不要なパラメータを削除
    xgb_params.pop('early_stopping_rounds', None)
    xgb_params.pop('n_estimators', None)
    
    # クロスバリデーション設定
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=config['general']['seed'])
    
    models = []
    oof_predictions = np.zeros((len(y_train), 18))  # 18クラス
    cv_scores = []
    
    X_features = X_train[feature_cols]
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_features, y_train, subjects)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        # データ分割
        X_fold_train = X_features.iloc[train_idx]
        X_fold_val = X_features.iloc[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]
        
        print(f"訓練サイズ: {len(X_fold_train)}, 検証サイズ: {len(X_fold_val)}")
        
        # DMatrixの作成
        dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
        dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
        
        # モデル訓練
        evals = [(dtrain, 'train'), (dval, 'valid')]
        
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=config['xgboost']['n_estimators'],
            evals=evals,
            early_stopping_rounds=config['xgboost']['early_stopping_rounds'],
            verbose_eval=config['training']['log_interval']
        )
        
        # 予測
        val_preds = model.predict(dval)
        oof_predictions[val_idx] = val_preds
        
        # 評価
        val_pred_classes = np.argmax(val_preds, axis=1)
        score, binary_f1, macro_f1 = competition_metric(y_fold_val, val_pred_classes)
        cv_scores.append(score)
        
        print(f"Fold {fold + 1} - スコア: {score:.4f} (Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f})")
        
        models.append(model)
    
    # 全体のCV性能
    oof_pred_classes = np.argmax(oof_predictions, axis=1)
    overall_score, overall_binary_f1, overall_macro_f1 = competition_metric(y_train, oof_pred_classes)
    
    print(f"\n{'='*60}")
    print("XGBoost クロスバリデーション結果")
    print(f"{'='*60}")
    print(f"全体スコア: {overall_score:.4f} ± {np.std(cv_scores):.4f}")
    print(f"Binary F1: {overall_binary_f1:.4f}")
    print(f"Macro F1: {overall_macro_f1:.4f}")
    print(f"各Foldスコア: {[f'{s:.4f}' for s in cv_scores]}")
    
    return models, oof_predictions, cv_scores


def ensemble_predictions(
    lgb_predictions: np.ndarray,
    xgb_predictions: np.ndarray,
    config: dict
) -> np.ndarray:
    """予測をアンサンブル"""
    
    method = config['ensemble']['method']
    
    if method == 'simple_average':
        return (lgb_predictions + xgb_predictions) / 2
    
    elif method == 'weighted_average':
        lgb_weight = config['ensemble']['weights']['lightgbm']
        xgb_weight = config['ensemble']['weights']['xgboost']
        total_weight = lgb_weight + xgb_weight
        
        return (lgb_predictions * lgb_weight + xgb_predictions * xgb_weight) / total_weight
    
    else:
        # デフォルトは単純平均
        return (lgb_predictions + xgb_predictions) / 2


def save_models_and_results(
    lgb_models: List,
    xgb_models: List,
    feature_cols: List[str],
    results: dict,
    config: dict
):
    """モデルと結果を保存"""
    
    # タイムスタンプ付きのディレクトリを作成
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(config['output']['model_dir']) / f'run_{timestamp}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n結果を保存中: {save_dir}")
    
    # LightGBMモデルを保存
    lgb_path = save_dir / 'lightgbm_models.pkl'
    with open(lgb_path, 'wb') as f:
        pickle.dump(lgb_models, f)
    
    # XGBoostモデルを保存
    xgb_path = save_dir / 'xgboost_models.pkl'
    with open(xgb_path, 'wb') as f:
        pickle.dump(xgb_models, f)
    
    # 特徴量カラムを保存
    features_path = save_dir / 'feature_columns.pkl'
    with open(features_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    
    # 結果を保存
    results_path = save_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 設定を保存
    config_path = save_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # ラベルエンコーダーを保存
    label_encoder = LabelEncoder()
    label_encoder.fit(list(GESTURE_MAPPER.keys()))
    le_path = save_dir / 'label_encoder.pkl'
    with open(le_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"モデルとを結果を保存しました: {save_dir}")
    
    return save_dir


def main():
    """メイン実行関数"""
    
    # 設定を読み込み
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    config = load_config(config_path)
    
    print("="*60)
    print("IMU改良モデル訓練スクリプト v2.0.0")
    print("="*60)
    
    # データの読み込みと前処理
    X_train, y_train, subjects, feature_cols = load_and_prepare_data(config)
    
    # LightGBMモデルの訓練
    lgb_models, lgb_oof, lgb_scores = train_lightgbm_models(
        X_train, y_train, subjects, feature_cols, config
    )
    
    # XGBoostモデルの訓練
    xgb_models, xgb_oof, xgb_scores = train_xgboost_models(
        X_train, y_train, subjects, feature_cols, config
    )
    
    # アンサンブル予測
    ensemble_oof = ensemble_predictions(lgb_oof, xgb_oof, config)
    ensemble_pred_classes = np.argmax(ensemble_oof, axis=1)
    
    # アンサンブルの評価
    ensemble_score, ensemble_binary_f1, ensemble_macro_f1 = competition_metric(
        y_train, ensemble_pred_classes
    )
    
    print(f"\n{'='*60}")
    print("アンサンブルモデルの結果")
    print(f"{'='*60}")
    print(f"スコア: {ensemble_score:.4f}")
    print(f"Binary F1: {ensemble_binary_f1:.4f}")
    print(f"Macro F1: {ensemble_macro_f1:.4f}")
    
    # 結果をまとめる
    results = {
        'lightgbm': {
            'mean_score': np.mean(lgb_scores),
            'std_score': np.std(lgb_scores),
            'scores': lgb_scores
        },
        'xgboost': {
            'mean_score': np.mean(xgb_scores),
            'std_score': np.std(xgb_scores),
            'scores': xgb_scores
        },
        'ensemble': {
            'score': ensemble_score,
            'binary_f1': ensemble_binary_f1,
            'macro_f1': ensemble_macro_f1
        },
        'feature_count': len(feature_cols),
        'training_samples': len(X_train),
        'timestamp': datetime.now().isoformat()
    }
    
    # モデルと結果を保存
    save_dir = save_models_and_results(
        lgb_models, xgb_models, feature_cols, results, config
    )
    
    print(f"\n訓練完了! 結果は {save_dir} に保存されました。")
    
    return save_dir


if __name__ == "__main__":
    main()