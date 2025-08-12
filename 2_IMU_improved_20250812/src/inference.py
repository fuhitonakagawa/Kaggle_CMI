"""
推論スクリプト
訓練済みモデルを使用してテストデータの予測を生成
"""

import os
import sys
import yaml
import json
import pickle
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import lightgbm as lgb

# Custom modules
from feature_engineering import extract_comprehensive_features
from postprocessing import apply_postprocessing, apply_test_time_augmentation


def load_models_and_config(model_dir: str) -> tuple:
    """
    保存されたモデルと設定を読み込む
    
    Args:
        model_dir: モデルが保存されているディレクトリ
    
    Returns:
        lgb_models, xgb_models, feature_cols, config, label_encoder
    """
    model_path = Path(model_dir)
    
    print(f"モデルを読み込み中: {model_path}")
    
    # LightGBMモデル
    with open(model_path / 'lightgbm_models.pkl', 'rb') as f:
        lgb_models = pickle.load(f)
    
    # XGBoostモデル
    with open(model_path / 'xgboost_models.pkl', 'rb') as f:
        xgb_models = pickle.load(f)
    
    # 特徴量カラム
    with open(model_path / 'feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    
    # 設定
    with open(model_path / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # ラベルエンコーダー
    with open(model_path / 'label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"✓ モデル読み込み完了")
    print(f"  - LightGBMモデル数: {len(lgb_models)}")
    print(f"  - XGBoostモデル数: {len(xgb_models)}")
    print(f"  - 特徴量数: {len(feature_cols)}")
    
    return lgb_models, xgb_models, feature_cols, config, label_encoder


def create_prediction_function(
    lgb_models: List,
    xgb_models: List,
    feature_cols: List[str],
    config: dict,
    label_encoder
):
    """
    Kaggle評価API用の予測関数を作成
    
    Returns:
        predict関数
    """
    
    def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
        """
        単一シーケンスの予測を行う
        
        Args:
            sequence: シーケンスデータ (Polars DataFrame)
            demographics: デモグラフィックデータ (Polars DataFrame)
        
        Returns:
            予測されたジェスチャー名
        """
        try:
            # Pandasに変換
            seq_df = sequence.to_pandas() if isinstance(sequence, pl.DataFrame) else sequence
            demo_df = demographics.to_pandas() if isinstance(demographics, pl.DataFrame) else demographics
            
            # IMUカラムのチェック
            imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
            if not all(col in seq_df.columns for col in imu_cols):
                # IMUセンサーがない場合はデフォルト予測
                return 'Wave hello'
            
            # 特徴量抽出
            features = extract_comprehensive_features(seq_df, demo_df, config)
            
            # 必要な特徴量のみを選択
            X_pred = features[feature_cols]
            
            # LightGBM予測
            lgb_predictions = []
            for model in lgb_models:
                pred = model.predict_proba(X_pred)
                lgb_predictions.append(pred)
            lgb_pred = np.mean(lgb_predictions, axis=0)
            
            # XGBoost予測
            xgb_predictions = []
            for model in xgb_models:
                dtest = xgb.DMatrix(X_pred)
                pred = model.predict(dtest)
                xgb_predictions.append(pred)
            xgb_pred = np.mean(xgb_predictions, axis=0)
            
            # アンサンブル
            ensemble_method = config['ensemble']['method']
            if ensemble_method == 'weighted_average':
                lgb_weight = config['ensemble']['weights']['lightgbm']
                xgb_weight = config['ensemble']['weights']['xgboost']
                total_weight = lgb_weight + xgb_weight
                final_pred = (lgb_pred * lgb_weight + xgb_pred * xgb_weight) / total_weight
            else:
                final_pred = (lgb_pred + xgb_pred) / 2
            
            # 後処理を適用
            if config['postprocessing']['enabled']:
                final_pred = apply_postprocessing(
                    final_pred,
                    list(label_encoder.classes_),
                    config['postprocessing']
                )
            
            # 最も確率の高いクラスを選択
            pred_idx = np.argmax(final_pred[0])
            pred_gesture = label_encoder.classes_[pred_idx]
            
            return pred_gesture
            
        except Exception as e:
            print(f"予測エラー: {e}")
            import traceback
            traceback.print_exc()
            # エラー時はデフォルト予測
            return 'Wave hello'
    
    return predict


def run_inference(model_dir: str) -> str:
    """
    推論を実行して提出ファイルを生成
    
    Args:
        model_dir: モデルディレクトリ
    
    Returns:
        提出ファイルのパス
    """
    
    # モデルと設定を読み込み
    lgb_models, xgb_models, feature_cols, config, label_encoder = load_models_and_config(model_dir)
    
    # 予測関数を作成
    predict_func = create_prediction_function(
        lgb_models, xgb_models, feature_cols, config, label_encoder
    )
    
    # Kaggle環境チェック
    if os.path.exists('/kaggle/input'):
        print("Kaggle環境で実行中...")
        
        # Kaggle評価APIを使用
        import kaggle_evaluation.cmi_inference_server
        
        inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict_func)
        
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            print("競技環境で推論サーバーを開始...")
            inference_server.serve()
        else:
            print("ローカルゲートウェイで推論を実行...")
            inference_server.run_local_gateway(
                data_paths=(
                    config['data']['test_path'],
                    config['data']['test_demographics_path'],
                )
            )
        
        submission_path = '/kaggle/working/submission.parquet'
        
    else:
        print("ローカル環境で実行中...")
        
        # テストデータを読み込み
        test_df = pl.read_csv(config['data']['test_path'])
        test_demographics = pl.read_csv(config['data']['test_demographics_path'])
        
        # シーケンスごとに予測
        predictions = []
        sequences = test_df.group_by('sequence_id', maintain_order=True)
        
        print(f"テストシーケンス数: {len(sequences)}")
        
        for seq_id, seq_data in sequences:
            sequence_id = seq_id[0] if isinstance(seq_id, tuple) else seq_id
            
            # 被験者IDを取得
            subject_id = seq_data['subject'][0]
            subject_demographics = test_demographics.filter(pl.col('subject') == subject_id)
            
            # 予測
            gesture = predict_func(seq_data, subject_demographics)
            
            predictions.append({
                'sequence_id': sequence_id,
                'gesture': gesture
            })
            
            print(f"  {sequence_id}: {gesture}")
        
        # 提出ファイルを作成
        submission_df = pd.DataFrame(predictions)
        
        # 保存
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        submission_dir = Path(config['output']['submission_dir'])
        submission_dir.mkdir(parents=True, exist_ok=True)
        
        submission_path = submission_dir / f'submission_{timestamp}.parquet'
        submission_df.to_parquet(submission_path, index=False)
        
        print(f"\n提出ファイルを保存: {submission_path}")
        
        # 統計情報を表示
        print("\n予測の分布:")
        print(submission_df['gesture'].value_counts())
    
    return str(submission_path)


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='推論スクリプト')
    parser.add_argument('model_dir', type=str, help='モデルディレクトリのパス')
    
    args = parser.parse_args()
    
    submission_path = run_inference(args.model_dir)
    
    print(f"\n推論完了! 提出ファイル: {submission_path}")
    
    return submission_path


if __name__ == "__main__":
    main()