import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .model import LightGBMModel
from .evaluate import competition_metric, print_evaluation_summary


class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = LightGBMModel(config)
        self.oof_predictions = None
        self.cv_scores = []
        
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: np.ndarray, 
        subjects: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Train models using stratified group k-fold cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            subjects: Subject IDs for grouping
            
        Returns:
            oof_predictions: Out-of-fold predictions
            cv_scores: Cross-validation scores for each fold
        """
        print(f"\n{'='*60}")
        print(f"Training LightGBM with {self.config['training']['n_folds']}-fold cross-validation")
        print(f"{'='*60}")
        
        # Prepare features (remove sequence_id if present)
        feature_cols = [col for col in X_train.columns if col != 'sequence_id']
        X_features = X_train[feature_cols]
        
        # Store feature columns in model
        self.model.feature_cols = feature_cols
        
        print(f"Number of features: {len(feature_cols)}")
        print(f"Number of samples: {len(X_features)}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        
        # Setup cross-validation
        cv = StratifiedGroupKFold(
            n_splits=self.config['training']['n_folds'],
            shuffle=True,
            random_state=self.config['training']['seed']
        )
        
        # Initialize results
        self.oof_predictions = np.zeros(len(y_train))
        self.cv_scores = []
        
        # Train on each fold
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_features, y_train, subjects)):
            # Split data
            X_fold_train = X_features.iloc[train_idx]
            X_fold_val = X_features.iloc[val_idx]
            y_fold_train = y_train[train_idx]
            y_fold_val = y_train[val_idx]
            
            # Train model for this fold
            val_preds = self.model.train_fold(
                X_fold_train, y_fold_train,
                X_fold_val, y_fold_val,
                fold
            )
            
            # Store OOF predictions
            self.oof_predictions[val_idx] = val_preds
            
            # Calculate metrics
            score, binary_f1, macro_f1 = competition_metric(y_fold_val, val_preds)
            self.cv_scores.append(score)
            
            print(f"Fold {fold + 1} - Competition Score: {score:.4f} "
                  f"(Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f})")
            
        # Overall CV performance
        self._print_cv_results(y_train)
        
        return self.oof_predictions, self.cv_scores
    
    def _print_cv_results(self, y_train: np.ndarray) -> None:
        """Print cross-validation results summary."""
        overall_score, overall_binary_f1, overall_macro_f1 = competition_metric(
            y_train, self.oof_predictions
        )
        
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"Overall Competition Score: {overall_score:.4f} ± {np.std(self.cv_scores):.4f}")
        print(f"Overall Binary F1: {overall_binary_f1:.4f}")
        print(f"Overall Macro F1: {overall_macro_f1:.4f}")
        print(f"Fold scores: {[f'{score:.4f}' for score in self.cv_scores]}")
        print(f"{'='*60}\n")
        
        # Detailed evaluation
        print_evaluation_summary(y_train, self.oof_predictions)
        
    def save_results(self, save_dir: Path) -> None:
        """Save training results."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        model_dir = save_dir / "models"
        self.model.save_models(model_dir)
        
        # Save OOF predictions
        if self.oof_predictions is not None:
            np.save(save_dir / "oof_predictions.npy", self.oof_predictions)
            
        # Save CV scores
        cv_results = {
            'cv_scores': self.cv_scores,
            'mean_cv_score': np.mean(self.cv_scores) if self.cv_scores else 0,
            'std_cv_score': np.std(self.cv_scores) if self.cv_scores else 0,
            'n_folds': self.config['training']['n_folds']
        }
        
        with open(save_dir / "cv_results.json", 'w') as f:
            json.dump(cv_results, f, indent=2)
            
        # Save feature importance
        if self.config['output'].get('save_feature_importance', True):
            importance_df = self.model.get_feature_importance()
            importance_df.to_csv(save_dir / "feature_importance.csv", index=False)
            
            # Print top features
            print("\nTop 20 Most Important Features:")
            print(importance_df.head(20).to_string(index=False))
            
        print(f"\n✓ Results saved to {save_dir}")
        
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions on test data."""
        # Use same feature columns as training
        if self.model.feature_cols:
            feature_cols = self.model.feature_cols
            X_features = X_test[feature_cols]
        else:
            # Remove sequence_id if present
            feature_cols = [col for col in X_test.columns if col != 'sequence_id']
            X_features = X_test[feature_cols]
            
        # Get predictions
        predictions = self.model.predict(X_features)
        
        return predictions
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Get probability predictions on test data."""
        # Use same feature columns as training
        if self.model.feature_cols:
            feature_cols = self.model.feature_cols
            X_features = X_test[feature_cols]
        else:
            feature_cols = [col for col in X_test.columns if col != 'sequence_id']
            X_features = X_test[feature_cols]
            
        # Get probability predictions
        predictions = self.model.predict_proba(X_features)
        
        return predictions