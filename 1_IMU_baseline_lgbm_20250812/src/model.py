from typing import Dict, Any, List, Optional
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, log_evaluation, early_stopping


class LightGBMModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lgbm_params = config['lgbm'].copy()
        self.models: List[LGBMClassifier] = []
        self.feature_cols: Optional[List[str]] = None
        
    def _create_model(self) -> LGBMClassifier:
        """Create a new LightGBM model instance."""
        params = self.lgbm_params.copy()
        
        # Remove non-LightGBM parameters
        if 'early_stopping_rounds' in params:
            del params['early_stopping_rounds']
        if 'eval_metric' in params:
            del params['eval_metric']
            
        return LGBMClassifier(**params)
    
    def train_fold(
        self, 
        X_train: pd.DataFrame, 
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        fold: int
    ) -> np.ndarray:
        """Train a model for a single fold."""
        print(f"\n--- Training Fold {fold + 1} ---")
        print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
        
        # Create and train model
        model = self._create_model()
        
        # Setup callbacks
        callbacks = [log_evaluation(period=10)]
        
        early_stop_rounds = self.config['lgbm'].get('early_stopping_rounds', 100)
        if early_stop_rounds:
            callbacks.append(early_stopping(stopping_rounds=early_stop_rounds, verbose=True))
            
        # Train
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_names=['train', 'valid'],
            eval_metric=self.config['lgbm'].get('eval_metric', 'multi_logloss'),
            callbacks=callbacks
        )
        
        # Store model
        self.models.append(model)
        
        # Get validation predictions
        val_preds = model.predict(X_val)
        
        print(f"Fold {fold + 1} training completed. Best iteration: {model.best_iteration_}")
        
        return val_preds
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using ensemble of models."""
        if not self.models:
            raise ValueError("No models have been trained yet")
            
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred_proba = model.predict_proba(X)
            pred_class = np.argmax(pred_proba, axis=1)
            predictions.append(pred_class)
            
        # Convert to array
        predictions = np.array(predictions)
        
        # Ensemble by majority voting
        from scipy.stats import mode
        final_predictions, _ = mode(predictions, axis=0)
        
        return final_predictions.flatten()
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions from ensemble."""
        if not self.models:
            raise ValueError("No models have been trained yet")
            
        # Average probabilities across all models
        prob_predictions = []
        for model in self.models:
            prob_predictions.append(model.predict_proba(X))
            
        # Average across models
        avg_proba = np.mean(prob_predictions, axis=0)
        
        return avg_proba
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get average feature importance across all models."""
        if not self.models or not self.feature_cols:
            raise ValueError("No models have been trained yet")
            
        # Collect importance from all models
        importance_list = []
        for model in self.models:
            importance_list.append(model.feature_importances_)
            
        # Average importance
        avg_importance = np.mean(importance_list, axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': avg_importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return importance_df
    
    def save_models(self, save_dir: Path) -> None:
        """Save trained models to disk."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.models):
            model_path = save_dir / f"lgbm_fold_{i}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                
        # Save feature columns
        if self.feature_cols:
            feature_cols_path = save_dir / "feature_cols.pkl"
            with open(feature_cols_path, 'wb') as f:
                pickle.dump(self.feature_cols, f)
                
        print(f"✓ Saved {len(self.models)} models to {save_dir}")
        
    def load_models(self, save_dir: Path) -> None:
        """Load models from disk."""
        save_dir = Path(save_dir)
        
        # Load all model files
        self.models = []
        for model_path in sorted(save_dir.glob("lgbm_fold_*.pkl")):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                self.models.append(model)
                
        # Load feature columns
        feature_cols_path = save_dir / "feature_cols.pkl"
        if feature_cols_path.exists():
            with open(feature_cols_path, 'rb') as f:
                self.feature_cols = pickle.load(f)
                
        print(f"✓ Loaded {len(self.models)} models from {save_dir}")