from typing import Dict, Any, List

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial.transform import Rotation as R


class FeatureEngineer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.acc_cols = config['features']['acc_cols']
        self.rot_cols = config['features']['rot_cols']
        self.use_world_acc = config['features']['use_world_acceleration']
        self.segment_count = config['features']['segment_count']
        
    def extract_features(self, sequence_info: Dict[str, Any]) -> pd.DataFrame:
        """Extract comprehensive features from a sequence."""
        sequence = sequence_info['data']
        demographics = sequence_info['demographics']
        
        # Convert to pandas for processing
        seq_df = sequence.to_pandas()
        demo_df = demographics.to_pandas()
        
        # Handle missing values
        acc_data = seq_df[self.acc_cols].copy()
        acc_data = acc_data.ffill().bfill().fillna(0)
        
        rot_data = seq_df[self.rot_cols].copy()
        rot_data = rot_data.ffill().bfill()
        
        # Handle quaternion missing values
        rot_data_clean = self._handle_quaternion_missing_values(rot_data.values)
        
        # Compute world acceleration if enabled
        if self.use_world_acc:
            world_acc_data = self._compute_world_acceleration(acc_data.values, rot_data_clean)
        else:
            world_acc_data = None
            
        # Initialize feature dictionary
        features = {}
        
        # Add sequence metadata
        features['sequence_length'] = len(seq_df)
        features['sequence_id'] = sequence_info['sequence_id']
        
        # Add demographics features
        if len(demo_df) > 0:
            demo_row = demo_df.iloc[0]
            features['age'] = demo_row.get('age', 0)
            features['adult_child'] = demo_row.get('adult_child', 0)
            features['sex'] = demo_row.get('sex', 0)
            features['handedness'] = demo_row.get('handedness', 0)
            features['height_cm'] = demo_row.get('height_cm', 0)
            features['shoulder_to_wrist_cm'] = demo_row.get('shoulder_to_wrist_cm', 0)
            features['elbow_to_wrist_cm'] = demo_row.get('elbow_to_wrist_cm', 0)
            
        # Define feature arrays
        feature_arrays = {
            'acc': acc_data.values,  # Device acceleration
            'rot': rot_data_clean,    # Rotation quaternion
        }
        
        if self.use_world_acc and world_acc_data is not None:
            feature_arrays['world_acc'] = world_acc_data  # World acceleration
            
        # Extract statistical features for each data source
        for source_name, array in feature_arrays.items():
            if array.ndim == 1:
                array = array.reshape(-1, 1)
                
            n_features = array.shape[1]
            
            for feat_idx in range(n_features):
                feat_data = array[:, feat_idx]
                
                # Create feature name
                if source_name == 'acc':
                    axis_names = ['x', 'y', 'z']
                    prefix = f"acc_{axis_names[feat_idx]}"
                elif source_name == 'rot':
                    comp_names = ['w', 'x', 'y', 'z']
                    prefix = f"rot_{comp_names[feat_idx]}"
                elif source_name == 'world_acc':
                    axis_names = ['x', 'y', 'z']
                    prefix = f"world_acc_{axis_names[feat_idx]}"
                else:
                    prefix = f"{source_name}_{feat_idx}"
                    
                # Extract statistical features
                features.update(self._extract_statistical_features(feat_data, prefix))
                
        # Compute magnitude features
        acc_magnitude = np.linalg.norm(acc_data.values, axis=1)
        features.update(self._extract_statistical_features(acc_magnitude, 'acc_magnitude'))
        
        if self.use_world_acc and world_acc_data is not None:
            world_acc_magnitude = np.linalg.norm(world_acc_data, axis=1)
            features.update(self._extract_statistical_features(world_acc_magnitude, 'world_acc_magnitude'))
            
            # Difference between device and world acceleration magnitudes
            acc_world_diff = acc_magnitude - world_acc_magnitude
            features.update(self._extract_statistical_features(acc_world_diff, 'acc_world_diff'))
            
        # Convert to DataFrame
        result_df = pd.DataFrame([features])
        
        # Handle any remaining NaN values
        result_df = result_df.fillna(0)
        
        return result_df
    
    def _handle_quaternion_missing_values(self, rot_data: np.ndarray) -> np.ndarray:
        """Handle missing values in quaternion data."""
        rot_cleaned = rot_data.copy()
        
        for i in range(len(rot_data)):
            row = rot_data[i]
            missing_count = np.isnan(row).sum()
            
            if missing_count == 0:
                # Normalize to unit quaternion
                norm = np.linalg.norm(row)
                if norm > 1e-8:
                    rot_cleaned[i] = row / norm
                else:
                    rot_cleaned[i] = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
                    
            elif missing_count == 1:
                # Reconstruct using unit quaternion constraint
                missing_idx = np.where(np.isnan(row))[0][0]
                valid_values = row[~np.isnan(row)]
                
                sum_squares = np.sum(valid_values**2)
                if sum_squares <= 1.0:
                    missing_value = np.sqrt(max(0, 1.0 - sum_squares))
                    # Choose sign for continuity
                    if i > 0 and not np.isnan(rot_cleaned[i-1, missing_idx]):
                        if rot_cleaned[i-1, missing_idx] < 0:
                            missing_value = -missing_value
                    rot_cleaned[i, missing_idx] = missing_value
                    rot_cleaned[i, ~np.isnan(row)] = valid_values
                else:
                    rot_cleaned[i] = [1.0, 0.0, 0.0, 0.0]
            else:
                # More than one missing value
                rot_cleaned[i] = [1.0, 0.0, 0.0, 0.0]
                
        return rot_cleaned
    
    def _compute_world_acceleration(self, acc: np.ndarray, rot: np.ndarray) -> np.ndarray:
        """Convert acceleration from device to world coordinates."""
        try:
            # Convert quaternion format from [w, x, y, z] to [x, y, z, w] for scipy
            rot_scipy = rot[:, [1, 2, 3, 0]]
            
            # Verify quaternions are valid
            norms = np.linalg.norm(rot_scipy, axis=1)
            if np.any(norms < 1e-8):
                mask = norms < 1e-8
                rot_scipy[mask] = [0.0, 0.0, 0.0, 1.0]  # Identity quaternion
                
            # Create rotation object and apply transformation
            r = R.from_quat(rot_scipy)
            acc_world = r.apply(acc)
            
        except Exception as e:
            print(f"Warning: World coordinate transformation failed: {e}")
            acc_world = acc.copy()
            
        return acc_world
    
    def _extract_statistical_features(self, data: np.ndarray, prefix: str) -> Dict[str, float]:
        """Extract statistical features from 1D time series."""
        features = {}
        
        # Basic statistics
        features[f'{prefix}_mean'] = np.mean(data)
        features[f'{prefix}_std'] = np.std(data)
        features[f'{prefix}_var'] = np.var(data)
        features[f'{prefix}_min'] = np.min(data)
        features[f'{prefix}_max'] = np.max(data)
        features[f'{prefix}_median'] = np.median(data)
        features[f'{prefix}_q25'] = np.percentile(data, 25)
        features[f'{prefix}_q75'] = np.percentile(data, 75)
        features[f'{prefix}_iqr'] = np.percentile(data, 75) - np.percentile(data, 25)
        
        # Range and boundary features
        features[f'{prefix}_range'] = np.max(data) - np.min(data)
        features[f'{prefix}_first'] = data[0] if len(data) > 0 else 0
        features[f'{prefix}_last'] = data[-1] if len(data) > 0 else 0
        features[f'{prefix}_delta'] = data[-1] - data[0] if len(data) > 0 else 0
        
        # Higher order moments
        if len(data) > 1 and np.std(data) > 1e-8:
            features[f'{prefix}_skew'] = pd.Series(data).skew()
            features[f'{prefix}_kurt'] = pd.Series(data).kurtosis()
        else:
            features[f'{prefix}_skew'] = 0
            features[f'{prefix}_kurt'] = 0
            
        # Differential features
        if len(data) > 1:
            diff_data = np.diff(data)
            features[f'{prefix}_diff_mean'] = np.mean(diff_data)
            features[f'{prefix}_diff_std'] = np.std(diff_data)
            features[f'{prefix}_n_changes'] = np.sum(np.abs(diff_data) > np.std(data) * 0.1)
        else:
            features[f'{prefix}_diff_mean'] = 0
            features[f'{prefix}_diff_std'] = 0
            features[f'{prefix}_n_changes'] = 0
            
        # Time correlation
        if len(data) > 2:
            time_indices = np.arange(len(data))
            try:
                corr_coef = np.corrcoef(time_indices, data)[0, 1]
                features[f'{prefix}_time_corr'] = corr_coef if not np.isnan(corr_coef) else 0
            except:
                features[f'{prefix}_time_corr'] = 0
        else:
            features[f'{prefix}_time_corr'] = 0
            
        # Segment features
        seq_len = len(data)
        if seq_len >= self.segment_count * 3:
            seg_size = seq_len // self.segment_count
            
            for seg_idx in range(self.segment_count):
                start_idx = seg_idx * seg_size
                end_idx = (seg_idx + 1) * seg_size if seg_idx < self.segment_count - 1 else seq_len
                segment = data[start_idx:end_idx]
                
                features[f'{prefix}_seg{seg_idx+1}_mean'] = np.mean(segment)
                features[f'{prefix}_seg{seg_idx+1}_std'] = np.std(segment)
                
            # Segment transitions
            for seg_idx in range(self.segment_count - 1):
                features[f'{prefix}_seg{seg_idx+1}_to_seg{seg_idx+2}'] = (
                    features[f'{prefix}_seg{seg_idx+2}_mean'] - features[f'{prefix}_seg{seg_idx+1}_mean']
                )
        else:
            # Not enough data for meaningful segments
            for seg_idx in range(self.segment_count):
                features[f'{prefix}_seg{seg_idx+1}_mean'] = features[f'{prefix}_mean']
                features[f'{prefix}_seg{seg_idx+1}_std'] = features[f'{prefix}_std']
            for seg_idx in range(self.segment_count - 1):
                features[f'{prefix}_seg{seg_idx+1}_to_seg{seg_idx+2}'] = 0
                
        return features
    
    def extract_all_features(self, sequences: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract features for all sequences."""
        features_list = []
        
        for i, sequence_info in enumerate(sequences):
            if i % 100 == 0:
                print(f"Processing sequence {i+1}/{len(sequences)}...")
                
            features = self.extract_features(sequence_info)
            features_list.append(features)
            
        # Combine all features
        all_features = pd.concat(features_list, ignore_index=True)
        
        print(f"✓ Extracted features shape: {all_features.shape}")
        
        return all_features