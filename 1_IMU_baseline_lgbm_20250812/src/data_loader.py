import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
import polars as pl


# Gesture mapping
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


class DataLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).parent
        
    def load_data(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Load training and test data with demographics."""
        print("Loading training data...")
        train_df = pl.read_csv(self.base_path / self.config['data']['train_path'])
        train_demographics = pl.read_csv(self.base_path / self.config['data']['train_demographics_path'])
        
        print("Loading test data...")
        test_df = pl.read_csv(self.base_path / self.config['data']['test_path'])
        test_demographics = pl.read_csv(self.base_path / self.config['data']['test_demographics_path'])
        
        print(f"✓ Train shape: {train_df.shape}")
        print(f"✓ Test shape: {test_df.shape}")
        
        return train_df, train_demographics, test_df, test_demographics
    
    def get_imu_columns(self, train_df: pl.DataFrame, test_df: pl.DataFrame) -> List[str]:
        """Get common IMU columns between train and test data."""
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        common_cols = train_cols.intersection(test_cols)
        
        # Filter to IMU-only columns (remove thermal and ToF sensors)
        imu_cols = [col for col in common_cols 
                   if not (col.startswith('thm_') or col.startswith('tof_'))]
        
        print(f"✓ Using {len(imu_cols)} common IMU columns")
        print(f"✓ Train-only columns: {len(train_cols - test_cols)} columns")
        print(f"✓ Test-only columns: {len(test_cols - train_cols)} columns")
        
        return imu_cols
    
    def prepare_sequences(
        self, 
        df: pl.DataFrame, 
        demographics: pl.DataFrame,
        imu_cols: List[str],
        is_train: bool = True
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray], Optional[np.ndarray], List[int]]:
        """Prepare sequences for feature extraction."""
        sequences = []
        labels = [] if is_train else None
        subjects = [] if is_train else None
        sequence_ids = []
        
        # Include gesture column for training data
        if is_train:
            cols = imu_cols + ['gesture'] if 'gesture' not in imu_cols else imu_cols
        else:
            cols = imu_cols
            
        # Group by sequence_id
        grouped = df.select(pl.col(cols)).group_by('sequence_id', maintain_order=True)
        
        for sequence_id, sequence_data in grouped:
            # Get sequence ID value
            seq_id_val = sequence_id[0] if isinstance(sequence_id, tuple) else sequence_id
            
            # Get subject demographics
            subject_id = sequence_data['subject'][0]
            subject_demographics = demographics.filter(pl.col('subject') == subject_id)
            
            # Prepare sequence info
            if is_train:
                imu_only_data = sequence_data.select(pl.col(imu_cols))
            else:
                imu_only_data = sequence_data
                
            sequence_info = {
                'data': imu_only_data,
                'demographics': subject_demographics,
                'sequence_id': seq_id_val,
                'subject_id': subject_id
            }
            sequences.append(sequence_info)
            
            # Get label for training data
            if is_train:
                gesture = sequence_data['gesture'][0]
                label = GESTURE_MAPPER[gesture]
                labels.append(label)
                subjects.append(subject_id)
                
            sequence_ids.append(seq_id_val)
        
        if is_train:
            return sequences, np.array(labels), np.array(subjects), sequence_ids
        else:
            return sequences, None, None, sequence_ids