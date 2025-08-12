from typing import Tuple

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate the competition metric (Binary F1 + Macro F1) / 2.
    
    Args:
        y_true: True labels (0-7: BFRB, 8-17: non-BFRB)
        y_pred: Predicted labels
        
    Returns:
        final_score: Competition score
        binary_f1: Binary F1 score for BFRB vs non-BFRB
        macro_f1: Macro F1 score for BFRB gestures
    """
    # Binary F1: BFRB vs non-BFRB
    binary_f1 = f1_score(
        np.where(y_true <= 7, 1, 0),
        np.where(y_pred <= 7, 1, 0),
        zero_division=0.0,
    )
    
    # Macro F1: specific gesture classification (only for BFRB gestures)
    # Map non-BFRB to 99 to exclude from macro calculation
    macro_f1 = f1_score(
        np.where(y_true <= 7, y_true, 99),
        np.where(y_pred <= 7, y_pred, 99),
        average="macro",
        zero_division=0.0,
    )
    
    # Final competition score
    final_score = 0.5 * (binary_f1 + macro_f1)
    
    return final_score, binary_f1, macro_f1


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate overall accuracy."""
    return accuracy_score(y_true, y_pred)


def calculate_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate per-class metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with per-class precision, recall, and F1 scores
    """
    from sklearn.metrics import classification_report
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    return report


def print_evaluation_summary(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print a comprehensive evaluation summary."""
    # Competition metric
    final_score, binary_f1, macro_f1 = competition_metric(y_true, y_pred)
    
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Competition Score: {final_score:.4f}")
    print(f"  - Binary F1 (BFRB vs non-BFRB): {binary_f1:.4f}")
    print(f"  - Macro F1 (BFRB classes): {macro_f1:.4f}")
    print(f"Overall Accuracy: {calculate_accuracy(y_true, y_pred):.4f}")
    
    # BFRB vs non-BFRB breakdown
    bfrb_true = np.where(y_true <= 7, 1, 0)
    bfrb_pred = np.where(y_pred <= 7, 1, 0)
    
    print("\nBFRB vs Non-BFRB:")
    print(f"  - True BFRB: {np.sum(bfrb_true)}")
    print(f"  - Predicted BFRB: {np.sum(bfrb_pred)}")
    print(f"  - Correct BFRB: {np.sum((bfrb_true == 1) & (bfrb_pred == 1))}")
    print(f"  - Correct Non-BFRB: {np.sum((bfrb_true == 0) & (bfrb_pred == 0))}")
    
    print("=" * 60)