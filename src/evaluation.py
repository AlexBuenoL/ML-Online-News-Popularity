import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any

class RegressionEvaluator:
    """
    Evaluation class for regression models.
    Designed to handle target variables that have been log-transformed (log1p).
    """
    
    def __init__(self, is_log_transformed: bool = True):
        """
        Args:
            is_log_transformed (bool): Flag indicating if the target variable 
                                       was transformed using np.log1p.
        """
        self.is_log_transformed = is_log_transformed

    def evaluate_predictions(self, 
                             y_true: np.ndarray, 
                             y_pred: np.ndarray, 
                             model_name: str = "Model") -> Dict[str, Any]:
        """
        Calculates standard regression metrics. If the target was log-transformed,
        it also computes MAE and RMSE in the original scale.

        Args:
            y_true (np.ndarray): Actual target values (in log space if transformed).
            y_pred (np.ndarray): Predicted target values.
            model_name (str): Model identifier.

        Returns:
            Dict[str, Any]: Dictionary containing all calculated metrics.
        """
        # Metrics in the log-transformed space
        r2 = r2_score(y_true, y_pred)
        mae_log = mean_absolute_error(y_true, y_pred)
        rmse_log = np.sqrt(mean_squared_error(y_true, y_pred))
        
        metrics = {
            "Model": model_name,
            "R2_Score": r2,
            "MAE_Log": mae_log,
            "RMSE_Log": rmse_log
        }

        # Metrics in the original space
        if self.is_log_transformed:
            # Reverse the np.log1p transformation
            y_true_raw = np.expm1(y_true)
            
            # Prevent overflow issues with extremely poor predictions with clipping
            y_pred_clipped = np.clip(y_pred, a_min=None, a_max=20.0) 
            y_pred_raw = np.expm1(y_pred_clipped)
            
            mae_raw = mean_absolute_error(y_true_raw, y_pred_raw)
            rmse_raw = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
            
            metrics["MAE_Raw"] = mae_raw
            metrics["RMSE_Raw"] = rmse_raw

        return metrics

    def print_report(self, metrics: Dict[str, Any]) -> None:
        """
        Prints a formatted report of the evaluated metrics.
        """
        print(f"--- Evaluation Report: {metrics['Model']} ---")
        print(f"R2 Score: {metrics['R2_Score']:.4f}")
        print(f"MAE (Log): {metrics['MAE_Log']:.4f}")
        print(f"RMSE (Log): {metrics['RMSE_Log']:.4f}")

        if "Train_Time_Seconds" in metrics:
            print(f"Training Time: {metrics['Train_Time_Seconds']:.2f} seconds")

        if "Grid_Search_Time_Seconds" in metrics:
            print(f"Grid Search Time: {metrics['Grid_Search_Time_Seconds']:.2f} seconds")

        if "CV_R2_Mean" in metrics:
            print(f"CV R2: {metrics['CV_R2_Mean']:.4f} +/- {metrics['CV_R2_Std']:.4f}")
            print(f"CV MAE: {metrics['CV_MAE_Mean']:.4f} +/- {metrics['CV_MAE_Std']:.4f}")
            print(f"CV RMSE: {metrics['CV_RMSE_Mean']:.4f} +/- {metrics['CV_RMSE_Std']:.4f}")
        
        if self.is_log_transformed:
            print(f"MAE (Shares): {metrics['MAE_Raw']:,.2f}")
            print(f"RMSE (Shares): {metrics['RMSE_Raw']:,.2f}")
        print("-" * 40 + "\n")