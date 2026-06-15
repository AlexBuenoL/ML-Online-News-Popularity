import os
import time
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge, Lasso

from evaluation import RegressionEvaluator

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Config variables
DATA_PATH = "data/preprocessed.csv"
MODEL_DIR = "models/"
METRICS_DIR = "metrics/"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS = 5

# Define grid of lambdas
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

def load_and_split_data(data_path: str, test_size: float, random_state: int):
    """Loads the dataset and isolates the test set."""
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    target_col = 'log_shares'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Isolate test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def execute_cv_for_model(ModelClass, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, evaluator: RegressionEvaluator):
    """
    Executes K-Fold CV across all lambdas. 
    Returns path data (Fold 1) and the optimal model's aggregated CV metrics.
    """
    feature_names = X_train.columns.tolist()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    path_metrics = []
    coef_path = []
    
    best_mean_val_r2 = -float('inf')
    best_alpha = None
    best_cv_aggregated_metrics = {}

    for alpha in ALPHAS:
        logger.info(f"[{model_name}] Evaluating lambda = {alpha}...")
        
        fold_val_r2 = []
        fold_metrics_list = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Initialize model with lambda
            if model_name == "Lasso":
                model = ModelClass(alpha=alpha, random_state=RANDOM_STATE, max_iter=20000)
            else:
                model = ModelClass(alpha=alpha, random_state=RANDOM_STATE)
            
            # Measure training time
            start_time = time.time()
            model.fit(X_cv_train, y_cv_train)
            train_time = time.time() - start_time
            
            # Predict
            train_preds = model.predict(X_cv_train)
            val_preds = model.predict(X_cv_val)
            
            # Evaluate
            train_metrics = evaluator.evaluate_predictions(y_cv_train.values, train_preds, f"{model_name}_Train")
            val_metrics = evaluator.evaluate_predictions(y_cv_val.values, val_preds, f"{model_name}_Val")
            
            # Add training time for aggregation
            val_metrics['Training_Time_s'] = train_time
            fold_metrics_list.append(val_metrics)
            fold_val_r2.append(val_metrics['R2_Score'])
            
            # Extract Fold 0 data for regularization effect analysis
            if fold_idx == 0:
                path_metrics.append({'Model': model_name, 'Alpha': alpha, 'Split': 'Train', **train_metrics})
                path_metrics.append({'Model': model_name, 'Alpha': alpha, 'Split': 'Validation', **val_metrics})
                
                coefs = {'Alpha': alpha}
                coefs.update({feat: coef for feat, coef in zip(feature_names, model.coef_)})
                coef_path.append(coefs)

        # Aggregate K-Fold metrics for current alpha
        mean_val_r2 = np.mean(fold_val_r2)
        
        # Check if this alpha provides the best cross-validation performance
        if mean_val_r2 > best_mean_val_r2:
            best_mean_val_r2 = mean_val_r2
            best_alpha = alpha
            
            # Calculate mean and stdev for all metrics
            best_cv_aggregated_metrics = {
                'Model': model_name,
                'Alpha': alpha,
                'R2_Mean': np.mean([m['R2_Score'] for m in fold_metrics_list]),
                'R2_Std': np.std([m['R2_Score'] for m in fold_metrics_list]),
                'MAE_log_Mean': np.mean([m['MAE_Log'] for m in fold_metrics_list]),
                'MAE_log_Std': np.std([m['MAE_Log'] for m in fold_metrics_list]),
                'RMSE_log_Mean': np.mean([m['RMSE_Log'] for m in fold_metrics_list]),
                'RMSE_log_Std': np.std([m['RMSE_Log'] for m in fold_metrics_list]),
                'MAE_orig_Mean': np.mean([m['MAE_Raw'] for m in fold_metrics_list]),
                'MAE_orig_Std': np.std([m['MAE_Raw'] for m in fold_metrics_list]),
                'RMSE_orig_Mean': np.mean([m['RMSE_Raw'] for m in fold_metrics_list]),
                'RMSE_orig_Std': np.std([m['RMSE_Raw'] for m in fold_metrics_list]),
                'Train_Time_Mean': np.mean([m['Training_Time_s'] for m in fold_metrics_list]),
                'Train_Time_Std': np.std([m['Training_Time_s'] for m in fold_metrics_list])
            }

    logger.info(f"Optimal {model_name} Alpha found via {N_SPLITS}-Fold CV: {best_alpha} (Val R2: {best_mean_val_r2:.4f})")
    
    return path_metrics, coef_path, best_cv_aggregated_metrics, best_alpha

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # 1. Data Loading
    X_train, X_test, y_train, y_test = load_and_split_data(
        DATA_PATH, TEST_SIZE, RANDOM_STATE
    )
    evaluator = RegressionEvaluator(is_log_transformed=True)

    logger.info("Starting Regularization CV analysis...")

    # 2. Execute K-Fold CV
    ridge_path_metrics, ridge_coefs, best_ridge_cv, best_ridge_alpha = execute_cv_for_model(
        Ridge, "Ridge", X_train, y_train, evaluator
    )
    
    lasso_path_metrics, lasso_coefs, best_lasso_cv, best_lasso_alpha = execute_cv_for_model(
        Lasso, "Lasso", X_train, y_train, evaluator
    )

    # Combine metrics
    all_path_metrics = pd.DataFrame(ridge_path_metrics + lasso_path_metrics)

    # Format the Best CV Metrics
    df_best_cv = pd.DataFrame([best_ridge_cv, best_lasso_cv])
    
    # 3. Retrain on the full dataset and evaluating on test
    logger.info("Retraining optimal models on entire training set for test evaluation...")
    
    # Ridge final
    final_ridge = Ridge(alpha=best_ridge_alpha, random_state=RANDOM_STATE)
    final_ridge.fit(X_train, y_train)
    ridge_test_preds = final_ridge.predict(X_test)
    ridge_test_metrics = evaluator.evaluate_predictions(y_test.values, ridge_test_preds, f"Ridge (a={best_ridge_alpha})")
    
    # Lasso final
    final_lasso = Lasso(alpha=best_lasso_alpha, random_state=RANDOM_STATE, max_iter=20000)
    final_lasso.fit(X_train, y_train)
    lasso_test_preds = final_lasso.predict(X_test)
    lasso_test_metrics = evaluator.evaluate_predictions(y_test.values, lasso_test_preds, f"Lasso (a={best_lasso_alpha})")
    
    df_test_metrics = pd.DataFrame([ridge_test_metrics, lasso_test_metrics])
    # 4. Export results
    logger.info("Exporting all metrics and coefficients...")

    # Regularization effects
    all_path_metrics = all_path_metrics.loc[:, ~all_path_metrics.columns.duplicated()] 
    all_path_metrics.to_csv(os.path.join(METRICS_DIR, "regularization_path_metrics.csv"), index=False)
    pd.DataFrame(ridge_coefs).to_csv(os.path.join(METRICS_DIR, "ridge_coefficient_path.csv"), index=False)
    pd.DataFrame(lasso_coefs).to_csv(os.path.join(METRICS_DIR, "lasso_coefficient_path.csv"), index=False)

    # CV and test results
    df_best_cv.to_csv(os.path.join(METRICS_DIR, "cv_best_metrics.csv"), index=False)
    df_test_metrics.to_csv(os.path.join(METRICS_DIR, "test_final_metrics.csv"), index=False)

    # C. Save final models
    joblib.dump(final_ridge, os.path.join(MODEL_DIR, "ridge_optimal.pkl"))
    joblib.dump(final_lasso, os.path.join(MODEL_DIR, "lasso_optimal.pkl"))

    logger.info("Regularized training and evaluation completed successfully.")
    
    # Print final test metrics
    evaluator.print_report(ridge_test_metrics)
    evaluator.print_report(lasso_test_metrics)

if __name__ == "__main__":
    main()