import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso

from evaluation import RegressionEvaluator

# Login setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Config variables
DATA_PATH = "../data/preprocessed.csv"
MODEL_DIR = "../models/"
METRICS_DIR = "../metrics/"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Define grid of lambdas
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

def load_and_split_data(data_path: str, test_size: float, random_state: int):
    """Loads the dataset and splits the data into train and test."""
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    target_col = 'log_shares'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # 1. Data loading
    X_train, X_test, y_train, y_test = load_and_split_data(
        DATA_PATH, TEST_SIZE, RANDOM_STATE
    )
    feature_names = X_train.columns.tolist()
    evaluator = RegressionEvaluator(is_log_transformed=True)

    # Store regularization paths
    performance_metrics = []
    ridge_coef_path = []
    lasso_coef_path = []

    best_ridge_val_r2 = -float('inf')
    best_ridge_model = None
    
    best_lasso_val_r2 = -float('inf')
    best_lasso_model = None

    logger.info("Starting Regularization analysis...")

    # 2. Iterate over alphas (lambdas)
    for alpha in ALPHAS:
        logger.info(f"--- Evaluating lambda = {alpha} ---")

        # Ridge
        ridge = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        ridge.fit(X_train, y_train)
        
        # Evaluate train and validation
        ridge_train_preds = ridge.predict(X_train)
        ridge_val_preds = ridge.predict(X_test)
        
        ridge_train_metrics = evaluator.evaluate_predictions(y_train.values, ridge_train_preds, f"Ridge_Train_a={alpha}")
        ridge_val_metrics = evaluator.evaluate_predictions(y_test.values, ridge_val_preds, f"Ridge_Val_a={alpha}")
        
        # Store metrics and coefficients
        performance_metrics.append({'Model': 'Ridge', 'Alpha': alpha, 'Split': 'Train', **ridge_train_metrics})
        performance_metrics.append({'Model': 'Ridge', 'Alpha': alpha, 'Split': 'Validation', **ridge_val_metrics})
        
        ridge_coefs = {'Alpha': alpha}
        ridge_coefs.update({feat: coef for feat, coef in zip(feature_names, ridge.coef_)})
        ridge_coef_path.append(ridge_coefs)

        # Store best model
        if ridge_val_metrics['R2_Score'] > best_ridge_val_r2:
            best_ridge_val_r2 = ridge_val_metrics['R2_Score']
            best_ridge_model = ridge

        # Lasso
        lasso = Lasso(alpha=alpha, random_state=RANDOM_STATE, max_iter=20000)
        lasso.fit(X_train, y_train)
        
        # Evaluate train and validation
        lasso_train_preds = lasso.predict(X_train)
        lasso_val_preds = lasso.predict(X_test)
        
        lasso_train_metrics = evaluator.evaluate_predictions(y_train.values, lasso_train_preds, f"Lasso_Train_a={alpha}")
        lasso_val_metrics = evaluator.evaluate_predictions(y_test.values, lasso_val_preds, f"Lasso_Val_a={alpha}")
        
        # Store metrics and coefficients
        performance_metrics.append({'Model': 'Lasso', 'Alpha': alpha, 'Split': 'Train', **lasso_train_metrics})
        performance_metrics.append({'Model': 'Lasso', 'Alpha': alpha, 'Split': 'Validation', **lasso_val_metrics})
        
        lasso_coefs = {'Alpha': alpha}
        lasso_coefs.update({feat: coef for feat, coef in zip(feature_names, lasso.coef_)})
        lasso_coef_path.append(lasso_coefs)

        # Store best model
        if lasso_val_metrics['R2_Score'] > best_lasso_val_r2:
            best_lasso_val_r2 = lasso_val_metrics['R2_Score']
            best_lasso_model = lasso

    # 3. Export results for analysis
    logger.info("Exporting metrics and coefficients...")

    # Save performance metrics
    df_perf = pd.DataFrame(performance_metrics)
    df_perf = df_perf.loc[:, ~df_perf.columns.duplicated()] 
    df_perf.to_csv(os.path.join(METRICS_DIR, "regularization_path_metrics.csv"), index=False)

    # Save coefficient paths
    pd.DataFrame(ridge_coef_path).to_csv(os.path.join(METRICS_DIR, "ridge_coefficient_path.csv"), index=False)
    pd.DataFrame(lasso_coef_path).to_csv(os.path.join(METRICS_DIR, "lasso_coefficient_path.csv"), index=False)

    # Save best model
    joblib.dump(best_ridge_model, os.path.join(MODEL_DIR, "ridge_optimal.pkl"))
    joblib.dump(best_lasso_model, os.path.join(MODEL_DIR, "lasso_optimal.pkl"))

    logger.info("Regularized training and evaluation completed successfully.")

if __name__ == "__main__":
    main()