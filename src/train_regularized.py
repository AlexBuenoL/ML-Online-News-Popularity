import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LassoCV

from evaluation import RegressionEvaluator

# Logger setup
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
CV_FOLDS = 5

# Define the list of lambdas to try
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

def load_and_split_data(data_path: str, test_size: float, random_state: int):
    """
    Loads the dataset and splits the data into train and test.
    """
    logger.info(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error(f"Dataset not found at {data_path}. Please check the path.")
        raise

    target_col = 'log_shares'
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' missing from dataset.")
        raise ValueError(f"Missing target column: {target_col}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info(f"Splitting data with test_size={test_size} and random_state={random_state}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def save_coefficients(model, feature_names: list, model_name: str):
    """
    Extracts model coefficients and saves them to a CSV for later analysis.
    """
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False).drop(columns=['Abs_Coefficient'])
    
    coef_path = os.path.join(METRICS_DIR, f"{model_name.lower()}_coefficients.csv")
    coef_df.to_csv(coef_path, index=False)
    logger.info(f"Coefficients for {model_name} saved to {coef_path}")

def main():
    # Ensure output directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # 1. Data loading
    X_train, X_test, y_train, y_test = load_and_split_data(
        data_path=DATA_PATH,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    feature_names = X_train.columns.tolist()
    evaluator = RegressionEvaluator(is_log_transformed=True)
    all_metrics = []

    # 2. RIDGE REGRESSION (L2 Regularization)
    logger.info("Initializing Ridge Regression model with CV...")
    ridge_model = RidgeCV(alphas=ALPHAS, cv=CV_FOLDS)
    
    logger.info("Training Ridge model...")
    ridge_model.fit(X_train, y_train)
    optimal_ridge_alpha = ridge_model.alpha_
    logger.info(f"Ridge training complete. Optimal Alpha (lambda) found: {optimal_ridge_alpha}")

    # Inference and evaluation
    ridge_preds = ridge_model.predict(X_test)
    ridge_metrics = evaluator.evaluate_predictions(
        y_true=y_test.values, 
        y_pred=ridge_preds, 
        model_name=f"Ridge (alpha={optimal_ridge_alpha})"
    )
    evaluator.print_report(ridge_metrics)
    all_metrics.append(ridge_metrics)

    # Save Artifacts
    joblib.dump(ridge_model, os.path.join(MODEL_DIR, "ridge_optimal.pkl"))
    save_coefficients(ridge_model, feature_names, "Ridge")


    # 3. LASSO REGRESSION (L1 Regularization)
    logger.info("Initializing Lasso Regression model with CV...")
    lasso_model = LassoCV(alphas=ALPHAS, cv=CV_FOLDS, random_state=RANDOM_STATE, max_iter=10000)
    
    logger.info("Training Lasso model...")
    lasso_model.fit(X_train, y_train)
    optimal_lasso_alpha = lasso_model.alpha_
    logger.info(f"Lasso training complete. Optimal Alpha (lambda) found: {optimal_lasso_alpha}")

    # Inference and evaluation
    lasso_preds = lasso_model.predict(X_test)
    lasso_metrics = evaluator.evaluate_predictions(
        y_true=y_test.values, 
        y_pred=lasso_preds, 
        model_name=f"Lasso (alpha={optimal_lasso_alpha})"
    )
    evaluator.print_report(lasso_metrics)
    all_metrics.append(lasso_metrics)

    # Save Artifacts
    joblib.dump(lasso_model, os.path.join(MODEL_DIR, "lasso_optimal.pkl"))
    save_coefficients(lasso_model, feature_names, "Lasso")


    # 4. Save Final Metrics
    metrics_path = os.path.join(METRICS_DIR, "model_metrics.csv")
    df_new_metrics = pd.DataFrame(all_metrics)
    
    if os.path.exists(metrics_path):
        df_existing = pd.read_csv(metrics_path)
        df_final_metrics = pd.concat([df_existing, df_new_metrics], ignore_index=True)
    else:
        df_final_metrics = df_new_metrics
        
    df_final_metrics.to_csv(metrics_path, index=False)
    logger.info(f"All metrics saved/appended to {metrics_path}")
    logger.info("Regularized models training finished successfully.")

if __name__ == "__main__":
    main()