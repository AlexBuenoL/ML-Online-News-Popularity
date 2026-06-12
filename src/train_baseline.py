import os
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

    # 2. Model initialization and training
    logger.info("Initializing OLS Regression model...")
    model = LinearRegression()

    logger.info("Training model...")
    model.fit(X_train, y_train)
    logger.info("Training complete.")

    # 3. Model inference
    logger.info("Generating predictions on the test set...")
    y_pred = model.predict(X_test)

    # 4. Evaluation
    logger.info("Evaluating predictions...")
    evaluator = RegressionEvaluator(is_log_transformed=True)
    metrics = evaluator.evaluate_predictions(
        y_true=y_test.values, 
        y_pred=y_pred, 
        model_name="OLS_Baseline"
    )
    
    # Print report
    evaluator.print_report(metrics)

    # 5. Save Model
    model_path = os.path.join(MODEL_DIR, "ols_baseline.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save Metrics
    metrics_path = os.path.join(METRICS_DIR, "baseline_metrics.csv")
    df_metrics = pd.DataFrame([metrics])
    
    # Append to existing metrics file if it exists, otherwise create it
    if os.path.exists(metrics_path):
        df_existing = pd.read_csv(metrics_path)
        df_metrics = pd.concat([df_existing, df_metrics], ignore_index=True)
    
    df_metrics.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info("Baseline training pipeline finished successfully.")

if __name__ == "__main__":
    main()