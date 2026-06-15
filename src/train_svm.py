import os
import logging
import joblib
from time import perf_counter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from evaluation import RegressionEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
DATA_PATH = "data/preprocessed.csv"
MODEL_DIR = "models/"
METRICS_DIR = "metrics/"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_and_split_data(data_path: str, test_size: float, random_state: int):
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


def report_model(model, X_test, y_test, evaluator, model_name, model_path, train_time_seconds):
    y_pred = model.predict(X_test)

    metrics = evaluator.evaluate_predictions(
        y_true=y_test.values,
        y_pred=y_pred,
        model_name=model_name
    )
    metrics["Train_Time_Seconds"] = train_time_seconds
    evaluator.print_report(metrics)
    logger.info(f"{model_name} training time: {metrics['Train_Time_Seconds']:.2f} seconds")
    joblib.dump(model, model_path)
    return metrics


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_and_split_data(DATA_PATH, TEST_SIZE, RANDOM_STATE)

    evaluator = RegressionEvaluator(is_log_transformed=True)
    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])

    # Define model specifications for different SVR kernels
    # With better hardware, we would perform a much more exhaustive hyperparameter search
    model_specs = [
        (
            "SVR_Linear",
            {
                'svr__kernel': 'linear',
                'svr__C': 10.0,
                'svr__epsilon': 0.1,
            },
            os.path.join(MODEL_DIR, "svr_linear_optimal.pkl")
        ),
        (
            "SVR_Poly",
            {
                'svr__kernel': 'poly',
                'svr__C': 10.0,
                'svr__degree': 3,
                'svr__gamma': 'scale',
                'svr__epsilon': 0.1,
            },
            os.path.join(MODEL_DIR, "svr_poly_optimal.pkl")
        ),
        (
            "SVR_RBF",
            {
                'svr__kernel': 'rbf',
                'svr__C': 10.0,
                'svr__gamma': 'scale',
                'svr__epsilon': 0.1,
            },
            os.path.join(MODEL_DIR, "svr_rbf_optimal.pkl")
        )
    ]

    # Train and evaluate each model
    all_metrics = []
    for model_name, params, model_path in model_specs:
        logger.info(f"Training {model_name}")
        model = base_pipeline.set_params(**params)
        train_start = perf_counter()
        model.fit(X_train, y_train)
        train_time_seconds = perf_counter() - train_start

        metrics = report_model(model, X_test, y_test, evaluator, model_name, model_path, train_time_seconds)
        all_metrics.append(metrics)

    # Save metrics to CSV
    metrics_path = os.path.join(METRICS_DIR, "model_metrics.csv")
    df_new_metrics = pd.DataFrame(all_metrics)
    if os.path.exists(metrics_path):
        df_existing = pd.read_csv(metrics_path)
        df_final_metrics = pd.concat([df_existing, df_new_metrics], ignore_index=True)
    else:
        df_final_metrics = df_new_metrics
    df_final_metrics.to_csv(metrics_path, index=False)
    logger.info(f"All SVM metrics saved/appended to {metrics_path}")


if __name__ == "__main__":
    main()