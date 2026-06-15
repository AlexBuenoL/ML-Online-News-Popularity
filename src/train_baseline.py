import os
import logging
from time import perf_counter

import joblib
import pandas as pd

from sklearn.model_selection import (
    cross_validate,
    cross_val_predict,
    KFold
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from evaluation import RegressionEvaluator

# Logger setup
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
RANDOM_STATE = 42
CV_FOLDS = 5


def load_data(data_path: str):
    """
    Loads the dataset.
    """
    logger.info(f"Loading data from {data_path}...")

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error(f"Dataset not found at {data_path}.")
        raise

    target_col = "log_shares"

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' missing from dataset."
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info(
        f"Dataset loaded. Shape: {X.shape}"
    )

    return X, y


def evaluate_with_cross_validation(
    model,
    X,
    y,
    cv_folds,
    random_state
):
    """
    Runs K-Fold CV on the full dataset and computes metrics
    both in log space and original shares space.
    """

    from sklearn.base import clone
    import numpy as np

    cv = KFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state
    )

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error"
        },
        n_jobs=-1,
        return_train_score=False
    )

    evaluator = RegressionEvaluator(
        is_log_transformed=True
    )

    mae_shares_scores = []
    rmse_shares_scores = []

    for train_idx, test_idx in cv.split(X):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        y_pred = fold_model.predict(X_test)

        fold_metrics = evaluator.evaluate_predictions(
            y_true=y_test.values,
            y_pred=y_pred,
            model_name="OLS_Baseline"
        )

        mae_shares_scores.append(
            fold_metrics["MAE_Raw"]
        )

        rmse_shares_scores.append(
            fold_metrics["RMSE_Raw"]
        )

    metrics = {
        "Model": "OLS_Baseline",
        "CV_Folds": cv_folds,

        # Log-space metrics
        "CV_R2_Mean": results["test_r2"].mean(),
        "CV_R2_Std": results["test_r2"].std(),

        "CV_MAE_Mean": -results["test_mae"].mean(),
        "CV_MAE_Std": results["test_mae"].std(),

        "CV_RMSE_Mean": -results["test_rmse"].mean(),
        "CV_RMSE_Std": results["test_rmse"].std(),

        # Shares-space metrics
        "CV_MAE_Shares_Mean": np.mean(mae_shares_scores),
        "CV_MAE_Shares_Std": np.std(mae_shares_scores),

        "CV_RMSE_Shares_Mean": np.mean(rmse_shares_scores),
        "CV_RMSE_Shares_Std": np.std(rmse_shares_scores),

        # Timing
        "CV_Fit_Time_Mean": results["fit_time"].mean(),
        "CV_Fit_Time_Std": results["fit_time"].std(),

        "CV_Score_Time_Mean": results["score_time"].mean(),
        "CV_Score_Time_Std": results["score_time"].std(),
    }

    return metrics

def main():

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    X, y = load_data(DATA_PATH)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

    logger.info(
        f"Running {CV_FOLDS}-fold CV on the full dataset..."
    )

    start_time = perf_counter()

    metrics = evaluate_with_cross_validation(
        model=model,
        X=X,
        y=y,
        cv_folds=CV_FOLDS,
        random_state=RANDOM_STATE
    )

    elapsed_time = perf_counter() - start_time


    # --------------------------------------------------
    # Logging
    # --------------------------------------------------
    logger.info(
        "CV Results - "
        f"R2: {metrics['CV_R2_Mean']:.4f} +/- {metrics['CV_R2_Std']:.4f} | "
        f"MAE(Log): {metrics['CV_MAE_Mean']:.4f} +/- {metrics['CV_MAE_Std']:.4f} | "
        f"RMSE(Log): {metrics['CV_RMSE_Mean']:.4f} +/- {metrics['CV_RMSE_Std']:.4f} | "
        f"Fit Time: {metrics['CV_Fit_Time_Mean']:.4f} +/- {metrics['CV_Fit_Time_Std']:.4f} | "
        f"MAE (Shares): {metrics['CV_MAE_Shares_Mean']:.4f} +/- {metrics['CV_MAE_Shares_Std']:.4f} | "
        f"RMSE (Shares): {metrics['CV_RMSE_Shares_Mean']:.4f} +/- {metrics['CV_RMSE_Shares_Std']:.4f}"
    )

    logger.info(
        "Training final model on the full dataset..."
    )

    model.fit(X, y)

    metrics["Train_Time_Seconds"] = elapsed_time

    model_path = os.path.join(
        MODEL_DIR,
        "ols_baseline.pkl"
    )

    joblib.dump(model, model_path)

    logger.info(
        f"Model saved to {model_path}"
    )

    metrics_path = os.path.join(
        METRICS_DIR,
        "baseline_metrics.csv"
    )

    df_metrics = pd.DataFrame([metrics])

    df_metrics.to_csv(
        metrics_path,
        index=False
    )

    logger.info(
        f"Metrics saved to {metrics_path}"
    )

    # --------------------------------------------------
    # Coefficients
    # --------------------------------------------------
    coefficients = model.named_steps["regressor"].coef_
    intercept = model.named_steps["regressor"].intercept_

    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": coefficients
    }).sort_values(
        "Coefficient",
        key=abs,
        ascending=False
    )

    logger.info(f"Intercept: {intercept:.6f}")

    logger.info("Top 5 Coefficients:")

    logger.info(
        "\n%s",
        coef_df.head(5).to_string(index=False)
    )

    logger.info(
        "OLS CV pipeline finished successfully."
    )


if __name__ == "__main__":
    main()