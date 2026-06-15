import os
import logging
from time import perf_counter

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

from evaluation import RegressionEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

DATA_PATH   = "data/preprocessed.csv"
MODEL_DIR   = "models/"
METRICS_DIR = "metrics/"
TEST_SIZE    = 0.2
RANDOM_STATE = 42
CV_FOLDS     = 5
PARAM_GRIDS = {
    "RandomForest": {
        "n_estimators":     [100, 200],
        "max_depth":        [None, 10, 20],
        "min_samples_split":[2, 5],
        "min_samples_leaf": [1, 2],
    },
    "GradientBoosting": {
        "n_estimators":  [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth":     [2, 3],
        "subsample":     [0.8, 1.0],
    },
}


def load_data(path: str):
    logger.info(f"Loading data from {path}...")
    df = pd.read_csv(path)

    if "log_shares" not in df.columns:
        raise ValueError("Missing target column: 'log_shares'")

    X = df.drop(columns=["log_shares"])
    y = df["log_shares"]

    logger.info(f"Dataset shape: {X.shape}")
    return X, y


def grid_search_cv(base_model, param_grid: dict, X_train, y_train):

    evaluator = RegressionEvaluator(is_log_transformed=True)
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Run grid search to find best hyperparameters
    logger.info("Running GridSearchCV...")
    gs_start = perf_counter()

    gs = GridSearchCV(
        estimator=clone(base_model),
        param_grid=param_grid,
        cv=kf,
        scoring="r2",
        n_jobs=-1,
        refit=False,
        verbose=0,
    )
    gs.fit(X_train, y_train)

    gs_time = perf_counter() - gs_start
    best_params = gs.best_params_
    logger.info(f"Best params: {best_params}  (grid-search took {gs_time:.2f}s)")

    # We run CV again with the best params to get mean/std metrics for logging
    best_base = clone(base_model).set_params(**best_params)

    r2_log_folds, mae_log_folds, rmse_log_folds = [], [], []
    mae_raw_folds, rmse_raw_folds, fit_time_folds = [], [], []

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        fold_model = clone(best_base)

        t0 = perf_counter()
        fold_model.fit(X_tr, y_tr)
        fit_time_folds.append(perf_counter() - t0)

        preds = fold_model.predict(X_val)

        # Get metrics
        metrics = evaluator.evaluate_predictions(y_val.values, preds)
        r2_log_folds.append(metrics["R2_Score"])
        mae_log_folds.append(metrics["MAE_Log"])
        rmse_log_folds.append(metrics["RMSE_Log"])
        mae_raw_folds.append(metrics["MAE_Raw"])
        rmse_raw_folds.append(metrics["RMSE_Raw"])

        logger.info(
            f"--- Fold {fold_i}/{CV_FOLDS} --- \n"
            f"R2: {r2_log_folds[-1]:.4f}\n"
            f"MAE(log): {mae_log_folds[-1]:.4f}\n"
            f"RMSE(log): {rmse_log_folds[-1]:.4f}\n"
            f"MAE(raw): {mae_raw_folds[-1]:,.1f}\n"
            f"RMSE(raw): {rmse_raw_folds[-1]:,.1f}\n"
            f"Train Time: {fit_time_folds[-1]:.2f}s"
        )

    cv_metrics = {
        "CV_R2_Log_Mean": np.mean(r2_log_folds),
        "CV_R2_Log_Std": np.std(r2_log_folds),
        "CV_MAE_Log_Mean": np.mean(mae_log_folds),
        "CV_MAE_Log_Std": np.std(mae_log_folds),
        "CV_RMSE_Log_Mean": np.mean(rmse_log_folds),
        "CV_RMSE_Log_Std": np.std(rmse_log_folds),
        "CV_MAE_Raw_Mean": np.mean(mae_raw_folds),
        "CV_MAE_Raw_Std": np.std(mae_raw_folds),
        "CV_RMSE_Raw_Mean": np.mean(rmse_raw_folds),
        "CV_RMSE_Raw_Std": np.std(rmse_raw_folds),
        "CV_FitTime_Mean": np.mean(fit_time_folds),
        "CV_FitTime_Std": np.std(fit_time_folds),
        "GridSearch_Time_Sec": gs_time,
    }

    logger.info("Refitting using best hyperparameters on the entire training set...")
    best_estimator = clone(best_base)
    best_estimator.fit(X_train, y_train)

    return best_estimator, best_params, cv_metrics


def evaluate_on_test(model, X_test, y_test) -> dict:
    evaluator = RegressionEvaluator(is_log_transformed=True)

    preds = model.predict(X_test)
    metrics = evaluator.evaluate_predictions(
        y_true=y_test.values,
        y_pred=preds,
        model_name="Test",
    )

    return {
        "Test_R2_Log": metrics["R2_Score"],
        "Test_MAE_Log": metrics["MAE_Log"],
        "Test_RMSE_Log": metrics["RMSE_Log"],
        "Test_MAE_Raw": metrics["MAE_Raw"],
        "Test_RMSE_Raw": metrics["RMSE_Raw"]
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    X, y = load_data(DATA_PATH)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    base_models = [
        ("RandomForest", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
        ("GradientBoosting", GradientBoostingRegressor(random_state=RANDOM_STATE)),
    ]

    all_metrics = []

    for name, base_model in base_models:
        logger.info(f"MODEL: {name}")

        # Run grid search CV to find best hyperparameters
        best_model, best_params, cv_metrics = grid_search_cv(
            base_model, PARAM_GRIDS[name], X_train, y_train
        )

        # Evaluate on test set
        test_metrics = evaluate_on_test(best_model, X_test, y_test)

        # Save the best model to disk
        model_path = os.path.join(MODEL_DIR, f"{name.lower()}_best.pkl")
        joblib.dump(best_model, model_path)
        logger.info(f"Saved model to {model_path}")

        # Get best hyperparameters and metrics for logging
        row = {"Model": name, **{f"BestParam_{k}": v for k, v in best_params.items()}}
        row.update(cv_metrics)
        row.update(test_metrics)
        all_metrics.append(row)

        # Summary log
        logger.info(
            f"SUMMARY ({name})\n"
            f"CV R2: {cv_metrics['CV_R2_Log_Mean']:.4f}, std: {cv_metrics['CV_R2_Log_Std']:.4f}\n"
            f"CV MAE (log): {cv_metrics['CV_MAE_Log_Mean']:.4f}, std: {cv_metrics['CV_MAE_Log_Std']:.4f}\n"
            f"CV RMSE (log): {cv_metrics['CV_RMSE_Log_Mean']:.4f}, std: {cv_metrics['CV_RMSE_Log_Std']:.4f}\n"
            f"CV MAE (raw): {cv_metrics['CV_MAE_Raw_Mean']:,.1f}, std: {cv_metrics['CV_MAE_Raw_Std']:,.1f}\n"
            f"CV RMSE (raw): {cv_metrics['CV_RMSE_Raw_Mean']:,.1f}, std: {cv_metrics['CV_RMSE_Raw_Std']:,.1f}\n"
            f"CV Train Time: {cv_metrics['CV_FitTime_Mean']:.2f}s, std: {cv_metrics['CV_FitTime_Std']:.2f}s\n"
            f"TEST R2: {test_metrics['Test_R2_Log']:.4f}\n"
            f"TEST MAE (log): {test_metrics['Test_MAE_Log']:.4f}\n"
            f"TEST RMSE (log):{test_metrics['Test_RMSE_Log']:.4f}\n"
            f"TEST MAE (raw): {test_metrics['Test_MAE_Raw']:,.1f}\n"
            f"TEST RMSE (raw):{test_metrics['Test_RMSE_Raw']:,.1f}"
        )

    # Save metrics to CSV
    df_metrics = pd.DataFrame(all_metrics)
    out_path = os.path.join(METRICS_DIR, "ensemble_metrics.csv")
    df_metrics.to_csv(out_path, index=False)
    logger.info(f"\nSaved metrics: {out_path}")
    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()