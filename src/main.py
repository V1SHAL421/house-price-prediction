import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from train_model import build_model, build_preprocessor, train_baseline_model, train_xgbboost_model
from data_preprocessing import preprocess_data, split_data
from train_model import train_lr_model
from visualise_results import visualise_results

dataset = "data/AmesHousing.csv"

df = pd.read_csv(dataset)

df_preprocessed = preprocess_data(df)

num_seeds = 10

baseline_rmse_log_base_list = []
baseline_rmse_dollar_base_list = []
lr_rmse_log_base_list = []
lr_rmse_dollar_base_list = []
lr_r2_val_list = []
ridge_rmse_log_base_list = []
ridge_rmse_dollar_base_list = []
ridge_r2_val_list = []
xgbboost_rmse_log_base_list = []
xgbboost_rmse_dollar_base_list = []
xgbboost_r2_val_list = []



for i in range(num_seeds):
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(df_preprocessed)

    baseline_val_pred = train_baseline_model(X_train, y_train, X_val)

    baseline_rmse_log_base = root_mean_squared_error(y_val, baseline_val_pred)
    baseline_rmse_dollar_base   = root_mean_squared_error(np.expm1(y_val), np.expm1(baseline_val_pred))

    print(f"Baseline log-RMSE: {baseline_rmse_log_base:.3f}")
    print(f"Baseline RMSE: {baseline_rmse_dollar_base:.3f}")

    baseline_rmse_log_base_list.append(baseline_rmse_log_base)
    baseline_rmse_dollar_base_list.append(baseline_rmse_dollar_base)

    preprocessor = build_preprocessor(X_train)

    lr_model = build_model(preprocessor, "lr")

    lr_val_pred = train_lr_model(lr_model, X_train, y_train, X_val)

    lr_rmse_log_base = root_mean_squared_error(y_val, lr_val_pred)
    lr_rmse_dollar_base   = root_mean_squared_error(np.expm1(y_val), np.expm1(lr_val_pred))
    lr_r2_val = r2_score(y_val, lr_val_pred)

    print(f"LR model log-RMSE: {lr_rmse_log_base:.3f}")
    print(f"LR model RMSE: {lr_rmse_dollar_base:.3f}")
    print(f"LR model Validation R²: {lr_r2_val:.3f}")

    lr_rmse_log_base_list.append(lr_rmse_log_base)
    lr_rmse_dollar_base_list.append(lr_rmse_dollar_base)
    lr_r2_val_list.append(lr_r2_val)

    ridge_model = build_model(preprocessor, "ridge")

    ridge_val_pred = train_lr_model(ridge_model, X_train, y_train, X_val)

    ridge_rmse_log_base = root_mean_squared_error(y_val, ridge_val_pred)
    ridge_rmse_dollar_base   = root_mean_squared_error(np.expm1(y_val), np.expm1(ridge_val_pred))
    ridge_r2_val = r2_score(y_val, ridge_val_pred)

    print(f"Ridge LR model log-RMSE: {ridge_rmse_log_base:.3f}")
    print(f"Ridge LR model RMSE: {ridge_rmse_dollar_base:.3f}")
    print(f"Ridge LR model Validation R²: {ridge_r2_val:.3f}")

    ridge_rmse_log_base_list.append(ridge_rmse_log_base)
    ridge_rmse_dollar_base_list.append(ridge_rmse_dollar_base)
    ridge_r2_val_list.append(ridge_r2_val)

    xgbboost_model = build_model(preprocessor, "xgb")
    xgb_val_pred = train_xgbboost_model(xgbboost_model, X_train, y_train, X_val)

    xgb_rmse_log_base = root_mean_squared_error(y_val, xgb_val_pred)
    xgb_rmse_dollar_base   = root_mean_squared_error(np.expm1(y_val), np.expm1(xgb_val_pred))
    xgb_r2_val = r2_score(y_val, xgb_val_pred)

    print(f"XGBoost Val log-RMSE: {xgb_rmse_log_base:.3f}")
    print(f"XGBoost Val RMSE: {xgb_rmse_dollar_base:.3f}")
    print(f"XGBoost Val R²: {xgb_r2_val:.3f}")

    xgbboost_rmse_log_base_list.append(xgb_rmse_log_base)
    xgbboost_rmse_dollar_base_list.append(xgb_rmse_dollar_base)
    xgbboost_r2_val_list.append(xgb_r2_val)

print(f"Baseline log-RMSE: {np.mean(baseline_rmse_log_base_list):.3f} +/- {np.std(baseline_rmse_log_base_list):.3f}")
print(f"Baseline RMSE: {np.mean(baseline_rmse_dollar_base_list):.3f} +/- {np.std(baseline_rmse_dollar_base_list):.3f}")
print(f"LR model log-RMSE: {np.mean(lr_rmse_log_base_list):.3f} +/- {np.std(lr_rmse_log_base_list):.3f}")
print(f"LR model RMSE: {np.mean(lr_rmse_dollar_base_list):.3f} +/- {np.std(lr_rmse_dollar_base_list):.3f}")
print(f"LR model Validation R²: {np.mean(lr_r2_val_list):.3f} +/- {np.std(lr_r2_val_list):.3f}")
print(f"Ridge LR model log-RMSE: {np.mean(ridge_rmse_log_base_list):.3f} +/- {np.std(ridge_rmse_log_base_list):.3f}")
print(f"Ridge LR model RMSE: {np.mean(ridge_rmse_dollar_base_list):.3f} +/- {np.std(ridge_rmse_dollar_base_list):.3f}")
print(f"Ridge LR model Validation R²: {np.mean(ridge_r2_val_list):.3f} +/- {np.std(ridge_r2_val_list):.3f}")
print(f"XGBoost Val log-RMSE: {np.mean(xgbboost_rmse_log_base_list):.3f} +/- {np.std(xgbboost_rmse_log_base_list):.3f}")
print(f"XGBoost Val RMSE: {np.mean(xgbboost_rmse_dollar_base_list):.3f} +/- {np.std(xgbboost_rmse_dollar_base_list):.3f}")
print(f"XGBoost Val R²: {np.mean(xgbboost_r2_val_list):.3f} +/- {np.std(xgbboost_r2_val_list):.3f}")

models = ["Baseline", "Linear Regression", "Ridge Regression", "XGBBoost"]

visualise_results("R²", models)