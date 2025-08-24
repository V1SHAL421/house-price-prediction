import pandas as pd
import numpy as np

from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from statistical_hypothesis_testing import test_hypotheses
from train_model import build_model, build_preprocessor, train_baseline_model, train_xgbboost_model
from data_preprocessing import preprocess_data, split_data
from train_model import train_lr_model
from visualise_results import visualise_results

dataset = "data/AmesHousing.csv"

df = pd.read_csv(dataset)

df_preprocessed = preprocess_data(df)

num_seeds = 100

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

    # print(f"Baseline log-RMSE: {baseline_rmse_log_base:.3f}")
    # print(f"Baseline RMSE: {baseline_rmse_dollar_base:.3f}")

    baseline_rmse_log_base_list.append(baseline_rmse_log_base)
    baseline_rmse_dollar_base_list.append(baseline_rmse_dollar_base)

    preprocessor = build_preprocessor(X_train)

    lr_model = build_model(preprocessor, "lr")

    lr_val_pred = train_lr_model(lr_model, X_train, y_train, X_val)

    lr_rmse_log_base = root_mean_squared_error(y_val, lr_val_pred)
    lr_rmse_dollar_base   = root_mean_squared_error(np.expm1(y_val), np.expm1(lr_val_pred))
    lr_r2_val = r2_score(y_val, lr_val_pred)

    # print(f"LR model log-RMSE: {lr_rmse_log_base:.3f}")
    # print(f"LR model RMSE: {lr_rmse_dollar_base:.3f}")
    # print(f"LR model Validation R²: {lr_r2_val:.3f}")

    lr_rmse_log_base_list.append(lr_rmse_log_base)
    lr_rmse_dollar_base_list.append(lr_rmse_dollar_base)
    lr_r2_val_list.append(lr_r2_val)

    ridge_model = build_model(preprocessor, "ridge")

    ridge_val_pred = train_lr_model(ridge_model, X_train, y_train, X_val)

    ridge_rmse_log_base = root_mean_squared_error(y_val, ridge_val_pred)
    ridge_rmse_dollar_base   = root_mean_squared_error(np.expm1(y_val), np.expm1(ridge_val_pred))
    ridge_r2_val = r2_score(y_val, ridge_val_pred)

    # print(f"Ridge LR model log-RMSE: {ridge_rmse_log_base:.3f}")
    # print(f"Ridge LR model RMSE: {ridge_rmse_dollar_base:.3f}")
    # print(f"Ridge LR model Validation R²: {ridge_r2_val:.3f}")

    ridge_rmse_log_base_list.append(ridge_rmse_log_base)
    ridge_rmse_dollar_base_list.append(ridge_rmse_dollar_base)
    ridge_r2_val_list.append(ridge_r2_val)

    xgbboost_model = build_model(preprocessor, "xgb")
    xgb_val_pred = train_xgbboost_model(xgbboost_model, X_train, y_train, X_val)

    xgb_rmse_log_base = root_mean_squared_error(y_val, xgb_val_pred)
    xgb_rmse_dollar_base   = root_mean_squared_error(np.expm1(y_val), np.expm1(xgb_val_pred))
    xgb_r2_val = r2_score(y_val, xgb_val_pred)

    # print(f"XGBoost Val log-RMSE: {xgb_rmse_log_base:.3f}")
    # print(f"XGBoost Val RMSE: {xgb_rmse_dollar_base:.3f}")
    # print(f"XGBoost Val R²: {xgb_r2_val:.3f}")

    xgbboost_rmse_log_base_list.append(xgb_rmse_log_base)
    xgbboost_rmse_dollar_base_list.append(xgb_rmse_dollar_base)
    xgbboost_r2_val_list.append(xgb_r2_val)

mean_baseline_log_rmse = np.mean(baseline_rmse_log_base_list)
std_baseline_log_rmse = np.std(baseline_rmse_log_base_list)
mean_baseline_rmse = np.mean(baseline_rmse_dollar_base_list)
std_baseline_rmse = np.std(baseline_rmse_dollar_base_list)
mean_lr_log_rmse = np.mean(lr_rmse_log_base_list)
std_lr_log_rmse = np.std(lr_rmse_log_base_list)
mean_lr_rmse = np.mean(lr_rmse_dollar_base_list)
std_lr_rmse = np.std(lr_rmse_dollar_base_list)
mean_lr_r2_val = np.mean(lr_r2_val_list)
std_lr_r2_val = np.std(lr_r2_val_list)
mean_ridge_log_rmse = np.mean(ridge_rmse_log_base_list)
std_ridge_log_rmse = np.std(ridge_rmse_log_base_list)
mean_ridge_rmse = np.mean(ridge_rmse_dollar_base_list)
std_ridge_rmse = np.std(ridge_rmse_dollar_base_list)
mean_ridge_r2_val = np.mean(ridge_r2_val_list)
std_ridge_r2_val = np.std(ridge_r2_val_list)
mean_xgbboost_log_rmse = np.mean(xgbboost_rmse_log_base_list)
std_xgbboost_log_rmse = np.std(xgbboost_rmse_log_base_list)
mean_xgbboost_rmse = np.mean(xgbboost_rmse_dollar_base_list)
std_xgbboost_rmse = np.std(xgbboost_rmse_dollar_base_list)
mean_xgbboost_r2_val = np.mean(xgbboost_r2_val_list)
std_xgbboost_r2_val = np.std(xgbboost_r2_val_list)

print(f"Baseline log-RMSE: {mean_baseline_log_rmse:.3f} +/- {std_baseline_log_rmse:.3f}")
print(f"Baseline RMSE: {mean_baseline_rmse:.3f} +/- {std_baseline_rmse:.3f}")
print(f"LR model log-RMSE: {mean_lr_log_rmse:.3f} +/- {std_lr_log_rmse:.3f}")
print(f"LR model RMSE: {mean_lr_rmse:.3f} +/- {std_lr_rmse:.3f}")
print(f"LR model Validation R²: {mean_lr_r2_val:.3f} +/- {std_lr_r2_val:.3f}")
print(f"Ridge LR model log-RMSE: {mean_ridge_log_rmse:.3f} +/- {std_ridge_log_rmse:.3f}")
print(f"Ridge LR model RMSE: {mean_ridge_rmse:.3f} +/- {std_ridge_rmse:.3f}")
print(f"Ridge LR model Validation R²: {mean_ridge_r2_val:.3f} +/- {std_ridge_r2_val:.3f}")
print(f"XGBoost model log-RMSE: {mean_xgbboost_log_rmse:.3f} +/- {std_xgbboost_log_rmse:.3f}")
print(f"XGBoost model RMSE: {mean_xgbboost_rmse:.3f} +/- {std_xgbboost_rmse:.3f}")
print(f"XGBoost model Validation R²: {mean_xgbboost_r2_val:.3f} +/- {std_xgbboost_r2_val:.3f}")

models = ["Baseline", "Linear Regression", "Ridge Regression", "XGBBoost"]

p_value_a, cohen_d_a, p_value_b, cohen_d_b = test_hypotheses(lr_rmse_log_base_list, ridge_rmse_log_base_list, xgbboost_rmse_log_base_list)

print("H1a Ridge vs LR: p =", p_value_a, "Cohen's d =", cohen_d_a)
print("H1b XGB vs LR: p =", p_value_b, "Cohen's d =", cohen_d_b)

log_rmse_means = [mean_baseline_log_rmse, mean_lr_log_rmse, mean_ridge_log_rmse, mean_xgbboost_log_rmse]
log_rmse_stds = [std_baseline_log_rmse, std_lr_log_rmse, std_ridge_log_rmse, std_xgbboost_log_rmse]
rmse_means = [mean_baseline_rmse, mean_lr_rmse, mean_ridge_rmse, mean_xgbboost_rmse]
rmse_stds = [std_baseline_rmse, std_lr_rmse, std_ridge_rmse, std_xgbboost_rmse]
r2_means = [0, mean_lr_r2_val, mean_ridge_r2_val, mean_xgbboost_r2_val]
r2_stds = [0, std_lr_r2_val, std_ridge_r2_val, std_xgbboost_r2_val]

visualise_results("R²", models, log_rmse_means, log_rmse_stds, rmse_means, rmse_stds, r2_means, r2_stds)
