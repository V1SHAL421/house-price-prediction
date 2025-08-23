import pandas as pd
import numpy as np

from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from train_model import build_model, train_baseline_model, train_xgbboost_model
from data_preprocessing import preprocess_data, split_data
from train_model import train_lr_model

dataset = "data/AmesHousing.csv"

df = pd.read_csv(dataset)

df_preprocessed = preprocess_data(df)

X_train, X_test, X_val, y_train, y_test, y_val = split_data(df_preprocessed)

baseline_val_pred = train_baseline_model(X_train, y_train, X_val)

baseline_rmse_log_base = root_mean_squared_error(y_val, baseline_val_pred)
baseline_rmse_dollar_base   = root_mean_squared_error(np.expm1(y_val), np.expm1(baseline_val_pred))

print(f"Baseline log-RMSE: {baseline_rmse_log_base:.3f}")
print(f"Baseline RMSE: {baseline_rmse_dollar_base:.3f}")

lr_model, preprocessor = build_model(X_train)

val_pred = train_lr_model(lr_model, X_train, y_train, X_val)

rmse_log_base = root_mean_squared_error(y_val, val_pred)
rmse_dollar_base   = root_mean_squared_error(np.expm1(y_val), np.expm1(val_pred))
r2_val = r2_score(y_val, val_pred)

print(f"LR model log-RMSE: {rmse_log_base:.3f}")
print(f"LR model RMSE: {rmse_dollar_base:.3f}")
print(f"LR model Validation R²: {r2_val:.3f}")

ridge_model, preprocessor = build_model(X_train, include_ridge=True)

val_pred = train_lr_model(ridge_model, X_train, y_train, X_val)

rmse_log_base = root_mean_squared_error(y_val, val_pred)
rmse_dollar_base   = root_mean_squared_error(np.expm1(y_val), np.expm1(val_pred))
r2_val = r2_score(y_val, val_pred)

print(f"Ridge LR model log-RMSE: {rmse_log_base:.3f}")
print(f"Ridge LR model RMSE: {rmse_dollar_base:.3f}")
print(f"Ridge LR model Validation R²: {r2_val:.3f}")

val_pred_xgb = train_xgbboost_model(preprocessor, X_train, y_train, X_val)

rmse_log_xgb = root_mean_squared_error(y_val, val_pred_xgb)
r2_val_xgb = r2_score(y_val, val_pred_xgb)

print(f"XGBoost Val log-RMSE: {rmse_log_xgb:.3f}")
print(f"XGBoost Val R²: {r2_val_xgb:.3f}") 