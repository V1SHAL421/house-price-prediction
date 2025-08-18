import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("data/AmesHousing.csv")

# print(df.head())

# Tidy column names
df.columns = (
    df.columns.str.strip() # Remove leading and trailing whitespace
    .str.lower()  # Convert to lowercase
    .str.replace(" ", "_")  # Replace spaces with underscores
    .str.replace("(", "")  # Remove opening parentheses
    .str.replace(")", "")  # Remove closing parentheses
)

# print(df.shape)

# df.info()

# Display the % missing per column
na_rate = df.isna().mean().sort_values(ascending=False)

# print(f"The % missing is {na_rate}")

"""
For the columns with more than 40% missing (extremely sparse columns) drop them
"""
high_na_cols = na_rate[na_rate > 0.4].index.intersection(df.columns)
df = df.drop(columns=high_na_cols)

df["log_saleprice"] = np.log1p(df["saleprice"])

# print(df.head())

print(f"The columns are {df.columns}")

y = df["log_saleprice"].copy()
X = df.drop(["saleprice", "log_saleprice"], axis=1, errors='ignore')

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())

])

model.fit(X_train, y_train)

val_pred = model.predict(X_val)

rmse_log_base = root_mean_squared_error(y_val, val_pred)
rmse_dollar_base   = root_mean_squared_error(np.expm1(y_val), np.expm1(val_pred))

print(f"The log-RMSE is {rmse_log_base:.3f}")
print(f"The RMSE is {rmse_dollar_base:.3f}")

baseline = DummyRegressor(strategy="mean")

baseline.fit(X_train, y_train)
baseline_val_pred = baseline.predict(X_val)

baseline_rmse_log_base = root_mean_squared_error(y_val, baseline_val_pred)
baseline_rmse_dollar_base   = root_mean_squared_error(np.expm1(y_val), np.expm1(baseline_val_pred))

print(f"The baseline log-RMSE is {baseline_rmse_log_base:.3f}")
print(f"The baseline RMSE is {baseline_rmse_dollar_base:.3f}")

r2_val = r2_score(y_val, val_pred)
print(f"Validation R²: {r2_val:.3f}")