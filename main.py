import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
df = df.drop(na_rate[na_rate > 0.4].index, axis=1)

df["log_saleprice"] = np.log1p(df["saleprice"])

# Impute median for columns of type integer or float
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop(["saleprice", "log_saleprice"])
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Impute mode for columns of type object or category
category_cols = df.select_dtypes(include=["object", "category"]).columns
df[category_cols] = df[category_cols].fillna(df[category_cols].mode().iloc[0])

# print(df.head())

y = df["log_saleprice"].copy()
X = df.drop(["saleprice", "log_saleprice"])

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)



