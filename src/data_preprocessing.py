import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generalise the dataset field names
    """
    df.columns = (
    df.columns.str.strip() # Remove leading and trailing whitespace
    .str.lower()  # Convert to lowercase
)
    # Display the % missing per column
    na_rate = df.isna().mean().sort_values(ascending=False)

    """
    For the columns with more than 40% missing (extremely sparse columns) drop them
    """
    # high_na_cols = na_rate[na_rate > 0.4].index.intersection(df.columns)
    high_na_cols = na_rate[na_rate > 0.4].index
    df = df.drop(columns=high_na_cols)

    df["log_saleprice"] = np.log1p(df["saleprice"])

    print(f"The columns are {df.columns}")

    return df

def split_data(df: pd.DataFrame):
    """
    Split the data into train, test and validation sets
    """
    df = df.dropna()
    y = df["log_saleprice"].copy()
    X = df.drop(["saleprice", "log_saleprice"], axis=1, errors='ignore')

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_test, X_val, y_train, y_test, y_val

    
