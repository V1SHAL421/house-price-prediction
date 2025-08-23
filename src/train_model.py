import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


def build_model(X_train, include_ridge=False) -> tuple[Pipeline, ColumnTransformer]:
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ]
    )
    
    if include_ridge:
        model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RidgeCV(alphas=np.logspace(-5, 5, 100), cv=5))
    ])
    
    else:
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ])

    return model, preprocessor

def train_lr_model(model: Pipeline, X_train, y_train, X_val):
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)

    return val_pred

def train_baseline_model(X_train, y_train, X_val):
    baseline = DummyRegressor(strategy="mean")

    baseline.fit(X_train, y_train)
    baseline_val_pred = baseline.predict(X_val)

    return baseline_val_pred

def train_xgbboost_model(preprocessor, X_train, y_train, X_val):
    xgb_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ))
    ])

    xgb_model.fit(X_train, y_train)
    val_pred_xgb = xgb_model.predict(X_val)

    return val_pred_xgb