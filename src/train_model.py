import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

def build_preprocessor(X_train) -> ColumnTransformer:
    """
    Build the preprocessor for the model
    """
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

def build_model(preprocessor: ColumnTransformer, estimator: str) -> Pipeline:
    """
    Build the model
    """
    estimators = {
        "lr": LinearRegression(),
        "ridge": RidgeCV(alphas=np.logspace(-5, 5, 100), cv=5),
        "xgb": XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1
        )
    }

    if estimator not in estimators:
        raise ValueError(f"Invalid estimator: {estimator}")

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", estimators[estimator])
    ])
    
    return model

def train_baseline_model(X_train, y_train, X_val):
    """
    Train the baseline model
    """
    baseline = DummyRegressor(strategy="mean")

    baseline.fit(X_train, y_train)
    baseline_val_pred = baseline.predict(X_val)

    return baseline_val_pred

def train_lr_model(model: Pipeline, X_train, y_train, X_val):
    """
    Train the linear regression model
    """
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)

    return val_pred


def train_xgbboost_model(xgb_model, X_train, y_train, X_val):
    """
    Train the XGBoost model
    """
    xgb_model.fit(X_train, y_train)
    val_pred_xgb = xgb_model.predict(X_val)

    return val_pred_xgb