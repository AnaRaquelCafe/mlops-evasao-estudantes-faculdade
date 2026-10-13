from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def build_preprocessor(X):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")

def candidate_pipelines(X, random_state=42):
    return {
        "logistic_regression": Pipeline([
            ("preprocessor", build_preprocessor(X)),
            ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", build_preprocessor(X)),
            ("classifier", RandomForestClassifier(
                n_estimators=250, min_samples_leaf=2,
                class_weight="balanced", random_state=random_state, n_jobs=-1
            )),
        ]),
    }
