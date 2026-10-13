# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Treino, avaliação e registro MLflow
# MAGIC Compara dois modelos e seleciona o maior macro-F1 como baseline inicial.

# COMMAND ----------
# MAGIC %pip install scikit-learn mlflow
# MAGIC %restart_python

# COMMAND ----------
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from mlflow.models import infer_signature

CATALOG = "main"
SCHEMA = "evasao_mlops"
TARGET_COLUMN = None  # Confira o target no notebook 01.

df = spark.table(f"{CATALOG}.{SCHEMA}.students_bronze").toPandas()
candidates = ["Target", "target", "target_name", "situacao", "Situação", "Status", "status", "resultado", "Resultado"]
target = TARGET_COLUMN or next((c for c in candidates if c in df.columns), None)
if target is None:
    raise ValueError(f"Defina TARGET_COLUMN. Colunas: {df.columns.tolist()}")

X = df.drop(columns=[target]).copy()
y = df[target].astype(str).copy()
X = X.dropna(axis=1, how="all")
X = X.drop(columns=[c for c in X.columns if c.strip().lower() in {"id", "student_id", "studentid", "id_aluno", "numero_aluno"}], errors="ignore")
numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical = [c for c in X.columns if c not in numeric]
stratify = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

def preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ], remainder="drop")

models = {
    "logistic_regression": Pipeline([
        ("preprocessor", preprocessor()),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced"))
    ]),
    "random_forest": Pipeline([
        ("preprocessor", preprocessor()),
        ("classifier", RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1))
    ])
}

# COMMAND ----------
mlflow.set_experiment(f"/Shared/evasao-faculdade-{SCHEMA}")
runs = []
for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0))
        }
        mlflow.log_params({"model_name": name, "target_column": target, "n_features": X_train.shape[1]})
        mlflow.log_metrics(metrics)
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", signature=signature, input_example=X_train.head(3))
        runs.append({"name": name, "run_id": run.info.run_id, "model": model, **metrics})
        print(f"\n{name}: {metrics}")
        print(classification_report(y_test, pred, zero_division=0))

# COMMAND ----------
best = max(runs, key=lambda row: row["f1_macro"])
display(pd.DataFrame([{k: v for k, v in row.items() if k != "model"} for row in runs]))
print("Melhor baseline:", best["name"], "| macro-F1:", best["f1_macro"])

# COMMAND ----------
# Registro Unity Catalog. Requer permissões CREATE MODEL no schema.
registered_model_name = f"{CATALOG}.{SCHEMA}.student_dropout_classifier"
try:
    mlflow.set_registry_uri("databricks-uc")
    version = mlflow.register_model(f"runs:/{best['run_id']}/model", registered_model_name)
    print("Modelo registrado:", registered_model_name, "versão:", version.version)
except Exception as exc:
    print("O tracking MLflow foi executado, mas o registro no Unity Catalog falhou.")
    print("Confira catálogo/schema, permissões CREATE MODEL e configuração do workspace.")
    print(str(exc)[:1200])
