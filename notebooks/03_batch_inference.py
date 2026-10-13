# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Inferência batch
# MAGIC Para uma demonstração realista, use uma tabela de alunos elegíveis para scoring sem a coluna-alvo.

# COMMAND ----------
# MAGIC %pip install scikit-learn mlflow
# MAGIC %restart_python

# COMMAND ----------
import mlflow
import pandas as pd

CATALOG = "main"
SCHEMA = "evasao_mlops"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.students_bronze"
TARGET_COLUMN = None
MODEL_URI = None  # Ex.: models:/main.evasao_mlops.student_dropout_classifier/1
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.student_dropout_predictions"

if not MODEL_URI:
    raise ValueError("Preencha MODEL_URI com a URI do modelo registrado.")

df = spark.table(SOURCE_TABLE).toPandas()
candidates = ["Target", "target", "target_name", "situacao", "Situação", "Status", "status", "resultado", "Resultado"]
target = TARGET_COLUMN or next((c for c in candidates if c in df.columns), None)
X = df.drop(columns=[target]).copy() if target else df.copy()
X = X.dropna(axis=1, how="all")
X = X.drop(columns=[c for c in X.columns if c.strip().lower() in {"id", "student_id", "studentid", "id_aluno", "numero_aluno"}], errors="ignore")

model = mlflow.pyfunc.load_model(MODEL_URI)
predictions = model.predict(X)
result = pd.DataFrame({"prediction": predictions.astype(str)})

# O wrapper pyfunc pode não expor predict_proba; confiança é opcional.
try:
    raw_model = mlflow.sklearn.load_model(MODEL_URI)
    probabilities = raw_model.predict_proba(X)
    result["confidence"] = probabilities.max(axis=1)
except Exception:
    pass

result = pd.concat([df.reset_index(drop=True), result], axis=1)
result["scored_at_utc"] = pd.Timestamp.utcnow().isoformat()
spark.createDataFrame(result).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(OUTPUT_TABLE)
print("Tabela de inferência criada:", OUTPUT_TABLE)
display(spark.table(OUTPUT_TABLE).limit(20))
