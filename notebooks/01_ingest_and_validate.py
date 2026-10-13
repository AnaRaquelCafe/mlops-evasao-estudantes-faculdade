# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Ingestão e validação
# MAGIC Envie `StudentsPrepared.xlsx` para um Volume do Unity Catalog antes de executar.

# COMMAND ----------
CATALOG = "main"
SCHEMA = "evasao_mlops"
VOLUME = "raw_files"
SOURCE_FILE = "StudentsPrepared.xlsx"
TARGET_COLUMN = None  # Se necessário, informe o nome exato da coluna-alvo.

# COMMAND ----------
# MAGIC %pip install openpyxl scikit-learn mlflow
# MAGIC %restart_python

# COMMAND ----------
import pandas as pd

source_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{SOURCE_FILE}"
print("Lendo:", source_path)
df = pd.read_excel(source_path) if SOURCE_FILE.lower().endswith((".xlsx", ".xls")) else pd.read_csv(source_path)
df.columns = [str(c).strip() for c in df.columns]

display(df.head(10))
print("Dimensões:", df.shape)
print("Colunas:", df.columns.tolist())
display(df.isna().sum().sort_values(ascending=False).to_frame("nulos"))

# COMMAND ----------
candidates = ["Target", "target", "target_name", "situacao", "Situação", "Status", "status", "resultado", "Resultado"]
target = TARGET_COLUMN or next((c for c in candidates if c in df.columns), None)
if target is None:
    raise ValueError(f"Defina TARGET_COLUMN. Colunas encontradas: {df.columns.tolist()}")
if df[target].isna().any() or df[target].nunique() < 2:
    raise ValueError(f"Target inválido: {target}")

print("Target:", target)
display(df[target].value_counts(dropna=False).to_frame("quantidade"))
print("Linhas duplicadas:", int(df.duplicated().sum()))

# COMMAND ----------
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")
spark.createDataFrame(df).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.students_bronze"
)
print(f"Tabela criada: {CATALOG}.{SCHEMA}.students_bronze")
