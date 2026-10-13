import pandas as pd

TARGET_CANDIDATES = [
    "Target", "target", "target_name", "situacao", "Situação",
    "Status", "status", "resultado", "Resultado"
]

def read_source_file(path: str) -> pd.DataFrame:
    """Read XLSX or CSV input, including a Databricks Volume path."""
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    elif path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError("Formato não suportado. Use .xlsx, .xls ou .csv.")
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise ValueError("A base está vazia.")
    return df

def detect_target_column(df: pd.DataFrame, requested: str | None = None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Coluna-alvo '{requested}' não encontrada. Colunas: {list(df.columns)}")
        return requested
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Não consegui identificar a coluna-alvo automaticamente. "
        f"Defina TARGET_COLUMN explicitamente. Colunas: {list(df.columns)}"
    )

def validate_dataset(df: pd.DataFrame, target: str) -> dict:
    if target not in df.columns:
        raise ValueError(f"Coluna-alvo ausente: {target}")
    if len(df) < 20:
        raise ValueError(f"A base tem apenas {len(df)} linhas; revise a origem antes de treinar.")
    if df[target].isna().any():
        raise ValueError(f"A coluna-alvo '{target}' possui valores nulos.")
    if df[target].nunique() < 2:
        raise ValueError("A coluna-alvo precisa conter pelo menos duas classes.")
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "target": target,
        "classes": sorted(df[target].astype(str).unique().tolist()),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
    }
