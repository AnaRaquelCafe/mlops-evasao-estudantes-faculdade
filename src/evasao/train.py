from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.base import clone
from .model import candidate_pipelines

def train_candidates(df, target, random_state=42):
    X = df.drop(columns=[target]).copy()
    y = df[target].astype(str).copy()
    X = X.dropna(axis=1, how="all")
    id_like = [c for c in X.columns if c.strip().lower() in {
        "id", "student_id", "studentid", "id_aluno", "numero_aluno"
    }]
    X = X.drop(columns=id_like, errors="ignore")
    if X.shape[1] == 0:
        raise ValueError("Nenhuma variável preditora restou após remover IDs e colunas vazias.")
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=stratify
    )
    results = {}
    best_name, best_model, best_score = None, None, -1
    for name, pipeline in candidate_pipelines(X_train, random_state).items():
        model = clone(pipeline)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
            "classification_report": classification_report(
                y_test, pred, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_test, pred, labels=sorted(y.unique())
            ).tolist(),
            "labels": sorted(y.unique()),
        }
        results[name] = {"model": model, "metrics": metrics}
        if metrics["f1_macro"] > best_score:
            best_name, best_model, best_score = name, model, metrics["f1_macro"]
    return {
        "best_name": best_name, "best_model": best_model,
        "best_metrics": results[best_name]["metrics"],
        "all_results": results, "X_test": X_test, "y_test": y_test,
        "feature_columns": X.columns.tolist(),
    }
