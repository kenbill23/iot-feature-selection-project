import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from imblearn.pipeline import Pipeline

# =====================================
# Load Dataset
# =====================================

df = pd.read_csv("IoT_Vulnerability.csv")

# =====================================
# Target
# =====================================

target = "Attack_Category"

# Hilangkan semua kolom target dari fitur
drop_cols = [
    "Attack_Category",
    "Label",
    "Attack_sub_category"
]

X = df.drop(
    columns=[c for c in drop_cols if c in df.columns],
    errors="ignore"
)

y = df[target]

# =====================================
# Train Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =====================================
# Best Pipeline
# =====================================

best_model = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "feature_selection",
        SelectFromModel(
            RandomForestClassifier(
                n_estimators=5,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
        )
    ),

    (
        "model",
        RandomForestClassifier(
            n_estimators=10,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
    )
])

# =====================================
# MLflow
# =====================================

mlflow.set_experiment(
    "IoT_Vulnerability_Classification"
)

with mlflow.start_run(run_name="Best_Embedded_Model"):

    best_model.fit(
        X_train,
        y_train
    )

    y_pred = best_model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro"
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    mlflow.log_params({

        "feature_selection_method": "SelectFromModel",

        "feature_selection__estimator__n_estimators": 5,
        "feature_selection__estimator__max_depth": 5,

        "model__n_estimators": 10,
        "model__max_depth": 5

    })

    mlflow.log_metrics({

        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1

    })

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model"
    )

    joblib.dump(
        best_model,
        "pipeline_terbaik.pkl"
    )

    print("\n=== HASIL EVALUASI ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    print("\nJumlah fitur training:", len(X.columns))
    print("Nama fitur pertama:", list(X.columns[:10]))

    print("\nModel berhasil disimpan sebagai pipeline_terbaik.pkl")