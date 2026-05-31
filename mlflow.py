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

# GANTI jika target berbeda
target = "Attack_Category"

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# Pipeline Terbaik
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
                n_jobs=1
            )
        )
    ),

    (
        "model",
        RandomForestClassifier(
            n_estimators=10,
            max_depth=5,
            random_state=42,
            n_jobs=1
        )
    )
])

# =====================================
# MLflow Tracking
# =====================================

mlflow.set_experiment(
    "IoT_Vulnerability_Classification"
)

with mlflow.start_run():

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

        "model__n_estimators": 10,
        "model__max_depth": 5,

        "feature_selection__estimator__n_estimators": 5,
        "feature_selection__estimator__max_depth": 5

    })

    mlflow.log_metrics({

        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1

    })

    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model"
    )

    joblib.dump(
        best_model,
        "pipeline_terbaik.pkl"
    )

    print("Model berhasil disimpan")

    print(classification_report(
        y_test,
        y_pred
    ))