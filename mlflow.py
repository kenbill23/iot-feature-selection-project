# ================================
# MLflow Tracking
# ================================

import mlflow
import mlflow.sklearn
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

mlflow.set_experiment("IoT_Vulnerability_Classification")

with mlflow.start_run():

    # Training model terbaik
    best_model.fit(X_train, y_train)

    # Prediksi
    y_pred = best_model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average='macro'
    )

    recall = recall_score(
        y_test,
        y_pred,
        average='macro'
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average='macro'
    )

    # Log parameter terbaik
    mlflow.log_params(best_model.get_params())

    # Log metrics
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1
    })

    # Log model
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model"
    )

    # Simpan model
    joblib.dump(
        best_model,
        "pipeline_terbaik.pkl"
    )

    print("Model berhasil disimpan.\n")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))