from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.audio_loader import create_training_index
from src.feature_extraction import (
    create_feature_dataset,
    get_model_feature_columns
)


MODEL_VERSION = "rf_audio_poc_v1"
MODEL_PATH = Path("models/model.pkl")


def prepare_training_data() -> tuple[pd.DataFrame, list]:
    """
    Maakt de volledige feature-dataset voor training.

    Stappen:
    1. WAV-bestanden zoeken
    2. Labels koppelen
    3. Features extraheren
    4. Ongeldige bestanden verwijderen
    5. Featurekolommen bepalen
    """

    training_index = create_training_index(
        normal_folder="data/normaal",
        abnormal_folder="data/afwijkend"
    )

    feature_df = create_feature_dataset(training_index)

    valid_feature_df = feature_df[feature_df["is_valid"] == True].copy()

    if valid_feature_df.empty:
        raise ValueError("Er zijn geen geldige audiobestanden gevonden voor training.")

    if valid_feature_df["label"].nunique() < 2:
        raise ValueError(
            "Er zijn minimaal twee klassen nodig voor supervised learning: normaal en afwijkend."
        )

    feature_columns = get_model_feature_columns(valid_feature_df)

    return valid_feature_df, feature_columns


def train_random_forest_model(
    feature_df: pd.DataFrame,
    feature_columns: list,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Traint een Random Forest-model.

    Output:
    dictionary met:
    - model
    - metrics
    - feature_columns
    - model_version
    """

    X = feature_df[feature_columns]
    y = feature_df["label"]

    class_counts = y.value_counts()

    if len(feature_df) < 10:
        raise ValueError(
            "Er zijn minder dan 10 geldige bestanden. Voeg meer normale en afwijkende WAV-bestanden toe."
        )

    stratify_value = y if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_value
    )

    model = RandomForestClassifier(
    n_estimators=500,
    random_state=random_state,
    class_weight={0: 1, 1: 2.5},
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt"
)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "test_samples": int(len(y_test)),
        "train_samples": int(len(y_train)),
        "normal_samples": int((feature_df["label"] == 0).sum()),
        "abnormal_samples": int((feature_df["label"] == 1).sum())
    }

    if y_proba is not None and len(set(y_test)) == 2:
        metrics["roc_auc"] = round(roc_auc_score(y_test, y_proba), 4)
    else:
        metrics["roc_auc"] = None

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    confusion_matrix_df = pd.DataFrame(
        cm,
        index=["Werkelijk normaal", "Werkelijk afwijkend"],
        columns=["Voorspeld normaal", "Voorspeld afwijkend"]
    )

    result = {
        "model": model,
        "metrics": metrics,
        "feature_columns": feature_columns,
        "model_version": MODEL_VERSION,
        "confusion_matrix": confusion_matrix_df
    }

    return result


def save_model(
    model,
    feature_columns: list,
    metrics: dict,
    model_version: str = MODEL_VERSION,
    model_path: Path = MODEL_PATH
) -> None:
    """
    Slaat het model en de bijbehorende metadata op.
    """

    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_package = {
        "model": model,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "model_version": model_version
    }

    joblib.dump(model_package, model_path)


def train_and_save_model() -> dict:
    """
    Complete training-pipeline.

    Deze functie gebruiken we vanuit Streamlit.
    """

    feature_df, feature_columns = prepare_training_data()

    training_result = train_random_forest_model(
        feature_df=feature_df,
        feature_columns=feature_columns
    )

    save_model(
        model=training_result["model"],
        feature_columns=training_result["feature_columns"],
        metrics=training_result["metrics"],
        model_version=training_result["model_version"]
    )

    training_result["feature_df"] = feature_df
    training_result["model_path"] = str(MODEL_PATH)

    return training_result


if __name__ == "__main__":
    result = train_and_save_model()

    print("Model succesvol getraind en opgeslagen.")
    print(f"Modelpad: {result['model_path']}")
    print("Metrics:")
    print(result["metrics"])