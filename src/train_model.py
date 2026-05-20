from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    confusion_matrix
)

from src.audio_loader import create_training_index
from src.feature_extraction import (
    create_segment_feature_dataset,
    get_model_feature_columns
)


MODEL_VERSION = "profile_asset_noise_iforest_v1"
MODEL_PATH = Path("models/model.pkl")

SEGMENT_DURATION_SECONDS = 3.0
HOP_DURATION_SECONDS = 1.5

FILE_NORMALITY_PERCENTILE = 35

MIN_NORMAL_FILES_PER_PROFILE = 5

MIN_RECALL_TARGET = 0.75
MAX_FALSE_ALARM_RATE_TARGET = 0.45
RED_MIN_PRECISION_TARGET = 0.70


def parse_profile_name(source_folder: str) -> dict:
    """
    Haalt asset-ID en ruisprofiel uit een mapnaam.

    Voorbeelden:
    id_04_0DB  -> asset_id = id_04, noise_profile = 0DB
    id_04_6DB  -> asset_id = id_04, noise_profile = 6DB
    id_04_m6DB -> asset_id = id_04, noise_profile = m6DB
    """

    parts = source_folder.split("_")

    if len(parts) >= 3:
        asset_id = "_".join(parts[:2])
        noise_profile = "_".join(parts[2:])
    else:
        asset_id = source_folder
        noise_profile = "unknown"

    return {
        "profile_name": source_folder,
        "asset_id": asset_id,
        "noise_profile": noise_profile
    }


def calculate_normality_score(raw_score: float, calibration_scores: np.ndarray) -> float:
    """
    Zet de ruwe Isolation Forest-score om naar een normaliteits-score van 0 tot 100.

    Hoe hoger de score, hoe normaler het segment lijkt ten opzichte van normale trainingssegmenten.
    """

    calibration_scores = np.asarray(calibration_scores)

    if len(calibration_scores) == 0:
        return 0.0

    sorted_scores = np.sort(calibration_scores)

    percentile_position = np.searchsorted(
        sorted_scores,
        raw_score,
        side="right"
    ) / len(sorted_scores)

    normality_score = percentile_position * 100

    return float(np.clip(normality_score, 0, 100))


def build_file_level_scores(
    model,
    segment_feature_df: pd.DataFrame,
    feature_columns: list,
    calibration_scores: np.ndarray
) -> pd.DataFrame:
    """
    Zet segmentscores om naar één score per bestand.

    Een WAV-bestand bestaat uit meerdere segmenten.
    Per segment berekenen we een normaliteits-score.
    De bestandsscore wordt gebaseerd op een lager percentiel van de segmenten,
    zodat korte afwijkende stukken niet verdwijnen in het gemiddelde.
    """

    records = []

    valid_df = segment_feature_df[segment_feature_df["is_valid"] == True].copy()

    if valid_df.empty:
        return pd.DataFrame()

    for file_path, group_df in valid_df.groupby("file_path"):
        X_segments = group_df[feature_columns]
        raw_scores = model.score_samples(X_segments)

        segment_normality_scores = [
            calculate_normality_score(
                raw_score=raw_score,
                calibration_scores=calibration_scores
            )
            for raw_score in raw_scores
        ]

        file_normality_score = float(
            np.percentile(segment_normality_scores, FILE_NORMALITY_PERCENTILE)
        )

        records.append({
            "file_path": file_path,
            "true_label": int(group_df["label"].iloc[0]),
            "class_name": group_df["class_name"].iloc[0],
            "source_folder": group_df["source_folder"].iloc[0],
            "normality_score": file_normality_score,
            "anomaly_score": 100 - file_normality_score,
            "segments_total": len(segment_normality_scores),
            "segment_normality_min": float(np.min(segment_normality_scores)),
            "segment_normality_mean": float(np.mean(segment_normality_scores)),
            "segment_normality_p20": float(np.percentile(segment_normality_scores, 20)),
            "segment_normality_p35": float(np.percentile(segment_normality_scores, 35)),
            "segment_normality_p50": float(np.percentile(segment_normality_scores, 50))
        })

    return pd.DataFrame(records)


def tune_thresholds(evaluation_df: pd.DataFrame) -> dict:
    """
    Zoekt automatisch twee thresholds:

    attention_threshold:
    - onder deze score is minimaal 'onderzoek nodig'

    defect_threshold:
    - onder deze score is 'defect / niet normaal'

    De tuning zoekt balans:
    - afwijkingen niet te veel missen;
    - valse meldingen beperken;
    - rode meldingen strenger maken dan oranje meldingen.
    """

    if evaluation_df.empty or evaluation_df["true_label"].nunique() < 2:
        return {
            "attention_threshold": 70,
            "defect_threshold": 35,
            "strategy": "default_insufficient_evaluation_data",
            "attention_accuracy": None,
            "attention_precision": None,
            "attention_recall": None,
            "attention_f1_score": None,
            "attention_f2_score": None,
            "attention_false_alarm_rate": None,
            "red_precision": None,
            "red_recall": None,
            "red_f1_score": None
        }

    y_true = evaluation_df["true_label"]

    attention_candidates = []

    for threshold in range(5, 96):
        y_pred = (evaluation_df["normality_score"] < threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        true_normal = cm[0, 0]
        false_alarm = cm[0, 1]

        false_alarm_rate = false_alarm / max((true_normal + false_alarm), 1)

        attention_candidates.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "f2_score": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
            "false_alarm_rate": false_alarm_rate
        })

    attention_df = pd.DataFrame(attention_candidates)

    balanced_candidates = attention_df[
        (attention_df["recall"] >= MIN_RECALL_TARGET) &
        (attention_df["false_alarm_rate"] <= MAX_FALSE_ALARM_RATE_TARGET)
    ].copy()

    if not balanced_candidates.empty:
        best_attention = balanced_candidates.sort_values(
            by=["f1_score", "precision", "accuracy"],
            ascending=False
        ).iloc[0]

        strategy = (
            f"balanced_min_recall_{MIN_RECALL_TARGET}"
            f"_max_false_alarm_{MAX_FALSE_ALARM_RATE_TARGET}"
        )

    else:
        recall_candidates = attention_df[
            attention_df["recall"] >= MIN_RECALL_TARGET
        ].copy()

        if not recall_candidates.empty:
            best_attention = recall_candidates.sort_values(
                by=["precision", "f1_score", "accuracy"],
                ascending=False
            ).iloc[0]

            strategy = f"highest_precision_with_min_recall_{MIN_RECALL_TARGET}"

        else:
            attention_df["missed_abnormal_rate"] = 1 - attention_df["recall"]
            attention_df["maintenance_cost"] = (
                2.0 * attention_df["missed_abnormal_rate"]
                + 1.0 * attention_df["false_alarm_rate"]
            )

            best_attention = attention_df.sort_values(
                by=["maintenance_cost", "false_alarm_rate", "precision"],
                ascending=[True, True, False]
            ).iloc[0]

            strategy = "lowest_maintenance_cost_fallback"

    attention_threshold = int(best_attention["threshold"])

    red_candidates = []

    for threshold in range(1, attention_threshold):
        y_red = (evaluation_df["normality_score"] < threshold).astype(int)

        red_candidates.append({
            "threshold": threshold,
            "precision": precision_score(y_true, y_red, zero_division=0),
            "recall": recall_score(y_true, y_red, zero_division=0),
            "f1_score": f1_score(y_true, y_red, zero_division=0)
        })

    red_df = pd.DataFrame(red_candidates)

    strong_red = red_df[
        red_df["precision"] >= RED_MIN_PRECISION_TARGET
    ].copy()

    if not strong_red.empty:
        best_red = strong_red.sort_values(
            by=["recall", "f1_score"],
            ascending=False
        ).iloc[0]
    else:
        best_red = red_df.sort_values(
            by=["precision", "f1_score"],
            ascending=False
        ).iloc[0]

    defect_threshold = int(best_red["threshold"])

    return {
        "attention_threshold": attention_threshold,
        "defect_threshold": defect_threshold,
        "strategy": strategy,
        "attention_accuracy": round(float(best_attention["accuracy"]), 4),
        "attention_precision": round(float(best_attention["precision"]), 4),
        "attention_recall": round(float(best_attention["recall"]), 4),
        "attention_f1_score": round(float(best_attention["f1_score"]), 4),
        "attention_f2_score": round(float(best_attention["f2_score"]), 4),
        "attention_false_alarm_rate": round(float(best_attention["false_alarm_rate"]), 4),
        "red_precision": round(float(best_red["precision"]), 4),
        "red_recall": round(float(best_red["recall"]), 4),
        "red_f1_score": round(float(best_red["f1_score"]), 4)
    }


def prepare_training_data() -> tuple[pd.DataFrame, list]:
    """
    Maakt segment-features voor alle normale en afwijkende bestanden.

    Training gebeurt later per profiel op alleen normale segmenten.
    """

    training_index = create_training_index(
        normal_folder="data/normaal",
        abnormal_folder="data/afwijkend"
    )

    segment_feature_df = create_segment_feature_dataset(
        training_index=training_index,
        segment_duration_seconds=SEGMENT_DURATION_SECONDS,
        hop_duration_seconds=HOP_DURATION_SECONDS
    )

    valid_segment_df = segment_feature_df[
        segment_feature_df["is_valid"] == True
    ].copy()

    if valid_segment_df.empty:
        raise ValueError("Er zijn geen geldige audiosegmenten gevonden.")

    normal_segment_df = valid_segment_df[
        valid_segment_df["class_name"] == "normaal"
    ].copy()

    if normal_segment_df.empty:
        raise ValueError(
            "Er zijn geen normale audiosegmenten gevonden. "
            "Voor anomaly detection zijn normale bestanden verplicht."
        )

    feature_columns = get_model_feature_columns(valid_segment_df)

    return valid_segment_df, feature_columns


def train_profile_model(
    profile_name: str,
    valid_segment_df: pd.DataFrame,
    feature_columns: list,
    random_state: int = 42
) -> dict | None:
    """
    Traint één profielmodel.

    Voorbeeld profiel:
    id_04_0DB

    Training:
    - alleen normale segmenten uit data/normaal/id_04_0DB

    Evaluatie:
    - normale testbestanden uit hetzelfde profiel
    - afwijkende bestanden uit data/afwijkend/id_04_0DB, als aanwezig
    """

    profile_info = parse_profile_name(profile_name)

    profile_df = valid_segment_df[
        valid_segment_df["source_folder"] == profile_name
    ].copy()

    normal_profile_df = profile_df[
        profile_df["class_name"] == "normaal"
    ].copy()

    abnormal_profile_df = profile_df[
        profile_df["class_name"] == "afwijkend"
    ].copy()

    normal_files = normal_profile_df["file_path"].drop_duplicates()

    if len(normal_files) < MIN_NORMAL_FILES_PER_PROFILE:
        return None

    if len(normal_files) >= 10:
        train_files, test_files = train_test_split(
            normal_files,
            test_size=0.2,
            random_state=random_state
        )
    else:
        train_files = normal_files
        test_files = normal_files

    train_segment_df = normal_profile_df[
        normal_profile_df["file_path"].isin(train_files)
    ].copy()

    test_normal_segment_df = normal_profile_df[
        normal_profile_df["file_path"].isin(test_files)
    ].copy()

    X_train = train_segment_df[feature_columns]

    model = IsolationForest(
        n_estimators=1000,
        contamination=0.01,
        max_samples=min(512, len(X_train)),
        max_features=1.0,
        bootstrap=False,
        random_state=random_state
    )

    model.fit(X_train)

    calibration_scores = model.score_samples(X_train)

    evaluation_segment_df = pd.concat(
        [test_normal_segment_df, abnormal_profile_df],
        ignore_index=True
    )

    evaluation_df = build_file_level_scores(
        model=model,
        segment_feature_df=evaluation_segment_df,
        feature_columns=feature_columns,
        calibration_scores=calibration_scores
    )

    threshold_result = tune_thresholds(evaluation_df)

    attention_threshold = threshold_result["attention_threshold"]
    defect_threshold = threshold_result["defect_threshold"]

    profile_metrics = {
        "profile_name": profile_name,
        "asset_id": profile_info["asset_id"],
        "noise_profile": profile_info["noise_profile"],
        "normal_files": int(normal_profile_df["file_path"].nunique()),
        "abnormal_files": int(abnormal_profile_df["file_path"].nunique()),
        "train_segments": int(len(train_segment_df)),
        "attention_threshold": attention_threshold,
        "defect_threshold": defect_threshold,
        "threshold_strategy": threshold_result["strategy"],
        "attention_false_alarm_rate": threshold_result.get("attention_false_alarm_rate"),
        "red_precision": threshold_result.get("red_precision"),
        "red_recall": threshold_result.get("red_recall")
    }

    if not evaluation_df.empty and evaluation_df["true_label"].nunique() == 2:
        y_true = evaluation_df["true_label"]
        y_pred = (evaluation_df["normality_score"] < attention_threshold).astype(int)
        y_score = evaluation_df["anomaly_score"] / 100

        profile_metrics.update({
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "f2_score": round(fbeta_score(y_true, y_pred, beta=2, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_true, y_score), 4)
        })
    else:
        profile_metrics.update({
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "f2_score": None,
            "roc_auc": None
        })

    return {
        "model": model,
        "profile_name": profile_name,
        "asset_id": profile_info["asset_id"],
        "noise_profile": profile_info["noise_profile"],
        "calibration_scores": calibration_scores,
        "attention_threshold": attention_threshold,
        "defect_threshold": defect_threshold,
        "metrics": profile_metrics,
        "evaluation_df": evaluation_df
    }


def train_all_profile_models(
    valid_segment_df: pd.DataFrame,
    feature_columns: list
) -> dict:
    """
    Traint per profiel een apart anomaly detection model.

    Profiel = source_folder, bijvoorbeeld:
    id_04_0DB
    id_04_6DB
    id_04_m6DB
    """

    normal_profiles = (
        valid_segment_df[valid_segment_df["class_name"] == "normaal"]["source_folder"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    profile_models = {}
    profile_metrics = []
    skipped_profiles = []

    for profile_name in normal_profiles:
        profile_result = train_profile_model(
            profile_name=profile_name,
            valid_segment_df=valid_segment_df,
            feature_columns=feature_columns
        )

        if profile_result is None:
            skipped_profiles.append(profile_name)
            continue

        profile_models[profile_name] = {
            "model": profile_result["model"],
            "asset_id": profile_result["asset_id"],
            "noise_profile": profile_result["noise_profile"],
            "calibration_scores": profile_result["calibration_scores"],
            "attention_threshold": profile_result["attention_threshold"],
            "defect_threshold": profile_result["defect_threshold"],
            "metrics": profile_result["metrics"]
        }

        profile_metrics.append(profile_result["metrics"])

    if not profile_models:
        raise ValueError(
            "Er konden geen profielmodellen worden getraind. "
            "Controleer of er voldoende normale bestanden per profielmap staan."
        )

    profile_metrics_df = pd.DataFrame(profile_metrics)

    return {
        "profile_models": profile_models,
        "profile_metrics_df": profile_metrics_df,
        "skipped_profiles": skipped_profiles
    }


def build_overall_metrics(profile_metrics_df: pd.DataFrame) -> dict:
    """
    Maakt globale metrics voor de app.

    Dit zijn gemiddelden over de profielmodellen waarvoor evaluatiemetrics beschikbaar zijn.
    """

    valid_metrics_df = profile_metrics_df.dropna(
        subset=["accuracy", "precision", "recall", "f1_score"],
        how="all"
    ).copy()

    if valid_metrics_df.empty:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "f2_score": None,
            "roc_auc": None,
            "profiles_trained": int(len(profile_metrics_df)),
            "normal_samples": int(profile_metrics_df["normal_files"].sum()),
            "abnormal_samples": int(profile_metrics_df["abnormal_files"].sum()),
            "train_samples": int(profile_metrics_df["train_segments"].sum()),
            "test_samples": None
        }

    metrics = {
        "accuracy": round(float(valid_metrics_df["accuracy"].mean()), 4),
        "precision": round(float(valid_metrics_df["precision"].mean()), 4),
        "recall": round(float(valid_metrics_df["recall"].mean()), 4),
        "f1_score": round(float(valid_metrics_df["f1_score"].mean()), 4),
        "f2_score": round(float(valid_metrics_df["f2_score"].mean()), 4),
        "roc_auc": round(float(valid_metrics_df["roc_auc"].mean()), 4),
        "profiles_trained": int(len(profile_metrics_df)),
        "normal_samples": int(profile_metrics_df["normal_files"].sum()),
        "abnormal_samples": int(profile_metrics_df["abnormal_files"].sum()),
        "train_samples": int(profile_metrics_df["train_segments"].sum()),
        "test_samples": None
    }

    return metrics


def save_model_package(
    profile_models: dict,
    feature_columns: list,
    metrics: dict,
    profile_metrics_df: pd.DataFrame,
    skipped_profiles: list,
    model_path: Path = MODEL_PATH
) -> None:
    """
    Slaat alle profielmodellen op in één modelbestand.
    """

    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_package = {
        "model_type": "profile_specific_segment_anomaly_detection",
        "model_version": MODEL_VERSION,
        "profile_models": profile_models,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "profile_metrics": profile_metrics_df.to_dict(orient="records"),
        "skipped_profiles": skipped_profiles,
        "segment_duration_seconds": SEGMENT_DURATION_SECONDS,
        "hop_duration_seconds": HOP_DURATION_SECONDS,
        "file_normality_percentile": FILE_NORMALITY_PERCENTILE,
        "min_normal_files_per_profile": MIN_NORMAL_FILES_PER_PROFILE
    }

    joblib.dump(model_package, model_path)


def train_and_save_model() -> dict:
    """
    Complete training-pipeline.

    Output blijft compatible met de Streamlit-app.
    """

    valid_segment_df, feature_columns = prepare_training_data()

    training_result = train_all_profile_models(
        valid_segment_df=valid_segment_df,
        feature_columns=feature_columns
    )

    profile_models = training_result["profile_models"]
    profile_metrics_df = training_result["profile_metrics_df"]
    skipped_profiles = training_result["skipped_profiles"]

    overall_metrics = build_overall_metrics(profile_metrics_df)

    save_model_package(
        profile_models=profile_models,
        feature_columns=feature_columns,
        metrics=overall_metrics,
        profile_metrics_df=profile_metrics_df,
        skipped_profiles=skipped_profiles
    )

    result = {
        "model": profile_models,
        "metrics": overall_metrics,
        "profile_metrics_df": profile_metrics_df,
        "feature_columns": feature_columns,
        "model_version": MODEL_VERSION,
        "model_path": str(MODEL_PATH),
        "feature_df": valid_segment_df,
        "skipped_profiles": skipped_profiles,
        "profile_models": profile_models,
        "confusion_matrix": pd.DataFrame(),
        "status_matrix": pd.DataFrame()
    }

    return result


if __name__ == "__main__":
    result = train_and_save_model()

    print("Profielspecifieke anomaly detection modellen succesvol getraind en opgeslagen.")
    print(f"Modelpad: {result['model_path']}")
    print("Globale metrics:")
    print(result["metrics"])
    print("Getrainde profielen:")
    print(result["profile_metrics_df"])