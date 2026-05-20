from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.feature_extraction import extract_segment_features_from_file


MODEL_PATH = Path("models/model.pkl")


def load_model_package(model_path: Path = MODEL_PATH) -> dict:
    """
    Laadt het modelpakket met meerdere profielmodellen.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Er is nog geen model gevonden op: {model_path}. Train eerst het model."
        )

    model_package = joblib.load(model_path)

    required_keys = [
        "profile_models",
        "feature_columns",
        "model_version"
    ]

    for key in required_keys:
        if key not in model_package:
            raise ValueError(
                f"Het modelbestand mist verplichte informatie: {key}. "
                "Train het profielspecifieke anomaly detection model opnieuw."
            )

    return model_package


def calculate_normality_score(raw_score: float, calibration_scores: np.ndarray) -> float:
    """
    Zet de ruwe Isolation Forest-score om naar een normaliteits-score van 0 tot 100.
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


def determine_status(
    normality_score: float,
    attention_threshold: int,
    defect_threshold: int
) -> dict:
    """
    Bepaalt status op basis van de thresholds van het gekozen profielmodel.
    """

    if normality_score >= attention_threshold:
        return {
            "status": "Normale situatie",
            "status_level": "green",
            "advice": "Geen directe actie nodig. Het geluidsfragment lijkt voldoende op het normale geluidsprofiel."
        }

    if normality_score >= defect_threshold:
        return {
            "status": "Onderzoek nodig",
            "status_level": "orange",
            "advice": "Controle of herhaalde meting aanbevolen. Eén of meerdere segmenten wijken mogelijk af."
        }

    return {
        "status": "Defect / niet normale situatie",
        "status_level": "red",
        "advice": "Inspectie aanbevolen. Eén of meerdere segmenten passen onvoldoende bij het normale geluidsprofiel."
    }


def score_segments_with_profile(
    segment_feature_df: pd.DataFrame,
    feature_columns: list,
    profile_name: str,
    profile_model_package: dict,
    file_normality_percentile: float
) -> dict:
    """
    Scoort één uploadbestand met één profielmodel.

    Voorbeeld:
    uploadbestand wordt gescoord met profielmodel id_04_6DB.
    """

    model = profile_model_package["model"]
    calibration_scores = profile_model_package["calibration_scores"]
    attention_threshold = int(profile_model_package["attention_threshold"])
    defect_threshold = int(profile_model_package["defect_threshold"])

    X_segments = segment_feature_df[feature_columns]
    raw_scores = model.score_samples(X_segments)

    segment_results = []

    for index, raw_score in enumerate(raw_scores):
        segment_row = segment_feature_df.iloc[index]

        segment_normality = calculate_normality_score(
            raw_score=float(raw_score),
            calibration_scores=calibration_scores
        )

        segment_anomaly = 100 - segment_normality

        if segment_normality >= attention_threshold:
            segment_status = "normaal"
        elif segment_normality >= defect_threshold:
            segment_status = "onderzoek nodig"
        else:
            segment_status = "defect / niet normaal"

        segment_results.append({
            "segment_index": int(segment_row["segment_index"]),
            "segment_start_seconds": float(segment_row["segment_start_seconds"]),
            "segment_end_seconds": float(segment_row["segment_end_seconds"]),
            "normality_score": round(segment_normality, 2),
            "anomaly_score": round(segment_anomaly, 2),
            "status": segment_status
        })

    segment_results_df = pd.DataFrame(segment_results)

    file_normality_score = float(
        np.percentile(
            segment_results_df["normality_score"],
            file_normality_percentile
        )
    )

    file_anomaly_score = 100 - file_normality_score

    worst_segment = segment_results_df.sort_values(
        by="normality_score",
        ascending=True
    ).iloc[0]

    status_info = determine_status(
        normality_score=file_normality_score,
        attention_threshold=attention_threshold,
        defect_threshold=defect_threshold
    )

    return {
        "profile_name": profile_name,
        "asset_id": profile_model_package.get("asset_id"),
        "noise_profile": profile_model_package.get("noise_profile"),
        "normality_score": round(file_normality_score, 2),
        "anomaly_score": round(file_anomaly_score, 2),
        "attention_threshold": attention_threshold,
        "defect_threshold": defect_threshold,
        "status": status_info["status"],
        "status_level": status_info["status_level"],
        "advice": status_info["advice"],
        "segment_results": segment_results,
        "worst_segment_start": float(worst_segment["segment_start_seconds"]),
        "worst_segment_end": float(worst_segment["segment_end_seconds"]),
        "segments_total": int(len(segment_results_df))
    }


def aggregate_feature_summary(
    segment_feature_df: pd.DataFrame,
    feature_columns: list
) -> dict:
    """
    Maakt een compacte feature-samenvatting voor weergave in de app.
    """

    summary = {}

    for column in feature_columns:
        if column in segment_feature_df.columns:
            summary[f"{column}_mean_over_segments"] = float(segment_feature_df[column].mean())

    return summary


def create_explanation(
    selected_asset_id: str,
    chosen_profile: str,
    chosen_noise_profile: str,
    normality_score: float,
    anomaly_score: float,
    status: str,
    attention_threshold: int,
    defect_threshold: int,
    segments_total: int,
    worst_segment_start: float,
    worst_segment_end: float
) -> str:
    """
    Maakt een uitleg voor de demo-interface.
    """

    return (
        f"Je hebt asset {selected_asset_id} geselecteerd. "
        f"De applicatie heeft het uploadbestand automatisch vergeleken met de beschikbare ruisprofielen "
        f"voor deze asset. Het best passende ruisprofiel is {chosen_noise_profile}. "
        f"Daarom is referentiemodel {chosen_profile} gebruikt. "
        f"Het bestand is opgeknipt in {segments_total} segment(en). "
        f"Het meest afwijkende segment zat rond {worst_segment_start:.1f}-{worst_segment_end:.1f} seconden. "
        f"Binnen dit referentieprofiel is het geluid {normality_score:.1f}% normaal. "
        f"De risicoscore is {anomaly_score:.1f}/100. "
        f"De aandachtgrens ligt op {attention_threshold}/100 en de defectgrens op {defect_threshold}/100. "
        f"Conclusie: {status}."
    )


def get_available_asset_ids(model_path: Path = MODEL_PATH) -> list:
    """
    Geeft de beschikbare asset-ID's terug uit het modelpakket.
    """

    model_package = load_model_package(model_path)
    profile_models = model_package["profile_models"]

    asset_ids = sorted({
        profile_info.get("asset_id")
        for profile_info in profile_models.values()
        if profile_info.get("asset_id")
    })

    return asset_ids


def predict_audio_file(
    file_input,
    selected_asset_id: str,
    model_path: Path = MODEL_PATH
) -> dict:
    """
    Voorspelt met profielspecifieke anomaly detection.

    Werkwijze:
    1. Gebruiker kiest asset-ID, bijvoorbeeld id_04.
    2. De app scoort het uploadbestand met alle ruisprofielen van id_04.
    3. De app kiest het profiel met de hoogste normaliteits-score.
    4. Dat profiel bepaalt de eindconclusie.
    """

    model_package = load_model_package(model_path)

    profile_models = model_package["profile_models"]
    feature_columns = model_package["feature_columns"]
    model_version = model_package["model_version"]

    segment_duration_seconds = float(model_package.get("segment_duration_seconds", 3.0))
    hop_duration_seconds = float(model_package.get("hop_duration_seconds", 1.5))
    file_normality_percentile = float(model_package.get("file_normality_percentile", 35))

    matching_profiles = {
        profile_name: profile_info
        for profile_name, profile_info in profile_models.items()
        if profile_info.get("asset_id") == selected_asset_id
    }

    if not matching_profiles:
        available_assets = sorted({
            profile_info.get("asset_id")
            for profile_info in profile_models.values()
            if profile_info.get("asset_id")
        })

        raise ValueError(
            f"Er zijn geen profielmodellen gevonden voor asset {selected_asset_id}. "
            f"Beschikbare assets: {available_assets}"
        )

    segment_feature_df = extract_segment_features_from_file(
        file_input=file_input,
        segment_duration_seconds=segment_duration_seconds,
        hop_duration_seconds=hop_duration_seconds
    )

    missing_columns = [
        column for column in feature_columns
        if column not in segment_feature_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Niet alle model-features konden uit de audiosegmenten worden gehaald. "
            f"Ontbrekende kolommen: {missing_columns}"
        )

    profile_scores = []

    for profile_name, profile_info in matching_profiles.items():
        profile_score = score_segments_with_profile(
            segment_feature_df=segment_feature_df,
            feature_columns=feature_columns,
            profile_name=profile_name,
            profile_model_package=profile_info,
            file_normality_percentile=file_normality_percentile
        )

        profile_scores.append(profile_score)

    profile_scores_df = pd.DataFrame(profile_scores)

    best_profile_row = profile_scores_df.sort_values(
        by="normality_score",
        ascending=False
    ).iloc[0]

    chosen_profile_name = best_profile_row["profile_name"]
    chosen_profile_score = next(
        item for item in profile_scores
        if item["profile_name"] == chosen_profile_name
    )

    normality_score = float(chosen_profile_score["normality_score"])
    anomaly_score = float(chosen_profile_score["anomaly_score"])

    if chosen_profile_score["status_level"] == "green":
        predicted_label = 0
        predicted_class = "normaal"
    elif chosen_profile_score["status_level"] == "orange":
        predicted_label = 2
        predicted_class = "onderzoek nodig"
    else:
        predicted_label = 1
        predicted_class = "defect / niet normaal"

    probability_normal = normality_score / 100
    probability_abnormal = anomaly_score / 100
    confidence_score = max(probability_normal, probability_abnormal)

    explanation = create_explanation(
        selected_asset_id=selected_asset_id,
        chosen_profile=chosen_profile_name,
        chosen_noise_profile=chosen_profile_score["noise_profile"],
        normality_score=normality_score,
        anomaly_score=anomaly_score,
        status=chosen_profile_score["status"],
        attention_threshold=int(chosen_profile_score["attention_threshold"]),
        defect_threshold=int(chosen_profile_score["defect_threshold"]),
        segments_total=int(chosen_profile_score["segments_total"]),
        worst_segment_start=float(chosen_profile_score["worst_segment_start"]),
        worst_segment_end=float(chosen_profile_score["worst_segment_end"])
    )

    feature_summary = aggregate_feature_summary(
        segment_feature_df=segment_feature_df,
        feature_columns=feature_columns
    )

    return {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "probability_normal": probability_normal,
        "probability_abnormal": probability_abnormal,
        "confidence_score": confidence_score,
        "normal_score": round(normality_score, 2),
        "normality_score": round(normality_score, 2),
        "anomaly_score": round(anomaly_score, 2),
        "status": chosen_profile_score["status"],
        "status_level": chosen_profile_score["status_level"],
        "advice": chosen_profile_score["advice"],
        "explanation": explanation,
        "model_version": model_version,
        "features": feature_summary,
        "segment_results": chosen_profile_score["segment_results"],
        "profile_scores": profile_scores,
        "selected_asset_id": selected_asset_id,
        "chosen_profile": chosen_profile_name,
        "chosen_noise_profile": chosen_profile_score["noise_profile"],
        "attention_threshold": int(chosen_profile_score["attention_threshold"]),
        "defect_threshold": int(chosen_profile_score["defect_threshold"]),
        "model_type": "profile_specific_segment_anomaly_detection"
    }