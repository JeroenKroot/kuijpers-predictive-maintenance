from pathlib import Path
import joblib
import pandas as pd

from src.feature_extraction import extract_features_from_file


MODEL_PATH = Path("models/model.pkl")

# Gevoeligere grens voor predictive maintenance.
# Bij 0.05 wordt een bestand eerder als afwijkend gezien dan bij 0.50.
# Deze grens sluit aan bij:
# - 0 t/m 5% afwijkend = normale situatie
# - >5% afwijkend = niet meer volledig normaal
ABNORMAL_THRESHOLD = 0.05


def load_model_package(model_path: Path = MODEL_PATH) -> dict:
    """
    Laadt het opgeslagen modelpakket.

    Het pakket bevat:
    - model
    - feature_columns
    - metrics
    - model_version
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Er is nog geen model gevonden op: {model_path}. Train eerst het model."
        )

    model_package = joblib.load(model_path)

    required_keys = ["model", "feature_columns", "model_version"]

    for key in required_keys:
        if key not in model_package:
            raise ValueError(f"Het modelbestand mist verplichte informatie: {key}")

    return model_package


def determine_status(anomaly_score: float) -> dict:
    """
    Vertaalt de anomaly score naar een duidelijke status.

    Nieuwe grenswaarden:
    - 0 t/m 5% afwijkend = normale situatie
    - >5 t/m 20% afwijkend = onderzoek nodig
    - >20% afwijkend = defect / niet normale situatie

    Dit komt overeen met:
    - 100 tot 95% normaal = normale situatie
    - 95 tot 80% normaal = onderzoek nodig
    - onder 80% normaal = defect / niet normaal
    """

    if anomaly_score <= 5:
        return {
            "status": "Normale situatie",
            "status_level": "green",
            "advice": "Geen directe actie nodig. Het geluidsfragment lijkt sterk op een normale situatie."
        }

    if anomaly_score <= 20:
        return {
            "status": "Onderzoek nodig",
            "status_level": "orange",
            "advice": "Controle of herhaalde meting aanbevolen. Het geluidsfragment wijkt mogelijk af van normaal gedrag."
        }

    return {
        "status": "Defect / niet normale situatie",
        "status_level": "red",
        "advice": "Inspectie aanbevolen. Het geluidsfragment lijkt onvoldoende op normaal gedrag."
    }


def create_explanation(
    predicted_class: str,
    anomaly_score: float,
    confidence_score: float,
    status: str
) -> str:
    """
    Maakt een korte zakelijke toelichting voor de demo-interface.
    """

    normal_score = 100 - anomaly_score

    if predicted_class == "normaal":
        return (
            f"Het model herkent vooral kenmerken die passen bij normaal geluid. "
            f"De normaliteits-score is {normal_score:.1f}/100. "
            f"De risicoscore is {anomaly_score:.1f}/100. "
            f"Status: {status}. "
            f"De confidence score is {confidence_score:.2f}."
        )

    if predicted_class == "onderzoek nodig":
        return (
            f"Het model ziet lichte afwijkingen ten opzichte van normaal geluid. "
            f"De normaliteits-score is {normal_score:.1f}/100. "
            f"De risicoscore is {anomaly_score:.1f}/100. "
            f"Status: {status}. "
            f"Een controlemeting of inspectie is aanbevolen."
        )

    return (
        f"Het model herkent kenmerken die onvoldoende passen bij normaal geluid. "
        f"De normaliteits-score is {normal_score:.1f}/100. "
        f"De risicoscore is {anomaly_score:.1f}/100. "
        f"Status: {status}. "
        f"Inspectie is aanbevolen."
    )


def predict_audio_file(file_input, model_path: Path = MODEL_PATH) -> dict:
    """
    Voorspelt of een WAV-bestand normaal, onderzoek nodig of defect / niet normaal is.

    Stappen:
    1. model laden
    2. features uit audio halen
    3. featurekolommen in juiste volgorde zetten
    4. kans op normaal en afwijkend berekenen
    5. anomaly score berekenen
    6. status en uitleg maken
    """

    model_package = load_model_package(model_path)

    model = model_package["model"]
    feature_columns = model_package["feature_columns"]
    model_version = model_package["model_version"]

    features = extract_features_from_file(file_input)
    feature_df = pd.DataFrame([features])

    missing_columns = [
        column for column in feature_columns
        if column not in feature_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Niet alle model-features konden uit het audiobestand worden gehaald. "
            f"Ontbrekende kolommen: {missing_columns}"
        )

    X = feature_df[feature_columns]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]

        # Labelafspraak:
        # 0 = normaal
        # 1 = afwijkend
        probability_normal = float(probabilities[0])
        probability_abnormal = float(probabilities[1])
    else:
        raw_prediction = int(model.predict(X)[0])
        probability_abnormal = float(raw_prediction)
        probability_normal = 1.0 - probability_abnormal

    anomaly_score = probability_abnormal * 100
    normal_score = probability_normal * 100

    # ------------------------------------------------------------
    # Nieuwe beoordelingsgrenzen
    # ------------------------------------------------------------
    # 0 t/m 5% afwijkend = normaal
    # >5 t/m 20% afwijkend = onderzoek nodig
    # >20% afwijkend = defect / niet normaal
    # ------------------------------------------------------------

    if anomaly_score <= 5:
        predicted_label = 0
        predicted_class = "normaal"
        confidence_score = probability_normal

    elif anomaly_score <= 20:
        predicted_label = 2
        predicted_class = "onderzoek nodig"
        confidence_score = max(probability_normal, probability_abnormal)

    else:
        predicted_label = 1
        predicted_class = "defect / niet normaal"
        confidence_score = probability_abnormal

    status_info = determine_status(anomaly_score)

    explanation = create_explanation(
        predicted_class=predicted_class,
        anomaly_score=anomaly_score,
        confidence_score=confidence_score,
        status=status_info["status"]
    )

    return {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "probability_normal": probability_normal,
        "probability_abnormal": probability_abnormal,
        "confidence_score": confidence_score,
        "normal_score": round(normal_score, 2),
        "anomaly_score": round(anomaly_score, 2),
        "status": status_info["status"],
        "status_level": status_info["status_level"],
        "advice": status_info["advice"],
        "explanation": explanation,
        "model_version": model_version,
        "features": features,
        "abnormal_threshold": ABNORMAL_THRESHOLD
    }