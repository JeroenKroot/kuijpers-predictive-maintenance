from pathlib import Path
from datetime import datetime
import pandas as pd


HISTORY_PATH = Path("history/analyse_history.csv")

HISTORY_COLUMNS = [
    "timestamp",
    "asset_id",
    "location",
    "filename_anonymized",
    "predicted_class",
    "anomaly_score",
    "confidence_score",
    "model_version",
    "explanation"
]


def ensure_history_file(history_path: Path = HISTORY_PATH) -> None:
    """
    Zorgt dat het history-bestand bestaat met de juiste kolommen.
    Als het bestand leeg of beschadigd is, wordt het opnieuw aangemaakt.
    """

    history_path.parent.mkdir(parents=True, exist_ok=True)

    if not history_path.exists() or history_path.stat().st_size == 0:
        empty_df = pd.DataFrame(columns=HISTORY_COLUMNS)
        empty_df.to_csv(history_path, index=False)
        return

    try:
        existing_df = pd.read_csv(history_path)

        for column in HISTORY_COLUMNS:
            if column not in existing_df.columns:
                existing_df[column] = ""

        existing_df = existing_df[HISTORY_COLUMNS]
        existing_df.to_csv(history_path, index=False)

    except pd.errors.EmptyDataError:
        empty_df = pd.DataFrame(columns=HISTORY_COLUMNS)
        empty_df.to_csv(history_path, index=False)


def anonymize_filename(prefix: str = "meting") -> str:
    """
    Maakt een geanonimiseerde bestandsnaam.

    Voorbeeld:
    meting_20260510_154501.wav
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.wav"


def load_analysis_history(history_path: Path = HISTORY_PATH) -> pd.DataFrame:
    """
    Laadt de analysehistorie.
    Als het bestand leeg is, wordt een lege tabel met kolommen teruggegeven.
    """

    ensure_history_file(history_path)

    try:
        history_df = pd.read_csv(history_path)
    except pd.errors.EmptyDataError:
        history_df = pd.DataFrame(columns=HISTORY_COLUMNS)

    for column in HISTORY_COLUMNS:
        if column not in history_df.columns:
            history_df[column] = ""

    history_df = history_df[HISTORY_COLUMNS]

    return history_df


def log_analysis_result(
    prediction: dict,
    filename_anonymized: str,
    asset_id: str = "Asset 001",
    location: str = "Locatie A",
    history_path: Path = HISTORY_PATH
) -> None:
    """
    Slaat één analyse op in de analysehistorie.

    Originele bestandsnaam, klantnaam, echte locatie en echte assetnaam
    worden niet opgeslagen.
    """

    ensure_history_file(history_path)

    new_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "asset_id": asset_id,
        "location": location,
        "filename_anonymized": filename_anonymized,
        "predicted_class": prediction.get("predicted_class", ""),
        "anomaly_score": prediction.get("anomaly_score", ""),
        "confidence_score": round(float(prediction.get("confidence_score", 0)), 4),
        "model_version": prediction.get("model_version", ""),
        "explanation": prediction.get("explanation", "")
    }

    try:
        history_df = pd.read_csv(history_path)
    except pd.errors.EmptyDataError:
        history_df = pd.DataFrame(columns=HISTORY_COLUMNS)

    for column in HISTORY_COLUMNS:
        if column not in history_df.columns:
            history_df[column] = ""

    history_df = history_df[HISTORY_COLUMNS]

    new_record_df = pd.DataFrame([new_record], columns=HISTORY_COLUMNS)

    history_df = pd.concat(
        [history_df, new_record_df],
        ignore_index=True
    )

    history_df.to_csv(history_path, index=False)


def clear_analysis_history(history_path: Path = HISTORY_PATH) -> None:
    """
    Maakt de analysehistorie leeg, maar behoudt de kolommen.
    Handig tijdens testen.
    """

    history_path.parent.mkdir(parents=True, exist_ok=True)

    empty_df = pd.DataFrame(columns=HISTORY_COLUMNS)
    empty_df.to_csv(history_path, index=False)