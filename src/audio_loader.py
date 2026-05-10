from pathlib import Path
import pandas as pd


# Basisinstellingen
SUPPORTED_AUDIO_EXTENSIONS = [".wav"]


def find_audio_files(folder_path: str) -> list[Path]:
    """
    Zoekt recursief naar WAV-bestanden in een map en alle onderliggende submappen.

    Voorbeeld:
    data/normaal/id_00_0DB/bestand.wav
    data/normaal/id_02_6DB/bestand.wav
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"De map bestaat niet: {folder}")

    audio_files = []

    for extension in SUPPORTED_AUDIO_EXTENSIONS:
        audio_files.extend(folder.rglob(f"*{extension}"))

    return sorted(audio_files)


def create_training_index(
    normal_folder: str = "data/normaal",
    abnormal_folder: str = "data/afwijkend"
) -> pd.DataFrame:
    """
    Maakt een overzicht van alle trainingsbestanden.

    Labelafspraak:
    - normaal = 0
    - afwijkend = 1

    De originele bestandsnamen worden alleen intern gebruikt.
    In de latere analysehistorie slaan we geanonimiseerde namen op.
    """

    normal_files = find_audio_files(normal_folder)
    abnormal_files = find_audio_files(abnormal_folder)

    records = []

    for file_path in normal_files:
        records.append({
            "file_path": str(file_path),
            "label": 0,
            "class_name": "normaal",
            "source_folder": file_path.parent.name
        })

    for file_path in abnormal_files:
        records.append({
            "file_path": str(file_path),
            "label": 1,
            "class_name": "afwijkend",
            "source_folder": file_path.parent.name
        })

    df = pd.DataFrame(records)

    return df


def summarize_training_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Geeft een korte samenvatting van het aantal bestanden per klasse.
    """

    if df.empty:
        return pd.DataFrame(columns=["class_name", "aantal_bestanden"])

    summary = (
        df.groupby("class_name")
        .size()
        .reset_index(name="aantal_bestanden")
        .sort_values("class_name")
    )

    return summary


def validate_training_index(df: pd.DataFrame) -> dict:
    """
    Controleert of er voldoende data aanwezig is.

    Deze controle is bewust eenvoudig.
    In latere stappen voegen we controles toe op audiokwaliteit.
    """

    result = {
        "is_valid": True,
        "messages": []
    }

    if df.empty:
        result["is_valid"] = False
        result["messages"].append("Er zijn geen WAV-bestanden gevonden.")

    if "normaal" not in df["class_name"].unique():
        result["is_valid"] = False
        result["messages"].append("Er zijn geen normale WAV-bestanden gevonden.")

    if "afwijkend" not in df["class_name"].unique():
        result["is_valid"] = False
        result["messages"].append("Er zijn geen afwijkende WAV-bestanden gevonden.")

    return result