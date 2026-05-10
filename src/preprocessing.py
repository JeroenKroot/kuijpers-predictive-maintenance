from pathlib import Path
import numpy as np
import librosa


# ------------------------------------------------------------
# Basisinstellingen voor de POC
# ------------------------------------------------------------

TARGET_SAMPLE_RATE = 16000
MIN_DURATION_SECONDS = 1.0
MAX_DURATION_SECONDS = 10.0


def load_audio_file(file_input, target_sample_rate: int = TARGET_SAMPLE_RATE):
    """
    Laadt een audiobestand in met librosa.

    Werkt met:
    - een bestandspad vanuit data/normaal of data/afwijkend;
    - een geüpload bestand vanuit Streamlit.

    Output:
    - audio: numpy-array met audiosignaal
    - sample_rate: vaste sample rate
    """

    try:
        audio, sample_rate = librosa.load(
            file_input,
            sr=target_sample_rate,
            mono=True
        )
    except Exception as error:
        raise ValueError(f"Het audiobestand kon niet worden ingelezen: {error}")

    return audio, sample_rate


def validate_audio(
    audio: np.ndarray,
    sample_rate: int,
    min_duration_seconds: float = MIN_DURATION_SECONDS
) -> dict:
    """
    Controleert of het audiobestand bruikbaar is.

    Controleert:
    - audio bestaat;
    - audio bevat samples;
    - audio is niet stil/leeg;
    - audio is lang genoeg.
    """

    result = {
        "is_valid": True,
        "messages": [],
        "duration_seconds": 0.0
    }

    if audio is None:
        result["is_valid"] = False
        result["messages"].append("Audio ontbreekt.")
        return result

    if len(audio) == 0:
        result["is_valid"] = False
        result["messages"].append("Het audiobestand bevat geen samples.")
        return result

    duration_seconds = len(audio) / sample_rate
    result["duration_seconds"] = round(duration_seconds, 2)

    if duration_seconds < min_duration_seconds:
        result["is_valid"] = False
        result["messages"].append(
            f"Het audiobestand is te kort. Minimale lengte is {min_duration_seconds} seconde."
        )

    max_amplitude = np.max(np.abs(audio))

    if max_amplitude == 0:
        result["is_valid"] = False
        result["messages"].append("Het audiobestand lijkt volledig stil te zijn.")

    if np.isnan(audio).any():
        result["is_valid"] = False
        result["messages"].append("Het audiobestand bevat ongeldige waarden.")

    return result


def normalize_volume(audio: np.ndarray) -> np.ndarray:
    """
    Normaliseert het volume.

    Hierdoor worden bestanden met verschillende opnamevolumes beter vergelijkbaar.
    """

    max_amplitude = np.max(np.abs(audio))

    if max_amplitude == 0:
        return audio

    normalized_audio = audio / max_amplitude

    return normalized_audio


def trim_or_pad_audio(
    audio: np.ndarray,
    sample_rate: int,
    max_duration_seconds: float = MAX_DURATION_SECONDS
) -> np.ndarray:
    """
    Maakt audio geschikt voor analyse.

    - Is audio langer dan de maximale lengte? Dan knippen we het af.
    - Is audio korter? Dan laten we het zoals het is.

    Voor deze POC kiezen we voor afknippen in plaats van complexe segmentatie.
    Segmentatie kan later worden toegevoegd.
    """

    max_samples = int(max_duration_seconds * sample_rate)

    if len(audio) > max_samples:
        return audio[:max_samples]

    return audio


def preprocess_audio(file_input) -> dict:
    """
    Complete preprocessing-pipeline.

    Deze functie wordt straks gebruikt voor:
    - trainingsbestanden;
    - geüploade testbestanden.

    Output is een dictionary met:
    - processed_audio
    - sample_rate
    - duration_seconds
    - is_valid
    - messages
    """

    audio, sample_rate = load_audio_file(file_input)

    validation_before = validate_audio(audio, sample_rate)

    if not validation_before["is_valid"]:
        return {
            "is_valid": False,
            "messages": validation_before["messages"],
            "processed_audio": None,
            "sample_rate": sample_rate,
            "duration_seconds": validation_before["duration_seconds"]
        }

    audio = normalize_volume(audio)
    audio = trim_or_pad_audio(audio, sample_rate)

    validation_after = validate_audio(audio, sample_rate)

    return {
        "is_valid": validation_after["is_valid"],
        "messages": validation_after["messages"],
        "processed_audio": audio,
        "sample_rate": sample_rate,
        "duration_seconds": round(len(audio) / sample_rate, 2)
    }


def preprocess_training_files(training_index):
    """
    Controleert en preprocesset alle trainingsbestanden.

    Deze functie gebruiken we vooral als controle:
    hoeveel bestanden zijn bruikbaar en hoeveel vallen af?
    """

    records = []

    for _, row in training_index.iterrows():
        file_path = row["file_path"]

        try:
            result = preprocess_audio(file_path)

            records.append({
                "file_path": file_path,
                "class_name": row["class_name"],
                "label": row["label"],
                "source_folder": row["source_folder"],
                "is_valid": result["is_valid"],
                "duration_seconds": result["duration_seconds"],
                "messages": " | ".join(result["messages"])
            })

        except Exception as error:
            records.append({
                "file_path": file_path,
                "class_name": row["class_name"],
                "label": row["label"],
                "source_folder": row["source_folder"],
                "is_valid": False,
                "duration_seconds": 0,
                "messages": str(error)
            })

    return records