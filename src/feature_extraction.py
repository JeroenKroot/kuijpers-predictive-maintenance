import numpy as np
import pandas as pd
import librosa
from streamlit import audio

from src.preprocessing import preprocess_audio


def calculate_band_energy(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Berekent eenvoudige energie per frequentiegebied.

    Dit helpt om verschil te zien tussen laagfrequente brom,
    middengebied en hogere mechanische geluiden.
    """

    fft_values = np.abs(np.fft.rfft(audio))
    fft_frequencies = np.fft.rfftfreq(len(audio), d=1 / sample_rate)

    total_energy = np.sum(fft_values ** 2)

    if total_energy == 0:
        total_energy = 1e-10

    bands = {
        "band_energy_0_500": (0, 500),
        "band_energy_500_2000": (500, 2000),
        "band_energy_2000_5000": (2000, 5000),
        "band_energy_5000_8000": (5000, 8000),
    }

    band_features = {}

    for band_name, (low_freq, high_freq) in bands.items():
        band_mask = (fft_frequencies >= low_freq) & (fft_frequencies < high_freq)
        band_energy = np.sum(fft_values[band_mask] ** 2)
        band_features[band_name] = band_energy / total_energy

    return band_features


def extract_features_from_audio(
    audio: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13
) -> dict:
    """
    Extraheert audiofeatures uit reeds gepreprocesste audio.

    Output:
    Een dictionary met vaste kolommen die later door het model gebruikt worden.
    """

    features = {}

    # ------------------------------------------------------------
    # Basiskenmerken
    # ------------------------------------------------------------

    features["duration_seconds"] = len(audio) / sample_rate
    features["mean_amplitude"] = float(np.mean(audio))
    features["std_amplitude"] = float(np.std(audio))
    features["max_amplitude"] = float(np.max(np.abs(audio)))
    features["min_amplitude"] = float(np.min(audio))
    features["peak_to_peak_amplitude"] = float(np.max(audio) - np.min(audio))

    # Crest factor: verhouding tussen piekwaarde en gemiddelde energie.
    # Dit helpt bij tikken, schrapen of korte piekgeluiden.
    rms_total = np.sqrt(np.mean(audio ** 2)) + 1e-10
    features["crest_factor"] = float(np.max(np.abs(audio)) / rms_total)


    # ------------------------------------------------------------
    # MFCC-features
    # ------------------------------------------------------------
    # MFCC's vatten de klankkleur van audio samen.
    # We slaan per MFCC zowel het gemiddelde als de spreiding op.
    # ------------------------------------------------------------

    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc
    )

    for i in range(n_mfcc):
        features[f"mfcc_{i + 1}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i + 1}_std"] = float(np.std(mfccs[i]))

    # ------------------------------------------------------------
    # Spectrale kenmerken
    # ------------------------------------------------------------

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate,
        roll_percent=0.85
    )

    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        y=audio
    )

    rms_energy = librosa.feature.rms(
        y=audio
    )

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
    y=audio,
    sr=sample_rate
)

    spectral_flatness = librosa.feature.spectral_flatness(
    y=audio
)

    features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
    features["spectral_centroid_std"] = float(np.std(spectral_centroid))

    features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
    features["spectral_rolloff_std"] = float(np.std(spectral_rolloff))

    features["zero_crossing_rate_mean"] = float(np.mean(zero_crossing_rate))
    features["zero_crossing_rate_std"] = float(np.std(zero_crossing_rate))

    features["rms_energy_mean"] = float(np.mean(rms_energy))
    features["rms_energy_std"] = float(np.std(rms_energy))
    features["rms_energy_min"] = float(np.min(rms_energy))
    features["rms_energy_max"] = float(np.max(rms_energy))
    features["rms_energy_p25"] = float(np.percentile(rms_energy, 25))
    features["rms_energy_p75"] = float(np.percentile(rms_energy, 75))

    features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
    features["spectral_bandwidth_std"] = float(np.std(spectral_bandwidth))

    features["spectral_flatness_mean"] = float(np.mean(spectral_flatness))
    features["spectral_flatness_std"] = float(np.std(spectral_flatness))

    # ------------------------------------------------------------
    # Frequentieband-energie
    # ------------------------------------------------------------

    band_features = calculate_band_energy(audio, sample_rate)
    features.update(band_features)

    return features


def extract_features_from_file(file_input) -> dict:
    """
    Complete pipeline voor één bestand:

    WAV-bestand
    ↓
    preprocessing
    ↓
    feature extraction

    Geeft alleen features terug als het bestand geldig is.
    """

    preprocessing_result = preprocess_audio(file_input)

    if not preprocessing_result["is_valid"]:
        raise ValueError(
            "Audio is niet geschikt voor feature extraction: "
            + " | ".join(preprocessing_result["messages"])
        )

    audio = preprocessing_result["processed_audio"]
    sample_rate = preprocessing_result["sample_rate"]

    features = extract_features_from_audio(audio, sample_rate)

    return features


def create_feature_dataset(training_index: pd.DataFrame) -> pd.DataFrame:
    """
    Maakt een volledige feature-dataset van alle trainingsbestanden.

    Input:
    training_index met:
    - file_path
    - label
    - class_name
    - source_folder

    Output:
    DataFrame met:
    - audiofeatures
    - label
    - class_name
    - source_folder
    """

    records = []

    for _, row in training_index.iterrows():
        file_path = row["file_path"]

        try:
            features = extract_features_from_file(file_path)

            features["label"] = row["label"]
            features["class_name"] = row["class_name"]
            features["source_folder"] = row["source_folder"]
            features["file_path"] = file_path
            features["is_valid"] = True
            features["error_message"] = ""

            records.append(features)

        except Exception as error:
            records.append({
                "label": row["label"],
                "class_name": row["class_name"],
                "source_folder": row["source_folder"],
                "file_path": file_path,
                "is_valid": False,
                "error_message": str(error)
            })

    feature_df = pd.DataFrame(records)

    return feature_df


def get_model_feature_columns(feature_df: pd.DataFrame) -> list:
    """
    Geeft de kolommen terug die straks als input voor het model gebruikt worden.

    We sluiten metadata en labels uit.
    """

    excluded_columns = [
        "label",
        "class_name",
        "source_folder",
        "file_path",
        "is_valid",
        "error_message"
    ]

    feature_columns = [
        column for column in feature_df.columns
        if column not in excluded_columns
    ]

    return feature_columns