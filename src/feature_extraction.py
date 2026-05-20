import numpy as np
import pandas as pd
import librosa

from src.preprocessing import preprocess_audio


def calculate_band_energy(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Berekent energie per frequentiegebied.
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
        band_features[band_name] = float(band_energy / total_energy)

    return band_features


def extract_features_from_audio(
    audio: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13
) -> dict:
    """
    Extraheert audiofeatures uit één audiofragment.
    """

    features = {}

    features["duration_seconds"] = len(audio) / sample_rate
    features["mean_amplitude"] = float(np.mean(audio))
    features["std_amplitude"] = float(np.std(audio))
    features["max_amplitude"] = float(np.max(np.abs(audio)))
    features["min_amplitude"] = float(np.min(audio))
    features["peak_to_peak_amplitude"] = float(np.max(audio) - np.min(audio))

    rms_total = np.sqrt(np.mean(audio ** 2)) + 1e-10
    features["crest_factor"] = float(np.max(np.abs(audio)) / rms_total)

    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc
    )

    for i in range(n_mfcc):
        features[f"mfcc_{i + 1}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i + 1}_std"] = float(np.std(mfccs[i]))

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate,
        roll_percent=0.85
    )

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sample_rate
    )

    spectral_flatness = librosa.feature.spectral_flatness(
        y=audio
    )

    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        y=audio
    )

    rms_energy = librosa.feature.rms(
        y=audio
    )

    features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
    features["spectral_centroid_std"] = float(np.std(spectral_centroid))

    features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
    features["spectral_rolloff_std"] = float(np.std(spectral_rolloff))

    features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
    features["spectral_bandwidth_std"] = float(np.std(spectral_bandwidth))

    features["spectral_flatness_mean"] = float(np.mean(spectral_flatness))
    features["spectral_flatness_std"] = float(np.std(spectral_flatness))

    features["zero_crossing_rate_mean"] = float(np.mean(zero_crossing_rate))
    features["zero_crossing_rate_std"] = float(np.std(zero_crossing_rate))

    features["rms_energy_mean"] = float(np.mean(rms_energy))
    features["rms_energy_std"] = float(np.std(rms_energy))
    features["rms_energy_min"] = float(np.min(rms_energy))
    features["rms_energy_max"] = float(np.max(rms_energy))
    features["rms_energy_p25"] = float(np.percentile(rms_energy, 25))
    features["rms_energy_p75"] = float(np.percentile(rms_energy, 75))

    band_features = calculate_band_energy(audio, sample_rate)
    features.update(band_features)

    return features


def extract_features_from_file(file_input) -> dict:
    """
    Maakt features van een volledig WAV-bestand.
    Deze functie blijft bestaan voor compatibiliteit met bestaande app-onderdelen.
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


def split_audio_into_segments(
    audio: np.ndarray,
    sample_rate: int,
    segment_duration_seconds: float = 2.0,
    hop_duration_seconds: float = 1.0
) -> list:
    """
    Knipt audio op in overlappende segmenten.

    Voorbeeld:
    - segmentduur 2 seconden
    - hop 1 seconde

    Dan kijkt de app naar:
    0-2 sec
    1-3 sec
    2-4 sec
    enzovoort.
    """

    segment_samples = int(segment_duration_seconds * sample_rate)
    hop_samples = int(hop_duration_seconds * sample_rate)

    if len(audio) <= segment_samples:
        return [
            {
                "segment_index": 0,
                "start_seconds": 0.0,
                "end_seconds": round(len(audio) / sample_rate, 2),
                "audio": audio
            }
        ]

    segments = []
    segment_index = 0

    for start_sample in range(0, len(audio) - segment_samples + 1, hop_samples):
        end_sample = start_sample + segment_samples

        segment_audio = audio[start_sample:end_sample]

        segments.append({
            "segment_index": segment_index,
            "start_seconds": round(start_sample / sample_rate, 2),
            "end_seconds": round(end_sample / sample_rate, 2),
            "audio": segment_audio
        })

        segment_index += 1

    return segments


def extract_segment_features_from_file(
    file_input,
    segment_duration_seconds: float = 2.0,
    hop_duration_seconds: float = 1.0
) -> pd.DataFrame:
    """
    Maakt features per segment van een WAV-bestand.

    Output:
    DataFrame met één rij per segment.
    """

    preprocessing_result = preprocess_audio(file_input)

    if not preprocessing_result["is_valid"]:
        raise ValueError(
            "Audio is niet geschikt voor segmentanalyse: "
            + " | ".join(preprocessing_result["messages"])
        )

    audio = preprocessing_result["processed_audio"]
    sample_rate = preprocessing_result["sample_rate"]

    segments = split_audio_into_segments(
        audio=audio,
        sample_rate=sample_rate,
        segment_duration_seconds=segment_duration_seconds,
        hop_duration_seconds=hop_duration_seconds
    )

    records = []

    for segment in segments:
        segment_features = extract_features_from_audio(
            audio=segment["audio"],
            sample_rate=sample_rate
        )

        segment_features["segment_index"] = segment["segment_index"]
        segment_features["segment_start_seconds"] = segment["start_seconds"]
        segment_features["segment_end_seconds"] = segment["end_seconds"]

        records.append(segment_features)

    return pd.DataFrame(records)


def create_feature_dataset(training_index: pd.DataFrame) -> pd.DataFrame:
    """
    Maakt een feature-dataset van alle trainingsbestanden.

    Dit is de bestand-gebaseerde feature extraction.
    Deze blijft bestaan voor datacontrole in de app.
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


def create_segment_feature_dataset(
    training_index: pd.DataFrame,
    segment_duration_seconds: float = 2.0,
    hop_duration_seconds: float = 1.0
) -> pd.DataFrame:
    """
    Maakt een segment-gebaseerde feature-dataset.

    Eén WAV-bestand kan dus meerdere rijen opleveren.
    """

    records = []

    for _, row in training_index.iterrows():
        file_path = row["file_path"]

        try:
            segment_df = extract_segment_features_from_file(
                file_input=file_path,
                segment_duration_seconds=segment_duration_seconds,
                hop_duration_seconds=hop_duration_seconds
            )

            segment_df["label"] = row["label"]
            segment_df["class_name"] = row["class_name"]
            segment_df["source_folder"] = row["source_folder"]
            segment_df["file_path"] = file_path
            segment_df["is_valid"] = True
            segment_df["error_message"] = ""

            records.extend(segment_df.to_dict(orient="records"))

        except Exception as error:
            records.append({
                "label": row["label"],
                "class_name": row["class_name"],
                "source_folder": row["source_folder"],
                "file_path": file_path,
                "segment_index": -1,
                "segment_start_seconds": 0,
                "segment_end_seconds": 0,
                "is_valid": False,
                "error_message": str(error)
            })

    return pd.DataFrame(records)


def get_model_feature_columns(feature_df: pd.DataFrame) -> list:
    """
    Geeft de kolommen terug die als modelinput gebruikt worden.
    Metadata en labels worden uitgesloten.
    """

    excluded_columns = [
        "label",
        "class_name",
        "source_folder",
        "file_path",
        "is_valid",
        "error_message",
        "segment_index",
        "segment_start_seconds",
        "segment_end_seconds"
    ]

    feature_columns = [
        column for column in feature_df.columns
        if column not in excluded_columns
    ]

    return feature_columns