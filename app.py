import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile

from src.audio_loader import (
    create_training_index,
    summarize_training_index,
    validate_training_index
)

from src.preprocessing import (
    preprocess_audio,
    preprocess_training_files,
    TARGET_SAMPLE_RATE,
    MIN_DURATION_SECONDS,
    MAX_DURATION_SECONDS
)

from src.feature_extraction import (
    create_feature_dataset,
    get_model_feature_columns
)

from src.train_model import (
    train_and_save_model,
    MODEL_PATH,
    MODEL_VERSION
)

from src.predict import predict_audio_file

from src.visualization import (
    create_waveform_figure,
    create_score_bar_html
)

from src.history_logger import (
    log_analysis_result,
    load_analysis_history,
    clear_analysis_history,
    anonymize_filename,
    HISTORY_PATH
)


# ------------------------------------------------------------
# Pagina-instellingen
# ------------------------------------------------------------

st.set_page_config(
    page_title="Kuijpers Predictive Maintenance",
    page_icon="🔊",
    layout="wide"
)


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1350px;
    }

    .kuijpers-header {
        background: linear-gradient(90deg, #003B5C 0%, #005B8C 55%, #009FE3 100%);
        padding: 26px 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 59, 92, 0.18);
    }

    .kuijpers-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .kuijpers-subtitle {
        font-size: 17px;
        opacity: 0.95;
    }

    .section-card {
        background-color: #FFFFFF;
        border: 1px solid #E1E7EC;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
    }

    .demo-card {
        background-color: #F5F8FA;
        border-left: 7px solid #009FE3;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 18px;
    }

    .small-muted {
        color: #667085;
        font-size: 14px;
    }

    .status-card {
        padding: 24px;
        border-radius: 18px;
        background-color: #FFFFFF;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.07);
        margin-bottom: 18px;
    }

    .status-title {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .status-subtitle {
        font-size: 17px;
        margin-bottom: 6px;
    }

    .upload-box {
        border: 2px dashed #009FE3;
        background-color: #F5FBFF;
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 16px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Helperfuncties
# ------------------------------------------------------------

def get_status_visuals(status_level: str) -> dict:
    """
    Koppelt statusniveau aan kleur, label en icoon.
    """

    if status_level == "green":
        return {
            "color": "#2EAD4B",
            "label": "Normale situatie",
            "icon": "✅"
        }

    if status_level == "orange":
        return {
            "color": "#F5A623",
            "label": "Onderzoek nodig",
            "icon": "⚠️"
        }

    return {
        "color": "#D0021B",
        "label": "Defect / niet normaal",
        "icon": "🚨"
    }


def load_training_data_safely():
    """
    Laadt trainingsdata zonder de app te laten crashen.
    """

    try:
        training_index = create_training_index(
            normal_folder="data/normaal",
            abnormal_folder="data/afwijkend"
        )

        validation = validate_training_index(training_index)
        summary = summarize_training_index(training_index)

        return training_index, validation, summary, None

    except Exception as error:
        return pd.DataFrame(), None, pd.DataFrame(), error


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

logo_path = Path("assets/kuijpers_logo.png")

header_col_logo, header_col_text = st.columns([1, 5])

with header_col_logo:
    if logo_path.exists():
        st.image(str(logo_path), width=145)
    else:
        st.markdown("")

with header_col_text:
    st.markdown(
        """
        <div class="kuijpers-header">
            <div class="kuijpers-title">Kuijpers Predictive Maintenance</div>
            <div class="kuijpers-subtitle">
                Proof of concept voor geluidsgebaseerde conditiebewaking van technische installaties
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.title("Demo-instellingen")

st.sidebar.markdown("### POC scope")
st.sidebar.write("Fase A — analyse van losse WAV-bestanden")

st.sidebar.markdown("### Audio-instellingen")
st.sidebar.write(f"Sample rate: `{TARGET_SAMPLE_RATE} Hz`")
st.sidebar.write(f"Minimale lengte: `{MIN_DURATION_SECONDS} sec.`")
st.sidebar.write(f"Maximale lengte: `{MAX_DURATION_SECONDS} sec.`")

st.sidebar.markdown("### Model")
st.sidebar.write(f"Modelversie: `{MODEL_VERSION}`")
st.sidebar.write("Modeltype: `Random Forest`")

if MODEL_PATH.exists():
    st.sidebar.success("Model beschikbaar")
else:
    st.sidebar.warning("Nog geen model getraind")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Deze demo gebruikt geanonimiseerde bestandsnamen en toont geen klantnamen, assetnamen of locaties."
)


# ------------------------------------------------------------
# Intro
# ------------------------------------------------------------

st.markdown(
    """
    <div class="demo-card">
        <b>Doel van deze demo:</b><br>
        Upload een WAV-bestand van een technische installatie. De applicatie analyseert het geluid,
        vergelijkt dit met gelabelde trainingsdata en geeft een indicatie of het geluid normaal is,
        onderzoek nodig heeft of mogelijk wijst op een defect / niet normale situatie.
    </div>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------

tab_demo, tab_training, tab_data, tab_info = st.tabs(
    [
        "Demo-analyse",
        "Modeltraining",
        "Data controle",
        "POC toelichting"
    ]
)


# ------------------------------------------------------------
# TAB 1 — Demo-analyse
# ------------------------------------------------------------

with tab_demo:
    st.markdown("## Demo-analyse van nieuw geluidsfragment")

    left_col, right_col = st.columns([1.05, 1])

    with left_col:
        st.markdown(
            """
            <div class="upload-box">
                <b>Upload geluidsfragment</b><br>
                Upload een WAV-bestand. De originele bestandsnaam wordt niet getoond in de analysehistorie
                of demo-uitkomst.
            </div>
            """,
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Kies een WAV-bestand",
            type=["wav"],
            label_visibility="collapsed"
        )

        anonymized_filename = anonymize_filename()

        if uploaded_file is not None:
            st.success("WAV-bestand ontvangen")
            st.write("Geanonimiseerde bestandsnaam:")
            st.code(anonymized_filename)

            st.audio(uploaded_file, format="audio/wav")

    with right_col:
        st.markdown(
            """
            <div class="section-card">
                <h3 style="margin-top: 0;">Beoordelingslogica</h3>
                <p>
                    De app rekent de modeluitkomst om naar een risicoscore.
                    Voor deze demo gelden de volgende grenzen:
                </p>
                <ul>
                    <li><b>0 t/m 5%</b> afwijkend: normale situatie</li>
                    <li><b>&gt;5 t/m 20%</b> afwijkend: onderzoek nodig</li>
                    <li><b>&gt;20%</b> afwijkend: defect / niet normale situatie</li>
                </ul>
                <p class="small-muted">
                    De grenzen zijn bewust gevoelig ingesteld, omdat bij predictive maintenance
                    vroegtijdig signaleren belangrijker is dan wachten tot een afwijking extreem duidelijk is.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(uploaded_file.read())
            temp_audio_path = temp_audio.name

        try:
            preprocessing_result = preprocess_audio(temp_audio_path)

            st.markdown("---")
            st.markdown("## Analyse-uitkomst")

            if not preprocessing_result["is_valid"]:
                st.error("Het uploadbestand is niet geschikt voor analyse.")
                for message in preprocessing_result["messages"]:
                    st.warning(message)

            elif not MODEL_PATH.exists():
                st.warning("Er is nog geen model gevonden. Train eerst het model via de tab Modeltraining.")

            else:
                prediction = predict_audio_file(temp_audio_path)

                # Analyse opslaan in CSV-historie.
                # Let op: originele bestandsnaam wordt niet opgeslagen.
                log_analysis_result(
                    prediction=prediction,
                    filename_anonymized=anonymized_filename,
                    asset_id="Asset 001",
                    location="Locatie A"
                )

                status_visuals = get_status_visuals(prediction["status_level"])

                anomaly_score = float(prediction["anomaly_score"])
                normal_score = float(prediction.get("normal_score", 100 - anomaly_score))
                confidence_score = float(prediction["confidence_score"])

                status_col, score_col = st.columns([1.15, 1])

                with status_col:
                    st.markdown(
                        f"""
                        <div class="status-card" style="border-left: 12px solid {status_visuals["color"]};">
                            <div class="status-title">
                                {status_visuals["icon"]} {prediction["status"]}
                            </div>
                            <div class="status-subtitle">
                                <b>Classificatie:</b> {prediction["predicted_class"]}
                            </div>
                            <div class="status-subtitle">
                                <b>Advies:</b> {prediction["advice"]}
                            </div>
                            <p style="margin-top: 14px;">
                                {prediction["explanation"]}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with score_col:
                    st.markdown(
                        """
                        <div class="section-card">
                            <h3 style="margin-top: 0;">Scores</h3>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        create_score_bar_html(
                            score=normal_score,
                            label="Normaliteits-score",
                            score_type="normal"
                        ),
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        create_score_bar_html(
                            score=anomaly_score,
                            label="Risicoscore",
                            score_type="risk"
                        ),
                        unsafe_allow_html=True
                    )

                    metric_a, metric_b = st.columns(2)
                    metric_a.metric("Kans normaal", f"{prediction['probability_normal']:.2f}")
                    metric_b.metric("Kans afwijkend", f"{prediction['probability_abnormal']:.2f}")

                    st.metric("Confidence score", f"{confidence_score:.2f}")

                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("## Audiovisualisatie")

                waveform_col, details_col = st.columns([1.4, 1])

                with waveform_col:
                    fig = create_waveform_figure(
                        preprocessing_result["processed_audio"],
                        preprocessing_result["sample_rate"]
                    )
                    st.pyplot(fig, use_container_width=True)

                with details_col:
                    st.markdown(
                        """
                        <div class="section-card">
                            <h3 style="margin-top: 0;">Technische audiocontrole</h3>
                        """,
                        unsafe_allow_html=True
                    )

                    st.metric("Sample rate", f"{preprocessing_result['sample_rate']} Hz")
                    st.metric("Duur na preprocessing", f"{preprocessing_result['duration_seconds']} sec.")
                    st.metric("Modelversie", prediction["model_version"])

                    st.markdown("</div>", unsafe_allow_html=True)

                with st.expander("Berekende audiofeatures bekijken"):
                    feature_df = pd.DataFrame([prediction["features"]])
                    st.dataframe(feature_df, use_container_width=True)

        except Exception as error:
            st.error(f"Het uploadbestand kon niet worden verwerkt of voorspeld: {error}")

    st.markdown("---")
    st.markdown("## Analysehistorie")

    history_df = load_analysis_history()

    if history_df.empty:
        st.info("Er is nog geen analysehistorie beschikbaar.")
    else:
        st.write(
            "Onderstaande tabel toont de laatst uitgevoerde analyses. "
            "Alle bestandsnamen, assets en locaties zijn geanonimiseerd."
        )

        history_display_df = history_df.sort_values(
            by="timestamp",
            ascending=False
        )

        st.dataframe(
            history_display_df,
            use_container_width=True
        )

        csv_data = history_display_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download analysehistorie als CSV",
            data=csv_data,
            file_name="analyse_history.csv",
            mime="text/csv"
        )

        with st.expander("Analysehistorie wissen tijdens testen"):
            st.warning(
                "Gebruik deze knop alleen tijdens ontwikkeling of testen. "
                "De CSV wordt leeggemaakt, maar de kolommen blijven bestaan."
            )

            if st.button("Analysehistorie leegmaken"):
                clear_analysis_history()
                st.success("Analysehistorie is leeggemaakt. Vernieuw de pagina om de lege tabel te zien.")


# ------------------------------------------------------------
# TAB 2 — Modeltraining
# ------------------------------------------------------------

with tab_training:
    st.markdown("## Supervised model trainen")

    st.write(
        "Train het Random Forest-model opnieuw op basis van de gelabelde bestanden "
        "in `data/normaal` en `data/afwijkend`."
    )

    st.markdown(
        """
        <div class="demo-card">
            <b>Let op:</b> opnieuw trainen is alleen nodig als je nieuwe trainingsbestanden toevoegt
            of features/modelinstellingen hebt aangepast.
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Train Random Forest-model", type="primary"):
        with st.spinner("Model wordt getraind. Features worden opnieuw berekend..."):
            try:
                training_result = train_and_save_model()

                metrics = training_result["metrics"]
                confusion_matrix_df = training_result["confusion_matrix"]
                feature_df = training_result["feature_df"]

                st.success(f"Model succesvol getraind en opgeslagen als `{training_result['model_path']}`.")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", metrics["accuracy"])
                col2.metric("Precision", metrics["precision"])
                col3.metric("Recall", metrics["recall"])
                col4.metric("F1-score", metrics["f1_score"])

                col5, col6, col7, col8 = st.columns(4)
                col5.metric("ROC-AUC", metrics["roc_auc"] if metrics["roc_auc"] is not None else "n.v.t.")
                col6.metric("Train samples", metrics["train_samples"])
                col7.metric("Test samples", metrics["test_samples"])
                col8.metric("Feature-kolommen", len(training_result["feature_columns"]))

                st.subheader("Verdeling trainingsdata")
                data_distribution = pd.DataFrame([
                    {"klasse": "normaal", "aantal": metrics["normal_samples"]},
                    {"klasse": "afwijkend", "aantal": metrics["abnormal_samples"]}
                ])
                st.dataframe(data_distribution, use_container_width=True)

                st.subheader("Confusion matrix")
                st.dataframe(confusion_matrix_df, use_container_width=True)

                with st.expander("Voorbeeld van gebruikte feature-dataset"):
                    columns_to_show = [
                        "class_name",
                        "source_folder",
                        "duration_seconds",
                        "rms_energy_mean",
                        "spectral_centroid_mean",
                        "spectral_rolloff_mean",
                        "zero_crossing_rate_mean",
                        "label"
                    ]

                    available_columns_to_show = [
                        column for column in columns_to_show
                        if column in feature_df.columns
                    ]

                    st.dataframe(
                        feature_df[available_columns_to_show].head(30),
                        use_container_width=True
                    )

                with st.expander("Gebruikte feature-kolommen"):
                    st.write(training_result["feature_columns"])

            except Exception as error:
                st.error(f"Het model kon niet worden getraind: {error}")

    if MODEL_PATH.exists():
        st.info(f"Er staat een opgeslagen model klaar: `{MODEL_PATH}`")
    else:
        st.warning("Er is nog geen opgeslagen model gevonden.")


# ------------------------------------------------------------
# TAB 3 — Data controle
# ------------------------------------------------------------

with tab_data:
    st.markdown("## Data controle")

    training_index, validation, summary, load_error = load_training_data_safely()

    if load_error is not None:
        st.error(f"Er ging iets mis bij het inlezen van de trainingsdata: {load_error}")

    elif training_index.empty:
        st.warning("Er zijn nog geen WAV-bestanden gevonden in de trainingsmappen.")

    else:
        if validation and validation["is_valid"]:
            st.success("Trainingsdata gevonden.")
        elif validation:
            st.warning("Er ontbreekt trainingsdata.")
            for message in validation["messages"]:
                st.warning(message)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Aantal bestanden per klasse")
            st.dataframe(summary, use_container_width=True)

        with col2:
            st.subheader("Voorbeeld gevonden bestanden")
            preview_df = training_index.copy()
            preview_df["file_path"] = preview_df["file_path"].apply(
                lambda x: str(Path(x).as_posix())
            )
            st.dataframe(preview_df.head(30), use_container_width=True)

        st.markdown("---")
        st.subheader("Preprocessing controleren")

        if st.button("Controleer preprocessing van trainingsbestanden"):
            with st.spinner("Trainingsbestanden worden gecontroleerd..."):
                preprocessing_records = preprocess_training_files(training_index)
                preprocessing_df = pd.DataFrame(preprocessing_records)

            valid_count = int(preprocessing_df["is_valid"].sum())
            total_count = len(preprocessing_df)
            invalid_count = total_count - valid_count

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Totaal bestanden", total_count)
            col_b.metric("Bruikbaar", valid_count)
            col_c.metric("Niet bruikbaar", invalid_count)

            if invalid_count == 0:
                st.success("Alle gecontroleerde bestanden zijn bruikbaar.")
            else:
                st.warning("Er zijn bestanden gevonden die niet bruikbaar zijn.")

            st.dataframe(preprocessing_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Feature extraction controleren")

        if st.button("Maak features van trainingsbestanden"):
            with st.spinner("Features worden gemaakt..."):
                feature_df = create_feature_dataset(training_index)

            valid_feature_df = feature_df[feature_df["is_valid"] == True].copy()
            invalid_feature_df = feature_df[feature_df["is_valid"] == False].copy()
            feature_columns = get_model_feature_columns(valid_feature_df)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Totaal bestanden", len(feature_df))
            col_b.metric("Features gelukt", len(valid_feature_df))
            col_c.metric("Aantal feature-kolommen", len(feature_columns))

            if len(invalid_feature_df) == 0:
                st.success("Voor alle bestanden zijn features gemaakt.")
            else:
                st.warning("Voor sommige bestanden konden geen features worden gemaakt.")
                st.dataframe(invalid_feature_df, use_container_width=True)

            columns_to_show = [
                "class_name",
                "source_folder",
                "duration_seconds",
                "rms_energy_mean",
                "spectral_centroid_mean",
                "spectral_rolloff_mean",
                "zero_crossing_rate_mean",
                "band_energy_0_500",
                "band_energy_500_2000",
                "band_energy_2000_5000",
                "band_energy_5000_8000",
                "label"
            ]

            available_columns_to_show = [
                column for column in columns_to_show
                if column in valid_feature_df.columns
            ]

            st.subheader("Voorbeeld feature-dataset")
            st.dataframe(
                valid_feature_df[available_columns_to_show].head(30),
                use_container_width=True
            )


# ------------------------------------------------------------
# TAB 4 — POC toelichting
# ------------------------------------------------------------

with tab_info:
    st.markdown("## POC toelichting")

    st.markdown(
        """
        ### Wat doet deze demo?

        Deze applicatie analyseert losse WAV-bestanden van technische installaties.
        Het model is vooraf getraind met gelabelde voorbeelden van normale en afwijkende geluiden.

        De demo voert deze stappen uit:

        1. WAV-bestand uploaden;
        2. audio preprocessing;
        3. feature extraction;
        4. voorspelling met supervised machine learning;
        5. vertaling naar normaliteits-score, risicoscore en advies;
        6. opslag van de analyse in een geanonimiseerde CSV-historie.

        ### Wat betekent de score?

        De risicoscore is gebaseerd op de kans dat het geluidsfragment afwijkt van normaal gedrag.

        - **0 t/m 5% afwijkend:** normale situatie;
        - **>5 t/m 20% afwijkend:** onderzoek nodig;
        - **>20% afwijkend:** defect / niet normale situatie.

        ### Analysehistorie

        Iedere succesvolle analyse wordt opgeslagen in:

        `history/analyse_history.csv`

        Daarbij worden alleen generieke waarden opgeslagen, zoals:

        - Asset 001;
        - Locatie A;
        - meting_YYYYMMDD_HHMMSS.wav.

        De originele bestandsnaam wordt niet opgeslagen.

        ### Wat zit bewust nog niet in fase A?

        Deze POC bevat nog geen:

        - live streaming;
        - sensorintegratie;
        - automatische werkorderkoppeling;
        - exacte defectclassificatie;
        - self-learning model in productie;
        - klant- of assetherkenning.

        ### Waarom deze aanpak?

        Deze fase toont eerst aan dat geluidsfragmenten technisch kunnen worden verwerkt en dat een supervised model
        verschil kan leren tussen normale en afwijkende geluiden.

        Daarmee vormt fase A een laagdrempelige opstap naar fase B en C.
        """
    )