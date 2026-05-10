# Kuijpers Predictive Maintenance Demo

Deze applicatie is een lokale Streamlit-demo voor geluidsgebaseerde predictive maintenance.

De demo richt zich op **fase A: analyse van losse WAV-bestanden / proof of concept**.

De applicatie analyseert een WAV-bestand en voorspelt of het geluid past bij:

- een normale situatie;
- een situatie waarbij onderzoek nodig is;
- een defect / niet normale situatie.

De demo gebruikt een supervised machine-learningmodel dat vooraf wordt getraind op gelabelde WAV-bestanden.

---

## 1. Doel van de applicatie

De applicatie toont aan dat het mogelijk is om met geluidsanalyse technische installaties te beoordelen op basis van losse WAV-bestanden.

De applicatie kan:

1. gelabelde WAV-bestanden verwerken;
2. audio preprocessen;
3. audiofeatures extraheren;
4. een supervised Random Forest-model trainen;
5. een nieuw WAV-bestand classificeren;
6. een normaliteits-score en risicoscore tonen;
7. een waveformgrafiek tonen;
8. analysehistorie geanonimiseerd opslaan;
9. lokaal draaien via Streamlit.

---

## 2. Scope van deze demo

Deze demo hoort bij fase A.

De applicatie bevat wel:

- upload van losse WAV-bestanden;
- preprocessing;
- feature extraction;
- supervised learning;
- voorspelling;
- visuele score;
- waveformgrafiek;
- geanonimiseerde analysehistorie.

De applicatie bevat nog niet:

- live streaming;
- sensorintegratie;
- automatische werkorderkoppeling;
- exacte defectclassificatie;
- self-learning gedrag in productie;
- klant- of assetherkenning.

---

## 3. Projectstructuur

De verwachte projectstructuur is:

```text
kuijpers_predictive_maintenance/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── normaal/
│   ├── afwijkend/
│   └── test/
├── models/
│   └── model.pkl
├── history/
│   └── analyse_history.csv
├── src/
│   ├── __init__.py
│   ├── audio_loader.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train_model.py
│   ├── predict.py
│   ├── history_logger.py
│   └── visualization.py
└── assets/
    └── kuijpers_logo.png