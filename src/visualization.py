import numpy as np
import matplotlib.pyplot as plt


def create_waveform_figure(audio: np.ndarray, sample_rate: int):
    """
    Maakt een waveformgrafiek van het audiosignaal.

    Input:
    - audio: numpy-array met audiosignaal
    - sample_rate: sample rate van de audio

    Output:
    - matplotlib figure voor Streamlit
    """

    duration = len(audio) / sample_rate
    time_axis = np.linspace(0, duration, num=len(audio))

    fig, ax = plt.subplots(figsize=(12, 3.5))

    ax.plot(time_axis, audio, linewidth=0.8)
    ax.set_title("Waveform van het geluidsfragment")
    ax.set_xlabel("Tijd in seconden")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig


def create_score_bar_html(score: float, label: str, score_type: str = "risk") -> str:
    """
    Maakt een eenvoudige HTML-scorebalk.

    score_type:
    - risk: hoge score is risicovoller
    - normal: hoge score is beter
    """

    score = max(0, min(100, score))

    if score_type == "normal":
        if score >= 95:
            color = "#2EAD4B"
        elif score >= 80:
            color = "#F5A623"
        else:
            color = "#D0021B"
    else:
        if score <= 5:
            color = "#2EAD4B"
        elif score <= 20:
            color = "#F5A623"
        else:
            color = "#D0021B"

    html = f"""
    <div style="margin-bottom: 18px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="font-weight: 600;">{label}</span>
            <span style="font-weight: 600;">{score:.1f}/100</span>
        </div>
        <div style="
            width: 100%;
            background-color: #E6EAF0;
            border-radius: 999px;
            height: 16px;
            overflow: hidden;
        ">
            <div style="
                width: {score}%;
                background-color: {color};
                height: 16px;
                border-radius: 999px;
            "></div>
        </div>
    </div>
    """

    return html