import matplotlib.pyplot as plt
from collections import defaultdict


def plot_emotions_over_time(segments_with_emotions):
    time_points = []
    emotion_series = defaultdict(list)

    for seg in segments_with_emotions:
        mid_time = (seg["start"] + seg["end"]) / 2
        time_points.append(mid_time)

        for emo, score in seg["emotions"].items():
            emotion_series[emo].append(score)

    plt.figure()
    for emo, scores in emotion_series.items():
        plt.plot(time_points[:len(scores)], scores, label=emo)

    plt.xlabel("Tiempo (segundos)")
    plt.ylabel("Probabilidad")
    plt.title("Evolución de emociones en el tiempo")
    plt.legend()
    plt.grid(True)
    plt.show()
