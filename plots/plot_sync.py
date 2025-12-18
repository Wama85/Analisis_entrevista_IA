# plots/plot_sync.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def _build_label_map(values: List[str]) -> Dict[str, int]:
    # Orden estable para que siempre se vean igual las etiquetas
    labels = sorted({v for v in values if v is not None})
    return {lab: i for i, lab in enumerate(labels)}


def plot_synchronized_emotions(sync_json_path: str) -> None:
    """
    Lee el JSON sincronizado y muestra una gráfica:
      - Emoción facial vs tiempo
      - Emoción del texto vs tiempo

    Espera una clave 'synchronized_timeline' con elementos que contengan:
      - time_sec
      - facial_emotion
      - text_emotion
    """
    p = Path(sync_json_path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    timeline = data.get("synchronized_timeline")
    if not isinstance(timeline, list):
        raise KeyError(
            f"El JSON no contiene 'synchronized_timeline'. Claves encontradas: {list(data.keys())}"
        )

    # Extraer series
    times: List[float] = []
    facial: List[str] = []
    text: List[str] = []

    for item in timeline:
        t = item.get("time_sec")
        fe = item.get("facial_emotion")
        te = item.get("text_emotion")

        if t is None:
            continue

        times.append(float(t))
        facial.append(str(fe) if fe is not None else "None")
        text.append(str(te) if te is not None else "None")

    # Mapear emociones a números para graficar en eje Y
    label_map = _build_label_map(facial + text)

    facial_y = [label_map.get(e, -1) for e in facial]
    text_y = [label_map.get(e, -1) for e in text]

    plt.figure()
    plt.plot(times, facial_y, label="Emocion facial")
    plt.plot(times, text_y, label="Emocion del texto")
    plt.yticks(list(label_map.values()), list(label_map.keys()))
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Emocion")
    plt.title("Evolucion de emociones (facial vs texto)")
    plt.legend()
    plt.grid(True)
    plt.show()
