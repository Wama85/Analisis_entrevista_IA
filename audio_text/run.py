from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict
from collections import defaultdict

import matplotlib.pyplot as plt

from audio_text.extract_audio import extract_audio
from audio_text.transcribe_whisper import transcribe_audio
from audio_text.text_emotion import build_emotion_classifier, predict_text_emotions


# ===============================
# CONFIGURACIÓN DE LOGGING
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def plot_emotions_over_time(segments_with_emotions):
    """
    Gráfica de emociones vs tiempo (usa el punto medio de cada segmento).
    """
    time_points = []
    emotion_series = defaultdict(list)

    for seg in segments_with_emotions:
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", 0.0) or 0.0)
        mid_time = (start + end) / 2.0
        time_points.append(mid_time)

        emotions = seg.get("emotions", {}) or {}
        for emo, score in emotions.items():
            emotion_series[emo].append(score)

    plt.figure()
    for emo, scores in emotion_series.items():
        plt.plot(time_points[:len(scores)], scores, label=emo)

    plt.xlabel("Tiempo (segundos)")
    plt.ylabel("Probabilidad")
    plt.title("Evolución temporal de emociones")
    plt.legend()
    plt.grid(True)
    plt.show()


def run(
        video_path: str,
        out_dir: str,
        whisper_model: str,
        language: str,
        make_plot: bool = True
) -> Dict[str, Any]:
    logger.info("Iniciando la ejecución del pipeline Grupo B")

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    wav_path = str(outp / "audio.wav")

    # ✅ JSON distinto por cada video
    video_name = Path(video_path).stem
    json_path = str(outp / f"{video_name}_group_b.json")

    # 1) Extract audio
    logger.info("Aquí se esta extrayendo el audio desde el video")
    logger.debug(f"Este es el video de entrada: {video_path}")
    extract_audio(video_path, wav_path, sr=16000)
    logger.info(f"Audio extraído de manera correcta: {wav_path}")

    # 2) Transcribe
    logger.info("Iniciando la transcripción del audio")
    logger.debug(f"Modelo Whisper: {whisper_model} | Idioma: {language}")
    tr = transcribe_audio(wav_path, model_size=whisper_model, language=language)
    text_full = tr.get("text", "")
    segments = tr.get("segments", [])
    logger.info("Transcripción completada")
    logger.debug(f"Texto transcrito (longitud): {len(text_full)} caracteres")
    logger.info(f"Segmentos detectados: {len(segments)}")

    # 3) Emotion from text (por segmento con timestamps)
    logger.info("Iniciando la parte de análisis emocional del texto (por segmento)")
    logger.warning(
        "El modelo de emociones utilizado está entrenado en inglés. "
        "Para mejores resultados en español se recomienda un modelo multilingüe."
    )

    clf = build_emotion_classifier()

    segments_with_emotions = []
    for seg in segments:
        start = seg.get("start")
        end = seg.get("end")
        seg_text = (seg.get("text") or "").strip()

        emotions = predict_text_emotions(clf, seg_text)

        segments_with_emotions.append({
            "start": start,
            "end": end,
            "text": seg_text,
            "emotions": emotions,
        })

    # ✅ SUMMARY GLOBAL (extra)
    emotion_accumulator = defaultdict(float)
    segment_count = 0
    duration_seconds = 0.0

    for seg in segments_with_emotions:
        if seg.get("end") is not None:
            duration_seconds = max(duration_seconds, float(seg["end"]))

        for emo, score in (seg.get("emotions") or {}).items():
            emotion_accumulator[emo] += float(score)
        segment_count += 1

    emotion_distribution = {
        emo: round(score / segment_count, 4)
        for emo, score in emotion_accumulator.items()
    } if segment_count > 0 else {}

    dominant_emotion = (
        max(emotion_distribution.items(), key=lambda x: x[1])[0]
        if emotion_distribution else None
    )

    # ✅ JSON con estructura (incluye timestamps en cada segmento)
    output = {
        "video": {
            "path": video_path,
            "audio_wav": wav_path,
        },
        "transcription": {
            "language": tr.get("language"),
            "text": text_full,  # 👈 mantenemos también el texto completo
            "segments": segments_with_emotions,  # 👈 con timestamps + emociones
        },
        "summary": {
            "duration_seconds": round(duration_seconds, 2),
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": emotion_distribution,
        }
    }

    logger.info("Guardando resultados en archivo JSON")
    logger.debug(f"Ruta del archivo JSON: {json_path}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("Ejecución del pipeline Grupo B finalizada correctamente")

    # ✅ Gráfica (extra)
    if make_plot:
        plot_emotions_over_time(segments_with_emotions)

    return output


def main():
    parser = argparse.ArgumentParser(description="Grupo B: Audio -> Whisper -> Text Emotion")
    parser.add_argument("--video", required=True, help="Path to input interview video (.mp4/.avi)")
    parser.add_argument("--out", default="outputs/audio_text", help="Output directory")
    parser.add_argument(
        "--whisper_model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size"
    )
    parser.add_argument("--lang", default="es", help="Language code (e.g., es, en). Use 'es' for Spanish.")
    parser.add_argument("--no_plot", action="store_true", help="No mostrar gráfica (solo JSON y consola)")
    args = parser.parse_args()

    logger.info("Argumentos recibidos desde línea de comandos")
    logger.debug(vars(args))

    result = run(args.video, args.out, args.whisper_model, args.lang, make_plot=(not args.no_plot))

    # ✅ PRINTS ORIGINALES (NO SE TOCAN)
    print("\n Ejecutado correctamente")
    print(f"Texto (primeros 200 chars): {result['transcription']['text'][:200]!r}")

    # Top-3 emociones globales (del summary) — extra útil
    dist = result.get("summary", {}).get("emotion_distribution", {}) or {}
    top3 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
    print("Top-3 emociones (global):", top3)


if __name__ == "__main__":
    main()
