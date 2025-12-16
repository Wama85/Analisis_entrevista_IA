from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

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


def run(video_path: str, out_dir: str, whisper_model: str, language: str) -> Dict[str, Any]:
    logger.info("Iniciando ejecución del pipeline Grupo B")

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Directorio de salida creado/verificado: {outp.resolve()}")

    wav_path = str(outp / "audio.wav")
    json_path = str(outp / "group_b_output.json")

    # 1) Extract audio
    logger.info("Extrayendo audio desde el video")
    logger.debug(f"Video de entrada: {video_path}")
    extract_audio(video_path, wav_path, sr=16000)
    logger.info(f"Audio extraído correctamente: {wav_path}")

    # 2) Transcribe
    logger.info("Iniciando transcripción de audio")
    logger.debug(f"Modelo Whisper: {whisper_model} | Idioma: {language}")
    tr = transcribe_audio(wav_path, model_size=whisper_model, language=language)
    text = tr["text"]
    logger.info("Transcripción completada")
    logger.debug(f"Texto transcrito (longitud): {len(text)} caracteres")

    # 3) Emotion from text
    logger.info("Iniciando análisis emocional del texto")
    logger.warning(
        "El modelo de emociones utilizado está entrenado en inglés. "
        "Para mejores resultados en español se recomienda un modelo multilingüe."
    )

    clf = build_emotion_classifier()
    text_emotions = predict_text_emotions(clf, text)
    logger.info("Análisis emocional finalizado")
    logger.debug(f"Emociones detectadas: {text_emotions}")

    output = {
        "input": {
            "video_path": video_path,
            "audio_wav": wav_path,
        },
        "transcription": {
            "language": tr.get("language"),
            "text": text,
            "segments": tr.get("segments", []),
        },
        "text_emotion": text_emotions,
    }

    logger.info("Guardando resultados en archivo JSON")
    logger.debug(f"Ruta del archivo JSON: {json_path}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("Ejecución del pipeline Grupo B finalizada correctamente")
    return output


def main():
    parser = argparse.ArgumentParser(description="Grupo B: Audio -> Whisper -> Text Emotion")
    parser.add_argument("--video", required=True, help="Path to input interview video (.mp4/.avi)")
    parser.add_argument("--out", default="outputs/group_b", help="Output directory")
    parser.add_argument(
        "--whisper_model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size"
    )
    parser.add_argument("--lang", default="es", help="Language code (e.g., es, en). Use 'es' for Spanish.")
    args = parser.parse_args()

    logger.info("Argumentos recibidos desde línea de comandos")
    logger.debug(vars(args))

    result = run(args.video, args.out, args.whisper_model, args.lang)

    # === PRINTS ORIGINALES (NO SE TOCAN) ===
    print("\n Grupo B ejecutado correctamente")
    print(f"Texto (primeros 200 chars): {result['transcription']['text'][:200]!r}")
    top3 = sorted(result["text_emotion"].items(), key=lambda x: x[1], reverse=True)[:3]
    print("Top-3 emociones (texto):", top3)


if __name__ == "__main__":
    main()
