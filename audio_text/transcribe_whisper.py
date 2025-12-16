from __future__ import annotations

from typing import Any, Dict, Optional
import logging
import whisper  # openai-whisper


# ===============================
# CONFIGURACIÓN DE LOGGING
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def transcribe_audio(
        wav_path: str,
        model_size: str = "small",
        language: Optional[str] = "es",
) -> Dict[str, Any]:
    """
    Transcribe audio using Whisper.

    Returns dict with:
      - text (full transcription)
      - segments (list with start/end/text)
      - language
    """

    # LOGGING (nuevo, no reemplaza prints)
    logger.info("Iniciando transcripción de audio con Whisper")
    logger.debug(f"Archivo de audio: {wav_path}")
    logger.debug(f"Modelo Whisper: {model_size}")
    logger.debug(f"Idioma configurado: {language}")

    # === CÓDIGO ORIGINAL (NO SE TOCA) ===
    model = whisper.load_model(model_size)

    # fp16 False helps on CPU / some GPUs
    result = model.transcribe(
        wav_path,
        language=language,
        fp16=False,
        verbose=False,
    )

    # LOGGING adicional
    logger.info("Transcripción finalizada correctamente")
    logger.debug(f"Idioma detectado: {result.get('language')}")
    logger.debug(f"Cantidad de segmentos: {len(result.get('segments', []))}")

    return {
        "text": (result.get("text") or "").strip(),
        "segments": result.get("segments", []),
        "language": result.get("language", language),
    }
