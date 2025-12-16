from __future__ import annotations
from typing import Dict
from transformers import pipeline
import logging


# ===============================
# CONFIGURACIÓN DE LOGGING
# ===============================
logging.basicConfig(
    level=logging.INFO,  # Cambia a DEBUG si quieres más detalle
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def build_emotion_classifier(
        model_name: str = "j-hartmann/emotion-english-distilroberta-base"
):
    """
    Construye un clasificador de emociones para texto.
    NOTA: El modelo está entrenado en inglés.
    """

    logger.info("Iniciando construcción del clasificador de emociones")
    logger.info(f"Modelo seleccionado: {model_name}")

    classifier = pipeline(
        task="text-classification",
        model=model_name,
        top_k=None,
        truncation=True,
    )

    logger.info("Clasificador de emociones creado correctamente")
    return classifier


def predict_text_emotions(classifier, text: str) -> Dict[str, float]:
    """
    Predice emociones a partir de un texto.
    Retorna un diccionario {emoción: probabilidad}.
    """

    logger.info("Iniciando análisis emocional del texto")

    if not text or not text.strip():
        logger.warning("El texto recibido está vacío. Se omite el análisis")
        return {}

    logger.debug(f"Texto recibido: {text}")

    logger.info("Ejecutando el modelo de emociones")
    output = classifier(text)

    logger.debug(f"Salida cruda del modelo: {output}")

    # Normalización de la salida
    if isinstance(output, list) and len(output) > 0 and isinstance(output[0], list):
        items = output[0]
        logger.debug("Formato de salida: lista anidada")
    elif isinstance(output, list):
        items = output
        logger.debug("Formato de salida: lista simple")
    else:
        logger.error("Formato de salida del modelo no reconocido")
        return {}

    emotions: Dict[str, float] = {}

    logger.info("Procesando resultados de emociones")
    for item in items:
        label = str(item.get("label"))
        score = float(item.get("score", 0.0))
        emotions[label] = score
        logger.debug(f"Emoción detectada: {label} | Score: {score:.4f}")

    logger.info("Análisis emocional finalizado correctamente")
    return emotions
