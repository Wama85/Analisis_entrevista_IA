import cv2
from deepface import DeepFace
import os
import logging
from typing import List, Dict, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extraer_frames(ruta_video: str, ruta_salida_frames: str) -> bool:
    """
    Extrae frames de un video y los guarda como archivos JPG.
    Esto genera la 'serie temporal de frames' para el análisis posterior.
    """
    logging.info(f"Iniciando extracción de frames desde: {ruta_video}")

    if not os.path.exists(ruta_salida_frames):
        os.makedirs(ruta_salida_frames)
        logging.info(f"Directorio creado para frames: {ruta_salida_frames}")

    cap = cv2.VideoCapture(ruta_video)

    if not cap.isOpened():
        logging.error(f"Error: No se pudo abrir el archivo de video en {ruta_video}. Verifique la ruta y el nombre.")
        return False

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Guardar el frame con un nombre secuencial de 5 dígitos (ej. frame_00000.jpg)
        nombre_frame = os.path.join(ruta_salida_frames, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(nombre_frame, frame)
        frame_count += 1

        if frame_count % 100 == 0:
            logging.info(f"Frames extraídos: {frame_count}")

    cap.release()
    logging.info(f"Extracción finalizada. Total de frames extraídos: {frame_count}")
    return True

def probar_deteccion_facial(ruta_imagen: str) -> List[Dict[str, Any]]:
    """
    Prueba la detección y análisis de emociones usando DeepFace en una imagen (el primer frame).
    """
    if not os.path.exists(ruta_imagen):
        logging.error(f"Error: La imagen de prueba no existe en {ruta_imagen}")
        return []

    logging.info(f"Iniciando prueba de detección facial con DeepFace en: {ruta_imagen}")

    try:
        resultados = DeepFace.analyze(img_path = ruta_imagen,
                                      actions = ['emotion', 'age', 'gender'],
                                      enforce_detection=True)

        logging.info("--- ✅ Prueba de Detección Facial Exitosa ---")
        for i, res in enumerate(resultados):
            logging.info(f"Rostro {i+1} detectado:")
            logging.info(f"  Emoción predominante: {res['dominant_emotion']}")
            logging.info(f"  Probabilidades de emociones: {res['emotion']}")

        return resultados

    except ValueError as e:
        if "Face could not be detected" in str(e):
            logging.warning(f"No se detectó ningún rostro en {ruta_imagen}. Asegúrese que el rostro sea visible y de frente.")
        else:
            logging.error(f"Error durante el análisis: {e}")
        return []
    except Exception as e:
        logging.error(f"Ocurrió un error inesperado con DeepFace: {e}")
        return []


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent
    VIDEO_DE_ENTRADA = BASE_DIR / "data" / "videos" / "mivideo.mp4"
    CARPETA_DE_FRAMES = "output_frames_dia1"

    logging.info("TAREA 1: Iniciando Módulo de Extracción de Frames (Grupo A)")
    if extraer_frames(VIDEO_DE_ENTRADA, CARPETA_DE_FRAMES):
        logging.info("Módulo de Extracción de Frames completado con éxito.")

        primer_frame = os.path.join(CARPETA_DE_FRAMES, "frame_00000.jpg")

        if os.path.exists(primer_frame):
            logging.info("\nTAREA 2: Iniciando Módulo de Prueba de Detección Facial (Grupo A)")
            probar_deteccion_facial(primer_frame)
        else:
            logging.warning("No se pudo encontrar el primer frame para la prueba de DeepFace.")

    else:
        logging.error("FALLA CRÍTICA: No se pudo extraer el video. Finalizando Día 1 para Grupo A.")