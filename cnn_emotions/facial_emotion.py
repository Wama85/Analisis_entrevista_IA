"""
Análisis de Emociones Frame por Frame - Anotación Directa
==========================================================
Extrae frames de videos y MIENTRAS los extrae, analiza las emociones
y dibuja los recuadros directamente en los frames.

FLUJO:
1. Extraer frame del video
2. Analizar emoción en ese frame
3. Dibujar recuadro y emoción
4. Guardar frame YA ANOTADO
5. Continuar con siguiente frame

Autor: Sistema de Reconocimiento Facial
Fecha: 16 de diciembre, 2025
"""

import cv2
from deepface import DeepFace
import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import numpy as np

def convertir_a_tipo_nativo(obj):
    """
    Convierte tipos NumPy a tipos nativos de Python para JSON.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convertir_a_tipo_nativo(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convertir_a_tipo_nativo(item) for item in obj]
    else:
        return obj

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class AnalizadorEmociones:
    """
    Clase para extraer frames con emociones anotadas directamente
    """

    def __init__(self, output_dir='resultados_emociones'):
        """
        Inicializa el analizador de emociones

        Args:
            output_dir: Directorio para guardar frames y reportes
        """
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / 'frames'
        self.reports_dir = self.output_dir / 'reports'

        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            'frames_totales': 0,
            'frames_con_rostros': 0,
            'frames_sin_rostros': 0,
            'rostros_totales': 0,
            'errores': []
        }

        self.fps_analisis = 0.0

        self.emociones_contador = {
            'angry': 0,
            'disgust': 0,
            'fear': 0,
            'happy': 0,
            'sad': 0,
            'surprise': 0,
            'neutral': 0
        }

        self.colores_emociones = {
            'angry': (0, 0, 255),      
            'disgust': (0, 128, 128),  
            'fear': (128, 0, 128),     
            'happy': (0, 255, 0),      
            'sad': (255, 0, 0),        
            'surprise': (0, 255, 255), 
            'neutral': (128, 128, 128) 
        }

        self.todos_los_analisis = []

    def dibujar_anotaciones(self, frame, rostros_info: List[Dict[str, Any]]) -> np.ndarray:
        """
        Dibuja recuadros y emociones en el frame
        ...
        """
        frame_anotado = frame.copy()

        for rostro in rostros_info:
            region = rostro.get('region', {})
            if not region:
                continue

            x = int(region.get('x', 0))
            y = int(region.get('y', 0))
            w = int(region.get('w', 0))
            h = int(region.get('h', 0))

            emocion = rostro.get('emocion_dominante', 'unknown')
            edad = rostro.get('edad', 0)
            genero = rostro.get('genero', 'unknown')

            emociones = rostro.get('emociones', {})
            probabilidad = emociones.get(emocion, 0)

            color = self.colores_emociones.get(emocion, (255, 255, 255))

            cv2.rectangle(frame_anotado, (x, y), (x+w, y+h), color, 3)

            texto_emocion = f"{emocion.upper()}"
            texto_prob = f"{probabilidad:.1f}%"
            texto_info = f"{genero}, {edad}a"

            (tw1, th1), _ = cv2.getTextSize(texto_emocion, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            (tw2, th2), _ = cv2.getTextSize(texto_prob, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            (tw3, th3), _ = cv2.getTextSize(texto_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            padding = 5
            cv2.rectangle(frame_anotado,
                          (x, y - th1 - th2 - th3 - padding*3),
                          (x + max(tw1, tw2, tw3) + padding*2, y),
                          (0, 0, 0), -1)

            cv2.putText(frame_anotado, texto_emocion,
                        (x + padding, y - th2 - th3 - padding*2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.putText(frame_anotado, texto_prob,
                        (x + padding, y - th3 - padding),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(frame_anotado, texto_info,
                        (x + padding, y - padding),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            centro_x = x + w // 2
            centro_y = y + h // 2
            cv2.circle(frame_anotado, (centro_x, centro_y), 3, color, -1)

        return frame_anotado

    def analizar_y_anotar_frame(self, frame, frame_count: int) -> Dict[str, Any]:
        """
        Analiza un frame y devuelve la información + frame anotado
        ...
        """
        timestamp_seconds = 0.0
        if self.fps_analisis > 0:
            timestamp_seconds = frame_count / self.fps_analisis 
            
        resultado = {
            'frame_name': f"frame_{frame_count:05d}.jpg",
            'frame_number': frame_count,
            'timestamp_seconds': round(timestamp_seconds, 3), 
            'num_rostros': 0,
            'rostros': [],
            'error': None
        }

        try:
            temp_path = self.frames_dir / f"temp_{frame_count}.jpg"
            cv2.imwrite(str(temp_path), frame)

            analisis = DeepFace.analyze(
                img_path=str(temp_path),
                actions=['emotion', 'age', 'gender'],
                enforce_detection=False,
                silent=True
            )

            temp_path.unlink()

            if isinstance(analisis, list):
                resultados_lista = analisis
            else:
                resultados_lista = [analisis]

            resultado['num_rostros'] = len(resultados_lista)

            for i, rostro in enumerate(resultados_lista):
                info_rostro = {
                    'rostro_id': i,
                    'emocion_dominante': str(rostro.get('dominant_emotion', 'unknown')),
                    'emociones': convertir_a_tipo_nativo(rostro.get('emotion', {})),
                    'edad': int(rostro.get('age', 0)),
                    'genero': str(rostro.get('dominant_gender', 'unknown')),
                    'confianza_genero': convertir_a_tipo_nativo(rostro.get('gender', {})),
                    'region': convertir_a_tipo_nativo(rostro.get('region', {}))
                }

                resultado['rostros'].append(info_rostro)

                emocion = info_rostro['emocion_dominante']
                if emocion in self.emociones_contador:
                    self.emociones_contador[emocion] += 1

            if resultado['num_rostros'] > 0:
                self.stats['frames_con_rostros'] += 1
                self.stats['rostros_totales'] += resultado['num_rostros']
            else:
                self.stats['frames_sin_rostros'] += 1

        except Exception as e:
            resultado['error'] = str(e)
            self.stats['errores'].append(f"Error en frame {frame_count}: {str(e)}")
            self.stats['frames_sin_rostros'] += 1

        return resultado

    def extraer_frames_con_emociones(self, ruta_video: str, fps_extraccion: Optional[int] = None) -> bool:
        """
        Extrae frames del video y DIRECTAMENTE los analiza y anota con emociones
        ...
        """
        logging.info(f"Iniciando extracción y análisis de frames desde: {ruta_video}")

        cap = cv2.VideoCapture(str(ruta_video))

        if not cap.isOpened():
            logging.error(f"Error: No se pudo abrir el video en {ruta_video}")
            return False

        fps_original = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duracion = total_frames / fps_original if fps_original > 0 else 0

        logging.info(f"Video - FPS: {fps_original:.2f}, Frames: {total_frames}, Duración: {duracion:.2f}s")

        if fps_extraccion is not None:
            frame_interval = int(fps_original / fps_extraccion)
            if frame_interval < 1:
                frame_interval = 1
            self.fps_analisis = fps_original / frame_interval
            logging.info(f"Extrayendo 1 frame cada {frame_interval} frames ({self.fps_analisis:.2f} fps)")
        else:
            frame_interval = 1
            self.fps_analisis = fps_original
            logging.info("Extrayendo todos los frames")

        logging.info("✨ Analizando y anotando emociones en tiempo real...")

        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_count % frame_interval == 0:
                analisis = self.analizar_y_anotar_frame(frame, extracted_count)

                if analisis['num_rostros'] > 0:
                    frame = self.dibujar_anotaciones(frame, analisis['rostros'])

                nombre_frame = self.frames_dir / f"frame_{extracted_count:05d}.jpg"
                cv2.imwrite(str(nombre_frame), frame)

                analisis['frame_path'] = str(nombre_frame)
                self.todos_los_analisis.append(analisis)

                extracted_count += 1

                if extracted_count % 10 == 0:
                    logging.info(f"  Procesados: {extracted_count} frames ({self.stats['frames_con_rostros']} con rostros)")

            frame_count += 1

        cap.release()
        self.stats['frames_totales'] = extracted_count

        logging.info(f"\n✓ Extracción y análisis completados")
        logging.info(f"  Total frames: {extracted_count}")
        logging.info(f"  Con rostros: {self.stats['frames_con_rostros']}")
        logging.info(f"  Sin rostros: {self.stats['frames_sin_rostros']}")

        return True

    def generar_reporte_json(self, nombre_archivo: str = 'emotion_analysis_report.json'):
        """
        Genera un reporte JSON completo con todos los resultados
        ...
        """
        if self.stats['frames_con_rostros'] > 0:
            promedio_rostros = self.stats['rostros_totales'] / self.stats['frames_con_rostros']
        else:
            promedio_rostros = 0

        if sum(self.emociones_contador.values()) > 0:
            emocion_mas_comun = max(self.emociones_contador.items(),
                                    key=lambda x: x[1])
        else:
            emocion_mas_comun = ('unknown', 0)

        reporte = {
            'fecha_analisis': datetime.now().isoformat(),
            'resumen': {
                'frames_totales': int(self.stats['frames_totales']),

                'frames_con_rostros': int(self.stats['frames_con_rostros']),
                'frames_sin_rostros': int(self.stats['frames_sin_rostros']),
                'rostros_totales_detectados': int(self.stats['rostros_totales']),
                'promedio_rostros_por_frame': float(round(promedio_rostros, 2)),
                'emocion_mas_comun': str(emocion_mas_comun[0]),
                'frecuencia_emocion_mas_comun': int(emocion_mas_comun[1])
            },
            'distribucion_emociones': {k: int(v) for k, v in self.emociones_contador.items()},
            'porcentaje_emociones': {k: float(v) for k, v in self._calcular_porcentajes_emociones().items()},
            'analisis_por_frame': self.todos_los_analisis,
            'estadisticas': {
                'total_errores': int(len(self.stats['errores'])),
                'errores': self.stats['errores'][:10]
            },
            'directorios': {
                'frames': str(self.frames_dir),
                'reportes': str(self.reports_dir)
            }
        }

        reporte = convertir_a_tipo_nativo(reporte)

        ruta_reporte = self.reports_dir / nombre_archivo
        with open(ruta_reporte, 'w', encoding='utf-8') as f:
            json.dump(reporte, f, indent=2, ensure_ascii=False)

        logging.info(f"\n✓ Reporte guardado en: {ruta_reporte}")
        return ruta_reporte

    def _calcular_porcentajes_emociones(self) -> Dict[str, float]:
        """Calcula el porcentaje de cada emoción"""
        total = sum(self.emociones_contador.values())

        if total == 0:
            return {emocion: 0.0 for emocion in self.emociones_contador}

        return {
            emocion: round((count / total) * 100, 2)
            for emocion, count in self.emociones_contador.items()
        }

    def imprimir_resumen(self):
        """Imprime un resumen visual del análisis"""
        logging.info("\n" + "="*60)
        logging.info("RESUMEN DEL ANÁLISIS DE EMOCIONES")
        logging.info("="*60)

        logging.info(f"\n📊 ESTADÍSTICAS GENERALES:")
        logging.info(f"  Frames totales: {self.stats['frames_totales']}")
        logging.info(f"  Frames con rostros: {self.stats['frames_con_rostros']}")
        logging.info(f"  Frames sin rostros: {self.stats['frames_sin_rostros']}")
        logging.info(f"  Rostros detectados: {self.stats['rostros_totales']}")

        if self.stats['frames_con_rostros'] > 0:
            promedio = self.stats['rostros_totales'] / self.stats['frames_con_rostros']
            logging.info(f"  Promedio rostros/frame: {promedio:.2f}")

        logging.info(f"\n😊 DISTRIBUCIÓN DE EMOCIONES:")
        porcentajes = self._calcular_porcentajes_emociones()

        emociones_ordenadas = sorted(
            self.emociones_contador.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for emocion, count in emociones_ordenadas:
            porcentaje = porcentajes[emocion]
            barra = "█" * int(porcentaje / 2)
            logging.info(f"  {emocion:10s}: {count:3d} ({porcentaje:5.1f}%) {barra}")

        if self.stats['errores']:
            logging.info(f"\n⚠️  Errores encontrados: {len(self.stats['errores'])}")
        else:
            logging.info(f"\n✓ No se encontraron errores")


def main():
    """
    Función principal - Ejemplo de uso completo
    """
    logging.info("="*60)
    logging.info("SISTEMA DE ANÁLISIS DE EMOCIONES")
    logging.info("Extracción + Análisis + Anotación EN TIEMPO REAL")
    logging.info("="*60)

    BASE_DIR = Path(__file__).resolve().parent.parent
    VIDEO_DE_ENTRADA = BASE_DIR / "data" / "videos" / "mivideo.mp4" 

    analizador = AnalizadorEmociones(output_dir='resultados_emociones')

    logging.info("\n📹 PASO 1: EXTRACCIÓN CON ANÁLISIS Y ANOTACIÓN")
    if not analizador.extraer_frames_con_emociones(VIDEO_DE_ENTRADA, fps_extraccion=5):
        logging.error("Error en el proceso. Finalizando.")
        return

    logging.info("\n📄 PASO 2: GENERACIÓN DE REPORTE")
    analizador.generar_reporte_json()

    analizador.imprimir_resumen()

    logging.info("\n" + "="*60)
    logging.info("✅ ANÁLISIS COMPLETADO")
    logging.info("="*60)
    logging.info(f"\nResultados disponibles en:")
    logging.info(f"  📁 Frames (con emociones): {analizador.frames_dir}")
    logging.info(f"  📊 Reportes: {analizador.reports_dir}")


if __name__ == "__main__":
    main()