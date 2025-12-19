from __future__ import annotations
from datetime import datetime
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict

import cv2
import numpy as np
from deepface import DeepFace

# ===============================
# CONFIGURACIÓN DE LOGGING
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


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


class AnalizadorEmocionesVideo:
    """
    Clase para extraer frames con emociones anotadas directamente
    """

    def __init__(self, output_dir: str):
        """
        Inicializa el analizador de emociones

        Args:
            output_dir: Directorio para guardar frames y reportes
        """
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / 'frames'
        self.reports_dir = self.output_dir / 'reports'

        # Crear directorios
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Estadísticas
        self.stats = {
            'frames_totales': 0,
            'frames_con_rostros': 0,
            'frames_sin_rostros': 0,
            'rostros_totales': 0,
            'errores': []
        }

        # Contadores de emociones
        self.emociones_contador = {
            'angry': 0,
            'disgust': 0,
            'fear': 0,
            'happy': 0,
            'sad': 0,
            'surprise': 0,
            'neutral': 0
        }

        # Colores para cada emoción (BGR para OpenCV)
        self.colores_emociones = {
            'angry': (0, 0, 255),      # Rojo
            'disgust': (0, 128, 128),  # Verde oscuro
            'fear': (128, 0, 128),     # Morado
            'happy': (0, 255, 0),      # Verde
            'sad': (255, 0, 0),        # Azul
            'surprise': (0, 255, 255), # Amarillo
            'neutral': (128, 128, 128) # Gris
        }

        # Lista para guardar todos los análisis
        self.todos_los_analisis = []

    def dibujar_anotaciones(self, frame: np.ndarray, rostros_info: List[Dict[str, Any]]) -> np.ndarray:
        """
        Dibuja recuadros y emociones en el frame

        Args:
            frame: Imagen OpenCV (numpy array)
            rostros_info: Lista con información de rostros detectados

        Returns:
            frame anotado con recuadros y texto
        """
        frame_anotado = frame.copy()

        for rostro in rostros_info:
            # Obtener región del rostro
            region = rostro.get('region', {})
            if not region:
                continue

            x = int(region.get('x', 0))
            y = int(region.get('y', 0))
            w = int(region.get('w', 0))
            h = int(region.get('h', 0))

            # Obtener emoción
            emocion = rostro.get('emocion_dominante', 'unknown')
            edad = rostro.get('edad', 0)
            genero = rostro.get('genero', 'unknown')

            # Obtener probabilidad de la emoción
            emociones = rostro.get('emociones', {})
            probabilidad = emociones.get(emocion, 0)

            # Color según la emoción
            color = self.colores_emociones.get(emocion, (255, 255, 255))

            # Dibujar rectángulo alrededor del rostro
            cv2.rectangle(frame_anotado, (x, y), (x+w, y+h), color, 3)

            # Preparar texto con la información
            texto_emocion = f"{emocion.upper()}"
            texto_prob = f"{probabilidad:.1f}%"
            texto_info = f"{genero}, {edad}a"

            # Calcular tamaño del texto
            (tw1, th1), _ = cv2.getTextSize(texto_emocion, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            (tw2, th2), _ = cv2.getTextSize(texto_prob, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            (tw3, th3), _ = cv2.getTextSize(texto_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Dibujar fondo negro para el texto
            padding = 5
            cv2.rectangle(frame_anotado,
                          (x, y - th1 - th2 - th3 - padding*3),
                          (x + max(tw1, tw2, tw3) + padding*2, y),
                          (0, 0, 0), -1)

            # Escribir texto de emoción
            cv2.putText(frame_anotado, texto_emocion,
                        (x + padding, y - th2 - th3 - padding*2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Escribir probabilidad
            cv2.putText(frame_anotado, texto_prob,
                        (x + padding, y - th3 - padding),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Escribir información adicional
            cv2.putText(frame_anotado, texto_info,
                        (x + padding, y - padding),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Dibujar punto en el centro del rostro
            centro_x = x + w // 2
            centro_y = y + h // 2
            cv2.circle(frame_anotado, (centro_x, centro_y), 3, color, -1)

        return frame_anotado

    def analizar_frame(self, frame: np.ndarray, frame_count: int) -> Dict[str, Any]:
        """
        Analiza un frame y devuelve la información

        Args:
            frame: Frame de OpenCV
            frame_count: Número del frame

        Returns:
            dict con información del análisis
        """
        resultado = {
            'frame_name': f"frame_{frame_count:05d}.jpg",
            'frame_number': frame_count,
            'timestamp': frame_count / 30.0,  # Asumiendo 30 FPS, ajustar según video
            'num_rostros': 0,
            'rostros': [],
            'error': None
        }

        try:
            # Guardar temporalmente el frame para análisis
            temp_path = self.frames_dir / f"temp_{frame_count}.jpg"
            cv2.imwrite(str(temp_path), frame)

            # Analizar con DeepFace
            analisis = DeepFace.analyze(
                img_path=str(temp_path),
                actions=['emotion', 'age', 'gender'],
                enforce_detection=False,
                silent=True
            )

            # Eliminar archivo temporal
            temp_path.unlink()

            # Procesar resultados
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

                # Actualizar contador de emociones
                emocion = info_rostro['emocion_dominante']
                if emocion in self.emociones_contador:
                    self.emociones_contador[emocion] += 1

            # Actualizar estadísticas
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

    def procesar_video(self, ruta_video: str, fps_extraccion: Optional[int] = None) -> bool:
        """
        Procesa el video frame por frame, analizando emociones

        Args:
            ruta_video: Ruta al archivo de video
            fps_extraccion: Frames por segundo a extraer (None = todos los frames)

        Returns:
            bool: True si exitoso, False si error
        """
        logger.info(f"Iniciando procesamiento del video: {ruta_video}")

        cap = cv2.VideoCapture(str(ruta_video))

        if not cap.isOpened():
            logger.error(f"Error: No se pudo abrir el video en {ruta_video}")
            return False

        # Obtener información del video
        fps_original = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duracion = total_frames / fps_original if fps_original > 0 else 0

        logger.info(f"Video - FPS: {fps_original:.2f}, Frames: {total_frames}, Duración: {duracion:.2f}s")

        # Calcular intervalo de extracción
        if fps_extraccion is not None:
            frame_interval = int(fps_original / fps_extraccion)
            if frame_interval < 1:
                frame_interval = 1
            logger.info(f"Extrayendo 1 frame cada {frame_interval} frames ({fps_extraccion} fps)")
        else:
            frame_interval = 1
            logger.info("Extrayendo todos los frames")

        logger.info("✨ Analizando y anotando emociones en tiempo real...")

        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Extraer frame según el intervalo
            if frame_count % frame_interval == 0:
                # 1. Analizar el frame
                analisis = self.analizar_frame(frame, extracted_count)
                analisis['timestamp'] = frame_count / fps_original if fps_original > 0 else 0

                # 2. Si hay rostros, dibujar anotaciones
                if analisis['num_rostros'] > 0:
                    frame_anotado = self.dibujar_anotaciones(frame, analisis['rostros'])
                else:
                    frame_anotado = frame

                # 3. Guardar frame (ya anotado si tenía rostros)
                nombre_frame = self.frames_dir / f"frame_{extracted_count:05d}.jpg"
                cv2.imwrite(str(nombre_frame), frame_anotado)

                # 4. Guardar análisis para el reporte
                analisis['frame_path'] = str(nombre_frame)
                self.todos_los_analisis.append(analisis)

                extracted_count += 1

                if extracted_count % 10 == 0:
                    logger.info(f"  Procesados: {extracted_count} frames ({self.stats['frames_con_rostros']} con rostros)")

            frame_count += 1

        cap.release()
        self.stats['frames_totales'] = extracted_count

        logger.info(f"\n✓ Procesamiento completado")
        logger.info(f"  Total frames: {extracted_count}")
        logger.info(f"  Con rostros: {self.stats['frames_con_rostros']}")
        logger.info(f"  Sin rostros: {self.stats['frames_sin_rostros']}")

        return True

    def generar_reporte(self, video_path: str, nombre_archivo: str = None) -> Dict[str, Any]:
        """
        Genera un reporte JSON completo con todos los resultados

        Args:
            video_path: Ruta del video original
            nombre_archivo: Nombre del archivo de reporte (opcional)

        Returns:
            dict con el reporte completo
        """
        # Calcular estadísticas adicionales
        if self.stats['frames_con_rostros'] > 0:
            promedio_rostros = self.stats['rostros_totales'] / self.stats['frames_con_rostros']
        else:
            promedio_rostros = 0

        # Encontrar emoción más común
        if sum(self.emociones_contador.values()) > 0:
            emocion_mas_comun = max(self.emociones_contador.items(),
                                    key=lambda x: x[1])
        else:
            emocion_mas_comun = ('unknown', 0)

        # Calcular porcentajes de emociones
        total_emociones = sum(self.emociones_contador.values())
        porcentajes_emociones = {}
        if total_emociones > 0:
            for emocion, count in self.emociones_contador.items():
                porcentajes_emociones[emocion] = round((count / total_emociones) * 100, 2)

        # Estructura del reporte
        reporte = {
            'metadata': {
                'fecha_analisis': datetime.now().isoformat(),
                'video_path': str(video_path),
                'video_nombre': Path(video_path).name,
                'directorios': {
                    'frames': str(self.frames_dir),
                    'reportes': str(self.reports_dir)
                }
            },
            'resumen': {
                'frames_totales': int(self.stats['frames_totales']),
                'frames_con_rostros': int(self.stats['frames_con_rostros']),
                'frames_sin_rostros': int(self.stats['frames_sin_rostros']),
                'rostros_totales_detectados': int(self.stats['rostros_totales']),
                'promedio_rostros_por_frame': float(round(promedio_rostros, 2)),
                'emocion_mas_comun': str(emocion_mas_comun[0]),
                'frecuencia_emocion_mas_comun': int(emocion_mas_comun[1])
            },
            'emociones': {
                'distribucion': {k: int(v) for k, v in self.emociones_contador.items()},
                'porcentajes': porcentajes_emociones
            },
            'analisis_por_frame': convertir_a_tipo_nativo(self.todos_los_analisis),
            'estadisticas': {
                'total_errores': int(len(self.stats['errores'])),
                'errores': self.stats['errores'][:10] if len(self.stats['errores']) > 0 else []
            }
        }

        # Convertir reporte completo antes de guardar
        reporte = convertir_a_tipo_nativo(reporte)

        # Determinar nombre del archivo
        if nombre_archivo is None:
            video_stem = Path(video_path).stem
            nombre_archivo = f"emociones_video_{video_stem}.json"

        # Guardar reporte
        ruta_reporte = self.reports_dir / nombre_archivo
        with open(ruta_reporte, 'w', encoding='utf-8') as f:
            json.dump(reporte, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✓ Reporte guardado en: {ruta_reporte}")
        return reporte

    def imprimir_resumen(self):
        """Imprime un resumen visual del análisis"""
        logger.info("\n" + "="*60)
        logger.info("RESUMEN DEL ANÁLISIS DE EMOCIONES EN VIDEO")
        logger.info("="*60)

        logger.info(f"\n📊 ESTADÍSTICAS GENERALES:")
        logger.info(f"  Frames totales: {self.stats['frames_totales']}")
        logger.info(f"  Frames con rostros: {self.stats['frames_con_rostros']}")
        logger.info(f"  Frames sin rostros: {self.stats['frames_sin_rostros']}")
        logger.info(f"  Rostros detectados: {self.stats['rostros_totales']}")

        if self.stats['frames_con_rostros'] > 0:
            promedio = self.stats['rostros_totales'] / self.stats['frames_con_rostros']
            logger.info(f"  Promedio rostros/frame: {promedio:.2f}")

        logger.info(f"\n😊 DISTRIBUCIÓN DE EMOCIONES:")

        # Calcular porcentajes
        total = sum(self.emociones_contador.values())
        if total > 0:
            # Ordenar por frecuencia
            emociones_ordenadas = sorted(
                self.emociones_contador.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for emocion, count in emociones_ordenadas:
                porcentaje = (count / total) * 100
                barra = "█" * int(porcentaje / 2)
                logger.info(f"  {emocion:10s}: {count:3d} ({porcentaje:5.1f}%) {barra}")
        else:
            logger.info("  No se detectaron emociones")

        if self.stats['errores']:
            logger.info(f"\n⚠️  Errores encontrados: {len(self.stats['errores'])}")
        else:
            logger.info(f"\n✓ No se encontraron errores")


def run(
        video_path: str,
        out_dir: str,
        fps_extraccion: int = 5,
        generate_report: bool = True,
        show_summary: bool = True
) -> Dict[str, Any]:
    """
    Función principal para procesar video y analizar emociones faciales

    Args:
        video_path: Ruta al archivo de video
        out_dir: Directorio de salida
        fps_extraccion: FPS para extracción de frames
        generate_report: Generar reporte JSON
        show_summary: Mostrar resumen en consola

    Returns:
        dict con los resultados del análisis
    """
    logger.info("Iniciando análisis de emociones faciales en video")

    # Crear directorio de salida
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Crear nombre específico para este video
    video_name = Path(video_path).stem
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    specific_output_dir = str(output_path / f"{video_name}_{run_id}")

    # Crear analizador
    logger.info(f"Creando analizador con directorio de salida: {specific_output_dir}")
    analizador = AnalizadorEmocionesVideo(output_dir=specific_output_dir)

    # Procesar video
    logger.info(f"Procesando video: {video_path}")
    logger.info(f"FPS de extracción: {fps_extraccion}")

    if not analizador.procesar_video(video_path, fps_extraccion=fps_extraccion):
        logger.error("Error en el procesamiento del video")
        return {}

    # Generar reporte
    reporte = {}
    if generate_report:
        logger.info("Generando reporte JSON")
        nombre_reporte = f"emociones_{video_name}.json"
        reporte = analizador.generar_reporte(video_path, nombre_reporte)

    # Mostrar resumen
    if show_summary:
        analizador.imprimir_resumen()

    # Estructura de retorno similar al de audio
    resultado = {
        'video': {
            'path': video_path,
            'output_dir': specific_output_dir,
        },
        'analysis': reporte.get('resumen', {}) if reporte else {},
        'emotions': reporte.get('emociones', {}) if reporte else {},
        'metadata': {
            'frames_dir': str(analizador.frames_dir),
            'reports_dir': str(analizador.reports_dir),
            'fecha_analisis': datetime.now().isoformat()
        }
    }

    logger.info("Análisis de video completado")
    return resultado


def main():
    """
    Función principal para ejecutar desde línea de comandos
    """
    parser = argparse.ArgumentParser(
        description="Video -> Facial Emotion Analysis"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video file (.mp4, .avi, .mov)"
    )
    parser.add_argument(
        "--out",
        default="outputs/video_emotion",
        help="Output directory (default: outputs/video_emotion)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="FPS for frame extraction (default: 5)"
    )
    parser.add_argument(
        "--no_report",
        action="store_true",
        help="No generar reporte JSON"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="No mostrar resumen en consola"
    )

    args = parser.parse_args()

    logger.info("Argumentos recibidos:")
    logger.info(f"  Video: {args.video}")
    logger.info(f"  Output: {args.out}")
    logger.info(f"  FPS: {args.fps}")
    logger.info(f"  Generar reporte: {not args.no_report}")
    logger.info(f"  Mostrar resumen: {not args.quiet}")

    # Ejecutar análisis
    result = run(
        video_path=args.video,
        out_dir=args.out,
        fps_extraccion=args.fps,
        generate_report=not args.no_report,
        show_summary=not args.quiet
    )

    # Mostrar información básica al final
    if result and not args.quiet:
        print("\n" + "="*60)
        print("EJECUCIÓN COMPLETADA")
        print("="*60)

        if 'analysis' in result:
            analysis = result['analysis']
            print(f"Frames procesados: {analysis.get('frames_totales', 0)}")
            print(f"Frames con rostros: {analysis.get('frames_con_rostros', 0)}")
            print(f"Emoción más común: {analysis.get('emocion_mas_comun', 'N/A')}")

        if 'emotions' in result and 'porcentajes' in result['emotions']:
            print("\nTop 3 emociones detectadas:")
            emociones_ordenadas = sorted(
                result['emotions']['porcentajes'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            for emocion, porcentaje in emociones_ordenadas:
                print(f"  {emocion}: {porcentaje:.1f}%")

        print(f"\nResultados guardados en: {result.get('metadata', {}).get('output_dir', 'N/A')}")


if __name__ == "__main__":
    main()