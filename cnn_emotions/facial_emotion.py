"""
Análisis de Emociones Frame por Frame - Versión Optimizada para Videos Completos
==================================================================================
Procesa TODO el video de manera eficiente con límite de tiempo/frames.

MEJORAS:
- Procesa TODO el video (no solo primeros segundos)
- Muestreo inteligente para mantener tiempo < 5 minutos
- Estimación de tiempo de procesamiento
- Opción de límite por frames totales o fps

Autor: Sistema de Reconocimiento Facial
Fecha: 17 de diciembre, 2025
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
import time

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

# Configuración de logging
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

        # Crear directorios
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Estadísticas
        self.stats = {
            'frames_totales': 0,
            'frames_con_rostros': 0,
            'frames_sin_rostros': 0,
            'rostros_totales': 0,
            'errores': [],
            'tiempo_total': 0,
            'tiempo_promedio_por_frame': 0
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

    def dibujar_anotaciones(self, frame, rostros_info: List[Dict[str, Any]]) -> np.ndarray:
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

        Args:
            frame: Frame de OpenCV
            frame_count: Número del frame

        Returns:
            dict con información del análisis
        """
        resultado = {
            'frame_name': f"frame_{frame_count:05d}.jpg",
            'frame_number': frame_count,
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

    def extraer_frames_con_emociones(
        self, 
        ruta_video: str, 
        max_frames: Optional[int] = None,
        fps_extraccion: Optional[float] = None,
        max_minutos: float = 5.0
    ) -> bool:
        """
        Extrae frames del video y los analiza con emociones.
        VERSIÓN MEJORADA: Procesa TODO el video con límites inteligentes.

        Args:
            ruta_video: Ruta al archivo de video
            max_frames: Número máximo de frames a procesar (None = usar fps_extraccion o max_minutos)
            fps_extraccion: Frames por segundo a extraer (None = calcular basado en max_minutos)
            max_minutos: Tiempo máximo de procesamiento en minutos (default: 5)

        Returns:
            bool: True si exitoso, False si error
        """
        logging.info(f"🎬 Iniciando análisis de video completo: {ruta_video}")

        cap = cv2.VideoCapture(str(ruta_video))

        if not cap.isOpened():
            logging.error(f"❌ Error: No se pudo abrir el video")
            return False

        # Obtener información del video
        fps_original = cap.get(cv2.CAP_PROP_FPS)
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duracion = total_frames_video / fps_original if fps_original > 0 else 0

        logging.info(f"📊 Información del video:")
        logging.info(f"   FPS: {fps_original:.2f}")
        logging.info(f"   Frames totales: {total_frames_video}")
        logging.info(f"   Duración: {duracion:.2f} segundos")

        # Calcular estrategia de muestreo
        if max_frames is not None:
            # Opción 1: Límite por frames
            frames_a_procesar = min(max_frames, total_frames_video)
            frame_interval = max(1, total_frames_video // frames_a_procesar)
            logging.info(f"\n⚙️  Modo: Límite por frames")
            logging.info(f"   Frames a procesar: {frames_a_procesar}")
            
        elif fps_extraccion is not None:
            # Opción 2: FPS específico
            frame_interval = max(1, int(fps_original / fps_extraccion))
            frames_a_procesar = total_frames_video // frame_interval
            logging.info(f"\n⚙️  Modo: FPS de extracción")
            logging.info(f"   FPS extracción: {fps_extraccion}")
            logging.info(f"   Frames a procesar: ~{frames_a_procesar}")
            
        else:
            # Opción 3: Basado en tiempo máximo
            tiempo_por_frame = 1.5  # segundos (estimación conservadora)
            max_segundos = max_minutos * 60
            frames_a_procesar = int(max_segundos / tiempo_por_frame)
            frames_a_procesar = min(frames_a_procesar, total_frames_video)
            frame_interval = max(1, total_frames_video // frames_a_procesar)
            
            logging.info(f"\n⚙️  Modo: Límite por tiempo")
            logging.info(f"   Tiempo máximo: {max_minutos} minutos")
            logging.info(f"   Frames a procesar: ~{frames_a_procesar}")

        logging.info(f"   Intervalo de frames: cada {frame_interval} frames")
        
        # Estimación de tiempo
        tiempo_estimado = frames_a_procesar * 1.5 / 60  # minutos
        logging.info(f"   ⏱️  Tiempo estimado: {tiempo_estimado:.1f} minutos")
        logging.info(f"   📍 Cobertura: TODO el video ({duracion:.1f}s)")

        logging.info("\n✨ Iniciando análisis...")
        
        inicio_total = time.time()
        frame_count = 0
        extracted_count = 0
        tiempos_procesamiento = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Muestreo uniforme a lo largo de TODO el video
            if frame_count % frame_interval == 0:
                inicio_frame = time.time()
                
                # 1. Analizar el frame
                analisis = self.analizar_y_anotar_frame(frame, extracted_count)

                # 2. Si hay rostros, dibujar anotaciones
                if analisis['num_rostros'] > 0:
                    frame = self.dibujar_anotaciones(frame, analisis['rostros'])

                # 3. Guardar frame (ya anotado si tenía rostros)
                nombre_frame = self.frames_dir / f"frame_{extracted_count:05d}.jpg"
                cv2.imwrite(str(nombre_frame), frame)

                # 4. Guardar análisis para el reporte
                analisis['frame_path'] = str(nombre_frame)
                self.todos_los_analisis.append(analisis)

                # Timing
                tiempo_frame = time.time() - inicio_frame
                tiempos_procesamiento.append(tiempo_frame)

                extracted_count += 1

                # Progreso cada 10 frames
                if extracted_count % 10 == 0:
                    progreso = (frame_count / total_frames_video) * 100
                    tiempo_transcurrido = (time.time() - inicio_total) / 60
                    tiempo_promedio = sum(tiempos_procesamiento) / len(tiempos_procesamiento)
                    frames_restantes = frames_a_procesar - extracted_count
                    tiempo_restante = (frames_restantes * tiempo_promedio) / 60
                    
                    logging.info(
                        f"   📊 Progreso: {progreso:.1f}% | "
                        f"Frames: {extracted_count}/{frames_a_procesar} | "
                        f"Tiempo: {tiempo_transcurrido:.1f}min | "
                        f"Restante: ~{tiempo_restante:.1f}min"
                    )

            frame_count += 1

        cap.release()
        
        tiempo_total = (time.time() - inicio_total) / 60
        
        self.stats['frames_totales'] = extracted_count
        self.stats['tiempo_total'] = tiempo_total
        self.stats['tiempo_promedio_por_frame'] = sum(tiempos_procesamiento) / len(tiempos_procesamiento) if tiempos_procesamiento else 0

        logging.info(f"\n✅ Análisis completado!")
        logging.info(f"   Frames procesados: {extracted_count}")
        logging.info(f"   Con rostros: {self.stats['frames_con_rostros']}")
        logging.info(f"   Sin rostros: {self.stats['frames_sin_rostros']}")
        logging.info(f"   ⏱️  Tiempo total: {tiempo_total:.2f} minutos")
        logging.info(f"   ⏱️  Promedio por frame: {self.stats['tiempo_promedio_por_frame']:.2f} segundos")

        return True

    def generar_reporte_json(self, nombre_archivo: str = 'emotion_analysis_report.json'):
        """
        Genera un reporte JSON completo con todos los resultados
        """
        if self.stats['frames_con_rostros'] > 0:
            promedio_rostros = self.stats['rostros_totales'] / self.stats['frames_con_rostros']
        else:
            promedio_rostros = 0

        if sum(self.emociones_contador.values()) > 0:
            emocion_mas_comun = max(self.emociones_contador.items(), key=lambda x: x[1])
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
                'frecuencia_emocion_mas_comun': int(emocion_mas_comun[1]),
                'tiempo_procesamiento_minutos': float(round(self.stats['tiempo_total'], 2)),
                'tiempo_promedio_por_frame_segundos': float(round(self.stats['tiempo_promedio_por_frame'], 2))
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

        logging.info(f"💾 Reporte guardado en: {ruta_reporte}")
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
        logging.info(f"  ⏱️  Tiempo total: {self.stats['tiempo_total']:.2f} minutos")

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
    Función principal con opciones de configuración
    """
    logging.info("="*60)
    logging.info("SISTEMA DE ANÁLISIS DE EMOCIONES - VERSIÓN OPTIMIZADA")
    logging.info("Procesa TODO el video con límites de tiempo")
    logging.info("="*60)

    BASE_DIR = Path(__file__).resolve().parent.parent
    VIDEO_DE_ENTRADA = BASE_DIR / "data" / "videos" / "Wilner.mp4"

    analizador = AnalizadorEmociones(output_dir='resultados_emociones')

    logging.info("\n📹 INICIANDO ANÁLISIS")
    
    # OPCIÓN 1: Límite por tiempo (5 minutos máximo)
    # if not analizador.extraer_frames_con_emociones(VIDEO_DE_ENTRADA, max_minutos=5.0):
    
    # OPCIÓN 2: Límite por FPS (2 fps = ~82 frames para video de 41s)
    if not analizador.extraer_frames_con_emociones(VIDEO_DE_ENTRADA, fps_extraccion=2):
    
    # OPCIÓN 3: Límite por frames totales (200 frames máximo)
    # if not analizador.extraer_frames_con_emociones(VIDEO_DE_ENTRADA, max_frames=200):
        
        logging.error("❌ Error en el proceso. Finalizando.")
        return

    logging.info("\n📄 GENERANDO REPORTE")
    analizador.generar_reporte_json()

    analizador.imprimir_resumen()

    logging.info("\n" + "="*60)
    logging.info("✅ ANÁLISIS COMPLETADO")
    logging.info("="*60)
    logging.info(f"\n📁 Resultados:")
    logging.info(f"  Frames: {analizador.frames_dir}")
    logging.info(f"  Reportes: {analizador.reports_dir}")


if __name__ == "__main__":
    main()