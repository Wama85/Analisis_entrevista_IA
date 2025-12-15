"""
Reconocimiento de Emociones Faciales con DeepFace
Proyecto: Prueba mínima de reconocimiento facial
"""

import cv2
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt
from pathlib import Path
import sys

class EmotionDetector:
    """
    Clase para detectar emociones en videos usando DeepFace
    """
    
    def __init__(self):
        """Inicializa el detector de emociones"""
        print("🎭 Inicializando detector de emociones...")
        self.emotions_detected = []
        
        # Colores para visualización (en formato BGR)
        self.emotion_colors = {
            'happy': (0, 255, 0),      # Verde
            'sad': (255, 0, 0),        # Azul
            'angry': (0, 0, 255),      # Rojo
            'surprise': (0, 255, 255), # Amarillo
            'fear': (255, 0, 255),     # Magenta
            'disgust': (128, 0, 128),  # Púrpura
            'neutral': (128, 128, 128) # Gris
        }
        
    def extract_frame_from_video(self, video_path, frame_number=0):
        """
        Extrae un frame específico del video
        
        Args:
            video_path: Ruta al archivo de video
            frame_number: Número del frame a extraer (0 = primer frame)
            
        Returns:
            frame: Imagen del frame extraído
        """
        print(f"📹 Cargando video: {video_path}")
        
        # Abrir el video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"❌ No se pudo abrir el video: {video_path}")
        
        # Obtener información del video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"   Total de frames: {total_frames}")
        print(f"   FPS: {fps}")
        
        # Posicionar en el frame deseado
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Leer el frame
        ret, frame = cap.read()
        
        cap.release()
        
        if not ret:
            raise ValueError(f"❌ No se pudo leer el frame {frame_number}")
        
        print(f"✅ Frame {frame_number} extraído correctamente")
        return frame
    
    def detect_emotion(self, frame):
        """
        Detecta emociones en un frame usando DeepFace
        
        Args:
            frame: Imagen del frame
            
        Returns:
            result: Diccionario con los resultados de la detección
        """
        print("\n🔍 Analizando emociones con DeepFace...")
        
        try:
            # Analizar el frame con DeepFace
            # enforce_detection=False permite continuar aunque no detecte rostro claramente
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            # Si el resultado es una lista, tomar el primer elemento
            if isinstance(result, list):
                result = result[0]
            
            # Extraer información
            dominant_emotion = result['dominant_emotion']
            emotion_scores = result['emotion']
            
            print(f"\n✨ Emoción dominante: {dominant_emotion.upper()}")
            print("\n📊 Scores de todas las emociones:")
            for emotion, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True):
                print(f"   {emotion:12s}: {score:6.2f}%")
            
            return result
            
        except Exception as e:
            print(f"❌ Error al analizar emociones: {e}")
            return None
    
    def visualize_result(self, frame, result, save_path='resultado_emociones.jpg'):
        """
        Visualiza el resultado con la emoción detectada
        
        Args:
            frame: Imagen del frame
            result: Resultado del análisis de DeepFace
            save_path: Ruta donde guardar la imagen resultante
        """
        if result is None:
            print("❌ No hay resultados para visualizar")
            return
        
        print(f"\n🎨 Creando visualización...")
        
        # Copiar el frame para no modificar el original
        frame_display = frame.copy()
        
        # Obtener información de la región facial si está disponible
        if 'region' in result:
            region = result['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # Dibujar rectángulo alrededor del rostro
            dominant_emotion = result['dominant_emotion']
            color = self.emotion_colors.get(dominant_emotion, (255, 255, 255))
            
            cv2.rectangle(frame_display, (x, y), (x+w, y+h), color, 3)
            
            # Añadir texto con la emoción
            label = f"{dominant_emotion.upper()}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            
            # Calcular tamaño del texto para el fondo
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Dibujar fondo para el texto
            cv2.rectangle(frame_display, 
                         (x, y - text_height - 10), 
                         (x + text_width, y), 
                         color, -1)
            
            # Dibujar texto
            cv2.putText(frame_display, label, (x, y - 5), 
                       font, font_scale, (255, 255, 255), thickness)
        
        # Crear figura con matplotlib
        plt.figure(figsize=(15, 5))
        
        # Subplot 1: Imagen original
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB))
        plt.title(f'Emoción detectada: {result["dominant_emotion"].upper()}', 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Subplot 2: Gráfico de barras con scores
        plt.subplot(1, 2, 2)
        emotions = list(result['emotion'].keys())
        scores = list(result['emotion'].values())
        
        # Ordenar por score
        sorted_pairs = sorted(zip(emotions, scores), key=lambda x: x[1], reverse=True)
        emotions, scores = zip(*sorted_pairs)
        
        colors = [self.emotion_colors.get(e, (128, 128, 128)) for e in emotions]
        colors_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in colors]  # BGR to RGB
        
        bars = plt.barh(emotions, scores, color=colors_rgb)
        plt.xlabel('Score (%)', fontsize=12)
        plt.title('Distribución de Emociones', fontsize=14, fontweight='bold')
        plt.xlim(0, 100)
        
        # Añadir valores en las barras
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(score + 1, i, f'{score:.1f}%', 
                    va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualización guardada en: {save_path}")
        
        return frame_display

def main():
    """Función principal"""
    print("=" * 60)
    print("  🎭 DETECTOR DE EMOCIONES FACIALES CON DEEPFACE")
    print("=" * 60)
    
    # Crear instancia del detector
    detector = EmotionDetector()
    
    # Ruta al video (puedes cambiar esto)
    video_path = "video_prueba.mp4"
    
    # Verificar si existe el archivo
    if not Path(video_path).exists():
        print(f"\n⚠️  El archivo '{video_path}' no existe.")
        print("\n💡 Opciones:")
        print("   1. Coloca un video llamado 'video_prueba.mp4' en el mismo directorio")
        print("   2. O especifica la ruta de tu video modificando la variable 'video_path'")
        print("\n📝 Para probar con video de webcam, puedes usar el script alternativo")
        return
    
    try:
        # Paso 1: Extraer un frame del video
        frame_number = 30  # Frame 30 (aproximadamente segundo 1 a 30fps)
        frame = detector.extract_frame_from_video(video_path, frame_number)
        
        # Paso 2: Detectar emociones en el frame
        result = detector.detect_emotion(frame)
        
        # Paso 3: Visualizar resultado
        if result:
            detector.visualize_result(frame, result, 'resultado_emociones.jpg')
            
            print("\n" + "=" * 60)
            print("  ✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
            print("=" * 60)
            print(f"\n📄 Resultado guardado en: resultado_emociones.jpg")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
