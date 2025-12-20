"""
Reconocimiento de Emociones en Tiempo Real con Webcam
Versión alternativa del detector que usa la cámara web
"""

import cv2
import numpy as np
from deepface import DeepFace
import time

class RealtimeEmotionDetector:
    """
    Detector de emociones en tiempo real usando webcam
    """
    
    def __init__(self):
        """Inicializa el detector"""
        print("🎭 Inicializando detector en tiempo real...")
        
        self.emotion_colors = {
            'happy': (0, 255, 0),
            'sad': (255, 0, 0),
            'angry': (0, 0, 255),
            'surprise': (0, 255, 255),
            'fear': (255, 0, 255),
            'disgust': (128, 0, 128),
            'neutral': (128, 128, 128)
        }
        
        # Emojis para cada emoción
        self.emotion_emoji = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'surprise': '😲',
            'fear': '😨',
            'disgust': '🤢',
            'neutral': '😐'
        }
        
        self.last_analysis_time = 0
        self.analysis_interval = 0.5  # Analizar cada 0.5 segundos
        self.current_emotion = None
        self.current_scores = {}
        
    def detect_from_webcam(self, camera_index=0):
        """
        Detecta emociones desde la webcam en tiempo real
        
        Args:
            camera_index: Índice de la cámara (0 para la cámara predeterminada)
        """
        print(f"📹 Abriendo cámara {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise ValueError(f"❌ No se pudo abrir la cámara {camera_index}")
        
        # Configurar resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n✅ Cámara iniciada correctamente")
        print("\n" + "="*60)
        print("  🎥 DETECTOR DE EMOCIONES EN TIEMPO REAL")
        print("="*60)
        print("\n💡 Instrucciones:")
        print("   • Mira a la cámara")
        print("   • Prueba diferentes expresiones faciales")
        print("   • Presiona 'q' o 'ESC' para salir")
        print("   • Presiona 's' para guardar captura")
        print("\n" + "="*60 + "\n")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ No se pudo capturar frame")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Crear copia para mostrar
                display_frame = frame.copy()
                
                # Analizar emociones periódicamente
                if current_time - self.last_analysis_time > self.analysis_interval:
                    try:
                        # Analizar frame
                        result = DeepFace.analyze(
                            frame,
                            actions=['emotion'],
                            enforce_detection=False,
                            silent=True
                        )
                        
                        if isinstance(result, list):
                            result = result[0]
                        
                        self.current_emotion = result['dominant_emotion']
                        self.current_scores = result['emotion']
                        
                        # Si hay región facial, guardarla
                        if 'region' in result:
                            self.face_region = result['region']
                        
                        self.last_analysis_time = current_time
                        
                    except Exception as e:
                        pass  # Continuar si hay error en la detección
                
                # Dibujar información en el frame
                self.draw_info(display_frame)
                
                # Mostrar frame
                cv2.imshow('Detector de Emociones', display_frame)
                
                # Manejar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' o ESC
                    print("\n👋 Cerrando detector...")
                    break
                elif key == ord('s'):  # Guardar captura
                    filename = f'captura_emocion_{int(time.time())}.jpg'
                    cv2.imwrite(filename, display_frame)
                    print(f"📸 Captura guardada: {filename}")
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Detector cerrado correctamente")
    
    def draw_info(self, frame):
        """
        Dibuja información de emociones en el frame
        
        Args:
            frame: Frame donde dibujar la información
        """
        if self.current_emotion is None:
            # Mensaje de espera
            cv2.putText(frame, "Analizando...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return
        
        # Obtener color de la emoción
        color = self.emotion_colors.get(self.current_emotion, (255, 255, 255))
        emoji = self.emotion_emoji.get(self.current_emotion, '')
        
        # Dibujar rectángulo de información (panel superior)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Título
        text = f"Emocion: {self.current_emotion.upper()} {emoji}"
        cv2.putText(frame, text, (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Mostrar top 3 emociones
        y_position = 70
        sorted_emotions = sorted(self.current_scores.items(), 
                                key=lambda x: x[1], reverse=True)[:3]
        
        for emotion, score in sorted_emotions:
            text = f"{emotion}: {score:.1f}%"
            em_color = self.emotion_colors.get(emotion, (255, 255, 255))
            cv2.putText(frame, text, (10, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, em_color, 1)
            y_position += 25
        
        # Instrucciones
        cv2.putText(frame, "Presiona 'q' para salir | 's' para captura", 
                   (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    """Función principal para webcam"""
    detector = RealtimeEmotionDetector()
    
    try:
        detector.detect_from_webcam(camera_index=0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
