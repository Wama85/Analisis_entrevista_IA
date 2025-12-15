"""
Generador de Video de Prueba
Crea un video simple para probar el detector de emociones
"""

import cv2
import numpy as np

def create_test_video_from_webcam(output_path='video_prueba.mp4', duration=10):
    """
    Crea un video de prueba capturando desde la webcam
    
    Args:
        output_path: Nombre del archivo de salida
        duration: Duración en segundos
    """
    print("🎥 Creando video de prueba desde webcam...")
    print(f"   Duración: {duration} segundos")
    print(f"   Salida: {output_path}")
    
    # Abrir webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ No se pudo abrir la webcam")
        return False
    
    # Configurar video writer
    fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("\n✅ Grabando... (Muestra diferentes expresiones)")
    print("\n💡 Sugerencias:")
    print("   • Sonríe (happy)")
    print("   • Cara triste (sad)")
    print("   • Cara enojada (angry)")
    print("   • Sorprendido (surprise)")
    print("   • Cara neutral")
    
    total_frames = int(fps * duration)
    frame_count = 0
    
    try:
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Agregar contador
            remaining = (total_frames - frame_count) / fps
            text = f"Quedan: {remaining:.1f}s"
            cv2.putText(frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Mostrar frame
            cv2.imshow('Grabando Video de Prueba', frame)
            
            # Escribir frame
            out.write(frame)
            frame_count += 1
            
            # ESC para cancelar
            if cv2.waitKey(1) & 0xFF == 27:
                print("\n⚠️  Grabación cancelada")
                break
        
        print(f"\n✅ Video creado exitosamente: {output_path}")
        print(f"   Frames grabados: {frame_count}")
        print(f"   Duración real: {frame_count/fps:.2f}s")
        
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    return True

def create_demo_video_with_text(output_path='video_prueba.mp4', duration=10):
    """
    Crea un video de demostración con texto (sin necesitar webcam)
    
    Args:
        output_path: Nombre del archivo de salida
        duration: Duración en segundos
    """
    print("🎬 Creando video de demostración...")
    print(f"   Duración: {duration} segundos")
    print(f"   Salida: {output_path}")
    
    # Configuración
    fps = 30.0
    width, height = 640, 480
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(fps * duration)
    
    emotions = ['HAPPY 😊', 'SAD 😢', 'ANGRY 😠', 'SURPRISE 😲', 
                'FEAR 😨', 'NEUTRAL 😐']
    
    frames_per_emotion = total_frames // len(emotions)
    
    print("\n✅ Generando frames...")
    
    for i in range(total_frames):
        # Crear frame negro
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Determinar emoción actual
        emotion_idx = i // frames_per_emotion
        if emotion_idx >= len(emotions):
            emotion_idx = len(emotions) - 1
        
        current_emotion = emotions[emotion_idx]
        
        # Color de fondo basado en emoción
        if 'HAPPY' in current_emotion:
            frame[:] = (100, 200, 100)  # Verde
        elif 'SAD' in current_emotion:
            frame[:] = (200, 100, 100)  # Azul
        elif 'ANGRY' in current_emotion:
            frame[:] = (100, 100, 200)  # Rojo
        elif 'SURPRISE' in current_emotion:
            frame[:] = (100, 200, 200)  # Amarillo
        elif 'FEAR' in current_emotion:
            frame[:] = (200, 100, 200)  # Magenta
        else:
            frame[:] = (150, 150, 150)  # Gris
        
        # Agregar texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = current_emotion
        
        # Calcular posición central
        text_size = cv2.getTextSize(text, font, 2, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        # Dibujar texto con sombra
        cv2.putText(frame, text, (text_x + 3, text_y + 3), 
                   font, 2, (0, 0, 0), 5)  # Sombra
        cv2.putText(frame, text, (text_x, text_y), 
                   font, 2, (255, 255, 255), 3)  # Texto
        
        # Agregar información
        info = f"Frame {i+1}/{total_frames} | {(i/fps):.1f}s"
        cv2.putText(frame, info, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    
    print(f"\n✅ Video demo creado: {output_path}")
    print("⚠️  Nota: Este es un video sintético con texto.")
    print("   Para mejores resultados, usa un video real con rostros.")
    
    return True

def show_menu():
    """Muestra menú de opciones"""
    print("\n" + "="*60)
    print("  🎬 GENERADOR DE VIDEO DE PRUEBA")
    print("="*60)
    print("\n¿Cómo deseas crear el video de prueba?")
    print("\n1. Grabar desde webcam (RECOMENDADO)")
    print("2. Generar video sintético con texto")
    print("3. Salir")
    
    choice = input("\nSelecciona una opción (1-3): ").strip()
    return choice

def main():
    """Función principal"""
    
    choice = show_menu()
    
    if choice == '1':
        print("\n📹 Opción: Grabar desde webcam")
        duration = input("Duración en segundos (default 10): ").strip()
        duration = int(duration) if duration.isdigit() else 10
        
        create_test_video_from_webcam('video_prueba.mp4', duration)
        
    elif choice == '2':
        print("\n🎨 Opción: Video sintético")
        duration = input("Duración en segundos (default 10): ").strip()
        duration = int(duration) if duration.isdigit() else 10
        
        create_demo_video_with_text('video_prueba.mp4', duration)
        
    elif choice == '3':
        print("\n👋 ¡Hasta luego!")
        return
    
    else:
        print("\n❌ Opción inválida")
        return
    
    print("\n" + "="*60)
    print("  ✅ LISTO PARA USAR")
    print("="*60)
    print("\nAhora puedes ejecutar el detector:")
    print("  python emotion_detection.py")

if __name__ == "__main__":
    main()
