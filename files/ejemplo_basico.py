"""
Ejemplo Básico de Detección de Emociones
Crea una imagen de prueba y detecta emociones en ella
"""

import cv2
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt

def create_sample_face_image():
    """
    Crea una imagen simple con un emoji/cara para probar
    (Esto es solo para demostración - en producción usarías fotos reales)
    """
    print("🎨 Creando imagen de muestra...")
    
    # Crear imagen blanca
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Dibujar cara simple (círculo)
    cv2.circle(img, (200, 200), 150, (255, 200, 100), -1)
    
    # Ojos
    cv2.circle(img, (150, 170), 20, (0, 0, 0), -1)
    cv2.circle(img, (250, 170), 20, (0, 0, 0), -1)
    
    # Sonrisa (arco)
    cv2.ellipse(img, (200, 220), (80, 60), 0, 0, 180, (0, 0, 0), 3)
    
    return img

def demo_basic_emotion_detection():
    """
    Demostración básica del proceso de detección
    """
    print("="*60)
    print("  🎭 EJEMPLO BÁSICO DE DETECCIÓN DE EMOCIONES")
    print("="*60)
    
    print("\n📝 Este script demuestra los pasos básicos:")
    print("   1. Crear/Cargar una imagen")
    print("   2. Analizar con DeepFace")
    print("   3. Mostrar resultados")
    
    # Paso 1: Crear imagen de muestra
    print("\n" + "-"*60)
    print("PASO 1: Crear imagen de muestra")
    print("-"*60)
    
    img = create_sample_face_image()
    cv2.imwrite('imagen_muestra.jpg', img)
    print("✅ Imagen creada: imagen_muestra.jpg")
    
    # Paso 2: Analizar con DeepFace
    print("\n" + "-"*60)
    print("PASO 2: Analizar con DeepFace")
    print("-"*60)
    print("🔍 Procesando imagen...")
    
    try:
        result = DeepFace.analyze(
            img,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )
        
        if isinstance(result, list):
            result = result[0]
        
        print("✅ Análisis completado")
        
        # Paso 3: Mostrar resultados
        print("\n" + "-"*60)
        print("PASO 3: Resultados del Análisis")
        print("-"*60)
        
        print(f"\n✨ EMOCIÓN DOMINANTE: {result['dominant_emotion'].upper()}")
        print("\n📊 Scores de confianza:")
        
        # Ordenar emociones por score
        sorted_emotions = sorted(
            result['emotion'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        print("\n   Emoción      | Score | Barra")
        print("   " + "-"*42)
        
        for emotion, score in sorted_emotions:
            bar = "█" * int(score / 5)  # Cada 5% = un bloque
            print(f"   {emotion:12s} | {score:5.1f}% | {bar}")
        
        # Crear visualización
        print("\n📈 Creando visualización...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Subplot 1: Imagen
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'Emoción: {result["dominant_emotion"].upper()}', 
                     fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Subplot 2: Gráfico de barras
        emotions = [e[0] for e in sorted_emotions]
        scores = [e[1] for e in sorted_emotions]
        
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(emotions))]
        
        ax2.barh(emotions, scores, color=colors)
        ax2.set_xlabel('Score de Confianza (%)', fontsize=11)
        ax2.set_title('Distribución de Emociones', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 100)
        
        for i, score in enumerate(scores):
            ax2.text(score + 1, i, f'{score:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('resultado_ejemplo_basico.jpg', dpi=150, bbox_inches='tight')
        
        print("✅ Visualización guardada: resultado_ejemplo_basico.jpg")
        
        # Explicación del resultado
        print("\n" + "="*60)
        print("  📖 INTERPRETACIÓN DE RESULTADOS")
        print("="*60)
        
        print(f"""
El modelo de DeepFace ha analizado la imagen y determinó que la
emoción dominante es: {result['dominant_emotion'].upper()}

¿Qué significa esto?
• DeepFace usa una red neuronal preentrenada para analizar
  expresiones faciales
• Asigna un score de confianza (0-100%) a cada una de las
  7 emociones básicas
• La emoción con el score más alto es la "dominante"

Scores de confianza:
• >70%: Alta confianza en la detección
• 40-70%: Confianza media
• <40%: Baja confianza (imagen ambigua)

En este caso:
• Emoción detectada: {result['dominant_emotion'].upper()}
• Score: {result['emotion'][result['dominant_emotion']]:.1f}%
""")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {e}")
        print("\n💡 Nota: Esta imagen es sintética y puede no ser detectada")
        print("   correctamente. Para mejores resultados, usa fotos reales")
        print("   de rostros humanos.")
        return False

def show_code_explanation():
    """
    Muestra explicación del código básico
    """
    print("\n" + "="*60)
    print("  💻 CÓDIGO BÁSICO DE DETECCIÓN")
    print("="*60)
    
    code = '''
# Importar librerías necesarias
import cv2
from deepface import DeepFace

# 1. Cargar imagen
imagen = cv2.imread('tu_imagen.jpg')

# 2. Analizar emociones
resultado = DeepFace.analyze(
    imagen,
    actions=['emotion'],    # Qué analizar
    enforce_detection=False # No fallar si no detecta rostro claramente
)

# 3. Obtener emoción dominante
emocion = resultado[0]['dominant_emotion']
scores = resultado[0]['emotion']

# 4. Mostrar resultado
print(f"Emoción detectada: {emocion}")
print(f"Score: {scores[emocion]:.1f}%")
'''
    
    print("\nEste es el código mínimo necesario:")
    print(code)
    
    print("\n📋 Explicación línea por línea:")
    print("""
1. cv2.imread() → Carga la imagen del disco
2. DeepFace.analyze() → Procesa la imagen:
   • Detecta rostros
   • Extrae características faciales
   • Clasifica la emoción
3. result[0]['dominant_emotion'] → Obtiene la emoción con mayor score
4. result[0]['emotion'] → Diccionario con todos los scores

¡Es así de simple! 🎉
""")

def main():
    """Función principal"""
    
    # Ejecutar demo
    success = demo_basic_emotion_detection()
    
    # Mostrar explicación del código
    show_code_explanation()
    
    if success:
        print("\n" + "="*60)
        print("  ✅ DEMO COMPLETADA EXITOSAMENTE")
        print("="*60)
        print("\n📁 Archivos generados:")
        print("   • imagen_muestra.jpg")
        print("   • resultado_ejemplo_basico.jpg")
        print("\n💡 Próximos pasos:")
        print("   1. Prueba con tus propias fotos")
        print("   2. Ejecuta emotion_detection_realtime.py para webcam")
        print("   3. Lee README.md para más información")
    
    print("\n👋 ¡Gracias por probar el detector de emociones!\n")

if __name__ == "__main__":
    main()
