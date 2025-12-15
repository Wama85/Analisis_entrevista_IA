# 🎭 Reconocimiento de Emociones Faciales con DeepFace

Proyecto de prueba mínima para detectar emociones faciales en videos usando DeepFace y modelos preentrenados.

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Archivos del Proyecto](#archivos-del-proyecto)
- [Cómo Funciona](#cómo-funciona)
- [Ejemplos de Salida](#ejemplos-de-salida)
- [Solución de Problemas](#solución-de-problemas)
- [Referencias](#referencias)

## 📝 Descripción

Este proyecto implementa un sistema de reconocimiento de emociones faciales que:
- Carga videos y extrae frames
- Utiliza DeepFace (con modelos preentrenados) para detectar 7 emociones básicas
- Muestra resultados visuales con gráficos y estadísticas

**Emociones detectadas:**
- 😊 Happy (Feliz)
- 😢 Sad (Triste)
- 😠 Angry (Enojado)
- 😲 Surprise (Sorprendido)
- 😨 Fear (Miedo)
- 🤢 Disgust (Asco)
- 😐 Neutral (Neutral)

## ✨ Características

### 🎯 Script Principal (`emotion_detection.py`)
- Carga un video y extrae un frame específico
- Analiza emociones usando DeepFace
- Genera visualización con imagen anotada y gráfico de barras
- Muestra scores de confianza para todas las emociones

### 📹 Detección en Tiempo Real (`emotion_detection_realtime.py`)
- Usa la webcam para detección en tiempo real
- Muestra emociones con overlay visual
- Permite guardar capturas
- Actualización fluida cada 0.5 segundos

### 📊 Análisis de Evolución (`emotion_evolution_analyzer.py`)
- Analiza múltiples frames del video
- Genera gráficos de evolución temporal
- Crea reportes estadísticos detallados
- Identifica momentos destacados

## 🔧 Requisitos

### Software Necesario

```bash
Python 3.8 - 3.11 (recomendado 3.10)
pip (gestor de paquetes de Python)
```

### Librerías Python

Todas las dependencias están en `requirements.txt`:

- **OpenCV**: Procesamiento de video/imagen
- **DeepFace**: Framework de análisis facial
- **TensorFlow**: Motor de deep learning
- **Matplotlib**: Visualización de resultados
- **NumPy**: Operaciones numéricas
- **Pandas**: Análisis de datos (para script de evolución)

## 📦 Instalación

### Paso 1: Clonar o Descargar el Proyecto

```bash
# Si usas git
git clone <url-del-repositorio>
cd emotion-detection-project

# O simplemente descarga los archivos en una carpeta
```

### Paso 2: Crear Entorno Virtual (Recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

**⚠️ Nota:** La primera instalación puede tardar varios minutos debido a TensorFlow.

### Paso 4: Verificar Instalación

```bash
python -c "import cv2; import deepface; print('✅ Todo listo!')"
```

## 🚀 Uso

### 1️⃣ Análisis de Video (Script Principal)

```bash
python emotion_detection.py
```

**Configuración:**
- Por defecto busca un archivo llamado `video_prueba.mp4`
- Para usar tu propio video, edita la línea en el código:
  ```python
  video_path = "tu_video.mp4"  # Cambia esto
  ```
- Ajusta el frame a analizar:
  ```python
  frame_number = 30  # Cambia esto (30 = ~1 segundo a 30fps)
  ```

**Salida:**
- `resultado_emociones.jpg` - Imagen con emoción detectada y gráfico

### 2️⃣ Detección en Tiempo Real (Webcam)

```bash
python emotion_detection_realtime.py
```

**Controles:**
- `q` o `ESC` - Salir
- `s` - Guardar captura de pantalla

**Requisitos:**
- Webcam conectada
- Buena iluminación
- Rostro visible de frente

### 3️⃣ Análisis de Evolución Temporal

```bash
python emotion_evolution_analyzer.py
```

**Configuración:**
- Ajusta el número de frames a analizar:
  ```python
  analyzer.analyze_frames(num_frames=15)  # Cambia 15 por el número deseado
  ```

**Salida:**
- `evolucion_emociones.png` - Gráficos de evolución temporal
- `reporte_emociones.txt` - Reporte estadístico completo

## 📁 Archivos del Proyecto

```
emotion-detection-project/
│
├── emotion_detection.py              # Script principal (análisis de frame único)
├── emotion_detection_realtime.py    # Detección en tiempo real con webcam
├── emotion_evolution_analyzer.py    # Análisis de múltiples frames
├── requirements.txt                  # Dependencias del proyecto
├── README.md                        # Este archivo
│
├── video_prueba.mp4                 # Tu video de prueba (no incluido)
│
└── Salidas generadas:
    ├── resultado_emociones.jpg      # Resultado del script principal
    ├── evolucion_emociones.png      # Gráficos de evolución
    ├── reporte_emociones.txt        # Reporte estadístico
    └── captura_emocion_*.jpg        # Capturas de webcam
```

## 🔬 Cómo Funciona

### Flujo del Proceso

```
1. ENTRADA
   └─> Video o Frame de Webcam
   
2. EXTRACCIÓN
   └─> OpenCV extrae frame(s) del video
   
3. DETECCIÓN FACIAL
   └─> DeepFace detecta rostros en el frame
   
4. ANÁLISIS DE EMOCIONES
   └─> Red neuronal preentrenada analiza expresión facial
   
5. CLASIFICACIÓN
   └─> Asigna scores a 7 emociones básicas
   
6. SALIDA
   └─> Visualización + Estadísticas
```

### Modelos Preentrenados

DeepFace utiliza modelos de deep learning ya entrenados:

1. **Detector de Rostros**: 
   - Encuentra y extrae regiones faciales
   - Usa arquitecturas como RetinaFace, MTCNN, etc.

2. **Clasificador de Emociones**:
   - Red neuronal convolucional (CNN)
   - Entrenada en datasets como FER-2013
   - Reconoce patrones en expresiones faciales

**Ventaja**: No necesitas entrenar nada, los modelos se descargan automáticamente en la primera ejecución.

### Emociones y Scores

Cada frame recibe 7 scores (0-100%):
```python
{
    'happy': 85.2,      # % de confianza
    'neutral': 12.3,
    'sad': 1.5,
    'angry': 0.5,
    'surprise': 0.3,
    'fear': 0.1,
    'disgust': 0.1
}
```

La **emoción dominante** es la que tiene el score más alto.

## 📊 Ejemplos de Salida

### Análisis de Frame Único

```
📹 Cargando video: video_prueba.mp4
   Total de frames: 450
   FPS: 30.0
✅ Frame 30 extraído correctamente

🔍 Analizando emociones con DeepFace...

✨ Emoción dominante: HAPPY

📊 Scores de todas las emociones:
   happy       :  92.34%
   neutral     :   5.21%
   surprise    :   1.45%
   sad         :   0.67%
   angry       :   0.22%
   fear        :   0.08%
   disgust     :   0.03%

✅ Visualización guardada en: resultado_emociones.jpg
```

### Análisis de Evolución

```
📊 ESTADÍSTICAS GENERALES
══════════════════════════════════════

🏆 Emoción dominante más frecuente: HAPPY
   Aparece en 12 de 15 frames (80.0%)

📊 Distribución de emociones dominantes:
   happy       : ████████████████████████ 80.0% (12 frames)
   neutral     : ████ 13.3% (2 frames)
   surprise    : ██ 6.7% (1 frames)
```

## 🐛 Solución de Problemas

### Problema: "No se pudo abrir el video"

**Solución:**
```python
# Verifica que el archivo exista
import os
print(os.path.exists("video_prueba.mp4"))

# Prueba con ruta absoluta
video_path = r"C:\Users\TuUsuario\Videos\video.mp4"
```

### Problema: "No se detecta rostro"

**Posibles causas:**
1. Rostro muy pequeño en el frame → Usa un frame con rostro más grande
2. Mala calidad de imagen → Mejora la iluminación
3. Ángulo extremo → Asegura que el rostro esté de frente

**Solución:**
```python
# El parámetro enforce_detection=False permite continuar
result = DeepFace.analyze(
    frame,
    actions=['emotion'],
    enforce_detection=False  # No lanza error si no detecta rostro claramente
)
```

### Problema: TensorFlow muy lento

**Solución CPU:**
```python
# Limita threads de TensorFlow
import os
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
os.environ['TF_NUM_INTEROP_THREADS'] = '4'
```

**Solución GPU (si tienes NVIDIA):**
```bash
# Instala versión GPU de TensorFlow
pip uninstall tensorflow
pip install tensorflow-gpu==2.15.0
```

### Problema: "Import error deepface"

**Solución:**
```bash
# Reinstala DeepFace
pip uninstall deepface
pip install deepface --no-cache-dir
```

### Problema: La webcam no funciona

**Solución:**
```python
# Prueba diferentes índices de cámara
detector.detect_from_webcam(camera_index=0)  # Cambia 0 por 1, 2, etc.
```

## 🎓 Conceptos Técnicos

### ¿Qué es DeepFace?

DeepFace es un framework de Python para análisis facial que incluye:
- **Detección de rostros**: Encuentra caras en imágenes
- **Reconocimiento facial**: Identifica personas
- **Análisis de atributos**: Edad, género, etnia
- **Análisis de emociones**: Detecta expresiones faciales

### Arquitectura del Sistema

```
┌─────────────────┐
│   INPUT VIDEO   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  OpenCV Frame   │
│   Extraction    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   DeepFace      │
│  Face Detection │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CNN Emotion    │
│  Classification │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Visualization  │
│    & Results    │
└─────────────────┘
```

### Modelos CNN para Emociones

Las redes neuronales convolucionales (CNN) aprenden características jerárquicas:

1. **Capas iniciales**: Detectan bordes, esquinas
2. **Capas medias**: Detectan partes faciales (ojos, boca)
3. **Capas finales**: Reconocen patrones de emociones

## 📚 Referencias

### Documentación Oficial
- [DeepFace GitHub](https://github.com/serengil/deepface)
- [OpenCV Python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [TensorFlow](https://www.tensorflow.org/)

### Papers Académicos
- **FER-2013**: Challenges in Representation Learning: Facial Expression Recognition Challenge
- **DeepFace**: Closing the Gap to Human-Level Performance in Face Verification

### Datasets de Emociones
- FER-2013: 35,000 imágenes de rostros etiquetados
- AffectNet: 1,000,000+ imágenes con anotaciones

## 💡 Consejos de Uso

### Para Mejores Resultados

1. **Iluminación**: Asegura buena iluminación frontal
2. **Resolución**: Usa videos de al menos 480p
3. **Ángulo**: Rostros de frente funcionan mejor
4. **Expresiones claras**: Las emociones sutiles son más difíciles de detectar
5. **Un rostro a la vez**: El sistema trabaja mejor con un solo rostro visible

### Optimización

```python
# Para videos largos, analiza menos frames
analyzer.analyze_frames(num_frames=10)  # En lugar de 50

# Reduce resolución si es muy lento
frame = cv2.resize(frame, (640, 480))
```

## 🤝 Contribuciones

Este es un proyecto educativo. Sugerencias de mejora:
- [ ] Agregar más modelos de detección
- [ ] Implementar tracking de rostros
- [ ] Exportar a formatos de video
- [ ] Dashboard web interactivo
- [ ] Análisis de múltiples rostros simultáneos

## 📄 Licencia

Proyecto educativo de código abierto.

## ✉️ Contacto

Para preguntas o problemas, consulta la documentación oficial de DeepFace.

---

**¡Disfruta explorando el reconocimiento de emociones! 🎭**
