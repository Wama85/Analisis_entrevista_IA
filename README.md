# Sistema Inteligente de Análisis de Entrevistas

Proyecto desarrollado para la materia **Sistemas Inteligentes** – Universidad Católica Boliviana “San Pablo”.

---
# Equipo de trabajo

* **Estudiante 1**: [Wilner Mena Aguilar]
* **Estudiante 2**: [Susan Cespedes Lazcano]
* **Estudiante 3**: [Isabella Montes]
* **Estudiante 4**: [Adriel Fenandez]


##  Descripción General

Este proyecto implementa un **sistema inteligente multimodal** capaz de analizar entrevistas en video combinando:

*  **Reconocimiento facial de emociones**
*  **Transcripción automática de audio**
*  **Análisis emocional del texto**
*  **Análisis temporal de emociones**

El objetivo principal es **integrar modelos de inteligencia artificial preentrenados** para obtener *insights* sobre el comportamiento emocional de una persona durante una entrevista, sin entrenar modelos desde cero.

---

## Objetivos del Proyecto

* Integrar múltiples modelos de IA en un solo pipeline funcional
* Analizar emociones faciales y textuales de forma conjunta
* Estudiar la evolución emocional a lo largo del tiempo
* Generar insights automáticos sobre congruencia emocional
* Demostrar un enfoque profesional de integración multimodal

---

## Arquitectura del Sistema

### Flujo General del Pipeline

1. Entrada: Video de entrevista
2. Extracción de frames del video
3. Análisis de emociones faciales (CNN)
4. Extracción de audio del video
5. Transcripción automática (ASR)
6. Análisis emocional del texto (NLP)
7. Organización temporal de emociones
8. Generación de insights y visualizaciones

---

## Modelos Utilizados (Preentrenados)

### Reconocimiento Facial

* **DeepFace**
* Tipo: CNN preentrenada
* Salida: Probabilidades de emociones por frame

### Transcripción de Audio

* **Whisper (OpenAI)**
* Tipo: Transformer encoder–decoder
* Salida: Texto transcrito de la entrevista

### Análisis de Texto

* **Transformer preentrenado** (BERT / RoBERTa para emociones)
* Salida: Emociones del discurso con probabilidades

### Análisis Temporal

* **GRU (Gated Recurrent Unit)**
* Entrada: Secuencia temporal de emociones
* Función: Analizar cambios y patrones emocionales

---

## Estructura del Proyecto

```
Sistema-Analisis-Entrevistas/
│
├── data/
│   ├── videos/
│   └── audio/
│
├── cnn_emotions/
│   └── facial_emotion.py
│
├── audio_text/
│   ├── extract_audio.py
│   └── plot_emotions.py
│   └── run.py
│   └── text_emotion.py
│   └── transcribe_whisper.py  
│
├── plots/
│   └── plot_sync.py
├── reports/
│   └── generate_insights.py
│
├── outputs/
│   
│
├── requirements.txt
├── README.md
└── main.py
└── api.py
└── index.html
└── Sync_emotion.py
└── create_Video_with_text.py

```

---

## Estructura de Datos Común (JSON)

```json
{
  "timestamp": 12.5,
  "facial_emotion": {
    "happy": 0.72,
    "sad": 0.08
  },
  "text_emotion": {
    "joy": 0.65,
    "anger": 0.12
  }
}
```

Esta estructura permite la integración entre los distintos módulos del sistema.

---

## Requisitos del Sistema

### Software

* Python 3.9+
* pip

### Librerías principales

```bash
pip install opencv-python deepface torch torchaudio
pip install transformers openai-whisper librosa
pip install numpy pandas matplotlib
```

---

## Ejecución del Proyecto

1. Clonar el repositorio:

```bash
git clone https://github.com/usuario/Sistema-Analisis-Entrevistas.git
cd Sistema-Analisis-Entrevistas
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Ejecutar el sistema:

```bash
python main.py --video data/videos/entrevista.mp4
```

---

## Validación del Sistema

* Se utilizaron **3–5 videos propios** creados por el equipo
* Las emociones fueron etiquetadas manualmente
* Se compararon las predicciones del sistema con los cambios reales observados

---

## Resultados y Análisis

El sistema permite:

* Detectar emociones predominantes
* Identificar cambios emocionales bruscos
* Analizar congruencia entre emoción facial y textual
* Generar visualizaciones temporales

---

## Limitaciones

* Dependencia de la calidad del audio y video
* Posibles sesgos culturales en modelos preentrenados
* No se analiza lenguaje corporal completo

---

## Trabajo Futuro

* Mejorar sincronización audio-video
* Incorporar más métricas de congruencia
* Desarrollar interfaz gráfica
* Evaluar más modelos de emociones

---

## Trabajo en Equipo

* Desarrollo colaborativo con rotación de roles
* Todos los integrantes programaron y documentaron
* Uso de control de versiones (Git) para seguimiento de contribuciones

---

## Contexto Académico

Proyecto desarrollado como **Práctica Integrada Avanzada** para la materia *Sistemas Inteligentes*.

---

##  Licencia

Este proyecto es de uso académico.
