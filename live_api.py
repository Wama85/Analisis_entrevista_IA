from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
import json
import logging
import cv2
import numpy as np
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import threading
import queue
import base64
import time

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="Live Emotion Detection API",
    description="API para detección de emociones en tiempo real desde la cámara web",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directorios
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
LIVE_SESSIONS_DIR = OUTPUTS_DIR / "live_sessions"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Crear directorios
for directory in [UPLOADS_DIR, OUTPUTS_DIR, LIVE_SESSIONS_DIR, STATIC_DIR, TEMPLATES_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Montar directorio estático
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ===============================
# MODELO DE DETECCIÓN
# ===============================
try:
    from keras.models import load_model
    MODEL_PATH = BASE_DIR / "modelos" / "emociones_cnn.h5"

    if MODEL_PATH.exists():
        model = load_model(str(MODEL_PATH))
        CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        logger.info(f"✓ Modelo cargado: {MODEL_PATH}")
    else:
        logger.warning("Modelo CNN no encontrado. Usando DeepFace como alternativa.")
        model = "deepface"
        CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        # Importar DeepFace
        try:
            from deepface import DeepFace
            logger.info("✓ DeepFace disponible como alternativa")
        except ImportError:
            logger.error("DeepFace no está instalado. Instala con: pip install deepface")
            model = None
except Exception as e:
    logger.error(f"Error cargando modelo: {e}")
    model = None
    CLASS_NAMES = []

# Detector de rostros
try:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
except:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')

# ===============================
# CLASES Y ESTADOS
# ===============================
class ConnectionManager:
    """Gestiona conexiones WebSocket"""
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.emotion_data_queue = queue.Queue()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Nueva conexión WebSocket. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Conexión cerrada. Total: {len(self.active_connections)}")

    async def send_json(self, data: dict):
        """Envía datos JSON a todos los clientes conectados"""
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.error(f"Error enviando datos: {e}")

class LiveDetectionSession:
    """Sesión individual de detección en vivo"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.camera = None
        self.is_running = False
        self.emotion_history: List[Dict] = []
        self.start_time = None
        self.stats = {
            'total_detections': 0,
            'emotion_counts': {},
            'last_emotion': None,
            'last_confidence': 0.0
        }

    def start_camera(self):
        """Inicia la cámara web"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                logger.error("No se pudo abrir la cámara")
                return False

            self.is_running = True
            self.start_time = datetime.now()
            logger.info(f"Sesión {self.session_id}: Cámara iniciada")
            return True

        except Exception as e:
            logger.error(f"Error iniciando cámara: {e}")
            return False

    def stop_camera(self):
        """Detiene la cámara"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None

        self.is_running = False
        logger.info(f"Sesión {self.session_id}: Cámara detenida")

    def get_frame(self):
        """Obtiene un frame de la cámara"""
        if self.camera is None or not self.is_running:
            return None

        ret, frame = self.camera.read()
        if not ret:
            return None

        return frame

    def process_frame(self, frame: np.ndarray) -> Dict:
        """Procesa un frame y detecta emociones"""
        # Detectar rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        emotions_data = []

        for (x, y, w, h) in faces:
            # Extraer rostro
            face_roi = frame[y:y+h, x:x+w]

            # Predecir emoción
            emotion_result = self.predict_emotion(face_roi)

            if emotion_result:
                emotion, confidence, all_emotions = emotion_result

                # Actualizar estadísticas
                self.update_stats(emotion, confidence)

                # Guardar en historial
                emotion_data = {
                    'timestamp': datetime.now().isoformat(),
                    'emotion': emotion,
                    'confidence': confidence,
                    'all_emotions': all_emotions,
                    'position': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
                }
                self.emotion_history.append(emotion_data)

                # Mantener historial limitado
                if len(self.emotion_history) > 1000:
                    self.emotion_history = self.emotion_history[-1000:]

                emotions_data.append(emotion_data)

                # Dibujar en frame
                color = self.get_emotion_color(emotion)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                text = f"{emotion}: {confidence:.1f}%"
                cv2.putText(frame, text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Convertir a JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            return {
                'frame': base64.b64encode(buffer).decode('utf-8'),
                'emotions': emotions_data,
                'timestamp': datetime.now().isoformat(),
                'session_id': self.session_id
            }

        return None

    def predict_emotion(self, face_img: np.ndarray):
        """Predice la emoción usando el modelo"""
        try:
            if model == "deepface":
                # Usar DeepFace
                from deepface import DeepFace
                analysis = DeepFace.analyze(
                    img_path=face_img,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )

                if isinstance(analysis, list):
                    analysis = analysis[0]

                emotion = analysis.get('dominant_emotion', 'neutral')
                emotions = analysis.get('emotion', {})
                confidence = float(emotions.get(emotion, 0))

                all_emotions = {k: float(v) for k, v in emotions.items()}

                return emotion, confidence, all_emotions

            elif model is not None:
                # Usar modelo CNN propio
                # Preprocesar imagen (ajusta según tu modelo)
                face_resized = cv2.resize(face_img, (48, 48))
                if len(face_resized.shape) == 3:
                    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                else:
                    face_gray = face_resized

                face_normalized = face_gray / 255.0
                face_processed = np.expand_dims(face_normalized, axis=-1)
                face_processed = np.expand_dims(face_processed, axis=0)

                # Predecir
                predictions = model.predict(face_processed, verbose=0)[0]
                emotion_idx = np.argmax(predictions)
                emotion = CLASS_NAMES[emotion_idx]
                confidence = float(predictions[emotion_idx] * 100)

                all_emotions = {
                    CLASS_NAMES[i]: float(pred * 100)
                    for i, pred in enumerate(predictions)
                }

                return emotion, confidence, all_emotions

            else:
                return None

        except Exception as e:
            logger.error(f"Error prediciendo emoción: {e}")
            return None

    def update_stats(self, emotion: str, confidence: float):
        """Actualiza estadísticas de la sesión"""
        self.stats['total_detections'] += 1
        self.stats['last_emotion'] = emotion
        self.stats['last_confidence'] = confidence

        if emotion not in self.stats['emotion_counts']:
            self.stats['emotion_counts'][emotion] = 0
        self.stats['emotion_counts'][emotion] += 1

    def get_emotion_color(self, emotion: str):
        """Devuelve color según la emoción"""
        colors = {
            'angry': (0, 0, 255),      # Rojo
            'disgust': (0, 128, 0),    # Verde
            'fear': (128, 0, 128),     # Morado
            'happy': (0, 255, 0),      # Verde claro
            'sad': (255, 0, 0),        # Azul
            'surprise': (0, 255, 255), # Amarillo
            'neutral': (128, 128, 128) # Gris
        }
        return colors.get(emotion, (255, 255, 255))

    def get_session_summary(self) -> Dict:
        """Obtiene resumen de la sesión"""
        if not self.emotion_history:
            return {}

        # Calcular estadísticas avanzadas
        emotion_counts = self.stats['emotion_counts']
        total = self.stats['total_detections']

        # Emoción predominante
        if emotion_counts:
            predominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        else:
            predominant_emotion = None

        # Porcentajes
        emotion_percentages = {
            emotion: (count / total * 100) if total > 0 else 0
            for emotion, count in emotion_counts.items()
        }

        return {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'duration_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'total_detections': total,
            'predominant_emotion': predominant_emotion,
            'emotion_counts': emotion_counts,
            'emotion_percentages': emotion_percentages,
            'last_emotion': self.stats['last_emotion'],
            'last_confidence': self.stats['last_confidence']
        }

    def save_session(self):
        """Guarda la sesión en un archivo JSON"""
        if not self.emotion_history:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"live_session_{self.session_id}_{timestamp}.json"
        filepath = LIVE_SESSIONS_DIR / filename

        session_data = {
            'metadata': {
                'session_id': self.session_id,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': datetime.now().isoformat(),
                'total_detections': self.stats['total_detections']
            },
            'summary': self.get_session_summary(),
            'emotion_history': self.emotion_history[-500:],  # Últimas 500 detecciones
            'raw_stats': self.stats
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Sesión guardada: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error guardando sesión: {e}")
            return None

# ===============================
# INSTANCIAS GLOBALES
# ===============================
connection_manager = ConnectionManager()
active_sessions: Dict[str, LiveDetectionSession] = {}

# ===============================
# RUTAS DE LA API
# ===============================

@app.get("/", response_class=HTMLResponse)
async def get_home():
    """Página principal"""
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>API de Detección de Emociones en Vivo</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                margin: 0;
                padding: 40px;
                color: #333;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            
            header {
                text-align: center;
                margin-bottom: 40px;
            }
            
            h1 {
                color: #2d3748;
                font-size: 2.5rem;
                margin-bottom: 10px;
            }
            
            .description {
                color: #718096;
                font-size: 1.1rem;
                line-height: 1.6;
            }
            
            .api-section {
                margin-bottom: 40px;
                padding: 25px;
                background: #f8f9fa;
                border-radius: 15px;
            }
            
            .api-title {
                display: flex;
                align-items: center;
                gap: 10px;
                color: #2d3748;
                margin-bottom: 20px;
            }
            
            .api-title i {
                color: #667eea;
                font-size: 1.5rem;
            }
            
            .endpoint {
                background: white;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 15px;
                border-left: 4px solid #667eea;
            }
            
            .method {
                display: inline-block;
                padding: 5px 15px;
                background: #667eea;
                color: white;
                border-radius: 20px;
                font-weight: bold;
                margin-right: 10px;
            }
            
            .path {
                font-family: monospace;
                font-size: 1.1rem;
                color: #2d3748;
            }
            
            .demo-buttons {
                display: flex;
                gap: 20px;
                justify-content: center;
                margin-top: 40px;
            }
            
            .demo-btn {
                padding: 15px 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 50px;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                gap: 10px;
                transition: all 0.3s ease;
            }
            
            .demo-btn:hover {
                transform: translateY(-3px);
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
            }
            
            footer {
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #e2e8f0;
                color: #718096;
            }
            
            code {
                background: #edf2f7;
                padding: 2px 8px;
                border-radius: 4px;
                font-family: monospace;
            }
        </style>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    </head>
    <body>
        <div class="container">
            <header>
                <h1><i class="fas fa-brain"></i> API de Detección de Emociones en Vivo</h1>
                <p class="description">
                    API REST para detección facial en tiempo real usando IA. 
                    Soporta WebSocket para streaming y REST para control.
                </p>
            </header>
            
            <div class="api-section">
                <h2 class="api-title"><i class="fas fa-plug"></i> Endpoints Disponibles</h2>
                
                <div class="endpoint">
                    <span class="method">GET</span>
                    <span class="path">/live/ui</span>
                    <p>Interfaz web completa para detección en vivo</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span>
                    <span class="path">/api/live/start</span>
                    <p>Inicia una nueva sesión de detección</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span>
                    <span class="path">/api/live/{session_id}/stop</span>
                    <p>Detiene una sesión activa</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span>
                    <span class="path">/api/live/{session_id}/stats</span>
                    <p>Obtiene estadísticas de una sesión</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span>
                    <span class="path">/api/live/sessions</span>
                    <p>Lista todas las sesiones activas</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">WebSocket</span>
                    <span class="path">/ws/live/{session_id}</span>
                    <p>Conexión WebSocket para streaming en tiempo real</p>
                </div>
            </div>
            
            <div class="demo-buttons">
                <a href="/live/ui" class="demo-btn">
                    <i class="fas fa-desktop"></i> Ir a la Interfaz Web
                </a>
                <a href="/docs" class="demo-btn">
                    <i class="fas fa-book"></i> Ver Documentación API
                </a>
            </div>
            
            <footer>
                <p>API v1.0.0 | Uso: <code>POST /api/live/start</code> para comenzar</p>
            </footer>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/live/ui", response_class=HTMLResponse)
async def get_live_interface():
    """Interfaz web para detección en vivo"""
    html_file = TEMPLATES_DIR / "live_interface.html"

    if html_file.exists():
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
    else:
        # Si no existe el archivo, generar uno básico
        html_content = """
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Detección en Vivo</title>
        </head>
        <body>
            <h1>Interfaz de Detección en Vivo</h1>
            <p>La interfaz completa estará disponible próximamente.</p>
        </body>
        </html>
        """

    return HTMLResponse(content=html_content)

@app.post("/api/live/start")
async def start_live_session():
    """Inicia una nueva sesión de detección en vivo"""
    session_id = str(uuid.uuid4())

    # Crear nueva sesión
    session = LiveDetectionSession(session_id)

    # Iniciar cámara
    if not session.start_camera():
        raise HTTPException(status_code=500, detail="No se pudo iniciar la cámara")

    # Guardar sesión activa
    active_sessions[session_id] = session

    logger.info(f"Sesión iniciada: {session_id}")

    return JSONResponse({
        "status": "success",
        "message": "Sesión iniciada",
        "session_id": session_id,
        "session_info": {
            "start_time": session.start_time.isoformat() if session.start_time else None,
            "is_running": session.is_running,
            "websocket_url": f"ws://localhost:8000/ws/live/{session_id}"
        }
    })

@app.post("/api/live/{session_id}/stop")
async def stop_live_session(session_id: str):
    """Detiene una sesión activa"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    session = active_sessions[session_id]

    # Detener cámara
    session.stop_camera()

    # Guardar sesión
    saved_file = session.save_session()

    # Remover de sesiones activas
    del active_sessions[session_id]

    logger.info(f"Sesión detenida: {session_id}")

    return JSONResponse({
        "status": "success",
        "message": "Sesión detenida",
        "session_id": session_id,
        "saved_file": saved_file,
        "session_summary": session.get_session_summary()
    })

@app.get("/api/live/{session_id}/stats")
async def get_session_stats(session_id: str):
    """Obtiene estadísticas de una sesión"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    session = active_sessions[session_id]

    return JSONResponse({
        "status": "success",
        "session_id": session_id,
        "is_running": session.is_running,
        "stats": session.get_session_summary(),
        "recent_detections": session.emotion_history[-10:] if session.emotion_history else []
    })

@app.get("/api/live/sessions")
async def list_active_sessions():
    """Lista todas las sesiones activas"""
    sessions_list = []

    for session_id, session in active_sessions.items():
        sessions_list.append({
            "session_id": session_id,
            "is_running": session.is_running,
            "start_time": session.start_time.isoformat() if session.start_time else None,
            "total_detections": session.stats['total_detections'],
            "last_emotion": session.stats['last_emotion']
        })

    return JSONResponse({
        "status": "success",
        "total_sessions": len(sessions_list),
        "sessions": sessions_list
    })

@app.websocket("/ws/live/{session_id}")
async def websocket_live_stream(websocket: WebSocket, session_id: str):
    """WebSocket para streaming en tiempo real"""
    if session_id not in active_sessions:
        await websocket.close(code=1008, reason="Sesión no encontrada")
        return

    session = active_sessions[session_id]

    # Conectar WebSocket
    await connection_manager.connect(websocket)

    try:
        # Enviar frames en tiempo real
        while session.is_running and websocket.client_state.CONNECTED:
            frame = session.get_frame()
            if frame is None:
                await asyncio.sleep(0.033)  # ~30 FPS
                continue

            # Procesar frame
            result = session.process_frame(frame)
            if result:
                await websocket.send_json(result)

            await asyncio.sleep(0.033)  # ~30 FPS

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        logger.info(f"WebSocket desconectado para sesión: {session_id}")
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}")
        connection_manager.disconnect(websocket)

@app.get("/api/live/{session_id}/frame")
async def get_single_frame(session_id: str):
    """Obtiene un solo frame con detección"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    session = active_sessions[session_id]

    frame = session.get_frame()
    if frame is None:
        raise HTTPException(status_code=500, detail="No se pudo obtener frame")

    result = session.process_frame(frame)
    if result is None:
        raise HTTPException(status_code=500, detail="Error procesando frame")

    return JSONResponse(result)

@app.get("/api/live/sessions/{session_id}/download")
async def download_session_data(session_id: str):
    """Descarga los datos de una sesión guardada"""
    # Buscar archivo de sesión
    session_files = list(LIVE_SESSIONS_DIR.glob(f"*{session_id}*.json"))

    if not session_files:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    latest_file = max(session_files, key=lambda x: x.stat().st_mtime)

    # Leer y devolver datos
    with open(latest_file, 'r', encoding='utf-8') as f:
        session_data = json.load(f)

    return JSONResponse(session_data)

@app.get("/health")
async def health_check():
    """Verifica el estado de la API"""
    model_status = "loaded" if model is not None else "not_loaded"

    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_status": model_status,
        "active_sessions": len(active_sessions),
        "api_version": "1.0.0"
    })

# ===============================
# MANTENER COMPATIBILIDAD CON TU API EXISTENTE
# ===============================
@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """
    Endpoint original para análisis de video precargado
    (Mantiene compatibilidad con tu API existente)
    """
    # Guardar video
    video_id = uuid.uuid4().hex
    video_path = UPLOADS_DIR / f"{video_id}_{file.filename}"

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info(f"Video recibido para análisis: {video_path}")

    try:
        # Aquí ejecutarías tu pipeline original
        # Por ahora, devolvemos un placeholder
        return JSONResponse({
            "status": "processing",
            "message": "Análisis iniciado",
            "video_id": video_id,
            "video_path": str(video_path),
            "estimated_time": "2-5 minutos"
        })

    except Exception as e:
        logger.error(f"Error en análisis de video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===============================
# FUNCIÓN PARA EJECUTAR
# ===============================
def run_api():
    """Ejecuta la API"""
    import uvicorn

    logger.info("=" * 60)
    logger.info("API DE DETECCIÓN DE EMOCIONES EN VIVO")
    logger.info("=" * 60)
    logger.info(f"URL: http://localhost:8000")
    logger.info(f"Documentación: http://localhost:8000/docs")
    logger.info(f"Interfaz web: http://localhost:8000/live/ui")
    logger.info("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )

if __name__ == "__main__":
    run_api()