from fastapi import FastAPI, WebSocket
import base64
import cv2
import numpy as np
import logging

from cnn_emotions.facial_emotion import predict_emotion_from_frame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.websocket("/live/camera")
async def live_camera(ws: WebSocket):
    await ws.accept()
    logger.info("Cliente conectado (camara)")

    try:
        while True:
            data = await ws.receive_text()  # base64 de imagen

            img_bytes = base64.b64decode(data)
            np_img = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            emotion = predict_emotion_from_frame(frame)

            await ws.send_json({
                "facial_emotion": emotion
            })

    except Exception as e:
        logger.error(f"Conexion cerrada o error: {e}")
