from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import uuid
import logging

# IMPORTA TU PIPELINE
from main import main as run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Emotion Analysis API")

BASE_UPLOADS = Path("uploads")
BASE_UPLOADS.mkdir(exist_ok=True)


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    # 1. Guardar video con nombre unico
    video_id = uuid.uuid4().hex
    video_path = BASE_UPLOADS / f"{video_id}_{file.filename}"

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info(f"Video recibido: {video_path}")

    # 2. Ejecutar pipeline (reutilizamos todo el main)
    try:
        run_pipeline(str(video_path))
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

    # 3. Rutas de salida esperadas
    video_name = video_path.stem

    result = {
        "video_id": video_id,
        "input_video": str(video_path),
        "outputs": {
            "facial_json": "resultados_emociones/reports/...",
            "audio_text_json": f"outputs/audio_text/{video_name}_text_audio.json",
            "sync_json": f"outputs/sync/synchronized_emotions.json",
            "insights_json": f"outputs/reports/{video_name}_insights.json",
            "final_video": f"outputs/sync/{video_name}_final.mp4"
        }
    }

    return result
