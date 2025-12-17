import whisper
import numpy as np
from moviepy.video.io.VideoFileClip import AudioFileClip

def transcribe_audio(wav_path, model_size="tiny", language="es"):
    """
    Versión para Isabella: Carga audio sin usar ffmpeg externo.
    """
    print(f"--- Cargando modelo Whisper ({model_size})... ---")
    model = whisper.load_model(model_size)

    print(f"--- Procesando audio para transcripción... ---")
    
    # En lugar de dejar que Whisper use ffmpeg, cargamos el audio con MoviePy
    audio_clip = AudioFileClip(wav_path)
    
    # Convertimos el audio a la forma que Whisper entiende (array de floats)
    # Whisper necesita una tasa de muestreo de 16000Hz
    audio_array = audio_clip.to_soundarray(fps=16000)
    
    # Si es estéreo, lo pasamos a mono
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)
    
    audio_array = audio_array.astype(np.float32)

    print(f"--- Iniciando transcripción (esto puede tardar)... ---")
    result = model.transcribe(
        audio_array,  # Le pasamos el array directamente, no el archivo
        language=language,
        verbose=False,
    )
    
    return result