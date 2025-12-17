import subprocess
import os
from pathlib import Path

def extract_audio(video_path: str, out_wav_path: str, sr: int = 16000) -> str:
    """
    Versión robusta para Isabella: Extrae audio usando el ffmpeg.exe de la carpeta local.
    """
    v = Path(video_path)
    out = Path(out_wav_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not v.exists():
        raise FileNotFoundError(f"Video no encontrado en: {v}")

    # Buscamos el ffmpeg.exe que pegaste en la carpeta principal
    # Si no, intentamos usar el comando global
    ffmpeg_exe = os.path.join(os.getcwd(), "ffmpeg.exe")
    if not os.path.exists(ffmpeg_exe):
        ffmpeg_exe = "ffmpeg"

    print(f"--- Intentando extraer audio con: {ffmpeg_exe} ---")

    cmd = [
        ffmpeg_exe, "-y", "-i", str(v),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sr), "-ac", "1", str(out)
    ]

    try:
        # shell=True es la clave para que Windows no de errores de permisos
        subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)
        print("--- Audio extraído con éxito vía FFmpeg ---")
        return str(out)
    except Exception as e:
        print(f"--- FFmpeg falló, intentando MoviePy como último recurso ---")
        try:
            from moviepy.video.io.VideoFileClip import VideoFileClip
            clip = VideoFileClip(str(v))
            if clip.audio is None:
                raise RuntimeError("El video no tiene pista de audio.")
            clip.audio.write_audiofile(str(out), fps=sr, nbytes=2, codec="pcm_s16le", ffmpeg_params=["-ac", "1"])
            clip.close()
            return str(out)
        except Exception as e2:
            raise RuntimeError(f"Error total: FFmpeg falló y MoviePy también. Detalles: {e2}")

def _ffmpeg_available():
    # Esta función ya no es tan crítica con el nuevo código, pero la dejamos por compatibilidad
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, shell=True)
        return True
    except:
        return os.path.exists(os.path.join(os.getcwd(), "ffmpeg.exe"))