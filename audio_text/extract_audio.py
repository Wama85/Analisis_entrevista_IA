from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False)
        return True
    except FileNotFoundError:
        return False


def extract_audio(video_path: str, out_wav_path: str, sr: int = 16000) -> str:
    """
    Extract audio from a video to WAV mono PCM at sr Hz.

    Tries ffmpeg first (recommended). If ffmpeg is not available, falls back to moviepy.
    Returns the output wav path.
    """
    v = Path(video_path)
    out = Path(out_wav_path)

    if out.exists():
        logger.info(f"El audio ya existe en {out}. Omitiendo extracción.")
        return str(out)

    if not v.exists():
        raise FileNotFoundError(f"Video not found: {v}")

    if _ffmpeg_available():
        # WAV: mono, 16-bit PCM, sample rate sr
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(v),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-acodec",
            "pcm_s16le",
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{proc.stderr}")
        return str(out)

    # Fallback: moviepy
    try:
        from moviepy.editor import VideoFileClip  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "moviepy not available. Install it or install ffmpeg."
        ) from e

    clip = VideoFileClip(str(v))
    if clip.audio is None:
        raise RuntimeError("Video has no audio track.")
    clip.audio.write_audiofile(str(out), fps=sr, nbytes=2, codec="pcm_s16le", ffmpeg_params=["-ac", "1"])
    clip.close()
    return str(out)
