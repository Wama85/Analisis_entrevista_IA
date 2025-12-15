from __future__ import annotations

from typing import Any, Dict, Optional

import whisper  # openai-whisper


def transcribe_audio(
        wav_path: str,
        model_size: str = "small",
        language: Optional[str] = "es",
) -> Dict[str, Any]:
    """
    Transcribe audio using Whisper.

    Returns dict with:
      - text (full transcription)
      - segments (list with start/end/text)
      - language
    """
    model = whisper.load_model(model_size)

    # fp16 False helps on CPU / some GPUs
    result = model.transcribe(
        wav_path,
        language=language,
        fp16=False,
        verbose=False,
    )
    return {
        "text": (result.get("text") or "").strip(),
        "segments": result.get("segments", []),
        "language": result.get("language", language),
    }
