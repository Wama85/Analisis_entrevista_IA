from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from audio_text.extract_audio import extract_audio
from audio_text.transcribe_whisper import transcribe_audio
from audio_text.text_emotion import build_emotion_classifier, predict_text_emotions


def run(video_path: str, out_dir: str, whisper_model: str, language: str) -> Dict[str, Any]:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    wav_path = str(outp / "audio.wav")
    json_path = str(outp / "group_b_output.json")

    # 1) Extract audio
    extract_audio(video_path, wav_path, sr=16000)

    # 2) Transcribe
    tr = transcribe_audio(wav_path, model_size=whisper_model, language=language)
    text = tr["text"]

    # 3) Emotion from text
    # NOTE: Change the model if you need Spanish/multilingual.
    clf = build_emotion_classifier()
    text_emotions = predict_text_emotions(clf, text)

    output = {
        "input": {
            "video_path": video_path,
            "audio_wav": wav_path,
        },
        "transcription": {
            "language": tr.get("language"),
            "text": text,
            "segments": tr.get("segments", []),
        },
        "text_emotion": text_emotions,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output


def main():
    parser = argparse.ArgumentParser(description="Grupo B: Audio -> Whisper -> Text Emotion")
    parser.add_argument("--video", required=True, help="Path to input interview video (.mp4/.avi)")
    parser.add_argument("--out", default="outputs/group_b", help="Output directory")
    parser.add_argument("--whisper_model", default="small", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size")
    parser.add_argument("--lang", default="es", help="Language code (e.g., es, en). Use 'es' for Spanish.")
    args = parser.parse_args()

    result = run(args.video, args.out, args.whisper_model, args.lang)

    # Short console summary
    print("\n✅ Grupo B ejecutado correctamente")
    print(f"Texto (primeros 200 chars): {result['transcription']['text'][:200]!r}")
    top3 = sorted(result["text_emotion"].items(), key=lambda x: x[1], reverse=True)[:3]
    print("Top-3 emociones (texto):", top3)


if __name__ == "__main__":
    main()
