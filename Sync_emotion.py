"""
Script para sincronizar análisis de emociones de video con audio.
VERSIÓN CORREGIDA - Calcula timestamps correctamente para frames muestreados.

FIXES:
- Calcula timestamps basándose en la distribución real de frames
- Maneja correctamente frames extraídos con intervalo
- Mejor matching de segmentos de audio
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def get_video_fps_and_duration(video_path: str) -> tuple[float, float, int]:
    """
    Obtiene el FPS, duración y frames totales del video.
    
    Returns:
        (fps, duration_seconds, total_frames)
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return fps, duration, frame_count
    except ImportError:
        print("  OpenCV no disponible, intentando con moviepy...")
    
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(video_path)
        fps = clip.fps
        duration = clip.duration
        frame_count = int(fps * duration)
        clip.close()
        return fps, duration, frame_count
    except ImportError:
        raise RuntimeError(
            "Instala opencv-python o moviepy:\n"
            "pip install opencv-python"
        )


def calculate_frame_timestamps_corrected(
    num_frames_analyzed: int,
    total_frames_video: int,
    fps: float,
    duration: float
) -> List[float]:
    """
    Calcula timestamps CORRECTOS para frames muestreados.
    
    Args:
        num_frames_analyzed: Número de frames que fueron analizados
        total_frames_video: Total de frames del video original
        fps: FPS del video
        duration: Duración del video
    
    Returns:
        Lista de timestamps en segundos para cada frame analizado
    """
    if num_frames_analyzed >= total_frames_video:
        # Todos los frames fueron analizados
        return [i / fps for i in range(num_frames_analyzed)]
    
    # Calcular el intervalo de muestreo
    interval = total_frames_video / num_frames_analyzed
    
    # Calcular timestamp para cada frame analizado
    timestamps = []
    for i in range(num_frames_analyzed):
        frame_index_in_video = int(i * interval)
        timestamp = frame_index_in_video / fps
        timestamps.append(timestamp)
    
    return timestamps


def find_audio_segment_for_timestamp(
    timestamp: float,
    audio_segments: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Encuentra el segmento de audio para un timestamp.
    """
    for segment in audio_segments:
        start = segment.get("start", 0)
        end = segment.get("end", float('inf'))
        if start <= timestamp <= end:
            return segment
    return None


def normalize_emotion_names(emotions: Dict[str, float]) -> Dict[str, float]:
    """
    Normaliza nombres de emociones.
    
    Visual: angry, disgust, fear, happy, sad, surprise, neutral
    Audio: joy, fear, sadness, neutral, disgust, anger, surprise
    
    Mapeo:
    - happy -> joy
    - sad -> sadness
    - angry -> anger
    """
    mapping = {
        "happy": "joy",
        "sad": "sadness",
        "angry": "anger",
    }
    
    normalized = {}
    for emo, score in emotions.items():
        normalized_name = mapping.get(emo, emo)
        normalized[normalized_name] = score
    
    return normalized


def sync_emotions(
    frames_json_path: str,
    audio_json_path: str,
    video_path: str,
    output_dir: str = "outputs/sync"
) -> Dict[str, Any]:
    """
    Sincroniza análisis de emociones visuales y de audio.
    VERSIÓN CORREGIDA.
    """
    print(" Iniciando sincronización de emociones (VERSIÓN CORREGIDA)...")
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Cargar JSONs
    print(f" Cargando {frames_json_path}...")
    with open(frames_json_path, 'r', encoding='utf-8') as f:
        frames_data = json.load(f)
    
    print(f" Cargando {audio_json_path}...")
    with open(audio_json_path, 'r', encoding='utf-8') as f:
        audio_data = json.load(f)
    
    # Obtener información del video
    print(f" Analizando video: {video_path}...")
    fps, duration, total_frames_video = get_video_fps_and_duration(video_path)
    print(f"   FPS: {fps:.2f}")
    print(f"   Duración: {duration:.2f} segundos")
    print(f"   Frames totales en video: {total_frames_video}")
    
    # Extraer datos
    frames_analysis = frames_data.get("analisis_por_frame", [])
    audio_segments = audio_data.get("transcription", {}).get("segments", [])
    
    num_frames_analyzed = len(frames_analysis)
    print(f" Frames analizados: {num_frames_analyzed}")
    print(f" Segmentos de audio: {len(audio_segments)}")
    
    # Calcular el intervalo de muestreo
    if num_frames_analyzed > 0:
        interval = total_frames_video / num_frames_analyzed
        print(f"   Intervalo de muestreo: ~{interval:.1f} frames")
        print(f"   (cada {interval:.1f} frames del video original)")
    
    # Calcular timestamps CORRECTOS
    print(" Calculando timestamps correctos...")
    frame_timestamps = calculate_frame_timestamps_corrected(
        num_frames_analyzed,
        total_frames_video,
        fps,
        duration
    )
    
    # Verificar cobertura
    if frame_timestamps:
        print(f"   Primer frame: {frame_timestamps[0]:.2f}s")
        print(f"   Último frame: {frame_timestamps[-1]:.2f}s")
        print(f"   Cobertura: TODO el video ")
    
    # Sincronizar
    print(" Sincronizando emociones con audio...")
    synchronized_data = []
    
    # Contador de segmentos
    segments_matched = defaultdict(int)
    
    for i, frame in enumerate(frames_analysis):
        timestamp = frame_timestamps[i]
        frame_num = frame.get("frame_number", i)
        
        # Emociones visuales
        visual_emotions = {}
        if frame.get("rostros") and len(frame["rostros"]) > 0:
            visual_emotions = normalize_emotion_names(
                frame["rostros"][0].get("emociones", {})
            )
        
        # Encontrar segmento de audio
        audio_segment = find_audio_segment_for_timestamp(timestamp, audio_segments)
        
        audio_emotions = {}
        audio_text = ""
        segment_info = "none"
        
        if audio_segment:
            audio_emotions = audio_segment.get("emotions", {})
            audio_text = audio_segment.get("text", "").strip()
            segment_start = audio_segment.get("start", 0)
            segment_info = f"seg_{segment_start:.1f}s"
            segments_matched[segment_info] += 1
        
        sync_entry = {
            "frame_number": frame_num,
            "timestamp": round(timestamp, 2),
            "visual_emotions": visual_emotions,
            "audio_emotions": audio_emotions,
            "audio_text": audio_text,
            "num_faces": frame.get("num_rostros", 0),
        }
        
        synchronized_data.append(sync_entry)
    
    # Mostrar estadísticas de matching
    print("\n Estadísticas de matching:")
    print(f"   Segmentos de audio: {len(audio_segments)}")
    print(f"   Segmentos matched: {len(segments_matched)}")
    for seg, count in sorted(segments_matched.items()):
        print(f"      {seg}: {count} frames")
    
    # Calcular estadísticas globales
    print(" Calculando estadísticas...")
    
    visual_emotion_totals = defaultdict(float)
    visual_count = 0
    for entry in synchronized_data:
        if entry["visual_emotions"]:
            for emo, score in entry["visual_emotions"].items():
                visual_emotion_totals[emo] += score
            visual_count += 1
    
    visual_emotion_avg = {
        emo: score / visual_count if visual_count > 0 else 0
        for emo, score in visual_emotion_totals.items()
    }
    
    audio_emotion_totals = defaultdict(float)
    audio_count = 0
    for entry in synchronized_data:
        if entry["audio_emotions"]:
            for emo, score in entry["audio_emotions"].items():
                audio_emotion_totals[emo] += score
            audio_count += 1
    
    audio_emotion_avg = {
        emo: score / audio_count if audio_count > 0 else 0
        for emo, score in audio_emotion_totals.items()
    }
    
    # Resultado final
    result = {
        "metadata": {
            "sync_timestamp": datetime.now().isoformat(),
            "video_path": video_path,
            "fps": fps,
            "duration_seconds": duration,
            "total_frames_video": total_frames_video,
            "total_frames_analyzed": num_frames_analyzed,
            "sampling_interval": total_frames_video / num_frames_analyzed if num_frames_analyzed > 0 else 0,
            "total_audio_segments": len(audio_segments),
        },
        "synchronized_timeline": synchronized_data,
        "statistics": {
            "visual_emotions_avg": visual_emotion_avg,
            "audio_emotions_avg": audio_emotion_avg,
            "segments_coverage": dict(segments_matched),
        },
        "original_data": {
            "frames_summary": frames_data.get("resumen", {}),
            "audio_summary": audio_data.get("summary", {}),
            "audio_segments": audio_segments,  # Incluir segmentos para referencia
        }
    }
    
    # Guardar resultado
    output_file = out_path / "synchronized_emotions.json"
    print(f"Guardando resultado en: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(" Sincronización completada!")
    return result


def plot_comparison(
    synchronized_data: List[Dict[str, Any]],
    output_path: str = "outputs/sync/emotion_comparison.png"
):
    """Visualización comparativa de emociones."""
    print(" Generando visualización comparativa...")
    
    timestamps = [entry["timestamp"] for entry in synchronized_data]
    
    all_emotions = set()
    for entry in synchronized_data:
        all_emotions.update(entry["visual_emotions"].keys())
        all_emotions.update(entry["audio_emotions"].keys())
    
    all_emotions = sorted(all_emotions)
    
    if not all_emotions:
        print(" No hay emociones para graficar")
        return
    
    fig, axes = plt.subplots(len(all_emotions), 1, figsize=(14, 2.5 * len(all_emotions)))
    
    if len(all_emotions) == 1:
        axes = [axes]
    
    colors = {
        "visual": "#FF6B6B",
        "audio": "#4ECDC4",
    }
    
    for idx, emotion in enumerate(all_emotions):
        ax = axes[idx]
        
        visual_scores = [
            entry["visual_emotions"].get(emotion, 0) 
            for entry in synchronized_data
        ]
        
        audio_scores = [
            entry["audio_emotions"].get(emotion, 0) * 100
            for entry in synchronized_data
        ]
        
        ax.plot(timestamps, visual_scores, 
                label=f'{emotion.capitalize()} (Visual)', 
                color=colors["visual"], linewidth=2, alpha=0.8)
        ax.plot(timestamps, audio_scores, 
                label=f'{emotion.capitalize()} (Audio)', 
                color=colors["audio"], linewidth=2, alpha=0.8)
        
        ax.set_ylabel('Probabilidad (%)', fontsize=10)
        ax.set_title(f'Emoción: {emotion.upper()}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(timestamps[0], timestamps[-1])
        
        if idx < len(all_emotions) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Tiempo (segundos)', fontsize=11)
    
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f" Gráfica guardada en: {output_path}")
    plt.close()


def plot_timeline_with_text(
    synchronized_data: List[Dict[str, Any]],
    output_path: str = "outputs/sync/timeline_with_text.png"
):
    """Timeline con emociones y texto."""
    print(" Generando timeline con texto...")
    
    timestamps = [entry["timestamp"] for entry in synchronized_data]
    
    visual_dominant = []
    audio_dominant = []
    
    for entry in synchronized_data:
        if entry["visual_emotions"]:
            v_emo = max(entry["visual_emotions"].items(), key=lambda x: x[1])
            visual_dominant.append(v_emo[0])
        else:
            visual_dominant.append("none")
        
        if entry["audio_emotions"]:
            a_emo = max(entry["audio_emotions"].items(), key=lambda x: x[1])
            audio_dominant.append(a_emo[0])
        else:
            audio_dominant.append("none")
    
    emotion_colors = {
        "joy": "#FFD93D",
        "sadness": "#6C5CE7",
        "anger": "#FF6B6B",
        "fear": "#A29BFE",
        "disgust": "#00B894",
        "surprise": "#FD79A8",
        "neutral": "#B2BEC3",
        "none": "#DFE6E9"
    }
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    
    for i in range(len(timestamps) - 1):
        color = emotion_colors.get(visual_dominant[i], "#DFE6E9")
        ax1.axvspan(timestamps[i], timestamps[i+1], color=color, alpha=0.7)
    
    ax1.set_ylabel('Emociones Visuales', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.grid(True, axis='x', alpha=0.3)
    
    for i in range(len(timestamps) - 1):
        color = emotion_colors.get(audio_dominant[i], "#DFE6E9")
        ax2.axvspan(timestamps[i], timestamps[i+1], color=color, alpha=0.7)
    
    prev_text = ""
    for entry in synchronized_data:
        if entry["audio_text"] and entry["audio_text"] != prev_text:
            text_display = entry["audio_text"][:50] + "..." if len(entry["audio_text"]) > 50 else entry["audio_text"]
            ax2.text(entry["timestamp"], 0.5, text_display,
                    fontsize=8, rotation=0, ha='left', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            prev_text = entry["audio_text"]
    
    ax2.set_ylabel('Emociones de Audio', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Tiempo (segundos)', fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.grid(True, axis='x', alpha=0.3)
    
    legend_patches = [
        mpatches.Patch(color=color, label=emo.capitalize())
        for emo, color in emotion_colors.items() if emo != "none"
    ]
    fig.legend(handles=legend_patches, loc='upper center', 
              ncol=7, bbox_to_anchor=(0.5, 0.98), fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f" Timeline guardado en: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Sincronizar emociones de video con audio (VERSIÓN CORREGIDA)"
    )
    parser.add_argument(
        "--frames",
        required=True,
        help="Ruta al JSON de análisis de frames"
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Ruta al JSON de análisis de audio"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Ruta al video original"
    )
    parser.add_argument(
        "--out",
        default="outputs/sync",
        help="Directorio de salida"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generar gráficas"
    )
    
    args = parser.parse_args()
    
    result = sync_emotions(
        frames_json_path=args.frames,
        audio_json_path=args.audio,
        video_path=args.video,
        output_dir=args.out
    )
    
    if args.plot:
        plot_comparison(
            result["synchronized_timeline"],
            output_path=f"{args.out}/emotion_comparison.png"
        )
        plot_timeline_with_text(
            result["synchronized_timeline"],
            output_path=f"{args.out}/timeline_with_text.png"
        )
    
    # Resumen
    print("\n" + "="*60)
    print(" RESUMEN DE SINCRONIZACIÓN")
    print("="*60)
    print(f" Frames sincronizados: {result['metadata']['total_frames_analyzed']}")
    print(f" Frames totales en video: {result['metadata']['total_frames_video']}")
    print(f" Intervalo de muestreo: ~{result['metadata']['sampling_interval']:.1f} frames")
    print(f" Segmentos de audio: {result['metadata']['total_audio_segments']}")
    print(f" Duración: {result['metadata']['duration_seconds']:.2f} segundos")
    
    print("\n Cobertura de segmentos:")
    for seg, count in sorted(result['statistics']['segments_coverage'].items()):
        print(f"   {seg}: {count} frames")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()