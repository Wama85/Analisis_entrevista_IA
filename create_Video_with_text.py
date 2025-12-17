"""
Script para generar video con texto sincronizado Y AUDIO ORIGINAL.
VERSIÓN 4.0 - CON AUDIO DEL VIDEO ORIGINAL

NUEVAS CARACTERÍSTICAS v4.0:
- ✅ Copia el audio del video original al video final
- ✅ Sincronización perfecta de audio y video
- ✅ Dos métodos: ffmpeg (rápido) o moviepy (más compatible)
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess
import shutil


def load_synchronized_data(json_path: str) -> Dict[str, Any]:
    """Carga el JSON de datos sincronizados."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_audio_segments_from_timeline(timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extrae segmentos únicos de audio desde el timeline sincronizado."""
    segments = []
    current_text = None
    
    for entry in timeline:
        text = entry.get("audio_text", "").strip()
        timestamp = entry.get("timestamp", 0)
        
        if text and text != current_text:
            if current_text is not None and segments:
                segments[-1]["end"] = timestamp
            
            segments.append({
                "start": timestamp,
                "end": float('inf'),
                "text": text,
                "emotions": entry.get("audio_emotions", {})
            })
            
            current_text = text
    
    return segments


def find_audio_segment_for_timestamp(timestamp: float, segments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Encuentra el segmento de audio que corresponde al timestamp dado."""
    for segment in segments:
        start = segment.get("start", 0)
        end = segment.get("end", float('inf'))
        
        if start <= timestamp <= end:
            return segment
    
    return None


def get_visual_data_for_timestamp(timestamp: float, timeline: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Obtiene los datos visuales (emociones faciales) para un timestamp."""
    if not timeline:
        return None
    
    closest_entry = min(timeline, key=lambda x: abs(x['timestamp'] - timestamp))
    
    if abs(closest_entry['timestamp'] - timestamp) > 0.5:
        return None
    
    return closest_entry


def get_dominant_emotion(emotions: Dict[str, float]) -> Tuple[str, float]:
    """Obtiene la emoción dominante."""
    if not emotions:
        return "neutral", 0.0
    return max(emotions.items(), key=lambda x: x[1])


def get_emotion_color(emotion: str) -> Tuple[int, int, int]:
    """Retorna un color BGR para cada emoción."""
    colors = {
        "joy": (0, 215, 255),
        "sadness": (255, 92, 108),
        "anger": (0, 0, 255),
        "fear": (255, 158, 162),
        "disgust": (148, 185, 0),
        "surprise": (168, 121, 253),
        "neutral": (195, 190, 178),
    }
    return colors.get(emotion.lower(), (200, 200, 200))


def wrap_text(text: str, max_width: int, font, draw) -> List[str]:
    """Divide el texto en múltiples líneas según el ancho máximo."""
    if not text:
        return []
    
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        width = bbox[2] - bbox[0]
        
        if width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


def add_text_to_frame(
    frame: np.ndarray,
    text: str,
    emotion: str,
    emotion_score: float,
    progress: float = 1.0,
    font_size: int = 24,
    show_no_data: bool = False,
    show_timestamp: bool = False,
    timestamp: float = 0.0
) -> np.ndarray:
    """Agrega texto y emoción al frame con efecto de escritura."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    # Cargar fuente
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        font_emotion = ImageFont.truetype("arial.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_emotion = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    h, w = frame.shape[:2]
    
    # === ZONA DE TEXTO (parte inferior) ===
    text_area_height = 150
    text_y_start = h - text_area_height
    
    # Fondo semi-transparente
    overlay = pil_img.copy()
    draw_overlay = ImageDraw.Draw(overlay)
    draw_overlay.rectangle(
        [(0, text_y_start), (w, h)],
        fill=(0, 0, 0, 180)
    )
    pil_img = Image.blend(pil_img, overlay, 0.7)
    draw = ImageDraw.Draw(pil_img)
    
    # Texto
    if text:
        max_text_width = w - 40
        lines = wrap_text(text, max_text_width, font, draw)
        
        if progress < 1.0:
            full_text = ' '.join(lines)
            chars_to_show = int(len(full_text) * progress)
            partial_text = full_text[:chars_to_show]
            lines = wrap_text(partial_text, max_text_width, font, draw)
        
        y_offset = text_y_start + 20
        for line in lines:
            draw.text((20, y_offset), line, font=font, fill=(255, 255, 255))
            y_offset += font_size + 5
    elif show_no_data:
        draw.text((20, text_y_start + 60), 
                 "Sin datos de audio en este segmento", 
                 font=font_small, fill=(150, 150, 150))
    
    # === EMOCIÓN ===
    if emotion and emotion_score > 0:
        emotion_color = get_emotion_color(emotion)
        emotion_text = f"{emotion.upper()}"
        emotion_score_text = f"{emotion_score:.1f}%"
        
        draw_overlay = ImageDraw.Draw(pil_img)
        draw_overlay.rectangle(
            [(10, 10), (250, 80)],
            fill=(emotion_color[2], emotion_color[1], emotion_color[0], 200),
            outline=(255, 255, 255),
            width=2
        )
        
        draw.text((20, 20), emotion_text, font=font_emotion, fill=(255, 255, 255))
        draw.text((20, 50), emotion_score_text, font=font_emotion, fill=(255, 255, 255))
    
    # === TIMESTAMP (debug) ===
    if show_timestamp:
        timestamp_text = f"T: {timestamp:.2f}s"
        draw.text((w - 120, 20), timestamp_text, font=font_small, fill=(200, 200, 200))
    
    frame_with_text = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    return frame_with_text


def add_audio_with_ffmpeg(video_no_audio: str, video_original: str, output_with_audio: str) -> bool:
    """
    Agrega el audio del video original al video generado usando ffmpeg.
    
    Args:
        video_no_audio: Video generado sin audio
        video_original: Video original con audio
        output_with_audio: Archivo de salida con audio
    
    Returns:
        True si exitoso, False si error
    """
    try:
        # Verificar si ffmpeg está disponible
        if not shutil.which("ffmpeg"):
            print("   ⚠️  ffmpeg no encontrado, intentando con moviepy...")
            return False
        
        print("   🔊 Copiando audio del video original con ffmpeg...")
        
        # Comando ffmpeg para copiar video y audio
        cmd = [
            "ffmpeg",
            "-i", video_no_audio,  # Video sin audio
            "-i", video_original,  # Video original con audio
            "-map", "0:v:0",  # Video del primer archivo
            "-map", "1:a:0",  # Audio del segundo archivo
            "-c:v", "copy",  # Copiar video sin recodificar
            "-c:a", "aac",  # Codificar audio en AAC
            "-shortest",  # Usar la duración del más corto
            "-y",  # Sobrescribir si existe
            output_with_audio
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            print("   ✅ Audio agregado exitosamente")
            return True
        else:
            print(f"   ❌ Error en ffmpeg: {result.stderr[:200]}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error al agregar audio: {e}")
        return False


def add_audio_with_moviepy(video_no_audio: str, video_original: str, output_with_audio: str) -> bool:
    """
    Agrega el audio del video original usando moviepy.
    
    Args:
        video_no_audio: Video generado sin audio
        video_original: Video original con audio
        output_with_audio: Archivo de salida con audio
    
    Returns:
        True si exitoso, False si error
    """
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip
        
        print("   🔊 Copiando audio del video original con moviepy...")
        
        # Cargar videos
        video_clip = VideoFileClip(video_no_audio)
        original_clip = VideoFileClip(video_original)
        
        # Extraer audio del original
        audio = original_clip.audio
        
        if audio is None:
            print("   ⚠️  El video original no tiene audio")
            # Simplemente copiar el video sin audio
            shutil.copy(video_no_audio, output_with_audio)
            return True
        
        # Agregar audio al nuevo video
        final_clip = video_clip.set_audio(audio)
        
        # Guardar
        final_clip.write_videofile(
            output_with_audio,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            logger=None
        )
        
        # Limpiar
        video_clip.close()
        original_clip.close()
        final_clip.close()
        
        print("   ✅ Audio agregado exitosamente")
        return True
        
    except ImportError:
        print("   ❌ moviepy no está instalado")
        print("   → Instala con: pip install moviepy")
        return False
    except Exception as e:
        print(f"   ❌ Error al agregar audio: {e}")
        return False


def create_video_with_text(
    video_path: str,
    sync_json_path: str,
    output_path: str = "outputs/sync/video_with_text.mp4",
    output_format: str = "mp4",
    typing_speed: float = 0.05,
    font_size: int = 28,
    debug: bool = False,
    include_audio: bool = True
):
    """
    Crea un video con texto sincronizado Y AUDIO ORIGINAL.
    VERSIÓN 4.0 - CON AUDIO.
    """
    print("🎬 Iniciando creación de video con texto Y AUDIO (VERSIÓN 4.0)...")
    
    # Cargar datos sincronizados
    print(f"📂 Cargando datos sincronizados: {sync_json_path}")
    data = load_synchronized_data(sync_json_path)
    timeline = data["synchronized_timeline"]
    fps = data["metadata"]["fps"]
    
    # Extraer segmentos de audio desde el timeline
    print("   🔍 Extrayendo segmentos de audio...")
    audio_segments = extract_audio_segments_from_timeline(timeline)
    
    print(f"   📝 Segmentos de texto encontrados: {len(audio_segments)}")
    
    if debug:
        print("\n   📋 Segmentos de audio:")
        for i, seg in enumerate(audio_segments):
            start = seg.get('start', 0)
            end = seg.get('end', float('inf'))
            text = seg.get('text', '')
            end_str = f"{end:.1f}s" if end != float('inf') else "fin"
            print(f"     [{i}] {start:.1f}s - {end_str}: {text[:60]}...")
        print()
    
    # Abrir video
    print(f"🎥 Abriendo video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"   Resolución: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duración: {duration:.2f} segundos")
    
    # Crear video temporal sin audio
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_format == "mp4":
        # Video temporal sin audio
        temp_video_path = output_path.parent / f"temp_{output_path.name}"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))
    elif output_format == "gif":
        frames_for_gif = []
    else:
        raise ValueError(f"Formato no soportado: {output_format}")
    
    print(f"📹 Generando video con texto sincronizado...")
    
    # Variables de estado
    current_text = ""
    text_start_time = 0
    last_emotion = "neutral"
    last_emotion_score = 0.0
    
    # Contadores
    frame_idx = 0
    frames_written = 0
    text_changes = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_idx / fps
        
        # Buscar segmento de audio
        current_audio_segment = find_audio_segment_for_timestamp(timestamp, audio_segments)
        
        if current_audio_segment:
            new_text = current_audio_segment.get("text", "").strip()
            
            if new_text and new_text != current_text:
                current_text = new_text
                text_start_time = timestamp
                text_changes += 1
                
                if debug and text_changes <= 10:
                    print(f"   📝 Cambio de texto en {timestamp:.2f}s: {current_text[:60]}...")
        
        # Buscar datos visuales
        visual_data = get_visual_data_for_timestamp(timestamp, timeline)
        
        if visual_data and visual_data.get("visual_emotions"):
            last_emotion, last_emotion_score = get_dominant_emotion(visual_data["visual_emotions"])
        
        # Calcular progreso de escritura
        if current_text:
            time_since_start = timestamp - text_start_time
            chars_per_second = 1.0 / typing_speed if typing_speed > 0 else float('inf')
            chars_shown = int(time_since_start * chars_per_second)
            progress = min(chars_shown / len(current_text), 1.0) if len(current_text) > 0 else 1.0
        else:
            progress = 1.0
        
        # Agregar texto al frame
        frame_with_text = add_text_to_frame(
            frame,
            current_text,
            last_emotion,
            last_emotion_score,
            progress,
            font_size,
            show_no_data=not current_text,
            show_timestamp=debug,
            timestamp=timestamp
        )
        
        # Escribir frame
        if output_format == "mp4":
            out.write(frame_with_text)
        elif output_format == "gif":
            frames_for_gif.append(cv2.cvtColor(frame_with_text, cv2.COLOR_BGR2RGB))
        
        frames_written += 1
        frame_idx += 1
        
        # Progreso
        if frame_idx % 30 == 0:
            progress_pct = (frame_idx / total_frames) * 100
            print(f"   Progreso: {progress_pct:.1f}% ({frame_idx}/{total_frames}) - Cambios de texto: {text_changes}")
    
    # Liberar recursos
    cap.release()
    
    if output_format == "mp4":
        out.release()
        print(f"✅ Video sin audio creado: {temp_video_path}")
        
        # Agregar audio del video original
        if include_audio:
            print("\n🔊 AGREGANDO AUDIO DEL VIDEO ORIGINAL...")
            
            # Intentar con ffmpeg primero (más rápido)
            success = add_audio_with_ffmpeg(
                str(temp_video_path),
                video_path,
                str(output_path)
            )
            
            # Si ffmpeg falla, intentar con moviepy
            if not success:
                success = add_audio_with_moviepy(
                    str(temp_video_path),
                    video_path,
                    str(output_path)
                )
            
            if success:
                # Eliminar video temporal
                temp_video_path.unlink()
                print(f"✅ Video final con audio: {output_path}")
            else:
                print(f"⚠️  No se pudo agregar audio")
                print(f"   Video sin audio disponible en: {temp_video_path}")
                # Renombrar temp a final
                temp_video_path.rename(output_path)
        else:
            # Renombrar temp a final
            temp_video_path.rename(output_path)
            print(f"✅ Video creado: {output_path}")
            
    elif output_format == "gif":
        print("🎨 Generando GIF...")
        from PIL import Image
        
        pil_frames = [Image.fromarray(f) for f in frames_for_gif]
        gif_path = output_path.with_suffix('.gif')
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / fps),
            loop=0
        )
        print(f"✅ GIF creado: {gif_path}")
    
    # Estadísticas finales
    print(f"\n📊 ESTADÍSTICAS FINALES:")
    print(f"   Frames procesados: {frames_written}/{total_frames}")
    print(f"   Cambios de texto detectados: {text_changes}")
    print(f"   Segmentos de texto: {len(audio_segments)}")
    print(f"   Audio incluido: {'✅ Sí' if include_audio and output_format == 'mp4' else '❌ No'}")
    
    if text_changes == 0:
        print(f"   ⚠️  ADVERTENCIA: No se detectaron cambios de texto")
    elif text_changes < len(audio_segments):
        print(f"   ⚠️  Se esperaban {len(audio_segments)} cambios pero solo se detectaron {text_changes}")
    else:
        print(f"   ✅ Todos los segmentos fueron procesados correctamente")
    
    print("🎉 Generación completada!")


def main():
    parser = argparse.ArgumentParser(
        description="Crear video con texto sincronizado Y AUDIO (V4.0)"
    )
    parser.add_argument("--video", required=True, help="Ruta al video original")
    parser.add_argument("--sync_json", required=True, help="Ruta al JSON sincronizado")
    parser.add_argument("--out", default="outputs/sync/video_with_audio.mp4", help="Ruta de salida")
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4", help="Formato de salida")
    parser.add_argument("--typing_speed", type=float, default=0.03, help="Velocidad de escritura")
    parser.add_argument("--font_size", type=int, default=28, help="Tamaño de fuente")
    parser.add_argument("--no_audio", action="store_true", help="No incluir audio (solo video)")
    parser.add_argument("--debug", action="store_true", help="Modo debug")
    
    args = parser.parse_args()
    
    create_video_with_text(
        args.video,
        args.sync_json,
        args.out,
        args.format,
        args.typing_speed,
        args.font_size,
        args.debug,
        include_audio=not args.no_audio
    )
    
    print("\n" + "="*70)
    print("✨ PROCESO COMPLETADO - VIDEO CON TEXTO Y AUDIO")
    print("="*70)


if __name__ == "__main__":
    main()