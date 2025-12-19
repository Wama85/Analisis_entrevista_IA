from pathlib import Path
import subprocess
import logging
import sys
import argparse
import os
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def run_command(cmd: list, descripcion: str):
    logger.info(descripcion)
    logger.debug(f"Comando: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Error en: {descripcion}")
        logger.error(result.stderr)
        sys.exit(1)

    logger.info(f"{descripcion} completado")


def run_facial_emotion_directly(video_path: str, out_dir: str = "outputs/facial_emotion"):
    """
    Ejecuta el análisis facial directamente sin subprocess
    Usando la nueva estructura de video_emotion.py
    """
    logger.info("Iniciando análisis de emociones faciales (nueva versión)")

    try:
        # Importar y ejecutar la función run del nuevo script
        import sys
        sys.path.insert(0, str(Path(__file__).parent))

        from cnn_emotions.facial_emotion import run as run_facial

        result = run_facial(
            video_path=video_path,
            out_dir=out_dir,
            fps_extraccion=5,
            generate_report=True,
            show_summary=False
        )

        # Encontrar el archivo JSON generado
        video_name = Path(video_path).stem
        output_base = Path(out_dir)

        # Buscar el directorio específico creado por el script
        pattern = f"{video_name}_*"
        matching_dirs = list(output_base.glob(pattern))

        if matching_dirs:
            latest_dir = max(matching_dirs, key=lambda x: x.stat().st_mtime)
            json_path = latest_dir / "reports" / f"emociones_{video_name}.json"

            if json_path.exists():
                logger.info(f"Reporte facial encontrado: {json_path}")
                return json_path

        # Si no lo encuentra, buscar alternativas
        json_candidates = list(output_base.rglob(f"*{video_name}*.json"))
        if json_candidates:
            return max(json_candidates, key=lambda x: x.stat().st_mtime)

    except Exception as e:
        logger.error(f"Error en análisis facial directo: {e}")
        # Fallback al comando anterior
        return None

    return None


def main(video_path: str):
    video = Path(video_path)

    if not video.exists():
        logger.error(f"El video no existe: {video}")
        sys.exit(1)

    video_name = video.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 60)
    logger.info("INICIANDO PIPELINE COMPLETO DE ANÁLISIS")
    logger.info(f"Video: {video.name}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info("=" * 60)

    # ===============================
    # 1) Emociones faciales (CNN) - NUEVA VERSIÓN
    # ===============================
    logger.info("\n📹 1) ANÁLISIS DE EMOCIONES FACIALES")

    # Opción 1: Usar la función directa (recomendado)
    facial_json = run_facial_emotion_directly(str(video))

    # Opción 2: Si falla la opción directa, usar el comando antiguo
    if facial_json is None or not facial_json.exists():
        logger.warning("Usando método de comando para análisis facial")

        # Crear directorio para resultados faciales
        facial_output = Path("outputs") / "facial_emotion"
        facial_output.mkdir(parents=True, exist_ok=True)

        run_command(
            [
                "python", "-m", "cnn_emotions.video_emotion",
                "--video", str(video),
                "--out", str(facial_output),
                "--fps", "5",
                "--quiet"
            ],
            "Análisis de emociones faciales"
        )

        # Buscar el JSON generado
        facial_json = facial_output / f"emociones_{video_name}.json"
        if not facial_json.exists():
            # Buscar en subdirectorios
            json_files = list(facial_output.rglob("*.json"))
            if json_files:
                facial_json = max(json_files, key=lambda x: x.stat().st_mtime)
            else:
                logger.error("No se encontró el JSON de análisis facial")
                sys.exit(1)

    logger.info(f"✓ Análisis facial completado: {facial_json}")

    # ===============================
    # 2) Audio → Texto + emociones
    # ===============================
    logger.info("\n🎵 2) ANÁLISIS DE AUDIO Y TEXTO")

    audio_output = Path("outputs") / "audio_text"
    audio_output.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "python", "-m", "audio_text.run",
            "--video", str(video),
            "--out", str(audio_output),
            "--lang", "es",
            "--whisper_model", "small",
            "--no_plot"
        ],
        "Extracción de audio y análisis de texto"
    )

    audio_json = audio_output / f"{video_name}_text_audio.json"
    if not audio_json.exists():
        # Buscar cualquier JSON de audio
        json_files = list(audio_output.rglob("*text_audio.json"))
        if json_files:
            audio_json = max(json_files, key=lambda x: x.stat().st_mtime)
        else:
            logger.error("No se encontró el JSON de análisis de audio")
            sys.exit(1)

    logger.info(f"✓ Análisis de audio completado: {audio_json}")

    # ===============================
    # 3) Sincronización multimodal
    # ===============================
    logger.info("\n🔄 3) SINCRONIZACIÓN MULTIMODAL")

    sync_output = Path("outputs") / "sync"
    sync_output.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "python", "Sync_emotion.py",
            "--frames", str(facial_json),
            "--audio", str(audio_json),
            "--video", str(video),
            "--out", str(sync_output)
        ],
        "Sincronización de emociones faciales y de texto"
    )

    sync_json = sync_output / "synchronized_emotions.json"
    if not sync_json.exists():
        # Buscar JSON de sincronización
        json_files = list(sync_output.rglob("*sync*.json"))
        if json_files:
            sync_json = max(json_files, key=lambda x: x.stat().st_mtime)
        else:
            logger.error("No se encontró el JSON de sincronización")
            sys.exit(1)

    logger.info(f"✓ Sincronización completada: {sync_json}")

    # ===============================
    # 4) Generar insights
    # ===============================
    logger.info("\n📊 4) GENERACIÓN DE INSIGHTS")

    reports_output = Path("outputs") / "reports"
    reports_output.mkdir(parents=True, exist_ok=True)

    insights_file = reports_output / f"{video_name}_insights_{timestamp}.json"

    run_command(
        [
            "python", "reports/generate_insights.py",
            "--sync_json", str(sync_json),
            "--out", str(insights_file)
        ],
        "Análisis de insights y reportes"
    )

    # ===============================
    # 5) Video final anotado
    # ===============================
    logger.info("\n🎬 5) GENERACIÓN DE VIDEO FINAL")

    final_video = sync_output / f"{video_name}_final_{timestamp}.mp4"

    run_command(
        [
            "python", "create_video_with_text.py",
            "--video", str(video),
            "--sync_json", str(sync_json),
            "--out", str(final_video),
            "--format", "mp4",
            "--typing_speed", "0.03",
            "--font_size", "28"
        ],
        "Creación de video final anotado"
    )

    # ===============================
    # 6) Resumen final
    # ===============================
    logger.info("\n" + "=" * 60)
    logger.info("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    logger.info("=" * 60)

    # Mostrar archivos generados
    logger.info("\n📁 ARCHIVOS GENERADOS:")
    logger.info(f"  Facial: {facial_json}")
    logger.info(f"  Audio: {audio_json}")
    logger.info(f"  Sincronización: {sync_json}")
    logger.info(f"  Insights: {insights_file}")
    logger.info(f"  Video final: {final_video}")

    # Abrir el video final si existe
    if final_video.exists():
        logger.info(f"\n🎥 Abriendo video final...")
        try:
            if os.name == 'nt':  # Windows
                os.startfile(final_video)
            elif os.name == 'posix':  # macOS o Linux
                if sys.platform == 'darwin':  # macOS
                    subprocess.run(['open', str(final_video)])
                else:  # Linux
                    subprocess.run(['xdg-open', str(final_video)])
            logger.info("✓ Video abierto exitosamente")
        except Exception as e:
            logger.warning(f"No se pudo abrir el video automáticamente: {e}")
            logger.info(f"Puedes abrirlo manualmente en: {final_video}")
    else:
        logger.warning(f"Video final no encontrado: {final_video}")

    # Mostrar estadísticas rápidas
    try:
        with open(sync_json, 'r', encoding='utf-8') as f:
            sync_data = json.load(f)

        if 'summary' in sync_data:
            summary = sync_data['summary']
            logger.info("\n📈 ESTADÍSTICAS RÁPIDAS:")
            logger.info(f"  Duración total: {summary.get('duration_seconds', 0):.1f}s")
            logger.info(f"  Emoción facial predominante: {summary.get('dominant_facial_emotion', 'N/A')}")
            logger.info(f"  Emoción textual predominante: {summary.get('dominant_text_emotion', 'N/A')}")
            logger.info(f"  Coincidencia emocional: {summary.get('emotional_match_percentage', 0):.1f}%")
    except Exception as e:
        logger.debug(f"No se pudieron cargar estadísticas: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline completo de análisis multimodal (facial + audio + texto)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py --video entrevista.mp4
  python main.py --video /ruta/completa/video.avi
  
Los resultados se guardarán en la carpeta 'outputs/'
        """
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Ruta al video de entrada (MP4, AVI, MOV, etc.)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Habilitar modo debug para más detalles"
    )

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Modo debug activado")

    # Verificar dependencias
    logger.debug("Verificando dependencias...")
    try:
        import cv2
        import numpy as np
        logger.debug("✓ OpenCV y NumPy instalados")
    except ImportError as e:
        logger.error(f"Falta dependencia: {e}")
        logger.info("Instala las dependencias con: pip install opencv-python numpy")
        sys.exit(1)

    main(args.video)