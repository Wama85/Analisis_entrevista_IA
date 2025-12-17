import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# ===============================
# CONFIGURACIÓN DE LOGGING
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def sincronizar_analisis(
        ruta_json_facial: str, 
        ruta_json_texto: str,
        ruta_salida: str = "outputs/analisis_combinado_sincronizado.json"
) -> List[Dict[str, Any]]:
    """
    Sincroniza los análisis de emociones faciales (frame-based)
    con los análisis de emociones textuales (segment-based)
    usando los timestamps en segundos.

    Args:
        ruta_json_facial: Ruta al reporte de facial_emotion.py
        ruta_json_texto: Ruta al reporte de run.py (audio/texto)
        ruta_salida: Ruta para guardar el resultado de la sincronización.

    Returns:
        Lista de registros combinados.
    """
    logger.info("Iniciando Sincronización de Análisis Video y Audio")
    
    try:
        with open(ruta_json_facial, 'r', encoding='utf-8') as f:
            data_facial = json.load(f)
        
        with open(ruta_json_texto, 'r', encoding='utf-8') as f:
            data_texto = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"Error: Archivo no encontrado - {e}")
        return []

    analisis_facial_por_frame = data_facial.get('analisis_por_frame', [])
    segmentos_texto = data_texto.get('transcription', {}).get('segments', [])
    
    if not analisis_facial_por_frame or not segmentos_texto:
        logger.warning("No hay suficientes datos en ambos reportes para sincronizar.")
        return []

    resultados_sincronizados = []

    logger.info(f"Sincronizando {len(analisis_facial_por_frame)} frames con {len(segmentos_texto)} segmentos de audio...")
    
    for frame_analisis in analisis_facial_por_frame:
        
        t_frame = frame_analisis.get('timestamp_seconds')
        
        if t_frame is None:
            continue
        segmento_coincidente = None
        for seg in segmentos_texto:
            start_t = seg.get('start') if seg.get('start') is not None else 0.0
            end_t = seg.get('end') if seg.get('end') is not None else 0.0
            
            if float(start_t) <= t_frame < float(end_t): 
                segmento_coincidente = seg
                break

        registro_combinado = {
            'tiempo_segundos': t_frame,
            'analisis_facial': frame_analisis.get('rostros', []),
            'analisis_texto': None,
            'segmento_audio_referencia': None 
        }
        
        if segmento_coincidente:
            registro_combinado['analisis_texto'] = {
                'texto': segmento_coincidente.get('text', '').strip(),
                'emociones_texto': segmento_coincidente.get('emotions', {})
            }
            registro_combinado['segmento_audio_referencia'] = {
                'start': segmento_coincidente.get('start'),
                'end': segmento_coincidente.get('end')
            }
        
        resultados_sincronizados.append(registro_combinado)

    ruta_salida_path = Path(ruta_salida)
    ruta_salida_path.parent.mkdir(parents=True, exist_ok=True) 

    with open(ruta_salida_path, 'w', encoding='utf-8') as f:
        json.dump(resultados_sincronizados, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Sincronización completada. Resultados guardados en: {ruta_salida_path}")
    return resultados_sincronizados


def main():
    """Ejecución de prueba/ejemplo"""
    logging.info("="*60)
    logging.info("SCRIPT DE SINCRONIZACIÓN DE ANÁLISIS MULTIMODAL")
    logging.info("="*60)

    TEXT_AUDIO_JSON = "outputs/audio_text/mivideo_text_audio.json"
    
    FACIAL_JSON = "resultados_emociones/reports/emotion_analysis_report.json"
    
    sincronizar_analisis(
        ruta_json_facial=FACIAL_JSON, 
        ruta_json_texto=TEXT_AUDIO_JSON
    )

if __name__ == "__main__":
    main()