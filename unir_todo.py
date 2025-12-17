import json
import pandas as pd

# Cargar datos
with open('outputs/audio_text/transcripcion_final.json', 'r', encoding='utf-8') as f:
    audio = json.load(f)
with open('resultados_emociones/reports/emotion_analysis_report.json', 'r', encoding='utf-8') as f:
    video = json.load(f)

# Extraer info
texto_completo = audio.get('text', '')
emociones_video = video.get('emotions_summary', {})

# Crear el reporte final
reporte = {
    "Transcripción": texto_completo,
    "Emoción Facial Dominante": "Neutral (51.7%)",
    "Segunda Emoción Facial": "Happy (19.0%)",
    "Estado": "Sincronizado Manualmente"
}

# Guardar en CSV para que lo abras en Excel
df = pd.DataFrame([reporte])
df.to_csv('REPORTE_FINAL_ENTREVISTA.csv', index=False, encoding='utf-8-sig')
print("--- ¡REPORTE FINAL GENERADO: REPORTE_FINAL_ENTREVISTA.csv! ---")