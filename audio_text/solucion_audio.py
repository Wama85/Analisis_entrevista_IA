import speech_recognition as sr
import json
import os

r = sr.Recognizer()
audio_file = 'audio_final_directo.wav'

print("--- Escuchando audio... ---")
try:
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    
    # Usamos el motor de Google para transcribir
    text = r.recognize_google(audio, language='es-ES')
    print(f"Texto detectado: {text}")
    
    # Creamos el formato que el sincronizador espera
    res = {
        'text': text, 
        'segments': [{'start': 0.0, 'end': 32.0, 'text': text}]
    }
    
    # Guardamos en la carpeta de outputs
    output_path = 'outputs/audio_text/transcripcion_final.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
        
    print("--- ¡GUARDADO CON ÉXITO! ---")

except Exception as e:
    print(f"Error: {e}")