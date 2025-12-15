"""
Análisis de Evolución de Emociones en Video
Extrae múltiples frames y muestra cómo cambian las emociones
"""

import cv2
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

class VideoEmotionAnalyzer:
    """
    Analiza emociones a lo largo de un video completo
    """
    
    def __init__(self, video_path):
        """
        Inicializa el analizador
        
        Args:
            video_path: Ruta al video a analizar
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        # Obtener información del video
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.total_frames / self.fps
        
        print(f"📹 Video cargado: {video_path}")
        print(f"   Duración: {self.duration:.2f} segundos")
        print(f"   FPS: {self.fps}")
        print(f"   Total frames: {self.total_frames}")
        
        self.results = []
        
    def analyze_frames(self, num_frames=10, frame_interval=None):
        """
        Analiza un número específico de frames del video
        
        Args:
            num_frames: Número de frames a analizar
            frame_interval: Intervalo entre frames (si es None, se distribuyen uniformemente)
        """
        print(f"\n🔍 Analizando {num_frames} frames del video...")
        
        # Calcular qué frames analizar
        if frame_interval is None:
            # Distribuir frames uniformemente
            frame_indices = np.linspace(0, self.total_frames - 1, num_frames, dtype=int)
        else:
            # Usar intervalo específico
            frame_indices = range(0, self.total_frames, frame_interval)[:num_frames]
        
        for i, frame_idx in enumerate(frame_indices):
            print(f"\n[{i+1}/{num_frames}] Analizando frame {frame_idx}...")
            
            # Posicionar en el frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            
            if not ret:
                print(f"   ⚠️  No se pudo leer el frame {frame_idx}")
                continue
            
            # Tiempo en el video
            time_seconds = frame_idx / self.fps
            
            try:
                # Analizar emociones
                result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                
                if isinstance(result, list):
                    result = result[0]
                
                # Guardar resultado
                emotion_data = {
                    'frame': frame_idx,
                    'time': time_seconds,
                    'dominant_emotion': result['dominant_emotion'],
                    **result['emotion']
                }
                
                self.results.append(emotion_data)
                
                print(f"   ✓ Emoción: {result['dominant_emotion'].upper()}")
                
            except Exception as e:
                print(f"   ✗ Error: {e}")
        
        print(f"\n✅ Análisis completado: {len(self.results)} frames analizados")
    
    def visualize_evolution(self, save_path='evolucion_emociones.png'):
        """
        Visualiza la evolución de las emociones a lo largo del tiempo
        
        Args:
            save_path: Ruta donde guardar el gráfico
        """
        if not self.results:
            print("❌ No hay resultados para visualizar")
            return
        
        print(f"\n📊 Creando visualización de evolución...")
        
        # Convertir a DataFrame
        df = pd.DataFrame(self.results)
        
        # Crear figura grande
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Subplot 1: Líneas de todas las emociones
        ax1 = axes[0]
        emotions = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']
        colors = ['green', 'blue', 'red', 'yellow', 'purple', 'brown', 'gray']
        
        for emotion, color in zip(emotions, colors):
            if emotion in df.columns:
                ax1.plot(df['time'], df[emotion], label=emotion.capitalize(), 
                        marker='o', linewidth=2, color=color)
        
        ax1.set_xlabel('Tiempo (segundos)', fontsize=12)
        ax1.set_ylabel('Score (%)', fontsize=12)
        ax1.set_title('Evolución de Todas las Emociones', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Subplot 2: Barras apiladas
        ax2 = axes[1]
        emotion_data = df[emotions].values
        time_points = df['time'].values
        
        ax2.bar(time_points, emotion_data[:, 0], label=emotions[0].capitalize(), 
               color=colors[0], alpha=0.8)
        
        bottom = emotion_data[:, 0]
        for i in range(1, len(emotions)):
            ax2.bar(time_points, emotion_data[:, i], bottom=bottom, 
                   label=emotions[i].capitalize(), color=colors[i], alpha=0.8)
            bottom += emotion_data[:, i]
        
        ax2.set_xlabel('Tiempo (segundos)', fontsize=12)
        ax2.set_ylabel('Score (%)', fontsize=12)
        ax2.set_title('Distribución de Emociones por Frame', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        
        # Subplot 3: Emoción dominante
        ax3 = axes[2]
        
        # Mapear emociones a números
        emotion_mapping = {emotion: i for i, emotion in enumerate(emotions)}
        dominant_values = [emotion_mapping[e] for e in df['dominant_emotion']]
        
        # Crear gráfico de dispersión con colores
        for emotion, color in zip(emotions, colors):
            mask = df['dominant_emotion'] == emotion
            if mask.any():
                ax3.scatter(df.loc[mask, 'time'], 
                          [emotion_mapping[emotion]] * mask.sum(),
                          c=color, s=200, label=emotion.capitalize(), 
                          alpha=0.7, edgecolors='black', linewidth=1)
        
        ax3.set_xlabel('Tiempo (segundos)', fontsize=12)
        ax3.set_ylabel('Emoción Dominante', fontsize=12)
        ax3.set_title('Emoción Dominante a lo Largo del Tiempo', fontsize=14, fontweight='bold')
        ax3.set_yticks(range(len(emotions)))
        ax3.set_yticklabels([e.capitalize() for e in emotions])
        ax3.legend(loc='upper right')
        ax3.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualización guardada en: {save_path}")
        
        return fig
    
    def generate_report(self, save_path='reporte_emociones.txt'):
        """
        Genera un reporte de texto con estadísticas
        
        Args:
            save_path: Ruta donde guardar el reporte
        """
        if not self.results:
            print("❌ No hay resultados para generar reporte")
            return
        
        df = pd.DataFrame(self.results)
        
        report = []
        report.append("=" * 70)
        report.append("  📊 REPORTE DE ANÁLISIS DE EMOCIONES EN VIDEO")
        report.append("=" * 70)
        report.append(f"\n📹 Video: {self.video_path}")
        report.append(f"⏱️  Duración: {self.duration:.2f} segundos")
        report.append(f"🎞️  Frames analizados: {len(self.results)}")
        
        report.append("\n" + "=" * 70)
        report.append("  📈 ESTADÍSTICAS GENERALES")
        report.append("=" * 70)
        
        # Emoción más frecuente
        emotion_counts = df['dominant_emotion'].value_counts()
        report.append(f"\n🏆 Emoción dominante más frecuente: {emotion_counts.index[0].upper()}")
        report.append(f"   Aparece en {emotion_counts.iloc[0]} de {len(self.results)} frames ({emotion_counts.iloc[0]/len(self.results)*100:.1f}%)")
        
        report.append("\n📊 Distribución de emociones dominantes:")
        for emotion, count in emotion_counts.items():
            percentage = (count / len(self.results)) * 100
            bar = "█" * int(percentage / 2)
            report.append(f"   {emotion:12s}: {bar} {percentage:.1f}% ({count} frames)")
        
        # Promedios
        report.append("\n" + "=" * 70)
        report.append("  📊 SCORES PROMEDIO DE TODAS LAS EMOCIONES")
        report.append("=" * 70 + "\n")
        
        emotions = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']
        for emotion in emotions:
            if emotion in df.columns:
                avg_score = df[emotion].mean()
                max_score = df[emotion].max()
                min_score = df[emotion].min()
                report.append(f"{emotion.capitalize():12s}: Promedio={avg_score:6.2f}%  Máx={max_score:6.2f}%  Mín={min_score:6.2f}%")
        
        # Momentos destacados
        report.append("\n" + "=" * 70)
        report.append("  ⭐ MOMENTOS DESTACADOS")
        report.append("=" * 70 + "\n")
        
        for emotion in emotions:
            if emotion in df.columns:
                max_idx = df[emotion].idxmax()
                max_time = df.loc[max_idx, 'time']
                max_value = df.loc[max_idx, emotion]
                report.append(f"📍 Máximo {emotion.upper()}: {max_value:.1f}% en t={max_time:.2f}s (frame {df.loc[max_idx, 'frame']})")
        
        report_text = "\n".join(report)
        
        # Guardar reporte
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n✅ Reporte guardado en: {save_path}")
        print("\n" + report_text)
    
    def __del__(self):
        """Liberar recursos"""
        if hasattr(self, 'cap'):
            self.cap.release()

def main():
    """Función principal"""
    print("=" * 70)
    print("  🎬 ANALIZADOR DE EVOLUCIÓN DE EMOCIONES EN VIDEO")
    print("=" * 70)
    
    # Ruta al video
    video_path = "video_prueba.mp4"
    
    if not Path(video_path).exists():
        print(f"\n⚠️  El archivo '{video_path}' no existe.")
        print("\n💡 Coloca un video en el directorio y ajusta 'video_path' en el código.")
        return
    
    try:
        # Crear analizador
        analyzer = VideoEmotionAnalyzer(video_path)
        
        # Analizar frames (ajusta num_frames según necesites)
        analyzer.analyze_frames(num_frames=15)
        
        # Generar visualizaciones
        analyzer.visualize_evolution('evolucion_emociones.png')
        
        # Generar reporte
        analyzer.generate_report('reporte_emociones.txt')
        
        print("\n" + "=" * 70)
        print("  ✅ ANÁLISIS COMPLETADO")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
