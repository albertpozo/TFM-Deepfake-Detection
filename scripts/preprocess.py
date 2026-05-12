import cv2
import torch
import os
import numpy as np
from pathlib import Path

# Configuraciones basadas en las instrucciones del tutor 
CLIP_LEN = 16        # Cantidad de frames por clip
FRAME_SIZE = (224, 224)
BASE_RAW_DIR = Path('data/raw')
PROCESSED_DIR = Path('preprocesado')

# Definimos las rutas de FF++
PATHS = {
    'real': BASE_RAW_DIR / 'original_sequences/youtube/c23/videos',
    'fake': BASE_RAW_DIR / 'manipulated_sequences/Deepfakes/c23/videos'
}

def preprocess_ff():
    PROCESSED_DIR.mkdir(exist_ok=True)
    
    for label_name, path in PATHS.items():
        if not path.exists():
            print(f"Saltando {label_name}: No se encontró la carpeta.")
            continue
            
        label = 0 if label_name == 'real' else 1
        videos = list(path.glob('*.mp4'))
        print(f"--- Procesando {len(videos)} vídeos {label_name} (Etiqueta: {label}) ---")

        for video_path in videos:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            
            while len(frames) < CLIP_LEN:
                ret, frame = cap.read()
                if not ret: break
                
                # Procesamiento individual (Punto 2 del tutor) 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, FRAME_SIZE)
                frame = frame / 255.0  # Normalización
                frames.append(frame)
            
            cap.release()

            if len(frames) == CLIP_LEN:
                # Convertir a tensor 4D: (C, T, H, W) para modelos 3D [cite: 46, 199]
                video_tensor = torch.tensor(np.array(frames), dtype=torch.float32)
                video_tensor = video_tensor.permute(3, 0, 1, 2)
                
                # Guardar el tensor y su etiqueta
                output_data = {'tensor': video_tensor, 'label': label}
                output_name = f"{video_path.stem}.pt"
                torch.save(output_data, PROCESSED_DIR / output_name)
                print(f"  > Tensor guardado: {output_name}")

if __name__ == "__main__":
    preprocess_ff()