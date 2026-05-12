import torch
from torch.utils.data import Dataset
import random

class FrameLevelDataset(Dataset):
    def __init__(self, file_paths, is_train=True):
        self.file_paths = file_paths
        self.is_train = is_train

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. Cargar el tensor original de vídeo [3, 16, 224, 224]
        data = torch.load(self.file_paths[idx])
        video_tensor = data['tensor']
        label = torch.tensor(data['label'], dtype=torch.float32)

        # 2. Selección del fotograma (Frame-level extraction)
        if self.is_train:
            # En entrenamiento, elegimos un frame aleatorio de los 16.
            # Esto actúa como Data Augmentation temporal.
            frame_idx = random.randint(0, video_tensor.size(1) - 1)
        else:
            # En validación y test, elegimos siempre el frame central (índice 8)
            # para que la evaluación sea determinista y justa.
            frame_idx = video_tensor.size(1) // 2

        # 3. Extraer el frame específico -> tensor [3, 224, 224]
        frame_tensor = video_tensor[:, frame_idx, :, :]

        # Adaptar la etiqueta para que sea [1]
        label = label.unsqueeze(0)

        return frame_tensor, label