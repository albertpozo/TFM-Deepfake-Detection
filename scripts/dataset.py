import torch
from pathlib import Path
from torch.utils.data import Dataset

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.files = list(self.root_dir.glob("*.pt")) # Busca tus archivos .pt

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = torch.load(file_path) # Carga el archivo procesado
        
        tensor = data['tensor']
        label = data['label']
        
        # Formato necesario para la clasificación binaria (float)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        
        return tensor, label