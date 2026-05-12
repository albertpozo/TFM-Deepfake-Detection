import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

class TD3DCNN(nn.Module):
    def __init__(self, num_classes=1, dropout_prob=0.5):
        super(TD3DCNN, self).__init__()
        
        # 1. Usamos un "backbone" robusto 3D (R3D_18) preentrenado
        # Esto garantiza que el modelo ya "sabe" ver vídeo..
        weights = R3D_18_Weights.DEFAULT
        self.backbone = r3d_18(weights=weights)
        
        # 2. Adaptación a CLASIFICACIÓN BINARIA (Real vs Fake)
        # La red original clasifica 400 acciones, nosotros queremos 1 salida.
        num_features = self.backbone.fc.in_features
        
        # 3. Implementación de "TD" (Temporal Dropout)
        # Añadimos Dropout
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),  # El Dropout temporal/estructural
            nn.Linear(num_features, num_classes)
        )
        
        # Sigmoide para que el resultado sea una probabilidad entre 0 y 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch, 3, Frames, Height, Width)
        x = self.backbone(x)
        return self.sigmoid(x)