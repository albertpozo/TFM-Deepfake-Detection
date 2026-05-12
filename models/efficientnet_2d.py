import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNet2D(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNet2D, self).__init__()
        
        # 1. Cargar el backbone de EfficientNet-B0 con pesos preentrenados en ImageNet
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = models.efficientnet_b0(weights=weights)
        
        # 2. Modificar la capa de clasificación final
        # Extraemos el número de características que entran a la última capa
        num_ftrs = self.model.classifier[1].in_features
        
        # Sustituimos el clasificador original (1000 clases) por uno binario (1 clase con Sigmoide)
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True), # Regularización para evitar sobreajuste
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x tiene dimensiones [Batch, Canales, Alto, Ancho] -> [B, 3, 224, 224]
        return self.model(x)