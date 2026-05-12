import torch
import torchvision.models.video as models

print("--- Verificando entorno del TFM ---")
# Punto 3 del tutor: Cargar un modelo de vídeo simple
try:
    model = models.r3d_18(weights='R3D_18_Weights.DEFAULT')
    model.eval()
    print("Modelo R3D_18 cargado correctamente.")
    
    # Simular entrada: 1 video, 3 canales (RGB), 16 frames, 224x224 px
    input_tensor = torch.randn(1, 3, 16, 224, 224) 
    output = model(input_tensor)
    print(f"Prueba exitosa. Forma de la salida: {output.shape}")
except Exception as e:
    print(f"Error en la configuración: {e}")