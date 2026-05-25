import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys
import os

# Importar los archivos anteriores
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.dataset import DeepfakeDataset
from models.td_3dcnn import TD3DCNN

# --- PARÁMETROS ESTÁNDAR (Petición de Paco) ---
BATCH_SIZE = 2      # Bajo para no saturar memoria
LR = 0.0001         # Velocidad de aprendizaje estándar
EPOCHS = 5          # "Unas cuantas épocas" para ver estabilidad
DATA_PATH = 'preprocesado' # Carpeta DE los tensores

def run_training():
    # 1. Preparar Datos
    dataset = DeepfakeDataset(DATA_PATH)
    
    # Separar Train (80%) y Validación (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"--- Iniciando con {len(dataset)} clips totales ---")

    # 2. Configurar Modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando: {device}")
    
    model = TD3DCNN().to(device)
    criterion = nn.BCELoss() # Función de pérdida para binaria (Real/Fake)
    optimizer = optim.Adam(model.parameters(), lr=LR) # Optimizador estándar

    # 3. Bucle de Entrenamiento
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Calcular media de error
        avg_loss = train_loss / len(train_loader)
        print(f"Época {epoch+1}/{EPOCHS} -> Loss Entrenamiento: {avg_loss:.4f}")

    # 4. Guardar resultado
    torch.save(model.state_dict(), 'models/td_3dcnn_piloto.pth')
    print("--- Ciclo cerrado: Modelo guardado correctamente ---")

if __name__ == "__main__":
    run_training()