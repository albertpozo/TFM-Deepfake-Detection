import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score
import sys
import os

# Asegurar que se encuentra la carpeta models y scripts [cite: 1, 26, 27]
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.dataset import DeepfakeDataset
from models.td_3dcnn import TD3DCNN

# --- CONFIGURACIÓN EXPERIMENTAL ---
BATCH_SIZE = 4 
LR = 1e-4
EPOCHS = 10 
DATA_PATH = 'preprocesado' # [cite: 1, 28]

def run_experiment():
    dataset = DeepfakeDataset(DATA_PATH)
    total_clips = len(dataset)
    
    # 7.2. SEPARACIÓN TRIPLE (70% Train / 15% Valid / 15% Test)
    train_size = int(0.7 * total_clips)
    val_size = int(0.15 * total_clips)
    test_size = total_clips - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 7.4. OPTIMIZACIÓN DE HARDWARE (RTX 5060)
    device = torch.device("cuda") 
    model = TD3DCNN().to(device) # [cite: 1, 72]
    
    criterion = nn.BCELoss() # Clasificación binaria
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Iniciando entrenamiento con {total_clips} clips...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 7.3. CÁLCULO DE MÉTRICAS (Accuracy y AUC)
        model.eval()
        y_true, y_probs = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                y_true.extend(labels.cpu().numpy())
                y_probs.extend(outputs.cpu().numpy())

        # Métricas de validación
        y_pred = [1 if p > 0.5 else 0 for p in y_probs]
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_probs)

        print(f"Época {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(train_loader):.4f} | Acc: {acc:.2f} | AUC: {auc:.2f}")

    # Guardado del modelo refinado
    torch.save(model.state_dict(), 'models/td_3dcnn_v1.pth')
    print("\n--- Entrenamiento finalizado y modelo v1 guardado ---")

if __name__ == "__main__":
    run_experiment()