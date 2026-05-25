import os
import glob
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score
import sys

# Importar el modelo 2D y el nuevo Dataset
sys.path.append(os.getcwd())
from models.efficientnet_2d import EfficientNet2D
from scripts.dataset_2d import FrameLevelDataset

def train_and_evaluate():
    # --- CONFIGURACIÓN ---
    DATA_DIR = os.path.join('preprocesado')
    LOG_DIR = os.path.join('logs')
    MODELS_DIR = os.path.join('models')
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    NUM_RUNS = 5
    EPOCHS = 25
    BATCH_SIZE = 16 # Batch size mayor al ser 2D (ocupa menos VRAM)
    LEARNING_RATE = 1e-4
    
    # Archivos de registro
    train_log_path = os.path.join(LOG_DIR, 'training_metrics_2D.csv')
    test_log_path = os.path.join(LOG_DIR, 'test_results_2D.csv')

    # Cargar lista de todos los tensores
    all_files = glob.glob(os.path.join(DATA_DIR, '*.pt'))
    total_files = len(all_files)
    
    if total_files == 0:
        print("Error: No se encontraron archivos .pt en la carpeta preprocesado/")
        return

    print(f"Iniciando comparativa 2D con {total_files} muestras...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}\n")

    # Inicializar CSVs
    with open(train_log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Run', 'Epoch', 'Train_Loss', 'Val_Loss', 'Val_Acc', 'Val_AUC'])

    with open(test_log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Run', 'Test_Acc', 'Test_AUC'])

    # --- BUCLE DE RUNS (5 Simulaciones) ---
    for run in range(1, NUM_RUNS + 1):
        print(f"========================================")
        print(f"       INICIANDO RUN {run}/{NUM_RUNS} (MODELO 2D)      ")
        print(f"========================================")

        # 1. Split estricto 70/15/15 con semilla dinámica por run para el reshuffle
        generator = torch.Generator().manual_seed(42 + run) 
        train_size = int(0.7 * total_files)
        val_size = int(0.15 * total_files)
        test_size = total_files - train_size - val_size

        train_files, val_files, test_files = random_split(
            all_files, [train_size, val_size, test_size], generator=generator
        )

        # 2. DataLoaders
        train_dataset = FrameLevelDataset(train_files, is_train=True)
        val_dataset = FrameLevelDataset(val_files, is_train=False)
        test_dataset = FrameLevelDataset(test_files, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 3. Inicializar Modelo, Loss y Optimizador
        model = EfficientNet2D(pretrained=True).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # --- BUCLE DE ÉPOCAS ---
        for epoch in range(1, EPOCHS + 1):
            # Fase de Entrenamiento
            model.train()
            train_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            
            train_loss /= len(train_loader.dataset)

            # Fase de Validación
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_loss /= len(val_loader.dataset)
            
            # Calcular métricas
            preds_binary = [1 if p >= 0.5 else 0 for p in all_preds]
            val_acc = accuracy_score(all_labels, preds_binary)
            val_auc = roc_auc_score(all_labels, all_preds)

            print(f"Run {run} | Época {epoch}/{EPOCHS} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")

            # Guardar logs de la época
            with open(train_log_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([run, epoch, train_loss, val_loss, val_acc, val_auc])

        # --- FASE DE TEST (Al final de la Run) ---
        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_preds.extend(outputs.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        test_preds_binary = [1 if p >= 0.5 else 0 for p in test_preds]
        test_acc = accuracy_score(test_labels, test_preds_binary)
        test_auc = roc_auc_score(test_labels, test_preds)

        print(f"\n--- RESULTADOS TEST (RUN {run}) ---")
        print(f"Accuracy: {test_acc:.4f} | AUC: {test_auc:.4f}\n")

        with open(test_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([run, test_acc, test_auc])

        # Guardar el modelo 2D de esta run
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, f'efficientnet2d_run{run}.pth'))

if __name__ == '__main__':
    train_and_evaluate()