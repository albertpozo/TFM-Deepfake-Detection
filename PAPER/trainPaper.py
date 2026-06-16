import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score
import sys
import os
import csv

# Configuración de rutas para importar módulos del proyecto
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.dataset import DeepfakeDataset
from models.td_3dcnn import TD3DCNN

# --- CONFIGURACIÓN DEL EXPERIMENTO PAPER ---
NUM_RUNS = 5          # Ejecución de 5 experimentos independientes
EPOCHS = 30           # Subo a 30 dar margen al scheduler a actuar
BATCH_SIZE = 4        # Mantenemos 4 para asegurar que no desborda la VRAM
LR = 1e-4
DATA_PATH = r'C:\Users\Albert\Documents\TFM\preprocesado'
LOG_DIR = 'logs_paper' #

def save_logs(filename, headers, data):
    """Guarda los datos en un archivo CSV."""
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(data)

def run_training():
    # Crear carpeta de logs si no existe
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # 1. Cargar dataset
    dataset = DeepfakeDataset(DATA_PATH)
    total_clips = len(dataset)

    # 2. Definir nombres de archivo DINÁMICOS
    experiment_name = "completo" # O el nombre que prefieras para distinguirlo de "medio"
    
    training_log_file = os.path.join(LOG_DIR, f'training_metrics_{experiment_name}_{total_clips}.csv')
    test_log_file = os.path.join(LOG_DIR, f'test_results_{experiment_name}_{total_clips}.csv')
    
    # Limpiar logs anteriores de ESTE experimento específico
    if os.path.exists(training_log_file): os.remove(training_log_file)
    if os.path.exists(test_log_file): os.remove(test_log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Iniciando Batería PAPER con {total_clips} clips en {device} ---")

    # Bucle de Repeticiones (RUNS)
    for run in range(1, NUM_RUNS + 1):
        print(f"\n[RUN {run}/{NUM_RUNS}] Iniciando iteración experimental...")
        
        # 3. División de Datos
        train_size = int(0.7 * total_clips)
        val_size = int(0.15 * total_clips)
        test_size = total_clips - train_size - val_size
        
        train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
        
        # num_workers=0 para evitar bloqueos estocásticos de multiprocesamiento
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        # 4. Inicializar Modelo Nuevo
        model = TD3DCNN().to(device)
        criterion = nn.BCELoss()
        
        # Regularización L2 (weight_decay) para evitar el sobreajuste
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        
        # Scheduler para evitar el estallido del gradiente
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        # 5. Bucle de Entrenamiento
        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss_acc = 0.0
            
            # Fase Train
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss_acc += loss.item()
            
            avg_train_loss = train_loss_acc / len(train_loader)

            # Fase Validación
            model.eval()
            y_true, y_probs = [], []
            val_loss_acc = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss_acc += loss.item()
                    y_true.extend(labels.cpu().numpy())
                    y_probs.extend(outputs.cpu().numpy())

            avg_val_loss = val_loss_acc / len(val_loader)
            y_pred = [1 if p > 0.5 else 0 for p in y_probs]
            val_acc = accuracy_score(y_true, y_pred)
            try:
                val_auc = roc_auc_score(y_true, y_probs)
            except:
                val_auc = 0.5

            # Actualizar el scheduler con la pérdida de validación actual
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Run {run} | Ep {epoch:02d} -> T.Loss: {avg_train_loss:.4f} | V.Loss: {avg_val_loss:.4f} | V.Acc: {val_acc:.2f} | V.AUC: {val_auc:.2f} | LR: {current_lr}")

            # Guardar logs (Mismo formato exacto que en el TFM)
            save_logs(training_log_file, 
                      ['Run', 'Epoch', 'Train_Loss', 'Val_Loss', 'Val_Acc', 'Val_AUC'],
                      [run, epoch, avg_train_loss, avg_val_loss, val_acc, val_auc])

        # 6. Evaluación Final en TEST
        print(f"Evaluando Run {run} en conjunto de TEST...")
        model.eval()
        test_y_true, test_y_probs = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_y_true.extend(labels.cpu().numpy())
                test_y_probs.extend(outputs.cpu().numpy())
        
        test_pred = [1 if p > 0.5 else 0 for p in test_y_probs]
        test_acc = accuracy_score(test_y_true, test_pred)
        try:
            test_auc = roc_auc_score(test_y_true, test_y_probs)
        except:
            test_auc = 0.5

        # Guardar resultados de Test
        save_logs(test_log_file,
                  ['Run', 'Test_Acc', 'Test_AUC'],
                  [run, test_acc, test_auc])
        
        # Guardar modelo
        torch.save(model.state_dict(), f'models/td_3dcnn_paper_{total_clips}_run{run}.pth')

    print("\n--- EXPERIMENTO FINALIZADO ---")
    print(f"Resultados guardados en: {os.path.abspath(LOG_DIR)}")

if __name__ == "__main__":
    run_training()