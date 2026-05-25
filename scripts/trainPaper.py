import torch
import torch.nn as nn
import torch.optim as optim
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader

# Importaciones adaptadas a la estructura de tu repositorio
from models.td_3dcnn import TD3DCNN
from scripts.dataset import DeepfakeDataset

def main():
    # ---------------------------------------------------------
    # 1. HIPERPARÁMETROS (Ajustados para el escenario N=1000)
    # ---------------------------------------------------------
    batch_size = 4  # Reducido drásticamente para evitar colapsar la VRAM
    initial_lr = 1e-4
    num_epochs = 30
    
    # Soporte CUDA para aceleración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Optimización de CuDNN comentada temporalmente para evitar el retardo inicial
    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = True 
        pass
        
    print(f"[INFO] Iniciando entrenamiento experimental para el Paper en: {device}")

    # ---------------------------------------------------------
    # 2. CARGA DE DATOS
    # ---------------------------------------------------------
    train_dataset = DeepfakeDataset(root_dir=r"C:\Users\Albert\Documents\TFM\preprocesado")
    val_dataset = DeepfakeDataset(root_dir=r"C:\Users\Albert\Documents\TFM\preprocesado")

    # Aceleración de E/S ajustada para evitar deadlocks (num_workers=0)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )

    # ---------------------------------------------------------
    # 3. MODELO, FUNCIÓN DE PÉRDIDA, OPTIMIZADOR Y SCHEDULER
    # ---------------------------------------------------------
    model = TD3DCNN().to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Añadimos weight_decay para incluir regularización L2 y frenar la inestabilidad
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)

    # LA CLAVE DEL PAPER: El Scheduler dinámico
    # Reducirá el LR a la mitad (factor=0.5) si el val_loss no mejora en 3 épocas (patience=3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
    )

    # ---------------------------------------------------------
    # 4. BUCLE DE ENTRENAMIENTO
    # ---------------------------------------------------------
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Entrenando Época {epoch+1}/{num_epochs}", dynamic_ncols=True, leave=False, file=sys.stdout):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Ajuste de dimensiones según si labels es [batch] o [batch, 1]
            loss = criterion(outputs, labels.float()) 
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # ---------------------------------------------------------
        # 5. BUCLE DE VALIDACIÓN
        # ---------------------------------------------------------
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validando Época {epoch+1}/{num_epochs}", dynamic_ncols=True, leave=False, file=sys.stdout):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Época {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr}")

        # ---------------------------------------------------------
        # 6. ACTUALIZACIÓN DEL SCHEDULER
        # ---------------------------------------------------------
        # Le pasamos el val_loss para que decida si debe intervenir
        scheduler.step(val_loss)

    print("[INFO] Entrenamiento experimental finalizado.")
    
    # Guardado de pesos específico para el paper
    torch.save(model.state_dict(), "models/td_3dcnn_paper_N1000.pth")
    print("[INFO] Pesos guardados en models/td_3dcnn_paper_N1000.pth")

if __name__ == "__main__":
    main()