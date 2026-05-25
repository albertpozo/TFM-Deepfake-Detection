import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Configuración estética
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

LOG_DIR = 'logs'
IMG_DIR = 'plots'

# --- CONFIGURACIÓN ---
NUM_VIDEOS = 1000  # <--- CAMBIAR NÚMERO SEGÚN EJECUCIÓN
# ---------------------

def plot_metrics():
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)

    # Construir nombres de archivo dinámicos
    # Busca 'training_metrics_400.csv' por ejemplo
    csv_path = os.path.join(LOG_DIR, f'training_metrics_{NUM_VIDEOS}.csv')
    
    if not os.path.exists(csv_path):
        print(f"ERROR: No se encuentra el archivo: {csv_path}")
        print(f"Verifica que en la carpeta 'logs' exista un archivo con ese nombre.")
        return

    # 1. Cargar datos
    train_df = pd.read_csv(csv_path)
    
    # 2. Gráfica de LOSS
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=train_df, x='Epoch', y='Train_Loss', label='Train Loss', color='blue')
    sns.lineplot(data=train_df, x='Epoch', y='Val_Loss', label='Validation Loss', color='orange', linestyle='--')
    plt.title(f'Evolución de Loss ({NUM_VIDEOS} vídeos - Promedio 5 Runs)')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(IMG_DIR, f'loss_evolution_{NUM_VIDEOS}.png'), dpi=300)
    plt.close()

    # 3. Gráfica de AUC y Accuracy
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=train_df, x='Epoch', y='Val_AUC', label='Validation AUC', color='green')
    sns.lineplot(data=train_df, x='Epoch', y='Val_Acc', label='Validation Accuracy', color='purple', linestyle='--')
    plt.title(f'Evolución de Métricas ({NUM_VIDEOS} vídeos - Promedio 5 Runs)')
    plt.xlabel('Época')
    plt.ylabel('Score (0-1)')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.savefig(os.path.join(IMG_DIR, f'metrics_evolution_{NUM_VIDEOS}.png'), dpi=300)
    plt.close()

    print(f"--- Gráficas guardadas en '{IMG_DIR}' con sufijo _{NUM_VIDEOS}.png ---")

def print_final_table():
    # Cargar resultados de TEST dinámicos
    csv_test_path = os.path.join(LOG_DIR, f'test_results_{NUM_VIDEOS}.csv')
    
    if not os.path.exists(csv_test_path):
        print(f"No se encontró archivo de test: {csv_test_path}")
        return

    test_df = pd.read_csv(csv_test_path)
    
    print("\n" + "="*40)
    print(f"TABLA RESUMEN ({NUM_VIDEOS} VÍDEOS)")
    print("="*40)
    print(test_df)
    print("-" * 40)
    
    avg_acc = test_df['Test_Acc'].mean()
    std_acc = test_df['Test_Acc'].std()
    avg_auc = test_df['Test_AUC'].mean()
    std_auc = test_df['Test_AUC'].std()
    
    print(f"Accuracy Promedio: {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"AUC Promedio:      {avg_auc:.4f} ± {std_auc:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    plot_metrics()
    print_final_table()