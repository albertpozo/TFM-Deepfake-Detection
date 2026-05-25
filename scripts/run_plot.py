import os
import re
import sys
import subprocess

# Ruta al archivo plot_results.py
TARGET_SCRIPT = os.path.join('scripts', 'plot_results.py')

def main():
    print("--- INICIANDO LANZADOR DE GRÁFICAS ---")
    
    # 1. Verificar que el archivo existe
    if not os.path.exists(TARGET_SCRIPT):
        print(f"ERROR CRÍTICO: No encuentro el archivo: {TARGET_SCRIPT}")
        return

    # 2. Pedir el número al usuario
    num_input = input("Introduce el número de vídeos (ej. 400, 600): ").strip()
    
    if not num_input.isdigit():
        print("Error: Por favor, introduce solo números.")
        return

    # 3. Leer el contenido del archivo plot_results.py
    print(f"Leyendo {TARGET_SCRIPT}...")
    with open(TARGET_SCRIPT, 'r', encoding='utf-8') as f:
        content = f.read()

    # 4. Buscar y reemplazar la línea NUM_VIDEOS = ...
    # Busca cualquier variante de "NUM_VIDEOS = 123"
    pattern = r"NUM_VIDEOS\s*=\s*\d+"
    replacement = f"NUM_VIDEOS = {num_input}"
    
    new_content, count = re.subn(pattern, replacement, content)

    if count == 0:
        print("ADVERTENCIA: No se encontró la línea 'NUM_VIDEOS = ...' en plot_results.py")
        print("NUM_VIDEOS = 200")
        return

    # 5. Guardar los cambios
    with open(TARGET_SCRIPT, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"--> Configuración actualizada a {num_input} vídeos.")

    # 6. Ejecutar el script modificado
    print(f"--> Lanzando generación de gráficas...")
    print("-" * 30)
    
    # Usamos sys.executable para asegurar que usa el mismo entorno python (tfm_env)
    subprocess.run([sys.executable, TARGET_SCRIPT])

if __name__ == "__main__":
    main()