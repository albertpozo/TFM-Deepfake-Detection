================================================================================
TRABAJO FIN DE MÁSTER (TFM)
DETECCIÓN DE MANIPULACIÓN DIGITAL EN IMÁGENES Y VÍDEOS MEDIANTE
TÉCNICAS DE VISIÓN POR COMPUTADOR
================================================================================

Autor: Albert (Ingeniería de Telecomunicaciones - Universidad de Alicante)

DESCRIPCIÓN GENERAL DEL PROYECTO
--------------------------------------------------------------------------------
Este repositorio contiene el código fuente completo, los scripts de automatización y la definición de las arquitecturas neuronales desarrolladas para este TFM. 
El proyecto aborda la problemática de la detección forense de vídeos generados mediante Inteligencia Artificial (Deepfakes).

A lo largo de la investigación, se plantea una comparativa empírica sistemática entre arquitecturas puramente espaciales (2D) y arquitecturas espacio-temporales (3D). El objetivo es demostrar que la extracción de características temporales y la evaluación de inconsistencias dinámicas entre fotogramas consecutivos resulta 
vital para generalizar la detección de falsificaciones modernas.

ESTRUCTURA DE DIRECTORIOS Y ARCHIVOS PRINCIPALES
--------------------------------------------------------------------------------
El repositorio está organizado de forma modular para separar el preprocesamiento de datos, las arquitecturas de red y el bucle de entrenamiento:

1. Directorio /models/ (Arquitecturas Neuronales)
   - td_3dcnn.py: Define la arquitectura principal del proyecto, una red espacio-temporal basada en una estructura ResNet-3D. Incluye mecanismos de regularización y un bloque clasificador adaptado para tensores [3, 16, 224, 224].
   - efficientnet_2d.py: Implementa el modelo "baseline" puramente bidimensional basado en EfficientNet-B0. Utiliza pesos preentrenados (ImageNet) y está adaptado para la extracción de características espaciales de fotogramas aislados ([3, 224, 224]).

2. Directorio /scripts/ (Automatización y Pipeline)
   - download.py: Script para la descarga controlada del dataset FaceForensics++.
   - preprocess.py: Pipeline de preprocesamiento de vídeo. Se encarga de la lectura de archivos .mp4, extracción de recortes faciales, normalización y empaquetado en tensores PyTorch (.pt) de 16 fotogramas para optimizar el cuello de botella de entrada/salida (I/O).
   - dataset.py: Define el DataLoader personalizado para alimentar los bloques 3D a la red TD-3DCNN durante el entrenamiento.
   - dataset_2d.py: DataLoader adaptado con extracción de fotograma dinámico para la comparativa con el modelo 2D, integrando data augmentation temporal.
   - trainTFM.py: Script principal de entrenamiento para el modelo 3D. Implementa el protocolo de validación cruzada iterativa (5-Runs) detallado en la memoria, el guardado automático de pesos y la métrica de AUC.
   - train2D_TFM.py: Equivalente orquestado para la experimentación con el modelo 2D.

3. Directorio /logs/ (Registro de Resultados)
   - Contiene los registros (archivos CSV) generados automáticamente por los scripts de entrenamiento. En ellos se almacena la evolución histórica del Loss, Accuracy y AUC por época para cada una de las 5 ejecuciones (Runs).

4. Archivos Raíz
   - graficas_resultados.mlx: Live Script interactivo de MATLAB. Este archivo procesa los CSV generados en la carpeta logs para trazar las representaciones gráficas evolutivas que se incluyen en el documento de memoria del TFM.
   - .gitignore: Archivo de exclusión para evitar la subida accidental de pesos masivos (.pth) y conjuntos de datos.

REQUISITOS DEL ENTORNO (WINDOWS 11)
--------------------------------------------------------------------------------
El entorno de desarrollo ha sido configurado y validado de forma nativa en Windows 11. 
Para la correcta ejecución de los scripts, se recomienda la creación de un entorno virtual de Python.

Dependencias principales de Python:
- torch, torchvision (Es imprescindible disponer de soporte CUDA para acelerar el entrenamiento en GPU, dados los elevados requerimientos de computación de las redes 3D).
- opencv-python (Para el procesamiento inicial de vídeo).
- scikit-learn (Para el cálculo analítico de la métrica AUC).
- numpy, pandas (Para manipulación de tensores matriciales y lectura/escritura de logs).
- tqdm (Para la visualización visual del progreso en terminal).

Adicionalmente, se requiere software de MATLAB instalado en el sistema operativo para ejecutar el script de generación de gráficas (.mlx).

NOTA SOBRE LOS DATOS (DATASETS) Y PESOS DEL MODELO
--------------------------------------------------------------------------------
Por restricciones de almacenamiento y licencias de distribución, ni el dataset original (FaceForensics++) ni los tensores preprocesados se incluyen en este repositorio. 
Para reproducir el estudio, el usuario debe descargar los datos utilizando los scripts proporcionados y ejecutar el pipeline de preprocesamiento localmente. 
Los pesos resultantes del entrenamiento de las redes neuronales (.pth) también quedan excluidos por superar holgadamente el límite de tamaño de la plataforma GitHub.
