\# Detección de manipulación digital en imágenes y vídeos (Deepfakes)



Este repositorio contiene el código principal desarrollado para el Trabajo Fin de Máster (TFM) titulado \*"Detección de manipulación digital en imágenes y vídeos mediante técnicas de visión por computador"\*. 



El proyecto explora y compara la eficacia de arquitecturas puramente espaciales frente a arquitecturas espacio-temporales para la detección de vídeos generados por Inteligencia Artificial, utilizando el dataset FaceForensics++.



\## Estructura del Proyecto



El código está organizado para separar la lógica de preprocesamiento, el modelado y la evaluación:



\* \*\*'models/'\*\*: Contiene las definiciones de las arquitecturas neuronales en PyTorch.

&#x20;   \* 'td\_3dcnn.py': Modelo principal espacio-temporal basado en una ResNet-3D con regularización temporal.

&#x20;   \* 'efficientnet\_2d.py': Modelo \*baseline\* bidimensional (puramente espacial) para la comparativa.

\* \*\*'scripts/'\*\*: Scripts de ejecución y automatización.

&#x20;   \* 'download.py': Descarga selectiva de vídeos desde el servidor oficial de FaceForensics++.

&#x20;   \* 'preprocess.py': Pipeline para convertir los vídeos '.mp4' en tensores '.pt' (extracción de clips de 16 \*frames\*).

&#x20;   \* 'dataset.py' / 'dataset\_2d.py': Clases 'DataLoader' personalizadas para alimentar los tensores a las redes 3D y 2D.

&#x20;   \* 'trainTFM.py': Script de orquestación principal para entrenar y validar el modelo 3D con iteraciones automáticas (\*5-Runs\*).

&#x20;   \* 'train2D\_TFM.py': Equivalente para el entrenamiento iterativo del modelo 2D.

\* \*\*'logs/'\*\*: Archivos CSV generados automáticamente con el historial de métricas (Loss, Accuracy, AUC).

\* \*\*'graficas\_resultados.mlx'\*\*: \*Live Script\* de MATLAB utilizado para procesar los CSV y generar las gráficas evolutivas presentadas en la memoria del TFM.



\## Requisitos y Entorno



El proyecto está diseñado para ejecutarse en un entorno virtual de Python bajo Windows 11. Las principales dependencias son:

\* 'torch', 'torchvision' (con soporte CUDA para aceleración GPU)

\* 'opencv-python'

\* 'scikit-learn'

\* 'numpy' y 'tqdm'

\* \*MATLAB\* (exclusivamente para la generación de gráficas a partir del archivo '.mlx').



> \*\*Nota sobre los datos:\*\* Por restricciones de licencia y tamaño, el dataset original de FaceForensics++ y los tensores preprocesados no se incluyen en este repositorio.

