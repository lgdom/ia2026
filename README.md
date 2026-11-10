# Tareas de Inteligencia Artificial - Visión y Redes Neuronales

Este repositorio contiene el desarrollo y las soluciones para las tareas de la asignatura de **Inteligencia Artificial**, enfocadas en procesamiento de imágenes y clasificación mediante redes neuronales convolucionales.

---

## Estructura del Repositorio

```text
.
├── tarea_1/
│   ├── Tarea_1_Canales_Espacios_Color.ipynb   # Notebook con separación de canales y espacios de color
│   └── imagenes/
│       └── ejemplo.jpg                       # Imagen de prueba
├── tarea_2/
│   ├── Tarea_2_Caltech101.ipynb              # Notebook con entrenamiento de las 3 redes y matrices de confusión
│   ├── train_models.py                       # Script Python de entrenamiento ejecutable
│   ├── graficas/                             # Matrices de confusión y comparativas (.png)
│   └── modelos/                              # Pesos entrenados de los modelos (.pth)
├── datos/                                    # 10 clases seleccionadas de Caltech-101 (2,998 imágenes)
├── requirements.txt                          # Dependencias del proyecto
├── .gitignore                                # Archivos ignorados para control de versiones
└── README.md                                 # Documentación del proyecto
```

---

## Tarea 1: Descomposición de Canales y Espacios de Color

### Objetivos:
1. **Descomposición tradicional (Matricial / Sin librerías de visión):** Extracción matemática de los canales Rojo, Verde y Azul indexando la matriz $I[:, :, c]$, visualizándolos en escala de grises y en color puro.
2. **Descomposición con OpenCV:** Uso de `cv2.split()`.
3. **Conversión a distintos espacios de color:** Representación y análisis en **Escala de Grises**, **HSV**, **CIE LAB** y **YCrCb**.

---

## Tarea 2: Clasificación de Imágenes y Transfer Learning (Caltech-101)

Se seleccionaron **10 categorías** del conjunto de datos Caltech-101 (`airplanes`, `Motorbikes`, `Faces_easy`, `watch`, `Leopards`, `bonsai`, `car_side`, `grand_piano`, `butterfly`, `sunflower`).

### Modelos implementados:
1. **Red Básica (3 capas desde cero):** 2 convoluciones + 1 capa densa de clasificación.
2. **Red Profunda (8 capas desde cero):** 6 capas convolucionales con Batch Normalization + 2 capas densas con Dropout.
3. **Transfer Learning (ResNet-18 preentrenado):** Ajuste fino de la última capa lineal sobre representaciones aprendidas en ImageNet.

### Resultados en Test:
- **Red Básica (3 capas):** ~96.90% de precisión.
- **Red Profunda (8 capas):** ~94.46% de precisión.
- **Transfer Learning (ResNet-18):** **~98.67% de precisión**.

Las matrices de confusión de cada modelo se encuentran graficadas dentro del notebook y en la carpeta `tarea_2/graficas/`.

---

## Instalación y Uso

1. Clonar el repositorio:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd <NOMBRE_DEL_REPOSITORIO>
   ```

2. Crear y activar un entorno virtual (recomendado):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # En Linux/Mac
   # o en Windows: .venv\Scripts\activate
   ```

3. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Abrir cualquiera de los notebooks en Jupyter o VS Code:
   ```bash
   jupyter notebook
   ```

