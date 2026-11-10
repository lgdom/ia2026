import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ==============================================================================
# CONFIGURACIÓN BÁSICA Y DISPOSITIVO
# ==============================================================================
torch.manual_seed(42)

# Usamos aceleración si está disponible (GPU en Mac o CUDA), si no CPU
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Dispositivo utilizado: {device}")

# ==============================================================================
# 1. CARGA Y PREPARACIÓN DE LAS 10 CLASES
# ==============================================================================
# Redimensionamos las imágenes a 128x128 para que sea rápido y ligero
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder('datos', transform=transform)
clases = dataset.classes
print(f"10 clases seleccionadas: {clases}")
print(f"Total de imágenes: {len(dataset)}")

# Separamos en 70% entrenamiento, 15% validación y 15% prueba
n_total = len(dataset)
n_train = int(0.70 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

train_data, val_data, test_data = random_split(
    dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# ==============================================================================
# 2. DEFINICIÓN DE LAS 3 REDES NEURONALES
# ==============================================================================

# RED 1: Básica desde cero con 3 capas entrenables (2 convolucionales + 1 lineal)
class RedBasica(nn.Module):
    def __init__(self):
        super().__init__()
        # Capa 1: Convolución
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 128 -> 64
        
        # Capa 2: Convolución
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 64 -> 32
        
        # Capa 3: Capa lineal de salida (10 clases)
        self.fc = nn.Linear(32 * 32 * 32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Aplanar
        x = self.fc(x)
        return x

# RED 2: Red profunda desde cero (8 capas entrenables con BatchNorm y Dropout)
class RedProfunda(nn.Module):
    def __init__(self):
        super().__init__()
        # Bloque convolucional 1 (2 capas)
        self.bloque1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 128 -> 64
        )
        # Bloque convolucional 2 (2 capas)
        self.bloque2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 64 -> 32
        )
        # Bloque convolucional 3 (2 capas)
        self.bloque3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 32 -> 16
        )
        # Clasificador (2 capas lineales)
        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.bloque1(x)
        x = self.bloque2(x)
        x = self.bloque3(x)
        x = self.clasificador(x)
        return x

# RED 3: Transfer Learning usando ResNet18 preentrenado
def crear_red_transfer():
    # Descargamos el modelo preentrenado en ImageNet
    modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Congelamos los pesos base para no modificarlos
    for param in modelo.parameters():
        param.requires_grad = False
        
    # Cambiamos la última capa para que clasifique nuestras 10 clases
    in_features = modelo.fc.in_features
    modelo.fc = nn.Linear(in_features, 10)
    return modelo

# ==============================================================================
# 3. FUNCIÓN DE ENTRENAMIENTO Y EVALUACIÓN
# ==============================================================================
def entrenar(modelo, épocas=5, lr=0.001, nombre="Modelo"):
    print(f"\n--- Entrenando {nombre} ({épocas} épocas) ---")
    modelo = modelo.to(device)
    criterio = nn.CrossEntropyLoss()
    # Solo optimizamos los parámetros que requieren gradiente
    parametros_a_entrenar = [p for p in modelo.parameters() if p.requires_grad]
    optimizador = optim.Adam(parametros_a_entrenar, lr=lr)

    for epoca in range(1, épocas + 1):
        modelo.train()
        suma_loss = 0.0
        aciertos = 0
        total = 0

        for imagenes, etiquetas in train_loader:
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
            
            optimizador.zero_grad()
            salidas = modelo(imagenes)
            loss = criterio(salidas, etiquetas)
            loss.backward()
            optimizador.step()

            suma_loss += loss.item() * imagenes.size(0)
            _, predicciones = torch.max(salidas, 1)
            aciertos += (predicciones == etiquetas).sum().item()
            total += etiquetas.size(0)

        loss_epoca = suma_loss / total
        acc_epoca = (aciertos / total) * 100
        print(f"Época {epoca}/{épocas} -> Pérdida: {loss_epoca:.4f} | Precisión (Train): {acc_epoca:.2f}%")

    return modelo

def evaluar(modelo):
    modelo.eval()
    y_reales = []
    y_predichas = []

    with torch.no_grad():
        for imagenes, etiquetas in test_loader:
            imagenes = imagenes.to(device)
            salidas = modelo(imagenes)
            _, predicciones = torch.max(salidas, 1)
            y_reales.extend(etiquetas.numpy())
            y_predichas.extend(predicciones.cpu().numpy())

    return np.array(y_reales), np.array(y_predichas)

# ==============================================================================
# 4. EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == '__main__':
    # 1. Red Básica
    red_1 = RedBasica()
    red_1 = entrenar(red_1, épocas=6, lr=0.001, nombre="Red Básica (3 Capas)")
    torch.save(red_1.state_dict(), 'modelos/red_basica.pth')
    y_test, pred_1 = evaluar(red_1)

    # 2. Red Profunda
    red_2 = RedProfunda()
    red_2 = entrenar(red_2, épocas=6, lr=0.0008, nombre="Red Profunda (8 Capas)")
    torch.save(red_2.state_dict(), 'modelos/red_profunda.pth')
    _, pred_2 = evaluar(red_2)

    # 3. Transfer Learning
    red_3 = crear_red_transfer()
    red_3 = entrenar(red_3, épocas=5, lr=0.001, nombre="Red Transfer Learning (ResNet18)")
    torch.save(red_3.state_dict(), 'modelos/red_transfer.pth')
    _, pred_3 = evaluar(red_3)

    # ==============================================================================
    # 5. MATRICES DE CONFUSIÓN Y GRÁFICAS
    # ==============================================================================
    cm1 = confusion_matrix(y_test, pred_1)
    cm2 = confusion_matrix(y_test, pred_2)
    cm3 = confusion_matrix(y_test, pred_3)

    acc1 = (pred_1 == y_test).mean() * 100
    acc2 = (pred_2 == y_test).mean() * 100
    acc3 = (pred_3 == y_test).mean() * 100

    print("\n================ RESUMEN DE RESULTADOS ================")
    print(f"1. Red Básica (3 capas):          {acc1:.2f}% de acierto en prueba")
    print(f"2. Red Profunda (8 capas):        {acc2:.2f}% de acierto en prueba")
    print(f"3. Transfer Learning (ResNet18):  {acc3:.2f}% de acierto en prueba")

    # Gráfico con las 3 matrices de confusión lado a lado
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    configuraciones = [
        (axes[0], cm1, f"Red Básica (3 Capas)\nAcierto: {acc1:.1f}%", "Blues"),
        (axes[1], cm2, f"Red Profunda (8 Capas)\nAcierto: {acc2:.1f}%", "Purples"),
        (axes[2], cm3, f"Transfer Learning (ResNet18)\nAcierto: {acc3:.1f}%", "Greens")
    ]

    for ax, cm, titulo, cmap in configuraciones:
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                    xticklabels=clases, yticklabels=clases, ax=ax)
        ax.set_title(titulo, fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Valor Real")
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('graficas/comparativa_matrices.png', dpi=200)
    plt.close()
    print("\nGráfica guardada en: graficas/comparativa_matrices.png")

    # Guardar cada matriz individualmente
    for cm, nombre, cmap in [(cm1, 'red_basica', 'Blues'), (cm2, 'red_profunda', 'Purples'), (cm3, 'red_transfer', 'Greens')]:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=True,
                    xticklabels=clases, yticklabels=clases)
        plt.title(f"Matriz de Confusión - {nombre.replace('_', ' ').title()}", fontsize=13)
        plt.xlabel("Predicción")
        plt.ylabel("Valor Real")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'graficas/matriz_{nombre}.png', dpi=200)
        plt.close()

    print("Todas las matrices individuales guardadas en la carpeta 'graficas/'.")
