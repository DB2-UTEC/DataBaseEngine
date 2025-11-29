# backend/multimedia/image_vector.py
"""
Módulo multimedia.image_vector
--------------------------------
Define el tipo de dato 'ImageVector' y provee funciones para
extraer características (features) de imágenes usando ResNet50.
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os


class ImageVector:
    """
    Representa una imagen como vector de características numéricas.
    """

    def __init__(self):
        # Carga el modelo de ResNet50 preentrenado (sin capa de clasificación)
        self.model = models.resnet50(weights='IMAGENET1K_V1')
        # Remover la capa de clasificación final
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()  # Modo evaluación
        
        # Preprocesamiento de imágenes para ResNet50
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def extract(self, img_path: str):
        """
        Dado el path de una imagen, retorna su vector de características.
        """
        try:
            # Cargar y preprocesar la imagen
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0)  # Añadir dimensión de batch
            
            # Extraer características
            with torch.no_grad():
                features = self.model(img_tensor)
                # Hacer pooling global (promedio) y aplanar
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                # Convertir a numpy: eliminar dimensiones de tamaño 1 y convertir a CPU
                features = features.squeeze().cpu().numpy()
            
            return features.flatten()
        except Exception as e:
            print(f"[ERROR] No se pudo procesar {img_path}: {e}")
            return None
