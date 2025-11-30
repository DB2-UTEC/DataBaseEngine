# multimedia/sift_features.py
"""
Módulo multimedia.sift_features
--------------------------------
Extracción de descriptores locales SIFT para imágenes.
"""

import cv2
import numpy as np
import os
from typing import List, Optional, Tuple


def extract_sift_features(image_path: str, max_keypoints: int = 100) -> Optional[np.ndarray]:
    """
    Extrae descriptores locales SIFT de una imagen.
    
    Args:
        image_path: Ruta a la imagen
        max_keypoints: Número máximo de keypoints a extraer (default: 100)
        
    Returns:
        Array de descriptores (n, 128) o None si no hay keypoints
    """
    try:
        # Leer imagen en escala de grises
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[ERROR] No se pudo cargar la imagen: {image_path}")
            return None
        
        # Crear detector SIFT con límite de keypoints
        sift = cv2.SIFT_create(nfeatures=max_keypoints)
        
        # Detectar keypoints y calcular descriptores
        keypoints, descriptors = sift.detectAndCompute(img, None)
        
        # Si no hay keypoints, retornar None
        if descriptors is None or len(descriptors) == 0:
            print(f"[WARN] No se encontraron keypoints en: {image_path}")
            return None
        
        # Limitar a max_keypoints si hay más (por si acaso)
        if len(descriptors) > max_keypoints:
            descriptors = descriptors[:max_keypoints]
        
        return descriptors.astype(np.float32)
    
    except Exception as e:
        print(f"[ERROR] Error extrayendo SIFT de {image_path}: {e}")
        return None


def extract_sift_from_dataset(image_paths: List[str], max_keypoints: int = 100) -> List[np.ndarray]:
    """
    Extrae todos los descriptores SIFT del dataset.
    
    Args:
        image_paths: Lista de rutas a imágenes
        max_keypoints: Número máximo de keypoints por imagen (default: 100)
        
    Returns:
        Lista de arrays de descriptores (uno por imagen)
    """
    all_descriptors = []
    
    for img_path in image_paths:
        descriptors = extract_sift_features(img_path, max_keypoints=max_keypoints)
        if descriptors is not None:
            all_descriptors.append(descriptors)
        else:
            # Agregar array vacío para mantener correspondencia con índices
            all_descriptors.append(np.array([]).reshape(0, 128))
    
    return all_descriptors


def get_image_paths(image_dir: str) -> List[str]:
    """
    Obtiene lista de rutas de imágenes válidas en un directorio.
    
    Args:
        image_dir: Directorio con imágenes
        
    Returns:
        Lista de rutas completas a imágenes
    """
    image_paths = []
    if not os.path.exists(image_dir):
        print(f"[ERROR] Directorio no existe: {image_dir}")
        return image_paths
    
    for fname in os.listdir(image_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            path = os.path.join(image_dir, fname)
            image_paths.append(path)
    
    return sorted(image_paths)

