# multimedia/bovw.py
"""
Módulo multimedia.bovw
--------------------------------
Bag of Visual Words: construcción de codebook, histogramas y TF-IDF.
"""

import numpy as np
from sklearn.cluster import KMeans
import pickle
import os
from typing import List, Optional, Tuple
from multimedia.config import VOCAB_SIZE


def build_visual_codebook(sift_descriptors_list: List[np.ndarray], 
                         vocab_size: int = None,
                         random_state: int = 42,
                         max_iter: int = 300,
                         n_init: int = 10) -> Tuple[KMeans, np.ndarray]:
    """
    Crea vocabulario visual usando K-Means sobre descriptores SIFT.
    
    Args:
        sift_descriptors_list: Lista de arrays de descriptores SIFT (uno por imagen)
        vocab_size: Tamaño del vocabulario visual (número de clusters)
        random_state: Semilla para reproducibilidad
        max_iter: Máximo número de iteraciones para K-Means
        n_init: Número de inicializaciones para K-Means
        
    Returns:
        Tupla (kmeans_model, training_descriptors)
        - kmeans_model: Modelo K-Means entrenado para asignar visual words
        - training_descriptors: Descriptores usados para entrenar (para actualización incremental)
    """
    # Usar valor por defecto de config si no se especifica
    if vocab_size is None:
        vocab_size = VOCAB_SIZE
    
    # Concatenar todos los descriptores de todas las imágenes
    print(f"[INFO] Construyendo codebook con vocab_size={vocab_size}...")
    
    all_descriptors = []
    for descriptors in sift_descriptors_list:
        if descriptors is not None and len(descriptors) > 0:
            all_descriptors.append(descriptors)
    
    if len(all_descriptors) == 0:
        raise ValueError("No hay descriptores SIFT disponibles para construir el codebook")
    
    # Concatenar en una sola matriz
    all_descriptors = np.vstack(all_descriptors)
    print(f"[INFO] Total de descriptores: {len(all_descriptors)}")
    
    # Entrenar K-Means
    print(f"[INFO] Entrenando K-Means con {vocab_size} clusters...")
    kmeans = KMeans(
        n_clusters=vocab_size,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
        verbose=0
    )
    kmeans.fit(all_descriptors)
    
    print(f"[OK] Codebook construido exitosamente")
    # Retornar también los descriptores para poder guardarlos
    return kmeans, all_descriptors


def images_to_histograms(image_descriptors_list: List[np.ndarray],
                        kmeans_model: KMeans,
                        vocab_size: int) -> np.ndarray:
    """
    Convierte descriptores SIFT de cada imagen a histograma de visual words.
    
    Args:
        image_descriptors_list: Lista de arrays de descriptores SIFT (uno por imagen)
        kmeans_model: Modelo K-Means entrenado
        vocab_size: Tamaño del vocabulario visual
        
    Returns:
        Matriz de histogramas (n_imágenes, vocab_size)
    """
    n_images = len(image_descriptors_list)
    histograms = np.zeros((n_images, vocab_size), dtype=np.float32)
    
    print(f"[INFO] Generando histogramas para {n_images} imágenes...")
    
    for i, descriptors in enumerate(image_descriptors_list):
        if descriptors is None or len(descriptors) == 0:
            # Imagen sin descriptores: histograma de ceros
            histograms[i] = np.zeros(vocab_size, dtype=np.float32)
            continue
        
        # Asignar cada descriptor al cluster más cercano (visual word)
        visual_words = kmeans_model.predict(descriptors)
        
        # Construir histograma de frecuencias
        hist, _ = np.histogram(visual_words, bins=vocab_size, range=(0, vocab_size))
        
        # Normalizar por el número de descriptores (TF: Term Frequency)
        if len(descriptors) > 0:
            hist = hist.astype(np.float32) / len(descriptors)
        
        histograms[i] = hist
    
    print(f"[OK] Histogramas generados: shape {histograms.shape}")
    return histograms


def apply_tfidf_to_histograms(histograms_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica ponderación TF-IDF a los histogramas.
    
    Args:
        histograms_matrix: Matriz de histogramas (n_imágenes, vocab_size)
        
    Returns:
        Tupla (histograms_tfidf, idf_weights)
        - histograms_tfidf: Matriz TF-IDF (n_imágenes, vocab_size)
        - idf_weights: Vector de pesos IDF (vocab_size,)
    """
    n_images, vocab_size = histograms_matrix.shape
    
    # Calcular IDF (Inverse Document Frequency)
    # Para cada visual word, contar en cuántas imágenes aparece
    # Un visual word "aparece" si su frecuencia > 0
    document_frequency = np.sum(histograms_matrix > 0, axis=0).astype(np.float32)
    
    # Evitar división por cero
    document_frequency = np.maximum(document_frequency, 1.0)
    
    # IDF = log(total_imágenes / imágenes_con_palabra)
    idf_weights = np.log(n_images / document_frequency)
    
    # Aplicar TF-IDF: multiplicar cada histograma (TF) por los pesos IDF
    histograms_tfidf = histograms_matrix * idf_weights[np.newaxis, :]
    
    print(f"[OK] TF-IDF aplicado: shape {histograms_tfidf.shape}")
    return histograms_tfidf, idf_weights


def save_codebook(kmeans_model: KMeans, filepath: str, training_descriptors: Optional[np.ndarray] = None):
    """
    Guarda el modelo K-Means en disco.
    
    Args:
        kmeans_model: Modelo K-Means entrenado
        filepath: Ruta donde guardar el codebook
        training_descriptors: Descriptores usados para entrenar (opcional, para actualización incremental)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = {
        'kmeans_model': kmeans_model,
        'training_descriptors': training_descriptors,
        'vocab_size': kmeans_model.n_clusters
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"[OK] Codebook guardado en {filepath}")


def load_codebook(filepath: str) -> Tuple[KMeans, Optional[np.ndarray]]:
    """
    Carga el modelo K-Means desde disco.
    
    Returns:
        Tupla (kmeans_model, training_descriptors)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Codebook no encontrado: {filepath}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Compatibilidad con formato antiguo (solo KMeans)
    if isinstance(data, dict):
        kmeans_model = data['kmeans_model']
        training_descriptors = data.get('training_descriptors', None)
    else:
        # Formato antiguo: solo el modelo
        kmeans_model = data
        training_descriptors = None
    
    print(f"[OK] Codebook cargado desde {filepath}")
    return kmeans_model, training_descriptors


def add_images_to_codebook(codebook_path: str, 
                           new_sift_descriptors_list: List[np.ndarray],
                           vocab_size: int = None,
                           random_state: int = 42,
                           max_iter: int = 300,
                           n_init: int = 10) -> KMeans:
    """
    Agrega nuevas imágenes al codebook existente de forma incremental.
    Re-entrena el codebook con los descriptores originales + los nuevos.
    
    Args:
        codebook_path: Ruta al codebook existente
        new_sift_descriptors_list: Lista de descriptores SIFT de las nuevas imágenes
        vocab_size: Tamaño del vocabulario (debe coincidir con el codebook existente). Si es None, usa VOCAB_SIZE de config.
        random_state: Semilla para reproducibilidad
        max_iter: Máximo número de iteraciones para K-Means
        n_init: Número de inicializaciones para K-Means
        
    Returns:
        Modelo K-Means re-entrenado
    """
    # Usar valor por defecto de config si no se especifica
    if vocab_size is None:
        vocab_size = VOCAB_SIZE
    
    # Cargar codebook existente
    kmeans_model, old_descriptors = load_codebook(codebook_path)
    
    if kmeans_model.n_clusters != vocab_size:
        raise ValueError(f"Vocab size mismatch: codebook tiene {kmeans_model.n_clusters}, se esperaba {vocab_size}")
    
    print(f"[INFO] Agregando {len(new_sift_descriptors_list)} nuevas imágenes al codebook...")
    
    # Recopilar nuevos descriptores
    new_descriptors = []
    for descriptors in new_sift_descriptors_list:
        if descriptors is not None and len(descriptors) > 0:
            new_descriptors.append(descriptors)
    
    if len(new_descriptors) == 0:
        print("[WARN] No hay descriptores nuevos para agregar")
        return kmeans_model
    
    new_descriptors = np.vstack(new_descriptors)
    print(f"[INFO] Nuevos descriptores: {len(new_descriptors)}")
    
    # Combinar con descriptores originales si existen
    if old_descriptors is not None and len(old_descriptors) > 0:
        print(f"[INFO] Combinando con {len(old_descriptors)} descriptores originales...")
        all_descriptors = np.vstack([old_descriptors, new_descriptors])
    else:
        print("[WARN] No se encontraron descriptores originales, usando solo los nuevos")
        all_descriptors = new_descriptors
    
    print(f"[INFO] Total de descriptores para re-entrenar: {len(all_descriptors)}")
    
    # Re-entrenar K-Means con todos los descriptores
    print(f"[INFO] Re-entrenando K-Means con {vocab_size} clusters...")
    kmeans = KMeans(
        n_clusters=vocab_size,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
        verbose=0
    )
    kmeans.fit(all_descriptors)
    
    print(f"[OK] Codebook actualizado exitosamente")
    return kmeans


def save_histograms(histograms: np.ndarray, idf_weights: np.ndarray, filepath: str):
    """Guarda histogramas e IDF en formato numpy."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez(filepath, histograms=histograms, idf_weights=idf_weights)
    print(f"[OK] Histogramas guardados en {filepath}")


def load_histograms(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Carga histogramas e IDF desde archivo numpy."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Histogramas no encontrados: {filepath}")
    data = np.load(filepath)
    histograms = data['histograms']
    idf_weights = data['idf_weights']
    print(f"[OK] Histogramas cargados desde {filepath}")
    return histograms, idf_weights

