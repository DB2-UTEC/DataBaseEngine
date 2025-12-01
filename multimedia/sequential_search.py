# multimedia/sequential_search.py
"""
Módulo multimedia.sequential_search
--------------------------------
Búsqueda KNN secuencial optimizada con heap para SIFT + BOVW.
"""

import heapq
import numpy as np
from typing import List, Tuple
from .sift_features import extract_sift_features


def sequential_knn_search(query_histogram: np.ndarray,
                          database_histograms: np.ndarray,
                          k: int = 10) -> List[Tuple[int, float]]:
    """
    Búsqueda secuencial KNN con optimización de heap.
    
    Args:
        query_histogram: Histograma TF-IDF de la imagen query (vocab_size,)
        database_histograms: Matriz de histogramas TF-IDF (n_imágenes, vocab_size)
        k: Número de resultados a retornar
        
    Returns:
        Lista de tuplas (índice, score) ordenadas por score descendente
    """
    if len(database_histograms) == 0:
        return []
    
    # Heap para mantener los k mejores resultados
    # Usamos min-heap con scores negativos para mantener los mayores
    heap = []
    
    query_norm = np.linalg.norm(query_histogram)
    if query_norm == 0:
        return []
    
    # Normalizar query
    query_normalized = query_histogram / query_norm
    
    # Búsqueda secuencial sobre todas las imágenes
    for idx, db_hist in enumerate(database_histograms):
        # Calcular similitud coseno
        db_norm = np.linalg.norm(db_hist)
        if db_norm == 0:
            continue
        
        db_normalized = db_hist / db_norm
        similarity = np.dot(query_normalized, db_normalized)
        
        # Mantener solo los k mejores
        if len(heap) < k:
            heapq.heappush(heap, (similarity, idx))
        else:
            # Si el score es mayor que el mínimo del heap, reemplazar
            if similarity > heap[0][0]:
                heapq.heapreplace(heap, (similarity, idx))
    
    # Ordenar por score descendente
    results = sorted(heap, key=lambda x: x[0], reverse=True)
    return [(idx, float(score)) for score, idx in results]


class SequentialSIFTSearch:
    """
    Clase para manejar búsqueda secuencial con SIFT + BOVW.
    """
    
    def __init__(self, histograms_tfidf: np.ndarray, image_paths: List[str], 
                 kmeans_model, idf_weights: np.ndarray):
        """
        Inicializa el buscador secuencial.
        
        Args:
            histograms_tfidf: Matriz de histogramas TF-IDF (n_imágenes, vocab_size)
            image_paths: Lista de rutas a imágenes (debe corresponder con histogramas)
            kmeans_model: Modelo K-Means entrenado para asignar visual words
            idf_weights: Pesos IDF para aplicar a nuevas imágenes
        """
        self.histograms = histograms_tfidf
        self.image_paths = image_paths
        self.kmeans_model = kmeans_model
        self.idf_weights = idf_weights
        self.vocab_size = len(idf_weights)
        
        if len(histograms_tfidf) != len(image_paths):
            raise ValueError("El número de histogramas debe coincidir con el número de imágenes")
    
    def _query_image_to_histogram_tfidf(self, query_image_path: str) -> np.ndarray:
        """
        Convierte una imagen query a histograma TF-IDF.
        
        Args:
            query_image_path: Ruta a la imagen query
            
        Returns:
            Histograma TF-IDF (vocab_size,)
        """
        # Extraer descriptores SIFT (máximo 100 keypoints)
        descriptors = extract_sift_features(query_image_path, max_keypoints=100)
        if descriptors is None or len(descriptors) == 0:
            # Retornar histograma de ceros si no hay descriptores
            return np.zeros(self.vocab_size, dtype=np.float32)
        
        # Asignar a visual words
        visual_words = self.kmeans_model.predict(descriptors)
        
        # Construir histograma (TF)
        hist, _ = np.histogram(visual_words, bins=self.vocab_size, range=(0, self.vocab_size))
        hist = hist.astype(np.float32) / len(descriptors)  # Normalizar
        
        # Aplicar IDF
        hist_tfidf = hist * self.idf_weights
        
        return hist_tfidf
    
    def search(self, query_image_path: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Busca las k imágenes más similares a la query.
        
        Args:
            query_image_path: Ruta a la imagen query
            k: Número de resultados a retornar
            
        Returns:
            Lista de tuplas (ruta_imagen, score) ordenadas por score descendente
        """
        # Convertir query a histograma TF-IDF
        query_hist = self._query_image_to_histogram_tfidf(query_image_path)
        
        # Búsqueda secuencial
        results = sequential_knn_search(query_hist, self.histograms, k)
        
        # Convertir índices a rutas de imágenes
        return [(self.image_paths[idx], score) for idx, score in results]

