# multimedia/sift_inverted_index.py
"""
Módulo multimedia.sift_inverted_index
--------------------------------
Índice invertido para búsqueda eficiente con SIFT + BOVW.
"""

import heapq
import numpy as np
import pickle
import os
from typing import List, Tuple, Dict
from .sift_features import extract_sift_features


class SIFTInvertedIndex:
    """
    Índice invertido para búsqueda eficiente con SIFT + BOVW.
    """
    
    def __init__(self, vocab_size: int):
        """
        Inicializa el índice invertido.
        
        Args:
            vocab_size: Tamaño del vocabulario visual
        """
        self.vocab_size = vocab_size
        self.index: Dict[int, List[Tuple[int, float]]] = {}  # visual_word_id -> [(image_id, tf_weight)]
        self.doc_norms: Dict[int, float] = {}  # image_id -> norma para similitud coseno
        self.idf: np.ndarray = None  # pesos IDF (vocab_size,)
        self.image_paths: List[str] = []  # image_id -> ruta de imagen
    
    def build_index(self, histograms_tfidf: np.ndarray, image_ids: List[str]):
        """
        Construye índice invertido desde histogramas TF-IDF.
        
        Args:
            histograms_tfidf: Matriz de histogramas TF-IDF (n_imágenes, vocab_size)
            image_ids: Lista de identificadores de imágenes (rutas o nombres)
        """
        n_images, vocab_size = histograms_tfidf.shape
        
        if vocab_size != self.vocab_size:
            raise ValueError(f"Vocab size mismatch: {vocab_size} != {self.vocab_size}")
        
        print(f"[INFO] Construyendo índice invertido para {n_images} imágenes...")
        
        # Inicializar posting lists
        self.index = {i: [] for i in range(vocab_size)}
        self.image_paths = image_ids.copy()
        
        # Construir posting lists: para cada visual word, guardar imágenes que lo contienen
        for img_id in range(n_images):
            hist = histograms_tfidf[img_id]
            
            # Calcular norma del documento para similitud coseno
            doc_norm = np.linalg.norm(hist)
            self.doc_norms[img_id] = doc_norm if doc_norm > 0 else 1.0
            
            # Para cada visual word con peso > 0, agregar a posting list
            for word_id in range(vocab_size):
                tf_weight = hist[word_id]
                if tf_weight > 0:
                    self.index[word_id].append((img_id, tf_weight))
        
        print(f"[OK] Índice invertido construido")
        print(f"[INFO] Posting lists promedio: {np.mean([len(self.index[i]) for i in range(vocab_size)]):.1f} entradas")
    
    def set_idf(self, idf_weights: np.ndarray):
        """Establece los pesos IDF."""
        if len(idf_weights) != self.vocab_size:
            raise ValueError(f"IDF size mismatch: {len(idf_weights)} != {self.vocab_size}")
        self.idf = idf_weights.copy()
    
    def search(self, query_histogram_tfidf: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """
        Búsqueda eficiente usando índice invertido.
        
        Args:
            query_histogram_tfidf: Histograma TF-IDF de la query (vocab_size,)
            k: Número de resultados a retornar
            
        Returns:
            Lista de tuplas (image_id, score) ordenadas por score descendente
        """
        if self.idf is None:
            raise ValueError("IDF weights no han sido establecidos. Use set_idf() primero.")
        
        # Normalizar query
        query_norm = np.linalg.norm(query_histogram_tfidf)
        if query_norm == 0:
            return []
        query_normalized = query_histogram_tfidf / query_norm
        
        # Acumulador de scores por imagen
        scores: Dict[int, float] = {}
        
        # Para cada visual word en la query, acceder a su posting list
        for word_id in range(self.vocab_size):
            query_weight = query_normalized[word_id]
            if query_weight == 0:
                continue
            
            # Procesar posting list de este visual word
            for img_id, tf_weight in self.index[word_id]:
                # Acumular score: query_weight * tf_weight
                if img_id not in scores:
                    scores[img_id] = 0.0
                scores[img_id] += query_weight * tf_weight
        
        # Normalizar scores por norma del documento (similitud coseno)
        for img_id in scores:
            scores[img_id] /= self.doc_norms[img_id]
        
        # Usar heap para mantener top-k
        heap = []
        for img_id, score in scores.items():
            if len(heap) < k:
                heapq.heappush(heap, (score, img_id))
            else:
                if score > heap[0][0]:
                    heapq.heapreplace(heap, (score, img_id))
        
        # Ordenar por score descendente
        results = sorted(heap, key=lambda x: x[0], reverse=True)
        return [(img_id, float(score)) for score, img_id in results]
    
    def search_by_image_path(self, query_image_path: str, kmeans_model, k: int = 10) -> List[Tuple[str, float]]:
        """
        Busca usando una imagen query directamente.
        
        Args:
            query_image_path: Ruta a la imagen query
            kmeans_model: Modelo K-Means para asignar visual words
            k: Número de resultados a retornar
            
        Returns:
            Lista de tuplas (ruta_imagen, score) ordenadas por score descendente
        """
        # Extraer descriptores SIFT (máximo 100 keypoints)
        descriptors = extract_sift_features(query_image_path, max_keypoints=100)
        if descriptors is None or len(descriptors) == 0:
            return []
        
        # Asignar a visual words
        visual_words = kmeans_model.predict(descriptors)
        
        # Construir histograma (TF)
        hist, _ = np.histogram(visual_words, bins=self.vocab_size, range=(0, self.vocab_size))
        hist = hist.astype(np.float32) / len(descriptors)
        
        # Aplicar IDF
        hist_tfidf = hist * self.idf
        
        # Búsqueda con índice invertido
        results = self.search(hist_tfidf, k)
        
        # Convertir image_ids a rutas
        return [(self.image_paths[img_id], score) for img_id, score in results]
    
    def save(self, filepath: str):
        """Guarda el índice invertido en disco."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            'index': self.index,
            'doc_norms': self.doc_norms,
            'idf': self.idf,
            'image_paths': self.image_paths,
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"[OK] Índice invertido guardado en {filepath}")
    
    def load(self, filepath: str):
        """Carga el índice invertido desde disco."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Índice invertido no encontrado: {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.index = data['index']
        self.doc_norms = data['doc_norms']
        self.idf = data['idf']
        self.image_paths = data['image_paths']
        self.vocab_size = data['vocab_size']
        print(f"[OK] Índice invertido cargado desde {filepath}")

