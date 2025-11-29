# backend/multimedia/vector_index.py
"""
Módulo multimedia.vector_index
--------------------------------
Implementa un índice vectorial básico para recuperación de imágenes
por similitud, usando la similitud del coseno y una cola de prioridad
para mantener los K resultados más cercanos.
"""

import heapq
import json
import numpy as np
from .metrics import cosine_similarity


class VectorIndex:
    """
    Índice vectorial simple para búsqueda KNN.
    Puede operar sobre embeddings globales (ej. ResNet50).
    """

    def __init__(self):
        self.index = {}  # {object_id: feature_vector}

    def insert(self, object_id: str, vector):
        """Inserta un nuevo vector al índice."""
        self.index[object_id] = np.array(vector, dtype=np.float32)

    def build_from_json(self, json_path: str):
        """Carga los vectores desde un archivo JSON generado por ImageVector."""
        with open(json_path, "r") as f:
            data = json.load(f)
            for obj_id, vec in data.items():
                self.insert(obj_id, vec)
        print(f"[OK] Cargados {len(self.index)} vectores en el índice.")

    def search(self, query_vector, k=5):
        """
        Realiza una búsqueda k-NN usando similitud de coseno.
        Devuelve una lista [(id, score), ...] ordenada por relevancia.
        """
        if not self.index:
            print("[WARN] El índice está vacío.")
            return []

        heap = []
        q = np.array(query_vector, dtype=np.float32)

        for obj_id, vec in self.index.items():
            sim = cosine_similarity(q, vec)
            if len(heap) < k:
                heapq.heappush(heap, (sim, obj_id))
            else:
                heapq.heappushpop(heap, (sim, obj_id))

        # Ordenamos por score descendente (mayor similitud primero)
        results = sorted(heap, key=lambda x: x[0], reverse=True)
        return [(obj_id, float(sim)) for sim, obj_id in results]
