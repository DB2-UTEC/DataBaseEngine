# backend/multimedia/metrics.py
"""
Módulo multimedia.metrics
--------------------------------
Define funciones de similitud, distancia y TF-IDF.
"""

import numpy as np
from math import log


def cosine_similarity(v1, v2):
    """Calcula la similitud de coseno entre dos vectores."""
    num = np.dot(v1, v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return num / denom


def euclidean_distance(v1, v2):
    """Distancia euclidiana entre dos vectores."""
    return np.linalg.norm(v1 - v2)


def tfidf(tf, df, N):
    """Cálculo básico de TF-IDF."""
    idf = log((N + 1) / (df + 1)) + 1
    return tf * idf
