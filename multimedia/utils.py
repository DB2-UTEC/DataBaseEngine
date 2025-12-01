# backend/multimedia/utils.py
"""
Módulo multimedia.utils
--------------------------------
Funciones de utilidad para manejo de vectores e índices.
"""

import os
import json
import numpy as np


def load_feature_database(path):
    """Carga el archivo JSON con los features de imágenes."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in data.items()}


def save_feature_database(data, path):
    """Guarda un diccionario de features (obj_id → vector) en JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, list):
            serializable[k] = v
        else:
            # Si es otro tipo (ej. tensor de PyTorch), convertir a numpy primero
            serializable[k] = np.array(v).tolist()
    with open(path, "w") as f:
        json.dump(serializable, f)
    print(f"[OK] Guardada base de features en {path}")


def normalize_vector(vec):
    """Normaliza un vector a norma 1 (unitario)."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec
