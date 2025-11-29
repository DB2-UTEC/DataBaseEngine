# backend/multimedia/demo.py
"""
Demo del módulo multimedia.
--------------------------------
Ejecuta la extracción de características, creación del índice y búsqueda por similitud.
"""

import os
from multimedia.image_vector import ImageVector
from multimedia.vector_index import VectorIndex
from multimedia.utils import save_feature_database
import json

IMAGE_DIR = "./multimedia/images"
DB_PATH = "./multimedia/database/features.json"


def build_feature_db():
    extractor = ImageVector()
    db = {}
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    for fname in os.listdir(IMAGE_DIR):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(IMAGE_DIR, fname)
            print(f"Extrayendo features de {fname}...")
            vec = extractor.extract(path)
            if vec is not None:
                db[fname] = vec.tolist()

    save_feature_database(db, DB_PATH)


def test_search(query_image_path, k=5):
    # Extraer vector de la imagen de consulta
    extractor = ImageVector()
    query_vec = extractor.extract(query_image_path)

    # Cargar índice
    index = VectorIndex()
    index.build_from_json(DB_PATH)

    # Buscar las k imágenes más similares
    results = index.search(query_vec, k)
    print("\n🔍 Resultados más similares:")
    for obj_id, score in results:
        print(f"{obj_id} → similitud = {score:.4f}")


if __name__ == "__main__":
    # Paso 1: generar base de features
    if not os.path.exists(DB_PATH):
        build_feature_db()

    # Paso 2: probar búsqueda
    test_query = os.path.join(IMAGE_DIR, os.listdir(IMAGE_DIR)[0])  # usa una imagen del dataset
    test_search(test_query, k=5)
