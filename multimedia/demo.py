# multimedia/demo.py
"""
Demo del módulo multimedia.
--------------------------------
Ejecuta la extracción de características, creación del índice y búsqueda por similitud.
Incluye demostraciones para ResNet50 y SIFT + Bag of Visual Words.
"""

import os
import time
import json
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from multimedia.image_vector import ImageVector
from multimedia.vector_index import VectorIndex
from multimedia.utils import save_feature_database
from multimedia.sift_features import extract_sift_features, extract_sift_from_dataset, get_image_paths
from multimedia.bovw import (
    build_visual_codebook, images_to_histograms, apply_tfidf_to_histograms,
    save_codebook, load_codebook, save_histograms, load_histograms,
    add_images_to_codebook
)
from multimedia.sequential_search import SequentialSIFTSearch
from multimedia.sift_inverted_index import SIFTInvertedIndex
from multimedia.config import MAX_KEYPOINTS, VOCAB_SIZE

IMAGE_DIR = "./data/imagenes"
DB_PATH = "./multimedia/database/features.json"

# Rutas para SIFT + BOVW
CODEBOOK_PATH = "./multimedia/database/codebook.pkl"
HISTOGRAMS_PATH = "./multimedia/database/histograms.npz"
INVERTED_INDEX_PATH = "./multimedia/database/inverted_index.pkl"


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


def demo_sift_sequential():
    """
    Demostrar búsqueda con SIFT + KNN secuencial.
    """
    print("\n" + "="*60)
    print("DEMO: SIFT + Búsqueda Secuencial")
    print("="*60)
    
    # Obtener rutas de imágenes
    image_paths = get_image_paths(IMAGE_DIR)
    if len(image_paths) == 0:
        print("[ERROR] No se encontraron imágenes en el directorio")
        return
    
    print(f"[INFO] Procesando {len(image_paths)} imágenes...")
    
    vocab_size = VOCAB_SIZE
    max_keypoints = MAX_KEYPOINTS
    
    # Paso 1: Extraer descriptores SIFT
    print(f"\n[PASO 1] Extrayendo descriptores SIFT (max_keypoints={max_keypoints})...")
    sift_descriptors_list = extract_sift_from_dataset(image_paths, max_keypoints=max_keypoints)
    
    # Paso 2: Construir o actualizar codebook
    if not os.path.exists(CODEBOOK_PATH):
        print(f"\n[PASO 2] Construyendo codebook visual (vocab_size={vocab_size})...")
        kmeans_model, training_descriptors = build_visual_codebook(sift_descriptors_list, vocab_size=vocab_size)
        save_codebook(kmeans_model, CODEBOOK_PATH, training_descriptors=training_descriptors)
    else:
        print("\n[PASO 2] Codebook existente encontrado...")
        kmeans_model, _ = load_codebook(CODEBOOK_PATH)
        vocab_size = kmeans_model.n_clusters
        
        # Verificar si hay nuevas imágenes que agregar
        # (En una implementación completa, se compararían las imágenes procesadas con las nuevas)
        # Por ahora, asumimos que si existe el codebook, las imágenes ya están procesadas
        print("[INFO] Usando codebook existente. Para agregar nuevas imágenes, use add_images_to_codebook()")
    
    # Paso 3: Generar histogramas (si no existen)
    if not os.path.exists(HISTOGRAMS_PATH):
        print("\n[PASO 3] Generando histogramas...")
        histograms = images_to_histograms(sift_descriptors_list, kmeans_model, vocab_size)
        histograms_tfidf, idf_weights = apply_tfidf_to_histograms(histograms)
        save_histograms(histograms_tfidf, idf_weights, HISTOGRAMS_PATH)
    else:
        print("\n[PASO 3] Cargando histogramas existentes...")
        histograms_tfidf, idf_weights = load_histograms(HISTOGRAMS_PATH)
    
    # Paso 4: Crear buscador secuencial
    print("\n[PASO 4] Inicializando buscador secuencial...")
    searcher = SequentialSIFTSearch(histograms_tfidf, image_paths, kmeans_model, idf_weights)
    
    # Paso 5: Realizar búsqueda
    query_path = image_paths[0]  # Usar primera imagen como query
    print(f"\n[PASO 5] Buscando imágenes similares a: {os.path.basename(query_path)}")
    
    start_time = time.time()
    results = searcher.search(query_path, k=5)
    elapsed_time = time.time() - start_time
    
    print(f"\n🔍 Resultados (tiempo: {elapsed_time:.4f}s):")
    for i, (img_path, score) in enumerate(results, 1):
        print(f"  {i}. {os.path.basename(img_path)} → similitud = {score:.4f}")
    
    # Visualizar resultados
    try:
        fig, axes = plt.subplots(1, len(results) + 1, figsize=(15, 3))
        axes[0].imshow(mpimg.imread(query_path))
        axes[0].set_title("Query")
        axes[0].axis('off')
        
        for i, (img_path, score) in enumerate(results, 1):
            axes[i].imshow(mpimg.imread(img_path))
            axes[i].set_title(f"#{i}\n{score:.3f}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig("./multimedia/database/demo_sift_sequential.png", dpi=150, bbox_inches='tight')
        print("\n[OK] Resultados guardados en: ./multimedia/database/demo_sift_sequential.png")
        plt.close()
    except Exception as e:
        print(f"[WARN] No se pudo visualizar: {e}")


def demo_sift_inverted_index():
    """
    Demostrar búsqueda con SIFT + índice invertido.
    """
    print("\n" + "="*60)
    print("DEMO: SIFT + Índice Invertido")
    print("="*60)
    
    # Obtener rutas de imágenes
    image_paths = get_image_paths(IMAGE_DIR)
    if len(image_paths) == 0:
        print("[ERROR] No se encontraron imágenes en el directorio")
        return
    
    # Cargar codebook
    if not os.path.exists(CODEBOOK_PATH):
        print("[ERROR] Codebook no encontrado. Ejecute demo_sift_sequential() primero.")
        return
    kmeans_model, _ = load_codebook(CODEBOOK_PATH)
    vocab_size = kmeans_model.n_clusters
    
    # Cargar histogramas
    if not os.path.exists(HISTOGRAMS_PATH):
        print("[ERROR] Histogramas no encontrados. Ejecute demo_sift_sequential() primero.")
        return
    histograms_tfidf, idf_weights = load_histograms(HISTOGRAMS_PATH)
    
    # Construir o cargar índice invertido
    if not os.path.exists(INVERTED_INDEX_PATH):
        print("\n[PASO 1] Construyendo índice invertido...")
        inverted_index = SIFTInvertedIndex(vocab_size)
        inverted_index.set_idf(idf_weights)
        inverted_index.build_index(histograms_tfidf, image_paths)
        inverted_index.save(INVERTED_INDEX_PATH)
    else:
        print("\n[PASO 1] Cargando índice invertido existente...")
        inverted_index = SIFTInvertedIndex(vocab_size)
        inverted_index.load(INVERTED_INDEX_PATH)
        inverted_index.set_idf(idf_weights)
    
    # Realizar búsqueda
    query_path = image_paths[0]  # Usar primera imagen como query
    print(f"\n[PASO 2] Buscando imágenes similares a: {os.path.basename(query_path)}")
    
    start_time = time.time()
    results = inverted_index.search_by_image_path(query_path, kmeans_model, k=5)
    elapsed_time = time.time() - start_time
    
    print(f"\n🔍 Resultados (tiempo: {elapsed_time:.4f}s):")
    for i, (img_path, score) in enumerate(results, 1):
        print(f"  {i}. {os.path.basename(img_path)} → similitud = {score:.4f}")
    
    # Visualizar resultados
    try:
        fig, axes = plt.subplots(1, len(results) + 1, figsize=(15, 3))
        axes[0].imshow(mpimg.imread(query_path))
        axes[0].set_title("Query")
        axes[0].axis('off')
        
        for i, (img_path, score) in enumerate(results, 1):
            axes[i].imshow(mpimg.imread(img_path))
            axes[i].set_title(f"#{i}\n{score:.3f}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig("./multimedia/database/demo_sift_inverted.png", dpi=150, bbox_inches='tight')
        print("\n[OK] Resultados guardados en: ./multimedia/database/demo_sift_inverted.png")
        plt.close()
    except Exception as e:
        print(f"[WARN] No se pudo visualizar: {e}")


def add_new_images_incremental(new_image_paths: List[str], vocab_size: int = None, max_keypoints: int = None):
    """
    Agrega nuevas imágenes al codebook existente de forma incremental.
    Solo procesa las nuevas imágenes y actualiza el codebook.
    
    Args:
        new_image_paths: Lista de rutas a las nuevas imágenes a agregar
        vocab_size: Tamaño del vocabulario (debe coincidir con el codebook existente). Si es None, usa VOCAB_SIZE de config.
        max_keypoints: Número máximo de keypoints por imagen. Si es None, usa MAX_KEYPOINTS de config.
    """
    # Usar valores por defecto de config si no se especifican
    if max_keypoints is None:
        max_keypoints = MAX_KEYPOINTS
    if vocab_size is None:
        vocab_size = VOCAB_SIZE
    
    print("\n" + "="*60)
    print("AGREGAR NUEVAS IMÁGENES AL CODEBOOK")
    print("="*60)
    
    if not os.path.exists(CODEBOOK_PATH):
        print("[ERROR] Codebook no encontrado. Use demo_sift_sequential() primero para crear el codebook inicial.")
        return
    
    if len(new_image_paths) == 0:
        print("[WARN] No hay nuevas imágenes para agregar")
        return
    
    print(f"[INFO] Agregando {len(new_image_paths)} nuevas imágenes...")
    
    # Cargar codebook existente
    kmeans_model, _ = load_codebook(CODEBOOK_PATH)
    existing_vocab_size = kmeans_model.n_clusters
    
    if existing_vocab_size != vocab_size:
        print(f"[WARN] Vocab size mismatch: codebook tiene {existing_vocab_size}, usando {vocab_size}")
        vocab_size = existing_vocab_size
    
    # Extraer descriptores SIFT de las nuevas imágenes
    print(f"\n[PASO 1] Extrayendo descriptores SIFT de nuevas imágenes (max_keypoints={max_keypoints})...")
    new_sift_descriptors_list = extract_sift_from_dataset(new_image_paths, max_keypoints=max_keypoints)
    
    # Actualizar codebook con las nuevas imágenes
    print("\n[PASO 2] Actualizando codebook con nuevas imágenes...")
    updated_kmeans_model = add_images_to_codebook(
        CODEBOOK_PATH,
        new_sift_descriptors_list,
        vocab_size=vocab_size
    )
    
    # Guardar codebook actualizado (con descriptores combinados)
    # Necesitamos cargar los descriptores originales y combinarlos
    _, old_descriptors = load_codebook(CODEBOOK_PATH)
    
    # Combinar descriptores
    new_descriptors = []
    for desc in new_sift_descriptors_list:
        if desc is not None and len(desc) > 0:
            new_descriptors.append(desc)
    if len(new_descriptors) > 0:
        new_descriptors = np.vstack(new_descriptors)
        if old_descriptors is not None and len(old_descriptors) > 0:
            combined_descriptors = np.vstack([old_descriptors, new_descriptors])
        else:
            combined_descriptors = new_descriptors
    else:
        combined_descriptors = old_descriptors
    
    save_codebook(updated_kmeans_model, CODEBOOK_PATH, training_descriptors=combined_descriptors)
    
    # Cargar histogramas existentes
    existing_histograms = None
    existing_image_paths = []
    if os.path.exists(HISTOGRAMS_PATH):
        existing_histograms, existing_idf = load_histograms(HISTOGRAMS_PATH)
        # Obtener rutas de imágenes existentes
        existing_image_paths = get_image_paths(IMAGE_DIR)
        # Filtrar solo las que ya estaban procesadas (asumiendo que todas las del directorio estaban)
        # En una implementación completa, se guardaría una lista de imágenes procesadas
        print(f"[INFO] Encontrados {len(existing_image_paths)} histogramas existentes")
    
    # Generar histogramas para las nuevas imágenes
    print("\n[PASO 3] Generando histogramas para nuevas imágenes...")
    new_histograms = images_to_histograms(new_sift_descriptors_list, updated_kmeans_model, vocab_size)
    
    # Combinar histogramas
    if existing_histograms is not None:
        print("[INFO] Combinando histogramas existentes con nuevos...")
        all_histograms = np.vstack([existing_histograms, new_histograms])
        all_image_paths = existing_image_paths + new_image_paths
    else:
        all_histograms = new_histograms
        all_image_paths = new_image_paths
    
    # Recalcular TF-IDF con todos los histogramas
    print("\n[PASO 4] Recalculando TF-IDF...")
    histograms_tfidf, idf_weights = apply_tfidf_to_histograms(all_histograms)
    
    # Guardar histogramas actualizados
    save_histograms(histograms_tfidf, idf_weights, HISTOGRAMS_PATH)
    
    print(f"\n[OK] {len(new_image_paths)} nuevas imágenes agregadas exitosamente")
    print(f"[INFO] Total de imágenes en el sistema: {len(all_image_paths)}")


def demo_compare_resnet_sift():
    """
    Comparar visualmente resultados de ResNet vs SIFT.
    """
    print("\n" + "="*60)
    print("DEMO: Comparación ResNet50 vs SIFT + BOVW")
    print("="*60)
    
    # Obtener rutas de imágenes
    image_paths = get_image_paths(IMAGE_DIR)
    if len(image_paths) == 0:
        print("[ERROR] No se encontraron imágenes en el directorio")
        return
    
    query_path = image_paths[0]
    print(f"Query: {os.path.basename(query_path)}")
    
    # Búsqueda con ResNet50
    print("\n[1] Búsqueda con ResNet50...")
    if not os.path.exists(DB_PATH):
        build_feature_db()

    extractor = ImageVector()
    query_vec = extractor.extract(query_path)
    
    index = VectorIndex()
    index.build_from_json(DB_PATH)
    
    start_time = time.time()
    resnet_results = index.search(query_vec, k=5)
    resnet_time = time.time() - start_time
    
    print(f"Tiempo: {resnet_time:.4f}s")
    for i, (obj_id, score) in enumerate(resnet_results, 1):
        print(f"  {i}. {obj_id} → {score:.4f}")
    
    # Búsqueda con SIFT + BOVW (secuencial)
    print("\n[2] Búsqueda con SIFT + BOVW (secuencial)...")
    
    # Cargar modelos necesarios
    if not os.path.exists(CODEBOOK_PATH) or not os.path.exists(HISTOGRAMS_PATH):
        print("[ERROR] Modelos SIFT no encontrados. Ejecute demo_sift_sequential() primero.")
        return
    
    kmeans_model, _ = load_codebook(CODEBOOK_PATH)
    histograms_tfidf, idf_weights = load_histograms(HISTOGRAMS_PATH)
    
    searcher = SequentialSIFTSearch(histograms_tfidf, image_paths, kmeans_model, idf_weights)
    
    start_time = time.time()
    sift_results = searcher.search(query_path, k=5)
    sift_time = time.time() - start_time
    
    print(f"Tiempo: {sift_time:.4f}s")
    for i, (img_path, score) in enumerate(sift_results, 1):
        print(f"  {i}. {os.path.basename(img_path)} → {score:.4f}")
    
    # Visualizar comparación
    try:
        fig, axes = plt.subplots(2, 6, figsize=(18, 6))
        
        # Query
        axes[0, 0].imshow(mpimg.imread(query_path))
        axes[0, 0].set_title("Query", fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        # ResNet resultados
        axes[0, 1].text(0.5, 0.5, "ResNet50\nResults", ha='center', va='center', 
                       fontsize=14, fontweight='bold', transform=axes[0, 1].transAxes)
        axes[0, 1].axis('off')
        
        for i, (obj_id, score) in enumerate(resnet_results[:5], 1):
            img_path = os.path.join(IMAGE_DIR, obj_id)
            if os.path.exists(img_path):
                axes[0, i+1].imshow(mpimg.imread(img_path))
                axes[0, i+1].set_title(f"#{i}\n{score:.3f}", fontsize=10)
                axes[0, i+1].axis('off')
        
        # SIFT resultados
        axes[1, 1].text(0.5, 0.5, "SIFT+BOVW\nResults", ha='center', va='center',
                       fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
        
        for i, (img_path, score) in enumerate(sift_results[:5], 1):
            if os.path.exists(img_path):
                axes[1, i+1].imshow(mpimg.imread(img_path))
                axes[1, i+1].set_title(f"#{i}\n{score:.3f}", fontsize=10)
                axes[1, i+1].axis('off')
        
        plt.suptitle(f"Comparación: ResNet50 ({resnet_time:.4f}s) vs SIFT+BOVW ({sift_time:.4f}s)", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("./multimedia/database/demo_comparison.png", dpi=150, bbox_inches='tight')
        print("\n[OK] Comparación guardada en: ./multimedia/database/demo_comparison.png")
        plt.close()
    except Exception as e:
        print(f"[WARN] No se pudo visualizar: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        demo_name = sys.argv[1]
        if demo_name == "resnet":
            # Demo original con ResNet50
            if not os.path.exists(DB_PATH):
                build_feature_db()
            test_query = os.path.join(IMAGE_DIR, os.listdir(IMAGE_DIR)[0])
            test_search(test_query, k=5)
        elif demo_name == "sift_seq":
            demo_sift_sequential()
        elif demo_name == "sift_inv":
            demo_sift_inverted_index()
        elif demo_name == "compare":
            demo_compare_resnet_sift()
        else:
            print("Uso: python demo.py [resnet|sift_seq|sift_inv|compare]")
    else:
        # Por defecto, ejecutar todas las demos
        print("Ejecutando todas las demostraciones...\n")
        
        # Demo ResNet50
        print("\n" + "="*60)
        print("DEMO: ResNet50 (Original)")
        print("="*60)
        if not os.path.exists(DB_PATH):
            build_feature_db()
        test_query = os.path.join(IMAGE_DIR, os.listdir(IMAGE_DIR)[0])
        test_search(test_query, k=5)
        
        # Demo SIFT secuencial
        demo_sift_sequential()
        
        # Demo SIFT índice invertido
        demo_sift_inverted_index()
        
        # Comparación
        demo_compare_resnet_sift()
        
        print("\n" + "="*60)
        print("Todas las demostraciones completadas!")
        print("="*60)
