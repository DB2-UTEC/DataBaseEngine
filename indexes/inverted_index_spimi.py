import os
import json
import csv
import math
import unicodedata
import hashlib
import pickle
import heapq
import re
import sys
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any

# Intenta importar nltk, si falla avisa al usuario
try:
    from nltk.stem.snowball import SnowballStemmer
    _NLTK_AVAILABLE = True
except ImportError:
    print("Aviso: nltk no disponible -> continuando SIN stemming. Para mejor calidad instalar: pip install nltk")
    SnowballStemmer = None
    _NLTK_AVAILABLE = False

# --- CONFIGURACIÓN ---
OUTPUT_DIR = '../data/indexes/inverted_index_spimi'
BLOCKS_DIR = os.path.join(OUTPUT_DIR, 'blocks')
BINARY_TERMS_DIR = os.path.join(OUTPUT_DIR, 'binary_terms')
VOCAB_MAP_PATH = os.path.join(OUTPUT_DIR, 'vocab_map.json')
DOC_NORMS_PATH = os.path.join(OUTPUT_DIR, 'doc_norms.json')
IDF_PATH = os.path.join(OUTPUT_DIR, 'idf.json')

# --- 1. PREPROCESAMIENTO ---

class Preprocessor:
    def __init__(self, stopwords_file: str = None):
        self.stemmer = SnowballStemmer("spanish") if _NLTK_AVAILABLE else None
        self.stopwords = set()
        if stopwords_file and os.path.exists(stopwords_file):
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                self.stopwords = set(line.strip().lower() for line in f if line.strip())
    
    def normalize_text(self, text):
        if not text: return ""
        text = str(text).lower()
        # Normalización Unicode para eliminar tildes (á -> a)
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(ch for ch in text if not unicodedata.combining(ch))
        # Mantener solo letras y números
        text = re.sub(r'[^0-9a-zñ\s]', ' ', text).strip()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        text = self.normalize_text(text)
        if not text: return []
        tokens = text.split() # Split simple es más rápido que re.findall para bloques grandes
        
        final_tokens = []
        for token in tokens:
            if self.stopwords and token in self.stopwords:
                continue
            # Stemming
            if self.stemmer:
                stem = self.stemmer.stem(token)
            else:
                stem = token
            if stem:
                final_tokens.append(stem)
        return final_tokens

# --- 2. GESTIÓN DE BINARIOS ---

def term_to_filename(term: str) -> str:
    # Usamos Hash SHA1 para evitar nombres de archivo inválidos o muy largos
    h = hashlib.sha1(term.encode('utf-8')).hexdigest()
    return f"{h}.pkl" 

# --- 3. INDEXADOR (SPIMI) ---

class SPIMIIndex:
    def __init__(self, datafilename: str, stopwords_file: str = None, max_terms_in_block: int = 100000, text_col: str = 'text', title_col: str = 'title', id_col: str = 'id'):
        self.datafilename = datafilename
        self.preproc = Preprocessor(stopwords_file)
        self.max_terms_in_block = max_terms_in_block # Ajustable según RAM disponible
        
        # Cada indice escribe sus outputs en OUTPUT_DIR/<basename>
        base = os.path.splitext(os.path.basename(datafilename))[0]
        self.output_subdir = os.path.join(OUTPUT_DIR, base)
        self.blocks_dir = os.path.join(self.output_subdir, 'blocks')
        self.binary_terms_dir = os.path.join(self.output_subdir, 'binary_terms')
        os.makedirs(self.blocks_dir, exist_ok=True)
        os.makedirs(self.output_subdir, exist_ok=True)

        self.total_docs = 0
        self.block_count = 0
        self.block_files = []
        self.text_col = text_col
        self.title_col = title_col
        self.id_col = id_col

        # rutas de metadatos por indice
        self.vocab_map_path = os.path.join(self.output_subdir, 'vocab_map.json')
        self.doc_norms_path =  os.path.join(self.output_subdir, 'doc_norms.json')
        self.idf_path = os.path.join(self.output_subdir, 'idf.json')    
    
    def _save_term_binary(self, term: str, postings: List[Dict[str, Any]]) -> str:
        """Guarda la posting list final de un término en un archivo binario individual."""
        os.makedirs(self.binary_terms_dir, exist_ok=True)
        filename = term_to_filename(term)
        path = os.path.join(self.binary_terms_dir, filename)
        with open(path, 'wb') as f:
            pickle.dump({'term': term, 'postings': postings}, f, protocol=pickle.HIGHEST_PROTOCOL)
        return filename
    
    def _load_term_binary_by_filename(self, filename: str) -> Dict[str, Any]:
        """Carga en RAM solo la posting list solicitada."""
        path = os.path.join(self.binary_terms_dir, filename)
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _write_block_disk(self, block_terms: Dict[str, List[Dict[str, Any]]]) -> str:
        """Escribe un bloque temporal en disco ordenado alfabéticamente."""
        path = os.path.join(self.blocks_dir, f"block_{self.block_count:04d}.jsonl")
        print(f"   --> Escribiendo Bloque {self.block_count} con {len(block_terms)} términos en {self.blocks_dir}...")
        with open(path, "w", encoding = "utf-8") as f:
            # Ordenamos los términos antes de escribir para facilitar el Merge
            sorted_terms = sorted(block_terms.keys())
            for term in sorted_terms:
                postings = block_terms[term]
                rec = {'term': term, 'postings': postings}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return path
    
    def build_blocks(self):
        """FASE 1: Inversión en memoria y escritura de bloques."""
        print(f">>> Iniciando Fase 1: Creación de Bloques SPIMI desde {self.datafilename}...")
        
        # Estructura en memoria: term -> list of {doc_id, freq}
        current_block = defaultdict(list)
        current_postings_count = 0
        
        try:
            with open(self.datafilename, 'r', encoding='utf-8') as fh:
                reader = csv.DictReader(fh)
                
                for row in reader:
                    # Detección flexible de columnas
                    doc_id = row.get(self.id_col)
                    title = row.get(self.title_col, '')
                    text = row.get(self.text_col, '')
                    
                    # Concatenar título y texto
                    full_text = f"{title} {text}"
                    
                    if not doc_id or not full_text.strip():
                        continue
                    
                    # Contamos el documento
                    self.total_docs += 1
                    tokens = self.preproc.tokenize_text(full_text)
                    
                    # Contamos frecuencia local en el documento
                    term_freqs = Counter(tokens)
                    
                    for term, freq in term_freqs.items():
                        current_block[term].append({'doc_id': doc_id, 'freq': freq})
                        current_postings_count += 1
                    
                    # Si superamos el límite de memoria (simulado por conteo de postings)
                    if current_postings_count >= self.max_terms_in_block:
                        path = self._write_block_disk(current_block)
                        self.block_files.append(path)
                        self.block_count += 1
                        current_block.clear()
                        current_postings_count = 0

                # Escribir el último bloque si quedó algo
                if current_block:
                    path = self._write_block_disk(current_block)
                    self.block_files.append(path)
                    self.block_count += 1
                    
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {self.datafilename}")
            sys.exit(1)

    def _open_block_stream(self, block_path: str):
        """Generador que lee un bloque línea por línea (stream) para no cargar todo en RAM."""
        f = open(block_path, 'r', encoding='utf-8')
        def gen():
            for line in f:
                if not line.strip(): continue
                yield json.loads(line)
            f.close()
        return gen()

    def merge_blocks(self):
        """FASE 2: Merge (k-way merge), cálculo de TF-IDF y Normas."""
        print(">>> Iniciando Fase 2: Fusión de Bloques y Cálculo de Pesos...")
        
        # Min-Heap para el algoritmo k-way merge
        heap = []
        iterators = []
        
        # Inicializar iteradores para cada bloque
        for i, path in enumerate(self.block_files):
            iterator = self._open_block_stream(path)
            try:
                first_rec = next(iterator) # {term: "...", postings: [...]}
                # Guardamos: (término, index_bloque, registro_actual, iterador)
                heapq.heappush(heap, (first_rec['term'], i, first_rec, iterator))
            except StopIteration:
                pass # Bloque vacío
        
        vocab_map = {} # term -> filename
        idf_map = {}   # term -> idf
        doc_norms = defaultdict(float) # doc_id -> sum(weight^2)
        
        N = self.total_docs
        
        while heap:
            # Extraer el término lexicográficamente menor de todos los bloques abiertos
            current_term, block_idx, current_rec, current_iter = heapq.heappop(heap)
            
            # Recolectar todas las postings para ESTE término de todos los bloques que lo tengan
            merged_postings =list(current_rec['postings'])
            
            # Avanzar el iterador del bloque que acabamos de usar
            try:
                next_rec = next(current_iter)
                heapq.heappush(heap, (next_rec['term'], block_idx, next_rec, current_iter))
            except StopIteration:
                pass # Ese bloque se terminó

            # Revisar si el siguiente en el heap es el MISMO término (viene de otro bloque)
            while heap and heap[0][0] == current_term:
                _, other_idx, other_rec, other_iter = heapq.heappop(heap)
                merged_postings.extend(other_rec['postings'])
                
                # Avanzar ese iterador también
                try:
                    nxt = next(other_iter)
                    heapq.heappush(heap, (nxt['term'], other_idx, nxt, other_iter))
                except StopIteration:
                    pass
            
            # --- PROCESAMIENTO DEL TÉRMINO UNIFICADO ---
            
            # 1. Consolidar por doc_id (sumar frecuencias)
            freq_map = defaultdict(int)
            for p in merged_postings:
                # Normalizamos doc_id y frecuencia
                doc_id = str(p.get('doc_id'))
                raw_freq = p.get('freq', 0)
                freq_map[doc_id] += raw_freq

            # Como leemos doc a doc, un doc_id solo aparece una vez por término normalmente.
            
            # 2. Calcular TF-IDF y actualizar Normas
            df = len(freq_map)
            idf = math.log10(N / df) if df > 0 and N > 0 else 0
            idf_map[current_term] = idf
            
            final_postings = []
            for doc_id, total_freq in freq_map.items():
                
                # TF Logarítmico: 1 + log10(f)
                tf = 1 + math.log10(total_freq) if total_freq > 0 else 0
                weight = tf * idf
                
                # Acumular cuadrado del peso para la norma del documento
                doc_norms[doc_id] += (weight ** 2)
                
                # Guardamos el peso pre-calculado para búsqueda rápida
                final_postings.append({'doc_id': doc_id, 'weight': weight})
            
            # 3. Guardar a disco (Binary File)
            # Ordenamos por doc_id para búsquedas más estructuradas si fuera necesario
            final_postings.sort(key=lambda x: str(x['doc_id']))
            filename = self._save_term_binary(current_term, final_postings)
            vocab_map[current_term] = filename

        # Finalizar cálculo de normas (Raíz cuadrada)
        print(">>> Finalizando cálculo de Normas Euclidianas...")
        final_doc_norms = {str(d): math.sqrt(val) for d, val in doc_norms.items()}
        
        # Guardar metadatos globales
        with open(self.vocab_map_path, "w") as f: json.dump(vocab_map, f)
        with open(self.idf_path, "w") as f: json.dump(idf_map, f)
        with open(self.doc_norms_path, "w") as f: json.dump(final_doc_norms, f)
        
        print("Indexación completada exitosamente.")

    def run(self):
        self.build_blocks()
        self.merge_blocks()

# --- 4. BUSCADOR (Consulta Vectorial) ---
    def load_metadata(self):
        """Carga vocab_map, idf y doc_norms para permitir búsquedas."""
        if not os.path.exists(self.vocab_map_path):
            raise Exception("El índice no existe. Ejecuta con --build primero.")
        with open(self.vocab_map_path, 'r') as f: self.vocab_map = json.load(f)
        with open(self.idf_path, 'r') as f: self.idf_map = json.load(f)
        with open(self.doc_norms_path, 'r') as f: self.doc_norms = json.load(f)
        if not hasattr(self, 'preproc') or self.preproc is None:
            self.preproc = Preprocessor()
        return True
    
    def search(self, query: str, k: int = 10, order: str = 'norm') -> List[Tuple[str, float]]:
        """Consulta vectorial (Cosine Similarity) utilizando los binarios ya generados."""
        # Vectorizar Query
        tokens = self.preproc.tokenize_text(query)
        q_term_counts = Counter(tokens)
        q_weights = {}
        q_norm_sq = 0.0
        for term, freq in q_term_counts.items():
            if term in self.idf_map:
                tf = 1 + math.log10(freq)
                idf = self.idf_map[term]
                w = tf * idf
                q_weights[term] = w
                q_norm_sq += w**2
        q_norm = math.sqrt(q_norm_sq)
        if q_norm == 0:
            return []
        # Acumuladores
        scores = defaultdict(float)
        for term, q_w in q_weights.items():
            filename = self.vocab_map.get(term)
            if filename:
                data = self._load_term_binary_by_filename(filename)
                if data:
                    for posting in data['postings']:
                        doc_id = posting['doc_id']
                        d_w = posting['weight']
                        scores[doc_id] += (q_w * d_w) # Producto punto
        results = []
        for doc_id, dot_product in scores.items():
            d_norm = self.doc_norms.get(doc_id, 0)
            if d_norm > 0:
                cosine = dot_product / (q_norm * d_norm)
                results.append((doc_id, cosine))
        # Ordenar por score (norm -> predeterminado como) y limitar a k
        if order == 'desc':
            results.sort(key=lambda x: x[1], reverse=True)
        elif order == 'asc':
            results.sort(key=lambda x: x[1])
        elif order == 'norm':
            # dejarlo como está, sin ordenar
            pass
        else:
            raise ValueError("order debe ser 'asc', 'desc' o 'norm'")
        if k is None:
            k = len(results)
        return results[:k]


# --- MAIN ---

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Motor de Búsqueda SPIMI")
    parser.add_argument("--data", type=str, default="../data/news_text_dataset.csv", help="CSV input")
    parser.add_argument("--col_text", type=str, default="text", help="Nombre de la columna de texto")
    parser.add_argument("--col_title", type=str, default="title", help="Nombre de la columna de título")
    parser.add_argument("--col_id", type=str, default="id", help="Nombre de la columna de ID de documento")

    parser.add_argument("--stopwords", type=str, default="../data/stopwords/spanish_stopwords.txt", help="archivo stopwords")
    parser.add_argument("--max_terms_in_block", type=int, default=20000, help="términos únicos por bloque (aprox)")
    parser.add_argument("--build", action="store_true", help="Construir el índice")
    parser.add_argument("--query", type=str, help="Ejecutar una consulta")
    parser.add_argument("--topk", type=int, help="Resultados a mostrar")
    parser.add_argument("--order", type=str, choices=['asc', 'desc', 'norm'], default='norm', help="Orden de resultados")
    
    args = parser.parse_args()

    # Si no hay argumentos, mostrar ayuda o ejecución por defecto
    if not args.build and not args.query:
        print("Modo interactivo.")
        print("1. Para construir índice: python script.py --build --data news_text_dataset.csv")
        print("2. Para buscar: python script.py --query 'tu consulta'")
    
    if args.build:
        print(f"Construyendo índice desde: {args.data}")
        idx = SPIMIIndex(args.data, args.stopwords, 
                         args.max_terms_in_block, 
                         text_col=args.col_text, 
                         title_col=args.col_title, 
                         id_col=args.col_id)
        idx.run()
        
    if args.query:
        idx = SPIMIIndex(args.data, args.stopwords, 
                         args.max_terms_in_block, 
                         text_col=args.col_text, 
                         title_col=args.col_title, 
                         id_col=args.col_id)    
        idx.load_metadata()
        res = idx.search(args.query, k=args.topk, order=args.order)
        print(f"\nResultados para: '{args.query}'")
        print("-" * 50)
        for i, (did, score) in enumerate(res, 1):
            print(f"{i}. Score: {score:.4f} | DocID: {did}")

## 
## codigo para construir
## python3 inverted_index_spimi.py --build  --data ../data/news_text_dataset.csv --stopwords ../data/stopwords/spanish_stopwords.txt --max_terms_in_block 20000 --col_text text --col_title title --col_id id

## codigo para buscar
## python3 inverted_index_spimi.py --query "tu consulta aquí"

## codigo para buscar & topk
## python3 inverted_index_spimi.py --query "tu consulta aquí" --topk 5

## codigo para buscar & topk & order
## python3 inverted_index_spimi.py --query "tu consulta aquí" --topk 5 --order desc