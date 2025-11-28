import os
import json
import csv
import math
import unicodedata
import hashlib
import pickle
import heapq
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any, Iterable

#from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re

# -- CONFIG
OUTPUT_DIR = '../data/indexes/inverted_index_spimi'
BLOCKS_DIR = os.path.join(OUTPUT_DIR, 'blocks')
BINARY_TERMS_DIR = os.path.join(OUTPUT_DIR, 'binary_terms')
VOCAB_MAP_PATH = os.path.join(OUTPUT_DIR, 'vocab_map.json')
DOC_STATS_PATH = os.path.join(OUTPUT_DIR, 'doc_stats.json')
IDF_PATH = os.path.join(OUTPUT_DIR, 'idf.json')
METADATA_PATH = os.path.join(OUTPUT_DIR, 'index_metadata.json')

# funciones basicas de preprocesamiento

stopwords_file = '../data/stopwords/spanish_stopwords.txt'

class Preprocessor:
    def __init__(self, stopwords_file: str = None):
        self.stemmer = SnowballStemmer("spanish")
        self.stopwords = set()
        if stopwords_file and os.path.exists(stopwords_file):
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                self.stopwords = set(line.strip().lower() for line in f if line.strip())
    
    def normalize_text(self, text):
        text = text.lower()
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(ch for ch in text if not unicodedata.combining(ch))
        # permite letras y números y espacios
        text = re.sub(r'[^0-9a-zñ\s]', ' ', text).strip()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        text = self.normalize_text(text)
        tokens = re.findall(r"[a-zñ0-9]+", text)
        if self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        return [self.stemmer.stem(token) for token in tokens]
    
    def compute_bow_with_positions(self, text):
        tokens = self.tokenize_text(text)
        result = {}
        for pos, token in enumerate(tokens):
            entry = result.setdefault(token, {'freq': 0, 'positions': []})
            entry['freq'] += 1
            entry['positions'].append(pos)
        return result
    
# -- utilidades de archivos binarios por termino

def term_to_filename(term: str) -> str:
    h = hashlib.sha1(term.encode('utf-8')).hexdigest()
    return f"{h}.pkl" 

def save_term_binary(term: str, postings: List[Dict[str, Any]]):
    os.makedirs(BINARY_TERMS_DIR, exist_ok=True)
    filename = term_to_filename(term)
    path = os.path.join(BINARY_TERMS_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump({'term': term, 'postings': postings}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return filename

def load_term_binary_by_filename(filename: str) -> Dict[str, Any]:
    path = os.path.join(BINARY_TERMS_DIR, filename)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

# Indexador SPIMI

class SPIMIIndex:
    def __init__(self, datafilename: str, stopwords_file: str = None, max_terms_in_block: int = 20000):
        self.datafilename = datafilename
        self.preproc = Preprocessor(stopwords_file)
        self.max_terms_in_block = max_terms_in_block
        os.makedirs(BLOCKS_DIR, exist_ok=True)
        os.makedirs(BINARY_TERMS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        self.doc_stats = {} 
        self.block_count = 0
        self.block_files = []

    def _write_block_jsonl(self, block_terms: Dict[str, List[Dict[str, Any]]], block_id: int) -> str:
        path = os.path.join(BLOCKS_DIR, f"block_{block_id:06d}.jsonl")
        with open(path, "w", encoding = "utf-8") as f:
            for term, postings in block_terms.items():
                rec = {'term': term, 'postings': postings}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return path
    
    def build_blocks(self):
        block_terms = {}
        unique_terms = 0
        with open(self.datafilename, 'r', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                doc_id = row.get('id') or row.get('doc_id') or row.get('document_id')
                text = row.get('text') or row.get('content') or row.get('body')
                if not doc_id or not text:
                    continue
                bow_pos = self.preproc.compute_bow_with_positions(text)
                self.doc_stats[doc_id] = sum(info['freq'] for info in bow_pos.values())
                
                for term, info in bow_pos.items():
                    if term not in block_terms:
                        unique_terms += 1
                        block_terms[term] = []
                    block_terms[term].append({'doc_id': doc_id, 'freq': info['freq'], 'positions': info['positions']})
                
                if unique_terms >= self.max_terms_in_block:
                    block_terms_sorted = dict(sorted(block_terms.items()))
                    path = self._write_block_jsonl(block_terms_sorted, self.block_count)
                    self.block_files.append(path)
                    self.block_count += 1
                    block_terms = {}
                    unique_terms = 0
        if block_terms:
            block_terms_sorted = dict(sorted(block_terms.items()))
            path = self._write_block_jsonl(block_terms_sorted, self.block_count)
            self.block_files.append(path)
            self.block_count += 1
            block_terms.clear()

    def _open_block_iter(self, block_path: str):
        f = open(block_path, 'r', encoding='utf-8')
        def gen():
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                yield rec['term'], rec['postings']
            f.close()
        return gen()
    
    def _merge_blocks_and_write_binary_terms(self):
        iterators = []
        for path in self.block_files:
            it = self._open_block_iter(path)
            try:
                term, postings = next(it)
                iterators.append({'term': term, 'postings': postings, 'iter': it, 'path': path})
            except StopIteration:
                continue
        heap = []
        for idx, itrec in enumerate(iterators):
            heapq.heappush(heap, (itrec['term'], idx))
        heapq.heapify(heap)

        vocab_map = {}
        doc_freqs = {}

        while heap:
            current_term, idx0 = heapq.heappop(heap)
            collected_postings = []
            itrec = iterators[idx0]
            collected_postings.extend(itrec['postings'])

            try:
                term_next, postings_next = next(itrec['iter'])
                iterators[idx0]['term'] = term_next
                iterators[idx0]['postings'] = postings_next
                heapq.heappush(heap, (term_next, idx0))
            except StopIteration:
                iterators[idx0]['term'] = None
            
            while heap and heap[0][0] == current_term:
                _, idx_i = heapq.heappop(heap)
                itrec_i = iterators[idx_i]
                collected_postings.extend(itrec_i['postings'])
                try:
                    tnext, pnext = next(itrec_i['iter'])
                    iterators[idx_i]['term'] = tnext
                    iterators[idx_i]['postings'] = pnext
                    heapq.heappush(heap, (tnext, idx_i))
                except StopIteration:
                    iterators[idx_i]['term'] = None

            postings_by_doc = {}
            for p in collected_postings:
                doc = p['doc_id']
                if doc not in postings_by_doc:
                    postings_by_doc[doc] = {'doc_id': doc, 'freq': p.get('freq', 0), 'positions': list(p.get('positions', []))}
                else:
                    postings_by_doc[doc]['freq'] += p.get('freq', 0)
                    postings_by_doc[doc]['positions'].extend(p.get('positions', []))
            merged_postings = []
            for docid, v in postings_by_doc.items():
                v['positions'] = sorted(v['positions'])
                merged_postings.append(v)
            try:
                merged_postings.sort(key=lambda x: x['doc_id'])
            except TypeError:
                merged_postings.sort(key=lambda x: str(x['doc_id']))

            filename = save_term_binary(current_term, merged_postings)
            vocab_map[current_term] = filename
            doc_freqs[current_term] = len(merged_postings)
        
        # calcular idf
        N = len(self.doc_stats)
        idf = {}
        for term, df in doc_freqs.items():
            idf_val = math.log(1.0 + (N / df)) if df > 0 else 0.0
            idf[term] = idf_val
        
        with open(VOCAB_MAP_PATH, "w", encoding = "utf-8") as f:
            json.dump(vocab_map, f, ensure_ascii=False)
        with open(DOC_STATS_PATH, "w", encoding = "utf-8") as f:
            json.dump(self.doc_stats, f, ensure_ascii=False)
        with open(IDF_PATH, "w", encoding = "utf-8") as f:
            json.dump(idf, f, ensure_ascii=False)
        meta = {
            'N': N,
            'num_terms': sum(doc_freqs.values()),
            'block_files': self.block_files,
            'binary_terms_dir': BINARY_TERMS_DIR
        }
        with open(METADATA_PATH, "w", encoding = "utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        return vocab_map, idf
    
    # -- consultas

    def load_vocab_map(self) -> Dict[str, str]:
        with open(VOCAB_MAP_PATH, "r", encoding = "utf-8") as f:
            vocab_map = json.load(f)
        return vocab_map

    def get_postings(self, term: str) -> List[Dict[str, Any]]:
        term_stem = self.preproc.stemmer.stem(term.lower())
        vocab = self.load_vocab_map()
        filename = vocab.get(term_stem)
        if not filename:
            return []
        rec = load_term_binary_by_filename(filename)
        return rec['postings']
    
    def compute_tfidf_for_doc(self, doc_id: str) -> Dict[str, float]:
        with open(IDF_PATH, "r", encoding = "utf-8") as f:
            idf = json.load(f)
        volab = self.load_vocab_map()
        tfidf = {}
        doc_len = self.doc_stats.get(doc_id)
        if not doc_len:
            return {}
        for term, filename in volab.items():
            rec = load_term_binary_by_filename(filename)
            for p in rec['postings']:
                if p['doc_id'] == doc_id:
                    tf = p['freq'] / doc_len
                    tfidf[term] = tf * idf.get(term, 0.0)
                    break
        return tfidf
    
    # -- Orquestador

    def build_index(self):
        self.build_blocks()
        vocab_map, idf = self._merge_blocks_and_write_binary_terms()
        return vocab_map, idf
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SPIMI (k-way merge) indexer")
    parser.add_argument("--data", type=str, default="../data/news_text_dataset.csv", help="CSV input")
    parser.add_argument("--stopwords", type=str, default="../data/stopwords/spanish_stopwords.txt", help="archivo stopwords")
    parser.add_argument("--max_terms_in_block", type=int, default=20000, help="términos únicos por bloque (aprox)")
    args = parser.parse_args()

    indexer = SPIMIIndex(args.data, stopwords_file=args.stopwords, max_terms_in_block=args.max_terms_in_block)
    vocab_map, idf = indexer.build_index()
    print("Index build complete.")
    print(f"Terms: {len(vocab_map)}, Documents: {len(indexer.doc_stats)}")
    
    print(f"Metadata saved to {METADATA_PATH}")


