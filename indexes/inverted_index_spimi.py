import os
import json
import csv
import math
import unicodedata
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any, Iterable

from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re

# funciones basicas de preprocesamiento

stopwords_file = 'data/stopwords/spanish_stopwords.txt'

class Preprocessor:
    @staticmethod
    def load_stopwords(file_path: str) -> set:
        with open(file_path, 'r', encoding='utf-8') as f:
            stopwords = set(f.read().splitlines())
            for i in list(stopwords):
                stopwords.add(i.lower())
        return stopwords
    
    def __init__(self, stopwords_file: str):
        self.stopwords = self.load_stopwords(stopwords_file)
        self.stemmer = SnowballStemmer("spanish")
    
    def normalize_text(self, text):
        text = text.lower()
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(ch for ch in text if not unicodedata.combining(ch))
        # permite letras y números y espacios
        text = re.sub(r'[^0-9a-zñ\s]', ' ', text)
        return text
    
    def preprocess(self, text):
        text = self.normalize_text(text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stopwords]
        tokens = [self.stemmer.stem(token) for token in tokens]
        return tokens

    def compute_tf(self, bow: Dict[str, int]) -> Dict[str, float]:
        total_terms = sum(bow.values())
        tf = {term: freq / total_terms for term, freq in bow.items()}
        return tf
    
    def compute_idf(self, doc_freq: Dict[str, int], total_docs: int) -> Dict[str, float]:
        idf = {term: math.log(total_docs / (df + 1)) for term, df in doc_freq.items()}
        return idf
    
    def compute_tfidf(self, bow: Dict[str, int], doc_freq: Dict[str, int], total_docs: int) -> Dict[str, float]:
        tf = self.compute_tf(bow)
        idf = self.compute_idf(doc_freq, total_docs)
        tfidf = {term: tf_val * idf.get(term, 0.0) for term, tf_val in tf.items()}
        return tfidf
    
    def compute_bow_with_positions(self, text):
        text = self.normalize_text(text)
        tokens = word_tokenize(text)
        filtered = [t for t in token if t and t not in self.stopwords]
        result = {}
        for pos, token in enumerate(filtered):
            stem = self.stemmer.stem(token)
            entry = result.setdefault(stem, {'freq': 0, 'positions': []})
            entry['freq'] += 1
            entry['positions'].append(pos)
        return result
    

# Indexador SPIMI

class SPIMIIndex:
    def __init__(self, datafilename: str , block_size: int = 1000, stopwords_file: str = stopwords_file):
        self.output_dir = 'data/indexes/inverted_index_spimi'
        self.block_dir = os.path.join(self.output_dir, 'blocks')
        self.datafilename = datafilename
        self.block_size = block_size
        os.makedirs(self.block_dir, exist_ok=True)
        self.Preprocessor_instance = Preprocessor(stopwords_file)
        self.index_filename = os.path.join(self.output_dir, 'final_index.json')
        self.doc_stats = {} 

    def _write_block(self, block_terms: dict, block_id: int):
        path = os.path.join(self.block_dir, f"block_{block_id}.json")
        with open(path, "w", encoding = "utf-8") as f:
            json.dump(block_terms, f, ensure_ascii=False)
        print(f"Written block {block_id} with {len(block_terms)} terms to {path}")
        return path
    
    def _merge_blocks(self, block_files: list):
        final_index = {}
        for block_file in block_files:
            with open(block_file, "r", encoding = "utf-8") as f:
                block_terms = json.load(f)
                for term, postings in block_terms.items():
                    if term not in final_index:
                        final_index[term] = []
                    final_index[term].extend(postings)
        
        for term, postings in final_index.items():
            postings_by_doc = {}
            for p in postings:
                doc = p['doc_id']
                if doc not in postings_by_doc:
                    postings_by_doc[doc] = {'doc_id': doc, 'freq': p.get('freg', 0), 'positions': list(p.get('positions', []))}
                else:
                    postings_by_doc[doc]['freq'] += p.get('freq', 0)
                    postings_by_doc[doc]['positions'].extend(p.get('positions', []))
            merged = list(postings_by_doc.values())
            merged.sort(key=lambda x: x['doc_id'])
            final_index[term] = merged
            
        final_index_sorted = dict(sorted(final_index.items()))
        final_index_path = os.path.join(self.output_dir, "final_index.json")
        with open(final_index_path, "w", encoding = "utf-8") as f:
            json.dump(final_index, f, ensure_ascii=False)
        print(f"Final index written to {final_index_path} with {len(final_index)} terms")
        return final_index_path

    def build_index(self):
        block_terms = defaultdict(list)
        block_id = 0
        doc_count = 0
        block_files = []
        
        with open(self.datafilename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_id = row.get('id') or row.get('doc_id') or row.get('document_id')
                text = row.get('text') or row.get('content') or row.get('body')
                if not doc_id or not text:
                    continue
                bow_pos = self.Preprocessor_instance.compute_bow_with_positions(text)
                self.doc_stats[doc_id] = sum(info['freq'] for info in bow_pos.values())

                for term, info in bow_pos.items():
                    block_terms[term].append({'doc_id': doc_id, 'freq': info['freq'], 'positions': info['positions']})
                
                doc_count += 1
                
                if doc_count >= self.block_size:
                    block_terms_sorted = dict(sorted(block_terms.items()))
                    block_file = self._write_block(block_terms_sorted, block_id)
                    block_files.append(block_file)
                    block_terms = defaultdict(list)
                    block_id += 1
                    doc_count = 0
        if block_terms:
            block_terms_sorted = self._write_block(block_terms, block_id)
            block_files.append(block_terms_sorted, block_id)
            block_files.append(block_file)  
        final_index_path = self._merge_blocks(block_files)
        return final_index_path
    
    def get_doc_stats(self):
        return self.doc_stats
    
    def load_index(self, index_path: str) -> Dict[str, List[Dict[str, Any]]]:
        with open(index_path, "r", encoding = "utf-8") as f:
            index = json.load(f)
        self._doc_to_terms = defaultdict(list)
        for term, postings in index.items():
            for posting in postings:
                self._doc_to_terms[posting['doc_id']].append(term)
        return index
    
    def query(self, term: str, index: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        term = self.Preprocessor_instance.stemmer.stem(term.lower())
        return index.get(term, [])
    
    def batch_query(self, terms: Iterable[str], index: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        results = {}
        for term in terms:
            stemmed_term = self.Preprocessor_instance.stemmer.stem(term.lower())
            results[term] = index.get(stemmed_term, [])
        return results
    
    def get_vocabulary(self, index: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        return list(index.keys())
    
    def get_index_size(self, index: Dict[str, List[Dict[str, Any]]]) -> int:
        return len(index)
    
    def get_postings_list_size(self, term: str, index: Dict[str, List[Dict[str, Any]]]) -> int:
        term = self.Preprocessor_instance.stemmer.stem(term.lower())
        return len(index.get(term, []))
    
    def get_total_postings(self, index: Dict[str, List[Dict[str, Any]]]) -> int:
        total = 0
        for postings in index.values():
            total += len(postings)
        return total
    
    def get_average_doc_length(self) -> float:
        total_length = sum(self.doc_stats.values())
        num_docs = len(self.doc_stats)
        return total_length / num_docs if num_docs > 0 else 0.0
    
    def get_max_doc_length(self) -> Tuple[str, int]:
        if not self.doc_stats:
            return None, 0
        max_doc = max(self.doc_stats.items(), key=lambda x: x[1])
        return max_doc
    
    def get_min_doc_length(self) -> Tuple[str, int]:
        if not self.doc_stats:
            return None, 0
        min_doc = min(self.doc_stats.items(), key=lambda x: x[1])
        return min_doc
    
    def get_total_documents(self) -> int:
        return len(self.doc_stats)
    
    def get_total_terms(self, index):
        return sum(sum(p['freq'] for p in postings) for postings in index.values())
    
    def get_term_frequencies(self, index: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
        term_frequencies = {}
        for term, postings in index.items():
            term_frequencies[term] = sum(posting['freq'] for posting in postings)
        return term_frequencies
    
    def get_document_frequency(self, term: str, index: Dict[str, List[Dict[str, Any]]]) -> int:
        term = self.Preprocessor_instance.stemmer.stem(term.lower())
        return len(index.get(term, []))
    
    def get_collection_frequency(self, term: str, index: Dict[str, List[Dict[str, Any]]]) -> int:
        term = self.Preprocessor_instance.stemmer.stem(term.lower())
        postings = index.get(term, [])
        return sum(posting['freq'] for posting in postings)
    
    def get_top_k_terms(self, k: int, index: Dict[str, List[Dict[str, Any]]]) -> List[Tuple[str, int]]:
        term_frequencies = self.get_term_frequencies(index)
        sorted_terms = sorted(term_frequencies.items(), key=lambda x: x[1], reverse=True)
        return sorted_terms[:k]
    
    def get_bottom_k_terms(self, k: int, index: Dict[str, List[Dict[str, Any]]]) -> List[Tuple[str, int]]:
        term_frequencies = self.get_term_frequencies(index)
        sorted_terms = sorted(term_frequencies.items(), key=lambda x: x[1])
        return sorted_terms[:k]
    
    def get_terms_in_doc(self, doc_id: str, index: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        if hasattr(self, '_doc_to_terms'):
            return self._doc_to_terms.get(doc_id, [])
        terms_in_doc = []
        for term, postings in index.items():
            for posting in postings:
                if posting['doc_id'] == doc_id:
                    terms_in_doc.append(term)
                    break
        return terms_in_doc
    
    def get_term_positions_in_doc(self, term: str, doc_id: str, index: Dict[str, List[Dict[str, Any]]]) -> List[int]:
        term = self.Preprocessor_instance.stemmer.stem(term.lower())
        postings = index.get(term, [])
        for posting in postings:
            if posting['doc_id'] == doc_id:
                return posting.get('positions', [])
        return []
    
    def get_terms_with_prefix(self, prefix: str, index: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        prefix_stem = self.Preprocessor_instance.stemmer.stem(prefix.lower())
        matching_terms = [term for term in index.keys() if term.startswith(prefix_stem)]
        return matching_terms
    
    def get_terms_with_suffix(self, suffix: str, index: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        suffix = self.Preprocessor_instance.stemmer.stem(suffix.lower())
        matching_terms = [term for term in index.keys() if term.endswith(suffix)]
        return matching_terms
    
    def get_terms_containing_substring(self, substring: str, index: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        substring = self.Preprocessor_instance.stemmer.stem(substring.lower())
        matching_terms = [term for term in index.keys() if substring in term]
        return matching_terms
    
    def get_postings_with_min_freq(self, min_freq: int, index: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        filtered_postings = {}
        for term, postings in index.items():
            filtered = [posting for posting in postings if posting['freq'] >= min_freq]
            if filtered:
                filtered_postings[term] = filtered
        return filtered_postings
    
    def get_postings_with_max_freq(self, max_freq: int, index: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        filtered_postings = {}
        for term, postings in index.items():
            filtered = [posting for posting in postings if posting['freq'] <= max_freq]
            if filtered:
                filtered_postings[term] = filtered
        return filtered_postings
    
    def get_terms_sorted_by_df(self, index: Dict[str, List[Dict[str, Any]]]) -> List[Tuple[str, int]]:
        term_dfs = [(term, len(postings)) for term, postings in index.items()]
        sorted_terms = sorted(term_dfs, key=lambda x: x[1], reverse=True)
        return sorted_terms
    
    def get_terms_sorted_by_cf(self, index: Dict[str, List[Dict[str, Any]]]) -> List[Tuple[str, int]]:
        term_cfs = []
        for term, postings in index.items():
            cf = sum(posting['freq'] for posting in postings)
            term_cfs.append((term, cf))
        sorted_terms = sorted(term_cfs, key=lambda x: x[1], reverse=True)
        return sorted_terms
    
    def get_documents_containing_term(self, term: str, index: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        term = self.Preprocessor_instance.stemmer.stem(term.lower())
        postings = index.get(term, [])
        return [posting['doc_id'] for posting in postings]
    
    def get_terms_in_document_frequency_range(self, min_df: int, max_df: int, index: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        matching_terms = []
        for term, postings in index.items():
            df = len(postings)
            if min_df <= df <= max_df:
                matching_terms.append(term)
        return matching_terms
    
    def get_terms_in_collection_frequency_range(self, min_cf: int, max_cf: int, index: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        matching_terms = []
        for term, postings in index.items():
            cf = sum(posting['freq'] for posting in postings)
            if min_cf <= cf <= max_cf:
                matching_terms.append(term)
        return matching_terms
    
    def get_postings_with_term_frequency_range(self, min_freq: int, max_freq: int, index: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        filtered_postings = {}
        for term, postings in index.items():
            filtered = [posting for posting in postings if min_freq <= posting['freq'] <= max_freq]
            if filtered:
                filtered_postings[term] = filtered
        return filtered_postings
    
    def clear_index(self):
        for filename in os.listdir(self.block_dir):
            file_path = os.path.join(self.block_dir, filename)
            os.remove(file_path)
        final_index_path = os.path.join(self.output_dir, "final_index.json")
        if os.path.exists(final_index_path):
            os.remove(final_index_path)
        print("Cleared all index files.")

    def get_index_statistics(self, index: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        stats = {
            "total_terms": self.get_index_size(index),
            "total_documents": self.get_total_documents(),
            "average_doc_length": self.get_average_doc_length(),
            "max_doc_length": self.get_max_doc_length(),
            "min_doc_length": self.get_min_doc_length(),
            "total_postings": self.get_total_postings(index)
        }
        return stats
    
    def save_doc_stats(self, path: str):
        with open(path, "w", encoding = "utf-8") as f:
            json.dump(self.doc_stats, f, ensure_ascii=False)
        print(f"Document statistics saved to {path}")
    
if __name__ == "__main__":
    datafilename = 'data/news_text_dataset.csv'  # Path to your input data file
    stopwords_file = 'data/stopwords/spanish_stopwords.txt'
    spimi_index = SPIMIIndex(datafilename, block_size=1000, stopwords_file=stopwords_file)
    spimi_index.build_index()
    


