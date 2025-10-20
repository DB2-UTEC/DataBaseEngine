import struct
import pickle
import os
import sys

# Agregar el directorio padre al path para importar FileManager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.file_manager import FileManager
from core.models import Table, Record
from typing import List, Any, Optional, Union

# Constantes de tamaño de bloque/índice
IDX_ENTRY_SIZE = struct.calcsize('ii')
IDX_PAGE_HEADER = struct.calcsize('i')
IDX_BLOCK_FACTOR = (4096 - IDX_PAGE_HEADER) // IDX_ENTRY_SIZE


class ISAMIndex:
    def __init__(self, data_filename: str, index_filename: str = None, file_manager=None, persist_path: str = None, table: Table = None):
        # metadatos
        self.data_filename = data_filename
        self.index_filename = index_filename or (data_filename.replace('.dat', '.idx') if data_filename else None)
        self.persist_path = persist_path or self.index_filename
        self.table = table

        # Inicializar FileManager si no se proporciona
        if not file_manager and table:
            self.file_manager = FileManager(data_filename, table)
        else:
            self.file_manager = file_manager

        # índices en memoria
        self.idx_l1 = []  # raiz (lista de (first_key, start_index_in_l2) )
        self.idx_l2 = []  # medio (lista de (first_key, start_index_in_l3) )
        self.idx_l3 = []  # hojas (lista ordenada de (key, pos))

        self.overflow = {}

        self.order = IDX_BLOCK_FACTOR

    @staticmethod
    def insert_pos(lista, key):
        # Busca donde insertar por clave
        lo = 0
        hi = len(lista)
        while lo < hi:
            mid = (lo + hi) // 2
            if lista[mid][0] < key:
                lo = mid + 1
            else:
                hi = mid
        return lo

    @staticmethod
    def busqueda_binaria(lista, target):
        # Busca índice del mayor
        lo = 0
        hi = len(lista) - 1
        result = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if lista[mid][0] <= target:
                result = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return result

    def recontruir2y1(self):
        # Reconstruye idx_l2 e idx_l1 desde idx_l3 (paginación lógica)
        self.idx_l2 = []
        self.idx_l1 = []
        if not self.idx_l3:
            return
        # L2: cada IDX_BLOCK_FACTOR entradas de idx_l3 forman una 'página' resumen
        for page_start in range(0, len(self.idx_l3), IDX_BLOCK_FACTOR):
            first_key = self.idx_l3[page_start][0]
            self.idx_l2.append((first_key, page_start))
        # L1: cada IDX_BLOCK_FACTOR páginas de idx_l2 forman un bloque de resumen
        for block_start in range(0, len(self.idx_l2), IDX_BLOCK_FACTOR):
            first_key = self.idx_l2[block_start][0]
            self.idx_l1.append((first_key, block_start))

    def is_empty(self):
        return len(self.idx_l3) == 0

    def insert(self, key, values):
        """
        Inserta un registro en ISAM.
        Si values es una lista, crea un Record y lo almacena en FileManager.
        Si values es una posición, asume que ya está almacenado.
        """
        if isinstance(values, list):
            # Crear Record y almacenarlo en FileManager
            if not self.file_manager:
                raise ValueError("FileManager no inicializado")
            
            record = Record(self.table, values)
            pos = self.file_manager.add_record(record)
        else:
            # values es una posición ya existente
            pos = values

        # Insertar en el índice ISAM
        self._insert_to_index(key, pos)

    def _insert_to_index(self, key, pos):
        """
        Inserta (key,pos) en el índice ISAM. Comportamiento mínimo ISAM con overflow:
        - Si la clave no existe -> se inserta en idx_l3 (base).
        - Si la clave ya existe en base -> pos se añade a self.overflow[key].
        (Así mantenemos una posición 'base' en idx_l3 y overflow encadenado).
        """
        if not self.idx_l3:
            # lista vacía
            self.idx_l3.insert(0, (key, pos))
            self.recontruir2y1()
            return

        i = self.insert_pos(self.idx_l3, key)
        # caso exacto de match en la posición i
        if i < len(self.idx_l3) and self.idx_l3[i][0] == key:
            base_pos = self.idx_l3[i][1]
            # overflow
            self.overflow.setdefault(key, [])
            # evitar duplicados exactos de pos
            if pos != base_pos and pos not in self.overflow[key]:
                self.overflow[key].append(pos)
            return
        # si el anterior elemento es el match
        if i > 0 and self.idx_l3[i - 1][0] == key:
            base_pos = self.idx_l3[i - 1][1]
            self.overflow.setdefault(key, [])
            if pos != base_pos and pos not in self.overflow[key]:
                self.overflow[key].append(pos)
            return

        self.idx_l3.insert(i, (key, pos))

        if key in self.overflow and not self.overflow[key]:
            self.overflow.pop(key, None)
        self.recontruir2y1()

    def bulk_insert(self, pairs):
        #Carga masiva: reemplaza idx_l3 con lista ordenada de (key,pos) y limpia overflow.
        self.idx_l3 = sorted(pairs, key=lambda x: x[0])
        self.overflow = {}
        self.recontruir2y1()

    def search(self, key):
        """
        Busca un registro por clave y retorna los valores del registro.
        """
        if not self.idx_l3:
            return None
        
        # Buscar en el índice
        i = self.insert_pos(self.idx_l3, key)
        pos = None
        
        if i < len(self.idx_l3) and self.idx_l3[i][0] == key:
            pos = self.idx_l3[i][1]
        elif i > 0 and self.idx_l3[i - 1][0] == key:
            pos = self.idx_l3[i - 1][1]
        
        if pos is not None and self.file_manager:
            # Leer el registro desde FileManager
            record = self.file_manager.read_record(pos)
            if record:
                return record.values
        
        return None

    def get_all_positions(self, key):
        # Retorna lista con [base_pos] + overflow (en ese orden).
        base = self.search(key)
        if base is None:
            return []
        res = [base]
        extra = self.overflow.get(key, [])
        if extra:
            res.extend(extra)
        return res

    def delete(self, key):
        """
        Elimina la entrada base:
        - Si hay overflow, promueve la primera overflow como nueva base.
        - Si no hay overflow, borra la entrada base.
        Retorna True si se eliminó (o promovió) algo.
        """
        if not self.idx_l3:
            return False
        i = self.insert_pos(self.idx_l3, key)
        # match en i
        if i < len(self.idx_l3) and self.idx_l3[i][0] == key:
            # existe base
            if self.overflow.get(key):
                # promover primer overflow como base
                promoted = self.overflow[key].pop(0)
                self.idx_l3[i] = (key, promoted)
                if not self.overflow[key]:
                    self.overflow.pop(key, None)
                return True
            else:
                # eliminar base
                self.idx_l3.pop(i)
                self.recontruir2y1()
                return True
        # buscar match en anteior
        if i > 0 and self.idx_l3[i - 1][0] == key:
            idx = i - 1
            if self.overflow.get(key):
                promoted = self.overflow[key].pop(0)
                self.idx_l3[idx] = (key, promoted)
                if not self.overflow[key]:
                    self.overflow.pop(key, None)
                return True
            else:
                self.idx_l3.pop(idx)
                self.recontruir2y1()
                return True
        return False

    def range_search(self, start_key, end_key):
        results = []
        if not self.idx_l3 or not self.file_manager:
            return results
        
        # localizar inicio
        i = self.insert_pos(self.idx_l3, start_key)
        if i > 0 and self.idx_l3[i - 1][0] >= start_key:
            i = i - 1
        n = len(self.idx_l3)
        j = i
        while j < n and self.idx_l3[j][0] <= end_key:
            key, base_pos = self.idx_l3[j]
            
            # Leer registro base
            record = self.file_manager.read_record(base_pos)
            if record:
                results.append(record)
            
            # Leer registros de overflow
            extras = self.overflow.get(key, [])
            for p in extras:
                overflow_record = self.file_manager.read_record(p)
                if overflow_record:
                    results.append(overflow_record)
            j += 1
        return results

    def update(self, key, pos):
        """
        - Si la clave está en base, actualiza base.
        - Si está en overflow pero no en base, la dejamos (no promovemos).
        - Si no está, se inserta.
        """
        if not self.idx_l3:
            self.idx_l3.append((key, pos))
            self.recontruir2y1()
            return True
        i = self.insert_pos(self.idx_l3, key)
        if i < len(self.idx_l3) and self.idx_l3[i][0] == key:
            self.idx_l3[i] = (key, pos)
            return True
        if i > 0 and self.idx_l3[i - 1][0] == key:
            self.idx_l3[i - 1] = (key, pos)
            return True
        # si no se encuentra, insertar como nueva base
        self.idx_l3.insert(i, (key, pos))
        self.recontruir2y1()
        return True

    # PERSISTENCIA ---------------------------------------------
    def save_to_file(self, path: str = None) -> bool:
        path = path or self.persist_path
        if not path:
            return False
        state = {
            'idx_l3': self.idx_l3,
            'idx_l2': self.idx_l2,
            'idx_l1': self.idx_l1,
            'overflow': self.overflow,
            'order': self.order
        }
        try:
            with open(path, 'wb') as f:
                pickle.dump(state, f)
            return True
        except Exception as e:
            print(f"[ISAM] Error guardando índice en '{path}': {e}")
            return False

    def load_from_file(self, path: str = None) -> bool:
        #Carga el estado desde disco. Retorna True si existía y se pudo cargar.
        
        path = path or self.persist_path
        if not path or not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            self.idx_l3 = state.get('idx_l3', [])
            self.idx_l2 = state.get('idx_l2', [])
            self.idx_l1 = state.get('idx_l1', [])
            self.overflow = state.get('overflow', {})
            self.order = state.get('order', self.order)
            # si idx_l2/l1 están vacíos, reconstruir
            if not self.idx_l2 or not self.idx_l1:
                self.recontruir2y1()
            return True
        except Exception as e:
            print(f"[ISAM] Error cargando índice desde '{path}': {e}")
            return False
    
    def get_all(self):
        """
        Retorna todos los registros almacenados en el índice ISAM.
        Útil para SELECT * FROM tabla.
        """
        results = []
        
        if not self.file_manager:
            return results
        
        # Recorrer todas las entradas en idx_l3 (hojas)
        for key, pos in self.idx_l3:
            # Leer el registro base
            record = self.file_manager.read_record(pos)
            if record:
                results.append(record)
            
            # Leer registros de overflow si existen
            if key in self.overflow:
                for overflow_pos in self.overflow[key]:
                    overflow_record = self.file_manager.read_record(overflow_pos)
                    if overflow_record:
                        results.append(overflow_record)
        
        return results

    def debug_print(self, max_show=10):
        print("ISAMIndex: entries:", len(self.idx_l3))
        print("L1:", self.idx_l1[:max_show])
        print("L2:", self.idx_l2[:max_show])
        print("L3 (primeras):", self.idx_l3[:max_show])
        print("Overflow (primeras keys):", list(self.overflow.items())[:max_show])
        print("L3 (primeras):", self.idx_l3[:max_show])
