import os
from core.models import Table, Record, Field
from core.file_manager import FileManager
from indexes.bplus import BPlusTree
from indexes.isam import ISAMIndex
from indexes.sequential_file import SequentialIndex
from indexes.rtree import RTreeIndex  # NUEVO IMPORT
from indexes.ExtendibleHashing import ExtendibleHashing
from typing import List, Union, Any


class DatabaseManager:
    def __init__(self, table: Table, filename: str, order: int = 4, index_type: str = 'bplus'):
        self.table = table
        self.filename = filename
        self.index_type = index_type

        # Crear nombres de archivos para datos e índice
        self.data_filename = filename
        self.index_filename = filename.replace('.dat', '.idx')

        # FileManager se usa para B+, ISAM, R-tree y Extendible Hashing
        if self.index_type in ('bplus', 'isam', 'rtree', 'extendiblehash'):
            self.file_manager = FileManager(self.data_filename, table)
        else:
            self.file_manager = None  # No usamos FileManager para Sequential

        # Inicializar índice según el tipo solicitado
        if index_type == 'isam':
            self.index = ISAMIndex(self.data_filename, index_filename=self.index_filename, file_manager=self.file_manager)
        elif index_type == 'sequential':
            self.index = SequentialIndex(self.data_filename, self.table)
        elif index_type == 'rtree':
            # Suponiendo que la tabla tiene campos adecuados para R-Tree
            spatial_fields = [field for field in table.fields if field.data_type in (float, int)]
            if len(spatial_fields) < 2:
                raise ValueError("R-Tree requires at least two spatial fields (int or float).")
            self.index = RTreeIndex(self.index_filename, spatial_fields, file_manager=self.file_manager)
        elif index_type == 'extendiblehash': 
            self.index = ExtendibleHashing(bucketSize=3, index_filename=self.index_filename)
        else:
            # Comportamiento por defecto: B+ Tree con persistencia
            self.index = BPlusTree(order=order, index_filename=self.index_filename)

        # Intentar cargar el índice desde archivo
        if not self.index.load_from_file():
            # Si el índice está vacío y NO es secuencial, cargamos desde FileManager
            if self.index_type in ('bplus', 'isam', 'extendiblehash'):  # Agregar extendiblehash
                self.load_index_from_file()

    def load_index_from_file(self):
        """Construye el índice B+/ISAM desde los registros existentes."""
        # Esta función NO aplica para SequentialIndex
        if self.index_type == 'sequential':
            print("SequentialIndex no requiere construcción manual de índice.")
            return

        if self.index.is_empty() and self.file_manager:
            if os.path.exists(self.file_manager.filename):
                print("Construyendo índice desde registros existentes...")
                idx = 0
                while True:
                    record = self.file_manager.read_record(idx)

                    if record is None:
                        break

                    if record.next == 0:  # Solo registros válidos (no eliminados)
                        self.index.insert(record.key, idx)
                    idx += 1
                self.file_manager.file_size = idx
                if self.index_type == 'extendiblehash':
                    print(f"Índice Extendible Hashing construido")
                else:
                    print(f"Índice construido con {len(self.index.traverse_leaves())} hojas/entradas.")

    def add_record(self, record: Record):
        """Añade un nuevo registro tanto al archivo como al índice."""
        
        if self.index_type == 'sequential':
            self.index.insert(record.key, record)
        elif self.index_type == 'rtree':
            pos = self.file_manager.add_record(record)
            self.index.insert(record, pos)
        elif self.index_type == 'extendiblehash':  # ¡NUEVO CASO!
            pos = self.file_manager.add_record(record)
            self.index.insert(record.key, pos)  # Extendible Hashing usa (key, position)
        else:
            # Lógica anterior para B+ e ISAM
            pos = self.file_manager.add_record(record)
            self.index.insert(record.key, pos)
            
        print(f"Registro con llave '{record.key}' añadido.")

    def get_record(self, key: Any) -> Union[Record, None]:
        """Busca un registro por su clave usando el índice."""
        
        if self.index_type == 'sequential':
            # SequentialIndex.search() devuelve el Record completo
            return self.index.search(key)
        elif self.index_type == ('rtree','extendiblehash'):
            pos = self.index.search(key)
            if pos is not None:
                return self.file_manager.read_record(pos)
            return None
        else:
            # Lógica anterior
            pos = self.index.search(key)
            if pos is not None:
                return self.file_manager.read_record(pos)
        return None

    def update_record(self, key: Any, new_values: List[Any]) -> bool:
        """Actualiza un registro existente."""
        
        if self.index_type == 'sequential':
            # Para Sequential, necesitamos eliminar y volver a insertar
            existing_record = self.get_record(key)
            if existing_record:
                # Crear nuevo registro con valores actualizados
                new_record = Record(self.table, new_values)
                # Eliminar el viejo y agregar el nuevo
                if self.remove_record(key):
                    self.add_record(new_record)
                    print(f"Registro con llave '{key}' actualizado en Sequential File.")
                    return True
            return False
        else:
            # Lógica anterior
            pos = self.index.search(key)
            if pos is not None:
                new_record = Record(self.table, new_values, pos=pos)
                self.file_manager._write_record_at_pos(new_record, pos)
                print(f"Registro con llave '{key}' actualizado.")
                return True
            return False

    def remove_record(self, key: Any) -> bool:
        """Elimina un registro tanto del archivo como del índice."""
        
        if self.index_type == 'sequential':
            # SequentialIndex.delete() hace la eliminación lógica
            result = self.index.delete(key)
            if result:
                print(f"Registro con llave '{key}' eliminado del Sequential File.")
            return result
        else:
            # Lógica anterior
            pos = self.index.search(key)
            if pos is not None:
                if self.file_manager.remove_record(pos):
                    self.index.delete(key)
                    print(f"Registro con llave '{key}' eliminado.")
                    return True
        return False

    def range_search(self, start_key: Any, end_key: Any) -> List[Record]:
        """Busca todos los registros en un rango de claves."""
        
        if self.index_type == 'sequential':
            # SequentialIndex.rangeSearch() devuelve la lista de Records
            return self.index.rangeSearch(start_key, end_key)
        else:
            # Lógica anterior
            found_records = []
            positions = self.index.range_search(start_key, end_key)
            for _, pos in positions:
                record = self.file_manager.read_record(pos)
                if record and record.next == 0:
                    found_records.append(record)
            return found_records

    def get_all(self) -> List[Record]:
        """Obtiene todos los registros válidos."""
        
        if self.index_type == 'sequential':
            # Para Sequential File, leer ambos archivos
            records = []
            
            # Leer archivo principal
            try:
                with open(self.index.data_filename, 'rb') as f:
                    while True:
                        data = f.read(self.index.record_size)
                        if not data:
                            break
                        record = Record.unpack(self.table, data)
                        if record.next == 0:  # Solo registros válidos
                            records.append(record)
            except FileNotFoundError:
                pass
            
            # Leer archivo auxiliar
            try:
                with open(self.index.aux_filename, 'rb') as f:
                    while True:
                        data = f.read(self.index.record_size)
                        if not data:
                            break
                        record = Record.unpack(self.table, data)
                        if record.next == 0:  # Solo registros válidos
                            records.append(record)
            except FileNotFoundError:
                pass
            
            return records
        else:
            return self.file_manager.get_all_records()

    def save_all(self):
        """Fuerza el guardado de todos los datos."""
        self.index.save_to_file()
        print("Datos guardados en memoria secundaria.")
 
    def get_index_info(self) -> dict:
        """Obtiene información sobre el estado del índice."""
        
        if self.index_type == 'sequential':
            # Información específica para Sequential File
            try:
                main_size = os.path.getsize(self.index.data_filename) // self.index.record_size if os.path.exists(self.index.data_filename) else 0
                aux_size = self.index.aux_records_count
                return {
                    'index_type': 'sequential',
                    'total_keys': main_size + aux_size,
                    'main_file_records': main_size,
                    'aux_file_records': aux_size,
                    'k_threshold': self.index.K_threshold,
                    'is_empty': self.index.is_empty()
                }
            except Exception as e:
                return {
                    'index_type': 'sequential',
                    'total_keys': 0,
                    'main_file_records': 0,
                    'aux_file_records': 0,
                    'k_threshold': self.index.K_threshold,
                    'is_empty': True,
                    'error': str(e)
                }
        else:
            # Lógica anterior para B+ e ISAM
            base_info = {'index_type': self.index_type}
            
            # Preferir traverse_leaves si existe (BPlusTree)
            if hasattr(self.index, 'traverse_leaves'):
                try:
                    leaves = self.index.traverse_leaves()
                    total_keys = sum(len(keys) for keys, _ in leaves)
                    leaf_nodes = len(leaves)
                    order = getattr(self.index, 'order', None)
                    return {
                        **base_info,
                        'total_keys': total_keys,
                        'leaf_nodes': leaf_nodes,
                        'order': order,
                        'is_empty': self.index.is_empty()
                    }
                except Exception:
                    pass

            # Fallback para ISAMIndex
            idx_l3 = getattr(self.index, 'idx_l3', None)
            overflow = getattr(self.index, 'overflow', None)
            order = getattr(self.index, 'order', None)

            if idx_l3 is not None:
                total_keys = len(idx_l3)
                if isinstance(overflow, dict):
                    extra_positions = sum(len(v) for v in overflow.values())
                else:
                    extra_positions = 0
                if order and order > 0:
                    leaf_nodes = (len(idx_l3) + order - 1) // order
                else:
                    leaf_nodes = len(idx_l3)
                return {
                    **base_info,
                    'total_keys': total_keys + extra_positions,
                    'leaf_nodes': leaf_nodes,
                    'order': order,
                    'is_empty': self.index.is_empty()
                }

            # Último recurso: valores por defecto
            return {
                **base_info,
                'total_keys': 0,
                'leaf_nodes': 0,
                'order': getattr(self.index, 'order', None),
                'is_empty': self.index.is_empty()
            }
