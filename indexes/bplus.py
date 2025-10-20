import pickle
import os
import json
import struct
from typing import List, Any, Optional
from core.file_manager import FileManager
from core.models import Table, Record, Field

class BPlusTreeNode:
    def __init__(self, order, is_leaf=False):
        self.order = order
        self.is_leaf = is_leaf
        self.keys = []
        self.children = []
        self.next = None  # Para enlazar hojas
        self.node_id = None  # ID único para persistencia


class BPlusTreePersistence:
    """Maneja la persistencia del árbol B+ (solo índice) usando un archivo pickle y metadatos binarios."""
    def __init__(self, index_filename: str, order: int = 4):
        self.index_filename = index_filename
        self.order = order
        self.node_counter = 0
        self.root_id = -1
        # Preparar metadatos si existen
        self._initialize_index_metadata()

    
    def _initialize_index_metadata(self):
        """Inicializa los metadatos del índice."""
        metadata_filename = self.index_filename + '.meta'
        try:
            with open(metadata_filename, 'rb') as f:
                data = f.read(8)  # 4 bytes para node_counter + 4 bytes para root_id
                if len(data) == 8:
                    self.node_counter = struct.unpack('i', data[:4])[0]
                    self.root_id = struct.unpack('i', data[4:8])[0]
                else:
                    self.node_counter = 0
                    self.root_id = -1
        except FileNotFoundError:
            self.node_counter = 0
            self.root_id = -1
            self._save_index_metadata()
    
    def _save_index_metadata(self):
        """Guarda los metadatos del índice."""
        metadata_filename = self.index_filename + '.meta'
        try:
            with open(metadata_filename, 'wb') as f:
                f.write(struct.pack('i', self.node_counter))
                f.write(struct.pack('i', self.root_id))
        except Exception as e:
            print(f"Error al guardar metadatos del índice: {e}")
    
    def _generate_node_id(self) -> int:
        """Genera un ID único para cada nodo."""
        self.node_counter += 1
        return self.node_counter
    
    def _assign_node_ids(self, node: BPlusTreeNode):
        """Asigna IDs únicos a todos los nodos del árbol."""
        if node.node_id is None:
            node.node_id = self._generate_node_id()
        
        if not node.is_leaf:
            for child in node.children:
                if isinstance(child, BPlusTreeNode):
                    self._assign_node_ids(child)
    
    
    def save_tree(self, tree: 'BPlusTree'):
        """Guarda el árbol B+ completo usando pickle para el índice."""
        # Asignar IDs a todos los nodos
        self._assign_node_ids(tree.root)
        self.root_id = tree.root.node_id
        
        # Serializar el árbol completo
        tree_data = {
            'root': tree.root,
            'order': tree.order,
            'node_counter': self.node_counter
        }
        
        with open(self.index_filename, 'wb') as f:
            pickle.dump(tree_data, f)
        
        # Guardar metadatos
        self._save_index_metadata()
    
    def load_tree(self) -> Optional['BPlusTree']:
        """Carga el árbol B+ desde archivo pickle."""
        if not os.path.exists(self.index_filename):
            return None
        
        try:
            with open(self.index_filename, 'rb') as f:
                tree_data = pickle.load(f)
            
            # Cargar el árbol únicamente con su estructura; el FileManager lo gestiona BPlusTree
            tree = BPlusTree(order=tree_data['order'], index_filename=self.index_filename)
            tree.root = tree_data['root']
            self.node_counter = tree_data.get('node_counter', 0)
            
            return tree
        except Exception as e:
            print(f"Error al cargar el árbol B+: {e}")
            return None
    


class BPlusTree:
    def __init__(self, order=4, index_filename: Optional[str] = None, table: Optional[Table] = None):
        self.root = BPlusTreeNode(order, is_leaf=True)
        self.order = order
        self.table = table
        self.index_filename = index_filename
        self.data_file_manager = None
        self.persistence = None

        # Configurar persistencia del índice
        if index_filename:
            # Asegurar directorio
            dir_name = os.path.dirname(index_filename)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self.persistence = BPlusTreePersistence(index_filename=index_filename, order=order)

        # Configurar almacenamiento de datos si hay tabla
        if index_filename and table:
            data_filename = index_filename.replace('.idx', '.dat')
            dir_name = os.path.dirname(data_filename)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            if not os.path.exists(data_filename):
                open(data_filename, 'wb').close()
            self.data_file_manager = FileManager(data_filename, table)

        # Si no se pasó Table, intentar cargar de metadatos para poder crear .dat al vuelo
        if index_filename and self.table is None:
            self._try_initialize_from_metadata()

        self._auto_save = True  # Guardar automáticamente después de cada operación

    def is_empty(self):
        """Check if the BPlus tree is empty."""
        return len(self.root.keys) == 0
    
    def add_record(self, record: Record) -> int:
        """Añade un registro a la tabla y actualiza el índice."""
        if not self.data_file_manager:
            # Inicializar FileManager usando la tabla del propio record
            if not self.index_filename:
                raise ValueError("No hay index_filename para derivar el archivo .dat")
            if not record or not getattr(record, 'table', None):
                raise ValueError("No hay Table asociado para inicializar FileManager")
            # Asegurar FM
            data_filename = self.index_filename.replace('.idx', '.dat')
            dir_name = os.path.dirname(data_filename)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            if not os.path.exists(data_filename):
                open(data_filename, 'wb').close()
            self.table = record.table
            self.data_file_manager = FileManager(data_filename, self.table)
        
        # Añadir el registro a la tabla
        pos = self.data_file_manager.add_record(record)
        
        # Actualizar el índice B+
        self.insert(record.key, pos)
        
        return pos
    
    def get_record(self, key: Any) -> Optional[Record]:
        """Obtiene un registro por su clave."""
        if not self.data_file_manager:
            raise ValueError("FileManager no inicializado")
        
        pos = self.search(key)
        if pos is not None:
            return self.data_file_manager.read_record(pos)
        return None
    
    def update_record(self, key: Any, new_values: List[Any]) -> bool:
        """Actualiza un registro existente."""
        if not self.data_file_manager:
            raise ValueError("FileManager no inicializado")
        
        pos = self.search(key)
        if pos is not None:
            # Crear nuevo registro con los valores actualizados
            new_record = Record(self.table, new_values)
            new_record.pos = pos
            
            # Escribir el registro actualizado
            self.data_file_manager._write_record_at_pos(new_record, pos)
            return True
        return False
    
    def delete_record(self, key: Any) -> bool:
        """Elimina un registro de la tabla y del índice."""
        if not self.data_file_manager:
            raise ValueError("FileManager no inicializado")
        
        pos = self.search(key)
        if pos is not None:
            # Eliminar del índice
            self.delete(key)
            
            # Eliminar de la tabla
            return self.data_file_manager.remove_record(pos)
        return False
    
    def get_all_records(self) -> List[Record]:
        """Obtiene todos los registros como lista de diccionarios {campo: valor}.
        Formato alineado a lo que retorna ISAM a través del FileManager.
        """
        if not self.data_file_manager:
            # Intentar inicializar desde metadatos si no hay Table
            if not self.table:
                self._try_initialize_from_metadata()
            if not self.table:
                # No podemos leer sin conocer el layout; retornar vacío en lugar de fallar
                return []
            self._ensure_file_manager(self.table)

        records = self.data_file_manager.get_all_records()
        if not records:
            return []

        # Convertir cada Record a dict usando los nombres de campos
        results: List[dict] = []
        for rec in records:
            try:
                field_names = [f.name for f in (rec.table.fields if hasattr(rec, 'table') and rec.table else self.table.fields)]
                results.append({name: value for name, value in zip(field_names, rec.values)})
            except Exception as e:
                # Fallback seguro, mostrar error real
                results.append({'data': str(rec.values), 'error': str(e)})
        return results


    
    def range_query(self, start_key: Any, end_key: Any) -> List[Record]:
        """Realiza una consulta por rango y devuelve los registros como lista de dicts (formato ISAM)."""
        if not self.data_file_manager:
            if not self.table:
                self._try_initialize_from_metadata()
            if not self.table:
                return []
            self._ensure_file_manager(self.table)

        positions = self.range_search(start_key, end_key)
        results = []
        for key, pos in positions:
            record = self.data_file_manager.read_record(pos)
            if record:
                try:
                    field_names = [f.name for f in (record.table.fields if hasattr(record, 'table') and record.table else self.table.fields)]
                    results.append({name: value for name, value in zip(field_names, record.values)})
                except Exception as e:
                    results.append({'data': str(record.values), 'error': str(e)})
        return results
    
    def load_from_file(self):
        """Carga el árbol desde el archivo de persistencia."""
        if self.persistence:
            loaded_tree = self.persistence.load_tree()
            if loaded_tree:
                self.root = loaded_tree.root
                self.order = loaded_tree.order
                if loaded_tree.persistence:
                    self.persistence.node_counter = loaded_tree.persistence.node_counter
                    self.persistence.root_id = loaded_tree.persistence.root_id
                return True
        return False
    
    def save_to_file(self):
        """Guarda el árbol en el archivo de persistencia."""
        if self.persistence:
            self.persistence.save_tree(self)
    
    def _auto_save_if_enabled(self):
        """Guarda automáticamente si está habilitado."""
        if self._auto_save and self.persistence:
            self.save_to_file()

    # -------------------------------
    # BÚSQUEDA
    # -------------------------------
    def search(self, key, node=None):
        # ISAM-style: return a list of dicts (one per record) for the given key
        if self.index_filename:
            print(f"[DEBUG][search] Usando archivo de índice: {self.index_filename}")
        else:
            print(f"[DEBUG][search] Sin archivo de índice asociado.")
        # Forzar tipo de clave si la tabla y el campo clave son int
        if self.table:
            key_field = self.table.key_field
            for f in self.table.fields:
                if f.name == key_field and f.data_type == int:
                    try:
                        key = int(key)
                    except Exception:
                        pass
                    break
        node = node or self.root
        pos = None
        if node.is_leaf:
            print(f"[DEBUG][search] Buscando clave: {key} (tipo: {type(key)}) en hoja con claves: {node.keys} (tipos: {[type(k) for k in node.keys]})")
            for i, item in enumerate(node.keys):
                if item == key:
                    print(f"[DEBUG][search] ¡Clave encontrada! Posición: {node.children[i]}")
                    pos = node.children[i]
                    break
            if pos is None:
                print(f"[DEBUG][search] Clave {key} NO encontrada en hoja.")
                return []
        else:
            for i, item in enumerate(node.keys):
                if key < item:
                    return self.search(key, node.children[i])
            return self.search(key, node.children[-1])

        # At this point, pos is the position of the record for the key
        if not self.data_file_manager:
            if not self.table:
                self._try_initialize_from_metadata()
            if not self.table:
                return []
            self._ensure_file_manager(self.table)
        record = self.data_file_manager.read_record(pos)
        if record:
            try:
                field_names = [f.name for f in (record.table.fields if hasattr(record, 'table') and record.table else self.table.fields)]   
                return [{name: value for name, value in zip(field_names, record.values)}][0]
            except Exception as e:
                return [{'data': str(record.values), 'error': str(e)}]
        return []

    def range_search(self, start, end):
        # Devuelve una lista de dicts (registros completos) igual que range_query
        if self.index_filename:
            print(f"[DEBUG][range_search] Usando archivo de índice: {self.index_filename}")
        else:
            print(f"[DEBUG][range_search] Sin archivo de índice asociado.")
        # Forzar tipo de clave si la tabla y el campo clave son int
        if self.table:
            key_field = self.table.key_field
            for f in self.table.fields:
                if f.name == key_field and f.data_type == int:
                    try:
                        start = int(start)
                        end = int(end)
                    except Exception:
                        pass
                    break
        # Buscar posiciones en el rango
        positions = []
        if self.is_empty():
            print(f"[DEBUG][range_search] El árbol está vacío.")
            return []
        node = self.root
        while not node.is_leaf:
            node = node.children[0]
        while node:
            print(f"[DEBUG][range_search] Hoja con claves: {node.keys} (tipos: {[type(k) for k in node.keys]}) | Buscando en rango: {start} - {end} (tipos: {type(start)} - {type(end)})")
            for i, key in enumerate(node.keys):
                if start <= key <= end:
                    print(f"[DEBUG][range_search] Clave {key} en rango, posición: {node.children[i]}")
                    positions.append((key, node.children[i]))
                elif key > end:
                    break
            node = node.next
        if not positions:
            print(f"[DEBUG][range_search] Ninguna clave encontrada en el rango.")
            return []
        # Convertir posiciones a registros completos (dicts)
        if not self.data_file_manager:
            if not self.table:
                self._try_initialize_from_metadata()
            if not self.table:
                return []
            self._ensure_file_manager(self.table)
        results = []
        for key, pos in positions:
            record = self.data_file_manager.read_record(pos)
            if record:
                try:
                    field_names = [f.name for f in (record.table.fields if hasattr(record, 'table') and record.table else self.table.fields)]
                    results.append({name: value for name, value in zip(field_names, record.values)})
                except Exception as e:
                    results.append({'data': str(record.values), 'error': str(e)})
        
        return results

    # -------------------------------
    # INSERCIÓN
    # -------------------------------
    def insert(self, key, pos):
        # Forzar tipo de clave si la tabla y el campo clave son int
        if self.table:
            key_field = self.table.key_field
            for f in self.table.fields:
                if f.name == key_field and f.data_type == int:
                    try:
                        key = int(key)
                    except Exception:
                        pass
                    break
        # Si pos es Record o valores, persistir primero
        if isinstance(pos, Record):
            pos = self.add_record(pos)
        elif isinstance(pos, (list, tuple)):
            values = list(pos)
            if not self.table:
                # Crear una tabla mínima para poder empacar el registro
                fields: List[Field] = []
                for i, v in enumerate(values):
                    if isinstance(v, int):
                        dtype, size = int, 0
                    elif isinstance(v, float):
                        dtype, size = float, 0
                    else:
                        dtype, size = str, 50
                    fields.append(Field(name=f"col{i}", data_type=dtype, size=size))
                base = os.path.splitext(os.path.basename(self.index_filename or 'tabla'))[0]
                for suf in ['.idx', '_btree', '_hash', '_isam']:
                    if base.endswith(suf):
                        base = base[: -len(suf)]
                self.table = Table(name=base or 'tabla', fields=fields, key_field=fields[0].name)
            # Asegurar FileManager y persistir
            if not self.data_file_manager:
                data_filename = self.index_filename.replace('.idx', '.dat') if self.index_filename else None
                if not data_filename:
                    raise ValueError("No hay index_filename para derivar el archivo .dat")
                dir_name = os.path.dirname(data_filename)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                if not os.path.exists(data_filename):
                    open(data_filename, 'wb').close()
                self.data_file_manager = FileManager(data_filename, self.table)
            pos = self.add_record(Record(self.table, values))

        root = self.root
        new_child = self._insert_recursive(root, key, pos)
        if new_child:
            new_root = BPlusTreeNode(self.order, is_leaf=False)
            new_root.keys = [new_child[0]]
            new_root.children = [root, new_child[1]]
            self.root = new_root
        # Guardar automáticamente el árbol en el archivo .idx después de cada inserción
        self.save_to_file()

    def _insert_recursive(self, node, key, pos):
        if node.is_leaf:
            # Actualizar si ya existe
            if key in node.keys:
                idx = node.keys.index(key)
                node.children[idx] = pos
                return None
            # Insertar ordenado
            i = 0
            while i < len(node.keys) and node.keys[i] < key:
                i += 1
            node.keys.insert(i, key)
            node.children.insert(i, pos)
            if len(node.keys) > self.order:
                return self._split_leaf(node)
            return None
        else:
            # Bajar recursivamente
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            new_child = self._insert_recursive(node.children[i], key, pos)
            if new_child:
                new_key, new_node = new_child
                node.keys.insert(i, new_key)
                node.children.insert(i + 1, new_node)
                if len(node.keys) > self.order:
                    return self._split_internal(node)
            return None

    def _split_leaf(self, node):
        mid = len(node.keys) // 2
        new_node = BPlusTreeNode(self.order, is_leaf=True)
        new_node.keys = node.keys[mid:]
        new_node.children = node.children[mid:]
        node.keys = node.keys[:mid]
        node.children = node.children[:mid]

        new_node.next = node.next
        node.next = new_node

        return new_node.keys[0], new_node

    # -------------------------------
    # Inicialización de almacenamiento (.dat)
    # -------------------------------
    def _data_filename(self) -> Optional[str]:
        if not self.index_filename:
            return None
        return self.index_filename.replace('.idx', '.dat')

    def _ensure_file_manager(self, table: Optional[Table] = None):
        """Asegura que exista un FileManager listo para escribir el .dat."""
        if self.data_file_manager:
            return
        if table is not None and self.table is None:
            self.table = table
        if not self.table:
            raise ValueError("No hay Table asociado para inicializar FileManager")
        data_filename = self._data_filename()
        if not data_filename:
            raise ValueError("No hay index_filename para derivar el archivo .dat")
        dir_name = os.path.dirname(data_filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        if not os.path.exists(data_filename):
            open(data_filename, 'wb').close()
        self.data_file_manager = FileManager(data_filename, self.table)

    def _derive_table_name(self) -> Optional[str]:
        if not self.index_filename:
            return None
        base = os.path.splitext(os.path.basename(self.index_filename))[0]
        for suf in ['_btree', '_hash', '_isam', '_rtree']:
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        return base or None

    def _try_initialize_from_metadata(self):
        """Intenta cargar Table desde data/tables_metadata.json y crear FileManager."""
        table_name = self._derive_table_name()
        if not table_name:
            return
        meta_path = os.path.join('data', 'tables_metadata.json')
        if not os.path.exists(meta_path):
            return
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            if table_name not in meta:
                return
            tbl_info = meta[table_name]
            fields_info = tbl_info.get('fields') or []
            key_field = tbl_info.get('key_field') or (fields_info[0]['name'] if fields_info else 'id')

            field_objs: List[Field] = []
            for fdef in fields_info:
                t = fdef.get('type')
                size = fdef.get('size', 50)
                if t in ('INT', int):
                    dtype = int
                    size = 0
                elif t in ('FLOAT', float):
                    dtype = float
                    size = 0
                else:
                    dtype = str
                    if not isinstance(size, int) or size <= 0:
                        size = 50
                field_objs.append(Field(name=fdef['name'], data_type=dtype, size=size))

            if field_objs:
                self.table = Table(name=table_name, fields=field_objs, key_field=key_field)
                self._ensure_file_manager(self.table)
        except Exception:
            # No romper flujo si falla
            pass

    

    def update(self, key, pos):
        """Update the position for an existing key."""
        if self.search(key) is not None:
            self._update_recursive(self.root, key, pos)
            # Guardar automáticamente después de la actualización
            self.save_to_file()
        else:
            # If key doesn't exist, insert it
            self.insert(key, pos)

    def _update_recursive(self, node, key, pos):
        if node.is_leaf:
            for i, item in enumerate(node.keys):
                if item == key:
                    node.children[i] = pos
                    return
        else:
            for i, item in enumerate(node.keys):
                if key < item:
                    self._update_recursive(node.children[i], key, pos)
                    return
            self._update_recursive(node.children[-1], key, pos)

    def _split_internal(self, node):
        mid = len(node.keys) // 2
        new_node = BPlusTreeNode(self.order, is_leaf=False)
        new_node.keys = node.keys[mid + 1:]
        new_node.children = node.children[mid + 1:]

        promoted_key = node.keys[mid]
        node.keys = node.keys[:mid]
        node.children = node.children[:mid + 1]

        return promoted_key, new_node

    # -------------------------------
    # ELIMINACIÓN
    # -------------------------------
    def delete(self, key):
        self._delete_recursive(self.root, key)
        # si la raíz se queda sin claves y no es hoja, se baja un nivel
        if not self.root.is_leaf and len(self.root.keys) == 0:
            self.root = self.root.children[0]
        # Guardar automáticamente el árbol en el archivo .idx después de la eliminación
        self.save_to_file()

    def _delete_recursive(self, node, key):
        if node.is_leaf:
            if key in node.keys:
                idx = node.keys.index(key)
                node.children.pop(idx)
                node.keys.pop(idx)
            return

        # nodo interno
        i = 0
        while i < len(node.keys) and key >= node.keys[i]:
            i += 1
        self._delete_recursive(node.children[i], key)

        # balancear si es necesario
        if len(node.children[i].keys) < (self.order + 1) // 2:
            self._rebalance(node, i)

    def _rebalance(self, parent, idx):
        child = parent.children[idx]
        if idx > 0:  # tiene hermano izquierdo
            left = parent.children[idx - 1]
            if len(left.keys) > (self.order + 1) // 2:
                # rotar desde la izquierda
                if child.is_leaf:
                    child.keys.insert(0, left.keys.pop(-1))
                    child.children.insert(0, left.children.pop(-1))
                    parent.keys[idx - 1] = child.keys[0]
                else:
                    child.keys.insert(0, parent.keys[idx - 1])
                    parent.keys[idx - 1] = left.keys.pop(-1)
                    child.children.insert(0, left.children.pop(-1))
                return
        if idx < len(parent.children) - 1:  # tiene hermano derecho
            right = parent.children[idx + 1]
            if len(right.keys) > (self.order + 1) // 2:
                # rotar desde la derecha
                if child.is_leaf:
                    child.keys.append(right.keys.pop(0))
                    child.children.append(right.children.pop(0))
                    parent.keys[idx] = right.keys[0]
                else:
                    child.keys.append(parent.keys[idx])
                    parent.keys[idx] = right.keys.pop(0)
                    child.children.append(right.children.pop(0))
                return

        # si no hay redistribución posible → merge
        if idx > 0:
            self._merge(parent, idx - 1)
        else:
            self._merge(parent, idx)

    def _merge(self, parent, idx):
        child = parent.children[idx]
        sibling = parent.children[idx + 1]

        if child.is_leaf:
            child.keys.extend(sibling.keys)
            child.children.extend(sibling.children)
            child.next = sibling.next
        else:
            child.keys.append(parent.keys[idx])
            child.keys.extend(sibling.keys)
            child.children.extend(sibling.children)

        parent.keys.pop(idx)
        parent.children.pop(idx + 1)

    # -------------------------------
    # UTILIDADES
    # -------------------------------
    def traverse_leaves(self):
        """Recorrido de todas las hojas encadenadas (para depuración)."""
        node = self.root
        while not node.is_leaf:
            node = node.children[0]
        result = []
        while node:
            result.append((node.keys, node.children))
            node = node.next
        return result
