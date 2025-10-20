import os
import struct
import heapq # Útil para el merge en _rebuild
from core.models import Table, Record
from typing import List, Any, Union

# K: Número de registros en el auxiliar antes de reconstruir 
K_THRESHOLD = 5 

class SequentialIndex:
    """
    Implementa la técnica de Archivo Secuencial Indexado con un 
    archivo auxiliar y reconstrucción por lotes (batch).
    Esta clase gestiona sus propios archivos (.dat y .aux) y NO 
    utiliza el FileManager genérico.
    """
    
    def __init__(self, data_filename: str, table: Table):
        self.table = table
        self.data_filename = data_filename
        self.aux_filename = data_filename.replace('.dat', '.aux')
        self.record_size = table.record_size
        
        # K_THRESHOLD es el 'K' de las especificaciones 
        self.K_threshold = K_THRESHOLD 
        
        # Crear archivos si no existen
        if not os.path.exists(self.data_filename):
            open(self.data_filename, 'wb').close()
        if not os.path.exists(self.aux_filename):
            open(self.aux_filename, 'wb').close()
            
        self.aux_records_count = self._get_aux_count()

    def _get_aux_count(self) -> int:
        """Helper para contar cuántos registros hay en el archivo auxiliar."""
        try:
            return os.path.getsize(self.aux_filename) // self.record_size
        except OSError:
            return 0

    def add(self, record: Record):
        """
        Añade un registro al archivo auxiliar.
        Si el auxiliar supera K, reconstruye el archivo principal.
        """
        # 1. Escribir el nuevo registro al FINAL del archivo auxiliar
        with open(self.aux_filename, 'ab') as f_aux:
            f_aux.write(record.pack())
        
        self.aux_records_count += 1
        
        # 2. Comprobar si hemos alcanzado el umbral K 
        if self.aux_records_count >= self.K_threshold:
            print(f"Límite K={self.K_threshold} alcanzado. Reconstruyendo archivo principal...")
            self._rebuild()

    def _rebuild(self):
        """
        Algoritmo de reconstrucción (merge).
        Fusiona el .dat (ordenado) con el .aux (desordenado) 
        en un nuevo archivo .dat ordenado.
        """
        temp_filename = self.data_filename + '.tmp'
        all_aux_records = []

        # 1. Leer TODOS los registros del archivo auxiliar y cargarlos a memoria
        try:
            with open(self.aux_filename, 'rb') as f_aux:
                while True:
                    data = f_aux.read(self.record_size)
                    if not data:
                        break
                    record = Record.unpack(self.table, data)
                    # Usamos 'next == 0' como "no borrado"
                    if record.next == 0: 
                        all_aux_records.append(record)
        except FileNotFoundError:
            pass # No hay archivo aux, no hay problema

        # 2. Ordenar los registros auxiliares en memoria
        all_aux_records.sort(key=lambda r: r.key)

        # 3. Fusionar (Merge) el archivo .dat ordenado y la lista aux ordenada
        try:
            with open(self.data_filename, 'rb') as f_main, \
                 open(temp_filename, 'wb') as f_temp:
                
                main_data = f_main.read(self.record_size)
                aux_idx = 0
                
                while main_data or aux_idx < len(all_aux_records):
                    main_record = None
                    if main_data:
                        main_record = Record.unpack(self.table, main_data)
                        # Ignorar registros borrados en el principal
                        if main_record.next != 0: 
                            main_data = f_main.read(self.record_size)
                            continue
                    
                    aux_record = all_aux_records[aux_idx] if aux_idx < len(all_aux_records) else None
                    
                    # Lógica de Merge
                    if main_record and (not aux_record or main_record.key <= aux_record.key):
                        # Escribir el registro del archivo principal
                        f_temp.write(main_record.pack())
                        main_data = f_main.read(self.record_size)
                    elif aux_record:
                        # Escribir el registro del archivo auxiliar
                        f_temp.write(aux_record.pack())
                        aux_idx += 1
                    else:
                        break # No hay más datos

        except FileNotFoundError:
             # Si el .dat no existe (primera vez), solo escribe el .aux
             with open(temp_filename, 'wb') as f_temp:
                 for record in all_aux_records:
                     f_temp.write(record.pack())

        # 4. Reemplazar archivos
        os.replace(temp_filename, self.data_filename)
        
        # 5. Limpiar el archivo auxiliar
        open(self.aux_filename, 'wb').close()
        self.aux_records_count = 0
        print("Reconstrucción completada.")

    def _search_record(self, key: Any) -> Union[Record, None]:
        """Método interno que busca y devuelve Record (para uso interno)"""
        # 1. Búsqueda binaria en el archivo principal (.dat)
        record = self._binary_search_data_file(key)
        if record:
            return record
            
        # 2. Búsqueda lineal en el archivo auxiliar (.aux)
        record = self._linear_search_aux_file(key)
        return record

    def search(self, key: Any) -> Any:
        """
        Busca por clave y devuelve el valor (lista de valores).
        En lugar del objeto Record completo.
        """
        record = self._search_record(key)  # Método interno que busca Record
        if record:
            return record.values  # Devuelve solo los valores
        return None


    def _binary_search_data_file(self, key: Any) -> Union[Record, None]:
        """Helper: Búsqueda binaria en el archivo .dat físicamente ordenado."""
        try:
            with open(self.data_filename, 'rb') as f:
                f.seek(0, os.SEEK_END)
                total_records = f.tell() // self.record_size
                low = 0
                high = total_records - 1
                
                while low <= high:
                    mid = (low + high) // 2
                    f.seek(mid * self.record_size)
                    data = f.read(self.record_size)
                    if not data:
                        break
                        
                    record = Record.unpack(self.table, data)
                    
                    if record.key == key:
                        # Encontrado, pero solo si no está borrado
                        return record if record.next == 0 else None
                    elif record.key < key:
                        low = mid + 1
                    else:
                        high = mid - 1
                        
        except FileNotFoundError:
            return None
        return None

    def _linear_search_aux_file(self, key: Any) -> Union[Record, None]:
        """Helper: Búsqueda lineal en el archivo .aux."""
        try:
            with open(self.aux_filename, 'rb') as f_aux:
                while True:
                    data = f_aux.read(self.record_size)
                    if not data:
                        break
                    record = Record.unpack(self.table, data)
                    if record.key == key:
                         # Encontrado, pero solo si no está borrado
                        return record if record.next == 0 else None
        except FileNotFoundError:
            return None
        return None

    def rangeSearch(self, begin_key: Any, end_key: Any) -> List[Record]:
        """
        Búsqueda por rango ORIGINAL que devuelve Records.
        Para uso interno por range_search().
        """
        results = []
        
        # 1. Búsqueda en .dat
        try:
            with open(self.data_filename, 'rb') as f_main:
                while True:
                    data = f_main.read(self.record_size)
                    if not data:
                        break
                    record = Record.unpack(self.table, data)
                    
                    if record.next == 0: # No borrado
                        if record.key > end_key:
                            # Como el .dat está ordenado, podemos parar aquí
                            break
                        if record.key >= begin_key:
                            results.append(record)
                            
        except FileNotFoundError:
            pass

        # 2. Búsqueda en .aux
        try:
            with open(self.aux_filename, 'rb') as f_aux:
                while True:
                    data = f_aux.read(self.record_size)
                    if not data:
                        break
                    record = Record.unpack(self.table, data)
                    
                    if record.next == 0: # No borrado
                        if begin_key <= record.key <= end_key:
                            results.append(record)
                            
        except FileNotFoundError:
            pass
            
        return results

    def range_search(self, begin_key: Any, end_key: Any) -> List[Any]:
        """
        Búsqueda por rango - nombre estandarizado.
        Devuelve lista de valores, no Records.
        """
        records = self.rangeSearch(begin_key, end_key)  # Llama al método original
        return [record.values for record in records]  # Convierte a valores

    def remove(self, key: Any) -> bool:
        """
        Eliminación mejorada: Elimina físicamente el registro y reconstruye inmediatamente.
        Esto evita problemas con tombstones y permite reinsertar la misma clave.
        """
        
        # 1. Buscar el registro en .dat
        found_in_dat = False
        try:
            with open(self.data_filename, 'r+b') as f:
                f.seek(0, os.SEEK_END)
                total_records = f.tell() // self.record_size
                low = 0
                high = total_records - 1
                
                while low <= high:
                    mid = (low + high) // 2
                    offset = mid * self.record_size
                    f.seek(offset)
                    data = f.read(self.record_size)
                    if not data: break
                        
                    record = Record.unpack(self.table, data)
                    
                    if record.key == key:
                        if record.next == 0: # Si no está ya borrado
                            record.next = -1 # Marcar como borrado
                            f.seek(offset) # Regresar a la posición
                            f.write(record.pack()) # Sobrescribir
                            found_in_dat = True
                            break
                        else:
                            return False # Ya estaba borrado
                    elif record.key < key:
                        low = mid + 1
                    else:
                        high = mid - 1
        except FileNotFoundError:
            pass
            
        # 2. Si no se encontró en .dat, buscar en .aux
        if not found_in_dat:
            try:
                with open(self.aux_filename, 'r+b') as f_aux:
                    offset = 0
                    while True:
                        data = f_aux.read(self.record_size)
                        if not data: break
                            
                        record = Record.unpack(self.table, data)
                        
                        if record.key == key:
                            if record.next == 0:
                                record.next = -1
                                f_aux.seek(offset)
                                f_aux.write(record.pack())
                                found_in_dat = True
                                break
                            else:
                                return False # Ya estaba borrado
                        
                        offset += self.record_size
            except FileNotFoundError:
                pass

        # 3. Si se encontró y marcó como borrado, reconstruir inmediatamente
        if found_in_dat:
            print(f"Registro con clave {key} marcado como borrado. Reconstruyendo archivo...")
            self._rebuild()
            return True

        return False # No se encontró

    # --- Métodos requeridos por la interfaz genérica de DatabaseManager ---
    # (Estos métodos son para que se parezca a BPlusTree e ISAM)

    def insert(self, key: Any, value: Any):
        """
        Inserta clave/valor en Sequential File.
        Ahora acepta (key, value) como las otras estructuras.
        """
        # Convertir value a Record si es necesario
        if isinstance(value, Record):
            record = value
        else:
            # Asumir que value es una lista de valores para el Record
            record = Record(self.table, value)
        
        # Llamar al método original add
        return self.add(record)

    def delete(self, key: Any) -> bool:
        """Wrapper para remove que devuelve el resultado."""
        return self.remove(key)

    def load_from_file(self) -> bool:
        """Simula la carga. Los archivos ya se manejan en __init__."""
        return os.path.exists(self.data_filename)

    def save_to_file(self):
        """
        Forzamos la reconstrucción para "guardar" todo ordenado.
        Esto es opcional, pero puede ser útil.
        """
        if self.aux_records_count > 0:
            print("Guardado forzado, iniciando reconstrucción...")
            self._rebuild()

    def is_empty(self) -> bool:
        return self._get_aux_count() == 0 and \
               (not os.path.exists(self.data_filename) or os.path.getsize(self.data_filename) == 0)

    def get_all(self) -> List[Record]:
        """
        Retorna TODOS los registros no borrados del archivo principal y auxiliar.
        Útil para SELECT * FROM tabla.
        """
        results = []
        
        # 1. Leer todos los registros del archivo principal (.dat)
        try:
            with open(self.data_filename, 'rb') as f_main:
                while True:
                    data = f_main.read(self.record_size)
                    if not data:
                        break
                    record = Record.unpack(self.table, data)
                    
                    # Solo incluir registros no borrados (next == 0)
                    if record.next == 0:
                        results.append(record)
                        
        except FileNotFoundError:
            pass  # Si no existe el archivo principal, continuar
        
        # 2. Leer todos los registros del archivo auxiliar (.aux)
        try:
            with open(self.aux_filename, 'rb') as f_aux:
                while True:
                    data = f_aux.read(self.record_size)
                    if not data:
                        break
                    record = Record.unpack(self.table, data)
                    
                    # Solo incluir registros no borrados (next == 0)
                    if record.next == 0:
                        results.append(record)
                        
        except FileNotFoundError:
            pass  # Si no existe el archivo auxiliar, continuar
        
        # 3. Ordenar los resultados por clave (opcional pero útil)
        results.sort(key=lambda r: r.key)
        
        return results