#!/usr/bin/env python3
"""
Executor SQL que toma ExecutionPlan y los ejecuta sobre las estructuras de datos.
Archivo reparado y consolidado.
"""

import os
import sys
import csv
import json
import re
from typing import Dict, List, Any, Optional, Union
from sql_parser import ExecutionPlan

# Agregar el directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexes.bplus import BPlusTree
from indexes.ExtendibleHashing import ExtendibleHashing
from indexes.isam import ISAMIndex
from core.databasemanager import DatabaseManager
from core.models import Table, Field, Record
from indexes.rtree import RTreeIndex
from indexes.sequential_file import SequentialIndex
from indexes.inverted_index_spimi import SPIMIIndex


class SQLExecutor:
    """Executor que ejecuta ExecutionPlan sobre las estructuras de datos."""
    
    def __init__(self, base_dir: str = "."):
        """Inicializa el executor."""
        self.base_dir = base_dir
        self.metadata_file = os.path.join(base_dir, 'data', 'tables_metadata.json')
        self.tables = {}  # Almacena metadatos de las tablas
        self.structures = {}  # Almacena las estructuras de datos activas
        
        # Cargar metadatos existentes
        self._load_metadata()
    
    def _load_metadata(self):
        """Carga metadatos de tablas desde archivo JSON y recarga estructuras."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.tables = json.load(f) or {}
                print(f"DEBUG Metadatos cargados: {list(self.tables.keys())}")
                # Recargar estructuras de índices
                for table_name, table_info in list(self.tables.items()):
                    try:
                        self._reload_structure(table_name, table_info)
                    except Exception as e:
                        print(f"ERROR recargando estructura {table_name}: {e}")
            except Exception as e:
                print(f"ERROR cargando metadatos: {e}")
    
    def _reload_structure(self, table_name, table_info):
        """Recarga estructura de índice desde archivos persistidos."""
        index_type = (table_info.get('index_type') or '').upper()
        fields = table_info.get('fields') or []
        key_field = table_info.get('key_field')

        try:
            print(f"DEBUG Recargando estructura: {table_name} ({index_type})")
            if index_type in ('FULLTEXT', 'SPIMI'):
                src = table_info.get('source') or table_info.get('source_file')
                text_col = table_info.get('text_col') or key_field or 'text'
                id_col = table_info.get('id_col') or key_field or 'id'
                if not src or not os.path.exists(src):
                    raise RuntimeError(f"Archivo fuente no encontrado para FULLTEXT: {src}")
                structure = SPIMIIndex(
                    datafilename=src,
                    max_terms_in_block=table_info.get('max_terms_in_block', 100000),
                    text_col=text_col,
                    title_col=None,
                    id_col=id_col
                )
                structure.load_metadata()
                self.structures[table_name] = structure
                print(f"OK Índice FULLTEXT (SPIMI) recargado: {table_name}")
                return
            structure = self._create_structure(table_name, index_type, fields, key_field)
            
            #  VERIFICAR que no sea None
            if structure is None:
                raise RuntimeError(f"No se pudo recargar estructura de {table_name}")
            
            self.structures[table_name] = structure
            print(f"OK Estructura recargada: {table_name} ({type(structure).__name__})")
        except Exception as e:
            print(f"ERROR recargando estructura {table_name}: {e}")
            import traceback
            traceback.print_exc()
            # No agregar a structures si falló
    
    def _save_metadata(self):
        """Guarda metadatos de tablas en archivo JSON (merge seguro)."""
        try:
            os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
            existing = {}
            if os.path.exists(self.metadata_file):
                try:
                    with open(self.metadata_file, 'r', encoding='utf-8') as f:
                        existing = json.load(f) or {}
                except Exception:
                    existing = {}
            merged = existing.copy()
            for tbl, data in (self.tables or {}).items():
                cur = merged.get(tbl, {}) or {}
                new = data or {}
                if new.get('fields') is None or new.get('fields') == []:
                    new_fields = cur.get('fields', [])
                else:
                    new_fields = new.get('fields')
                merged_entry = cur.copy()
                merged_entry.update({k: v for k, v in new.items() if v is not None})
                merged_entry['fields'] = new_fields
                merged[tbl] = merged_entry
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
            self.tables = merged
            print(f"DEBUG Metadatos guardados (merge): {list(self.tables.keys())}")
        except Exception as e:
            print(f"ERROR guardando metadatos: {e}")

    def execute(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """
        Ejecuta un ExecutionPlan - VERIFICAR ENLACE DELETE.
        """
        print(f"DEBUG execute: {getattr(plan, 'operation', None)}")
        if not plan or not hasattr(plan, 'operation'):
            return {'success': False, 'error': 'Plan de ejecución inválido'}
        try:
            operation = plan.operation
            print(f" Operación a ejecutar: {operation}")
            if operation == 'CREATE_TABLE':
                result = self._execute_create_table(plan)
            elif operation == 'SELECT':
                result = self._execute_select(plan)
            elif operation == 'INSERT':
                result = self._execute_insert(plan)
            elif operation == 'UPDATE':
                result = self._execute_update(plan)
            elif operation == 'DELETE':
                result = self._execute_delete(plan)
            else:
                result = {'success': False, 'error': f'Operación no soportada: {operation}'}
            print(f" Resultado de {operation}: {result.get('success')}")
            if 'success' not in result:
                result['success'] = False
                if 'error' not in result:
                    result['error'] = 'Error desconocido'
            
            return result

        except Exception as e:
            print(f" EXCEPCIÓN en execute: {e}")
            return {'success': False, 'error': f'Error ejecutando operación: {str(e)}'}

    # -------------------------
    # Helpers
    # -------------------------
    def _plan_get(self, plan, name, default=None):
        if plan is None:
            return default
        try:
            if isinstance(plan, dict):
                return plan.get(name, default)
        except Exception:
            pass
        if hasattr(plan, name):
            return getattr(plan, name)
        if hasattr(plan, "data") and isinstance(plan.data, dict):
            return plan.data.get(name, default)
        if hasattr(plan, "__dict__") and isinstance(plan.__dict__, dict):
            return plan.__dict__.get(name, default)
        return default

    def _extract_limit_from_plan(self, plan) -> Optional[int]:
        names = ["limit", "limit_clause", "limit_value", "limit_raw"]
        for n in names:
            v = self._plan_get(plan, n, None)
            if v is None:
                continue
            if isinstance(v, int):
                return v
            try:
                s = str(v)
                m = re.search(r'(\d+)', s)
                if m:
                    return int(m.group(1))
            except Exception:
                pass
            if isinstance(v, (list, tuple)):
                for e in v:
                    if isinstance(e, int):
                        return e
                    try:
                        s = str(e)
                        m = re.search(r'(\d+)', s)
                        if m:
                            return int(m.group(1))
                    except Exception:
                        pass
        try:
            if hasattr(plan, "__dict__"):
                for v in plan.__dict__.values():
                    if isinstance(v, int):
                        return v
                    try:
                        s = str(v)
                        m = re.search(r'(\d+)', s)
                        if m:
                            return int(m.group(1))
                    except Exception:
                        pass
        except Exception:
            pass
        for attr in ("raw", "sql", "text"):
            s = self._plan_get(plan, attr, None)
            if isinstance(s, str):
                m = re.search(r'LIMIT\s+(\d+)', s, re.I)
                if m:
                    return int(m.group(1))
        return None

    def _normalize_hits(self, raw_hits) -> Dict[str, float]:
        """
        Normaliza la salida del índice a {id: score}.
        Acepta: dict(id->score), list of (id,score), list of id.
        """
        hits: Dict[str, float] = {}
        if raw_hits is None:
            return hits
        if isinstance(raw_hits, dict):
            for k, v in raw_hits.items():
                try:
                    hits[str(k)] = float(v) if v is not None else 1.0
                except Exception:
                    hits[str(k)] = 1.0
            return hits
        if isinstance(raw_hits, (list, tuple)):
            for item in raw_hits:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    try:
                        hits[str(item[0])] = float(item[1]) if item[1] is not None else 1.0
                    except Exception:
                        hits[str(item[0])] = 1.0
                else:
                    hits[str(item)] = 1.0
            return hits
        try:
            hits[str(raw_hits)] = 1.0
        except Exception:
            pass
        return hits

    # -------------------------
    # SELECT / WHERE
    # -------------------------
    def _execute_select(self, plan: ExecutionPlan) -> Dict[str, Any]:
        def _flatten_select(sel):
            if sel is None:
                return ['*']
            flat = []
            for e in sel:
                if isinstance(e, list):
                    flat.extend(_flatten_select(e))
                else:
                    flat.append(e)
            return flat or ['*']

        try:
            table_name = self._plan_get(plan, "table_name", self._plan_get(plan, "table", None))
            raw_select = self._plan_get(plan, "select_list", self._plan_get(plan, "select", None))
            select_list = _flatten_select(raw_select)
            where_clause = self._plan_get(plan, "where_clause", self._plan_get(plan, "where", None))
            plan_limit = self._extract_limit_from_plan(plan)

            if not table_name:
                return {'success': False, 'error': 'No table specified in SELECT'}

            table_info = self.tables.get(table_name, {})
            index_type = (table_info.get('index_type') or '').upper()
            structure = self.structures.get(table_name)
            if structure is None and table_info:
                try:
                    self._reload_structure(table_name, table_info)
                    structure = self.structures.get(table_name)
                except Exception as e:
                    print(f"DEBUG reload failed: {e}")

            if structure is None:
                return {'success': False, 'error': f'Estructura de {table_name} no cargada. Tablas disponibles: {list(self.structures.keys())}'}

            if where_clause:
                results = self._execute_where_clause(structure, where_clause, index_type, plan_limit)
            else:
                results = self._select_all(structure, index_type)

            if results is None:
                results = []
            if not isinstance(results, list):
                results = [results]

            if select_list and select_list != ['*'] and results:
                projected = []
                for r in results:
                    if isinstance(r, dict):
                        projected.append({k: r.get(k) for k in select_list if k in r})
                    else:
                        projected.append(r)
                results = projected

            if plan_limit and isinstance(results, list):
                results = results[:plan_limit]

            count = len(results) if isinstance(results, list) else (1 if results else 0)

            return {
                'success': True,
                'results': results,
                'count': count,
                'table_name': table_name,
                'index_type': index_type
            }
        except Exception as e:
            print(f"ERROR en _execute_select: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': f'Error en _execute_select: {e}'}

    def _select_all(self, structure, index_type: str) -> List[Dict[str, Any]]:
        print(f"DEBUG _select_all: tipo={index_type}, estructura={type(structure).__name__}")
        try:
            if index_type in ['SEQ', 'SEQUENTIAL']:
                if not hasattr(structure, 'get_all'):
                    return []
                records = structure.get_all()
                if not isinstance(records, list):
                    return []
                results = []
                for record in records:
                    if hasattr(record, 'values') and hasattr(record, 'table'):
                        field_names = [f.name for f in record.table.fields]
                        results.append(dict(zip(field_names, record.values)))
                    elif isinstance(record, dict):
                        results.append(record)
                    elif isinstance(record, (list, tuple)):
                        # fallback: cannot know field_names here
                        results.append({'data': list(record)})
                    else:
                        results.append({'data': str(record)})
                return results

            elif index_type == 'BTREE':
                if hasattr(structure, 'get_all_records'):
                    records = structure.get_all_records()
                    return [r if isinstance(r, dict) else {'data': str(r)} for r in (records or [])]
                return []

            elif index_type in ('ISAM', 'EXTENDIBLEHASH', 'RTREE'):
                if hasattr(structure, 'get_all'):
                    records = structure.get_all()
                    out = []
                    for record in (records or []):
                        if hasattr(record, 'values') and hasattr(record, 'table'):
                            field_names = [f.name for f in record.table.fields]
                            out.append(dict(zip(field_names, record.values)))
                        elif isinstance(record, dict):
                            out.append(record)
                        elif isinstance(record, (list, tuple)):
                            out.append({'data': list(record)})
                        else:
                            out.append({'data': str(record)})
                    return out
                return []

            elif index_type in ('FULLTEXT', 'SPIMI'):
                tbl_name = None
                for tn, s in self.structures.items():
                    if s is structure:
                        tbl_name = tn
                        break
                if tbl_name:
                    tbl_info = self.tables.get(tbl_name, {})
                    src = tbl_info.get('source') or tbl_info.get('source_file')
                    if src and os.path.exists(src):
                        with open(src, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            return [row for row in reader]
                return []

            else:
                return []
        except Exception as e:
            print(f"ERROR en _select_all: {e}")
            import traceback
            traceback.print_exc()
            return []

    # -------------------------
    # CREATE TABLE
    # -------------------------
    def _create_table_from_file(self, table_name: str, plan: ExecutionPlan) -> Dict[str, Any]:
        """Crea tabla desde archivo CSV - VERSIÓN CORREGIDA."""
        file_path = plan.data['source']
        index_type = (plan.data.get('index_type') or '').upper()
        key_field = plan.data.get('key_field')

        print(f"DEBUG _create_table_from_file: {table_name}, {file_path}, {index_type}, {key_field}")
        
        # DEBUG DETALLADO de rutas
        print(f"DEBUG Ruta solicitada: {file_path}")
        print(f"DEBUG Ruta absoluta: {os.path.abspath(file_path)}")
        print(f"DEBUG Existe?: {os.path.exists(file_path)}")
        print(f"DEBUG Directorio actual: {os.getcwd()}")
        print(f"DEBUG Archivos en directorio actual: {os.listdir('.')}")
        if os.path.exists('data'):
            print(f"DEBUG Archivos en data/: {os.listdir('data')}")
        if os.path.exists('data/imagenes/'):
            print(f"DEBUG Archivos en data/imagenes/: {os.listdir('data')}")
        if not os.path.exists(file_path):
            return {'success': False, 'error': f'Archivo no encontrado: {file_path}. Ruta absoluta: {os.path.abspath(file_path)}'}

        # Forzar que file_path sea absoluta dentro base_dir/data si no absoluta
        if not os.path.isabs(file_path):
            abs_path = os.path.abspath(os.path.join(self.base_dir, file_path))
        else:
            abs_path = os.path.abspath(file_path)
        # If path not inside base_dir/data try to resolve against base_dir/data
        base_data = os.path.abspath(os.path.join(self.base_dir, 'data'))
        if os.path.commonpath([base_data, abs_path]) != base_data:
            # try base_data + basename
            candidate = os.path.join(base_data, os.path.basename(file_path))
            if os.path.exists(candidate):
                abs_path = candidate
            else:
                return {'success': False, 'error': f'Archivo fuera de {base_data} o no encontrado: {file_path}'}
        if not os.path.exists(abs_path):
            return {'success': False, 'error': f'Archivo no encontrado: {abs_path}'}

        if index_type in ('FULLTEXT', 'SPIMI'):
            text_col = None
            try:
                with open(abs_path, 'r', encoding='utf-8') as fh:
                    reader = csv.DictReader(fh)
                    headers = reader.fieldnames or []
                    headers_l = [h.lower() for h in headers]
                    candidates = ('text', 'content', 'body', 'description', 'title', 'titulo', 'name', 'nombre')
                    for candidate in candidates:
                        if candidate in headers_l:
                            text_col = headers[headers_l.index(candidate)]
                            break
                    if text_col is None and headers:
                        for h in headers:
                            if key_field and h == key_field:
                                continue
                            text_col = h
                            break
            except Exception as e:
                print(f"WARN: No se pudo leer headers CSV para detectar text_col: {e}")
            if text_col is None:
                text_col = key_field or 'text'
            id_col = key_field if key_field else 'id'
            spimi = SPIMIIndex(datafilename=abs_path, max_terms_in_block=1000, text_col=text_col, title_col=None, id_col=id_col)
            spimi.run()
            self.structures[table_name] = spimi

            # inferir fields
            inferred_fields = []
            try:
                with open(abs_path, 'r', encoding='utf-8') as fh:
                    rdr = csv.DictReader(fh)
                    hdrs = rdr.fieldnames or []
                    for h in hdrs:
                        inferred_fields.append({'name': h, 'type': 'VARCHAR', 'size': 0, 'index': None})
            except Exception:
                inferred_fields = []

            fields_to_store = plan.data.get('fields') if plan.data.get('fields') else inferred_fields

            self.tables[table_name] = {
                'table_name': table_name,
                'fields': fields_to_store,
                'index_type': index_type,
                'key_field': key_field,
                'source': abs_path,
                'text_col': text_col,
                'id_col': id_col
            }
            self._save_metadata()
            return {'success': True, 'message': f'Table {table_name} creada con índice FULLTEXT desde {abs_path} (text_col={text_col})'}

        # non-fulltext: inferir esquema y crear estructura
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                field_names = reader.fieldnames or []
                first_row = next(reader, None)
            if not field_names:
                return {'success': False, 'error': f'Archivo CSV vacío o sin encabezados: {abs_path}'}
            fields = []
            for col_name in field_names:
                # Determinar tipo basado en nombre de columna
                if col_name.lower() in ['id', 'codigo', 'numero', 'usuario_id']:
                    data_type = 'INT'
                    size = 0
                elif col_name.lower() in ['precio', 'valor', 'costo', 'rating', 'total', 'ubicacion_x', 'ubicacion_y']:
                    data_type = 'FLOAT'
                    size = 0
                else:
                    data_type = 'VARCHAR'
                    size = 50
                fields.append({
                    'name': col_name,
                    'type': data_type,
                    'size': size,
                    'index': None
                })
            # Guardar metadatos de la tabla
            self.tables[table_name] = {
                'table_name': table_name,
                'fields': fields,
                'index_type': index_type,
                'key_field': key_field,
                'source': abs_path
            }
            structure = self._create_structure(table_name, index_type, fields, key_field)
            self.structures[table_name] = structure
            # Cargar datos del CSV
            record_count = self._load_data_from_csv(table_name, abs_path, fields, structure, index_type, key_field)
            self._save_metadata()
            return {
                'success': True,
                'message': f'Tabla "{table_name}" creada exitosamente desde "{abs_path}"',
                'rows_loaded': record_count,
                'fields': len(fields)
            }
        except Exception as e:
            return {'success': False, 'error': f'Error creando tabla desde archivo: {str(e)}'}

    def _execute_create_table(self, plan: ExecutionPlan) -> Dict[str, Any]:
        table_name = plan.data['table_name']
        if plan.data.get('source'):
            result = self._create_table_from_file(table_name, plan)
        else:
            result = self._create_table_from_schema(table_name, plan)
        if result.get('success'):
            self._save_metadata()
        return result

    def _create_table_from_schema(self, table_name: str, plan: ExecutionPlan) -> Dict[str, Any]:
        """Crea tabla desde esquema definido."""
        fields_data = plan.data['fields']
        try:
            # Crear campos
            fields = []
            key_field = None
            index_type = 'SEQ'  # Por defecto
            for field_data in fields_data:
                name = field_data['name']
                data_type = field_data['type']
                size = field_data.get('size', 0)
                field_index = field_data.get('index')
                
                # Determinar tipo de Python
                if data_type == 'INT':
                    type_class = int
                elif data_type == 'VARCHAR':
                    type_class = str
                elif data_type == 'FLOAT':
                    type_class = float
                elif data_type == 'DATE':
                    type_class = str
                elif data_type == 'ARRAY[FLOAT]':
                    type_class = list
                else:
                    type_class = str  # Por defecto
                
                # Si tiene índice y es el primero, usarlo como índice principal
                if field_index and key_field is None:
                    key_field = name
                    index_type = field_index
                
                fields.append({
                    'name': name,
                    'type': data_type,
                    'size': size,
                    'index': field_index
                })
            if not key_field:
                key_field = fields[0]['name'] if fields else 'id'
            
            # Guardar metadatos de la tabla
            self.tables[table_name] = {
                'table_name': table_name,
                'fields': fields,
                'index_type': index_type,
                'key_field': key_field,
                'source': None
            }
            self._save_metadata()
            return {
                'success': True,
                'message': f'Tabla "{table_name}" creada exitosamente con esquema',
                'fields': len(fields),
                'index_type': index_type
            }

        except Exception as e:
            return {'success': False, 'error': f'Error creando tabla desde esquema: {e}'}

    # -------------------------
    # Core structures creation & loading
    # -------------------------
    def _create_structure(self, table_name: str, index_type: str, fields: List, key_field: str):
        """Crea estructura de datos REAL"""
        index_type = (index_type or '').upper()
        print(f"DEBUG Creando estructura REAL: {index_type} para {table_name}")
        print(f"DEBUG Campos recibidos: {fields}")
        print(f"DEBUG Key field: {key_field}")
        
        try:
            if index_type in ('SEQ', 'SEQUENTIAL'):
                # Crear objeto Table con los campos
                table_fields = []
                for field_info in fields:
                    # Convertir tipos string a clases Python
                    field_type = field_info.get('type', 'VARCHAR')
                    if field_type in ('INT', int):
                        data_type = int
                    elif field_type in ('FLOAT', float):
                        data_type = float
                    else:  # VARCHAR y otros
                        data_type = str
                    
                    table_fields.append(Field(
                        name=field_info['name'],
                        data_type=data_type,
                        size=field_info.get('size', 50)
                    ))
                
                # Crear objeto Table
                table_obj = Table(name=table_name, fields=table_fields, key_field=key_field)
                # Crear directorio data/ si no existe
                os.makedirs(os.path.join(self.base_dir, 'data'), exist_ok=True)
                structure = ExtendibleHashing(
                    bucketSize=3, 
                    index_filename=os.path.join('data', f"{table_name}_hash.idx"), 
                    table=table_obj  # ✅ Pasar la tabla al constructor
                )
                print(f"OK Sequential File creado: {type(structure)}")
            elif index_type == 'BTREE':
                os.makedirs(os.path.join(self.base_dir, 'data'), exist_ok=True)
                structure = BPlusTree(order=4, index_filename=os.path.join('data', f"{table_name}_btree.idx"))
                print(f"OK B+ Tree creado: {type(structure)}")
            elif index_type == 'ISAM':
                # Crear objeto Table con los campos
                table_fields = []
                for field_info in fields:
                    # Convertir tipos string a clases Python
                    field_type = field_info.get('type', 'VARCHAR')
                    if field_type in ('INT', int):
                        data_type = int
                    elif field_type in ('FLOAT', float):
                        data_type = float
                    else:  # VARCHAR y otros
                        data_type = str
                    
                    table_fields.append(Field(
                        name=field_info['name'],
                        data_type=data_type,
                        size=field_info.get('size', 50)
                    ))
                
                # Crear objeto Table
                table_obj = Table(name=table_name, fields=table_fields, key_field=key_field)
                os.makedirs(os.path.join(self.base_dir, 'data'), exist_ok=True)
                structure = ISAMIndex(os.path.join('data', f"{table_name}_isam.dat"), table=table_obj)
                print(f"OK ISAM creado: {type(structure)}")

            elif index_type == 'EXTENDIBLEHASH':
                os.makedirs(os.path.join(self.base_dir, 'data'), exist_ok=True)
                structure = ExtendibleHashing(bucketSize=3, index_filename=os.path.join('data', f"{table_name}_hash.idx"))
                print(f"OK Extendible Hashing creado: {type(structure)}")

            elif index_type == 'RTREE':
                # Para R-tree necesitamos identificar campos espaciales
                spatial_fields = []
                for field_info in fields:
                    if field_info.get('type') in ('ARRAY[FLOAT]',) or 'ubicacion' in field_info['name'].lower() or 'lat' in field_info['name'].lower() or 'lon' in field_info['name'].lower():
                        spatial_fields.append(field_info)
                numeric_fields = [f for f in fields if f.get('type') in ('FLOAT', 'INT', float, int)]
                if len(spatial_fields) < 2:
                    if len(numeric_fields) >= 2:
                        spatial_fields = numeric_fields[:2]
                    else:
                        raise ValueError("R-tree requiere al menos 2 campos numéricos")
                spatial_field_objects = []
                for field_info in spatial_fields[:2]:  # Solo necesitamos 2 campos para coordenadas
                    # Convertir tipo string a clase Python
                    field_type = field_info.get('type', 'FLOAT')
                    if field_type in ('INT', int):
                        data_type = int
                    elif field_type in ('FLOAT', float):
                        data_type = float
                    else:
                        data_type = str
                    
                    spatial_field_objects.append(Field(
                        name=field_info['name'],
                        data_type=data_type,
                        size=field_info.get('size', 0)
                    ))
                    print(f"DEBUG Campo R-tree: {field_info['name']} -> {data_type}")
                os.makedirs(os.path.join(self.base_dir, 'data'), exist_ok=True)
                structure = RTreeIndex(
                    index_filename=os.path.join('data', f"{table_name}_rtree.idx"),
                    fields=spatial_field_objects,
                    max_children=4
                )
                print(f"OK R-tree creado: {type(structure)}")
                
            elif index_type in ('FULLTEXT', 'SPIMI'):
                raise ValueError("FULLTEXT/SPIMI debe ser creado desde archivo CSV")
            else:
                raise ValueError(f"Tipo de índice no soportado: {index_type}")
        except Exception as e:
            print(f"ERROR creando estructura real: {e}")
            import traceback
            traceback.print_exc()
            raise  # LANZAR excepción en lugar de retornar None

    def _load_data_from_csv(self, table_name, file_path, fields, structure, index_type, key_field):
        """Carga datos desde CSV Y construye el índice."""
        print(f"DEBUG Cargando datos desde {file_path}")
        count = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        values = []
                        for field_info in fields:
                            field_name = field_info['name']
                            field_type = field_info['type']
                            raw_value = row.get(field_name, '')
                            if field_type == 'INT':
                                value = int(raw_value) if raw_value else 0
                            elif field_type == 'FLOAT':
                                value = float(raw_value) if raw_value else 0.0
                            else:
                                value = str(raw_value)
                            values.append(value)
                        # **INSERTAR en la estructura de índice**
                        key = values[0] if values else None
                        if index_type in ['SEQ', 'SEQUENTIAL']:
                            # CREAR CAMPOS CON LOS TIPOS CORRECTOS (no todo str)
                            table_fields = []
                            for field_info in fields:
                                ft = field_info['type']
                                if ft == 'INT':
                                    python_type = int
                                elif ft == 'FLOAT':
                                    python_type = float
                                else:
                                    python_type = str
                                table_fields.append(Field(
                                    name=field_info['name'], 
                                    data_type=python_type, 
                                    size=field_info.get('size', 50)
                                ))
                            table_obj = Table(table_name, table_fields, key_field)
                            record = Record(table_obj, values)
                            print(f"DEBUG Insertando registro {count}: {values} (tipos: {[type(v).__name__ for v in values]})")
                            structure.add(record)
                        elif index_type == 'BTREE':
                            # Insertar en B+ Tree
                            structure.insert(key, values)
                        elif index_type == 'ISAM':
                            structure.insert(key, values)
                        elif index_type == 'EXTENDIBLEHASH':
                            structure.insert(key, values)
                        elif index_type == 'RTREE':
                            coords = [v for v in values if isinstance(v, (int, float))]
                            if len(coords) >= 2:
                                structure.insert(coords, values)
                        count += 1
                    except Exception as e:
                        print(f"ERROR cargando fila {count}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
        except Exception as e:
            print(f"ERROR leyendo CSV: {e}")
        print(f"DEBUG Cargados {count} registros en {table_name}")
        return count

    # -------------------------
    # WHERE clause executor
    # -------------------------
    def _execute_where_clause(self, structure, where_clause, index_type, limit_value=None):
        """
        Ejecuta cláusula WHERE USANDO los índices para optimizar.
        
        Args:
            structure: Estructura de datos (índice)
            where_clause: Diccionario con la condición WHERE
            index_type: Tipo de índice
            limit_value: Valor de LIMIT si está presente
        """
        condition_type = where_clause['type']
        field = where_clause.get('field')
        # Obtener información de la tabla para saber cuál es el key_field
        table_name = None
        for tn, ti in self.tables.items():
            if self.structures.get(tn) is structure:
                table_name = tn
                break
        if not table_name:
            print("ERROR: No se encontró tabla para la estructura")
            return []
        table_info = self.tables.get(table_name, {})
        key_field = table_info.get('key_field')
        fields_info = table_info.get('fields', [])

        # BÚSQUEDA POR IGUALDAD
        if condition_type == 'comparison':
            value = where_clause.get('value')
            operator = where_clause.get('operator')
            if operator == '=' and field == key_field:
                if index_type in ['BTREE', 'EXTENDIBLEHASH', 'ISAM']:
                    result = structure.search(value)
                    if result:
                        if isinstance(result, dict):
                            return [result]
                        elif isinstance(result, (list, tuple)):
                            field_names = [f['name'] for f in fields_info]
                            return [dict(zip(field_names, result))]
                        else:
                            return [{'data': str(result)}]
                    return []
                if index_type in ['SEQ', 'SEQUENTIAL']:
                    record = structure.search(value)
                    if record:
                        if hasattr(record, 'values') and hasattr(record, 'table'):
                            field_names = [f.name for f in record.table.fields]
                            return [dict(zip(field_names, record.values))]
                        elif isinstance(record, (list, tuple)):
                            field_names = [f['name'] for f in fields_info]
                            return [dict(zip(field_names, record))]
                        elif isinstance(record, dict):
                            return [record]
                        else:
                            return [{'data': str(record)}]
                    return []
                # fallback naive scan
                return self._scan_with_field_condition(structure, field, operator, value, index_type)
            else:
                return self._scan_with_field_condition(structure, field, operator, value, index_type)

        elif condition_type == 'between':
            start, end = where_clause.get('start'), where_clause.get('end')
            if field == key_field and index_type == 'BTREE':
                positions = structure.range_search(start, end)
                return positions or []
            return self._scan_with_range_condition(structure, field, start, end, index_type)

        elif condition_type == 'spatial':
            point = where_clause.get('point')
            radius_or_k = where_clause.get('radius') or where_clause.get('k', 10)
            if index_type == 'RTREE':
                ids = structure.spatial_search(point, radius_or_k)
                results = []
                for item in ids or []:
                    if isinstance(item, dict):
                        results.append(item)
                    elif isinstance(item, (list, tuple)):
                        field_names = [f['name'] for f in fields_info]
                        results.append(dict(zip(field_names, item)))
                return results
            else:
                return []

        # BÚSQUEDA MULTIMEDIA - Búsqueda por similitud de imágenes
        elif condition_type == 'multimedia':
            print(f"DEBUG _execute_where_clause: Detectada búsqueda multimedia")
            query_path = where_clause.get('query_path')
            if not query_path:
                print(f"ERROR: No se encontró query_path en la condición multimedia")
                return []
            print(f"DEBUG _execute_where_clause: Ejecutando búsqueda multimedia en tabla {table_name}")
            # Usar limit_value pasado como parámetro, o 10 por defecto
            search_limit = limit_value if limit_value and isinstance(limit_value, int) else 10
            print(f"DEBUG _execute_where_clause: Usando limit={search_limit} para búsqueda multimedia")
            return self._execute_multimedia_search(table_name, query_path, search_limit)
        
        # BÚSQUEDA FULLTEXT
        elif condition_type == 'fulltext':
            query = where_clause.get('query') or where_clause.get('term') or ''
            print(f"DEBUG _execute_where_clause: Detectada búsqueda fulltext por query '{query}' en campo '{field}'")
            if index_type in ('FULLTEXT', 'SPIMI'):
                try:
                    structure.load_metadata()
                except Exception:
                    pass
                search_limit = limit_value if isinstance(limit_value, int) and limit_value > 0 else 100
                raw_hits = None
                try:
                    # algunos índices aceptan k parameter, otros no
                    try:
                        raw_hits = structure.search(query, k=search_limit)
                    except TypeError:
                        raw_hits = structure.search(query)
                except Exception as e:
                    print(f"WARN: fulltext search failed: {e}")
                    raw_hits = None

                hits_map = self._normalize_hits(raw_hits)
                # si no hay hits -> fallback por scan simple
                if not hits_map:
                    print("WARN: SPIMI search returned no hits; using naive CSV scan fallback")
                    src = table_info.get('source') or table_info.get('source_file')
                    if not src or not os.path.exists(src):
                        return []
                    qterms = [t.lower() for t in str(query).split() if t.strip()]
                    doc_ids = []
                    id_field_name = key_field or table_info.get('id_col') or 'id'
                    with open(src, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for r in reader:
                            txt = str(r.get(field, '')).lower()
                            if all(term in txt for term in qterms):
                                rid = str(r.get(id_field_name, '')).strip()
                                if rid:
                                    doc_ids.append(rid)
                    doc_ids = doc_ids[:search_limit]
                    # mapear a filas
                    results = []
                    with open(src, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        rows_by_id = {str(r.get(key_field or table_info.get('id_col') or 'id')): r for r in reader}
                    for did in doc_ids:
                        row = rows_by_id.get(did)
                        if row:
                            row['score'] = 1.0
                            results.append(row)
                    return results

                # ordenar hits por score descendente
                ordered = sorted(hits_map.items(), key=lambda x: x[1], reverse=True)
                doc_ids = [str(d) for d, s in ordered][:search_limit]

                src = table_info.get('source') or table_info.get('source_file')
                if not src or not os.path.exists(src):
                    print(f"ERROR: No se encontró archivo fuente para tabla {table_name}")
                    return []
                results = []
                with open(src, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    id_field_name = key_field or table_info.get('id_col') or 'id'
                    rows_by_id = {}
                    for r in reader:
                        rid = str(r.get(id_field_name))
                        if rid:
                            rows_by_id[rid] = r
                    for did in doc_ids:
                        row = rows_by_id.get(did)
                        if row:
                            row = dict(row)
                            row['score'] = float(hits_map.get(did, 0.0))
                            results.append(row)
                return results
            else:
                print(f"WARN: Búsqueda fulltext solicitada pero índice es {index_type}")
                return []

        return []

    # -------------------------
    # Scans
    # -------------------------
    def _scan_with_field_condition(self, structure, field, operator, value, index_type):
        """Realiza un scan completo para buscar por un campo que NO es clave."""
        print(f"DEBUG Realizando scan completo: {field} {operator} {value}")
        # Obtener todos los registros
        all_records = self._select_all(structure, index_type)
        # Filtrar por condición
        results = []
        for record in all_records:
            if isinstance(record, dict) and field in record:
                record_value = record[field]
                try:
                    # Aplicar operador
                    if operator == '=' and record_value == value:
                        results.append(record)
                    elif operator == '>' and record_value > value:
                        results.append(record)
                    elif operator == '<' and record_value < value:
                        results.append(record)
                    elif operator == '>=' and record_value >= value:
                        results.append(record)
                    elif operator == '<=' and record_value <= value:
                        results.append(record)
                    elif operator in ('!=', '<>') and record_value != value:
                        results.append(record)
                except Exception:
                    continue
        print(f"DEBUG Scan completado: {len(results)} registros encontrados")
        return results
    
    def _scan_with_range_condition(self, structure, field, start, end, index_type):
        """Realiza un scan completo para BETWEEN en campo NO clave."""
        print(f"DEBUG Realizando scan completo para rango: {field} BETWEEN {start} AND {end}")
        
        all_records = self._select_all(structure, index_type)
        
        results = []
        for record in all_records:
            if isinstance(record, dict) and field in record:
                try:
                    rv = record[field]
                    if start <= rv <= end:
                        results.append(record)
                except Exception:
                    continue
        print(f"DEBUG Scan de rango completado: {len(results)} registros encontrados")
        return results

    # -------------------------
    # INSERT / UPDATE / DELETE (simplificados)
    # -------------------------
    def _execute_insert(self, plan: ExecutionPlan) -> Dict[str, Any]:
        table_name = plan.data['table_name']
        values = plan.data['values']
        if table_name not in self.tables:
            return {'success': False, 'error': f'Tabla "{table_name}" no existe'}
        try:
            table_info = self.tables[table_name]
            structure = self.structures[table_name]
            key_field = table_info.get('key_field')
            key_index = next((i for i, f in enumerate(table_info.get('fields', [])) if f['name'] == key_field), 0)
            key_value = values[key_index] if key_index < len(values) else None
            if key_value is None:
                return {'success': False, 'error': 'No se pudo determinar clave primaria'}
            if hasattr(structure, 'insert'):
                structure.insert(key_value, values)
            else:
                return {'success': False, 'error': 'Estructura no soporta insert'}
            return {'success': True, 'message': f'Registro insertado en "{table_name}" con clave {key_value}', 'values': values}
        except Exception as e:
            return {'success': False, 'error': f'Error insertando registro: {e}'}

    def _execute_update(self, plan: ExecutionPlan) -> Dict[str, Any]:
        return {'success': False, 'error': 'UPDATE no implementado aún'}

    def _execute_delete(self, plan: ExecutionPlan) -> Dict[str, Any]:
        table_name = plan.data.get('table_name')
        where_clause = plan.data.get('where_clause')
        if table_name not in self.tables:
            return {'success': False, 'error': f'Tabla "{table_name}" no existe'}
        try:
            structure = self.structures[table_name]
            if not where_clause:
                return {'success': False, 'error': 'DELETE sin WHERE no implementado'}
            if where_clause.get('type') == 'comparison':
                field = where_clause.get('field')
                value = where_clause.get('value')
                operator = where_clause.get('operator')
                if operator == '=' and field == self.tables[table_name].get('key_field'):
                    existing = structure.search(value)
                    if existing:
                        if hasattr(structure, 'delete'):
                            structure.delete(value)
                            return {'success': True, 'message': f'Registro con clave {value} eliminado de "{table_name}"'}
                        return {'success': False, 'error': 'Estructura no soporta delete'}
                    return {'success': False, 'error': f'Clave {value} no encontrada'}
            return {'success': False, 'error': 'Tipo de condición no soportado'}
        except Exception as e:
            return {'success': False, 'error': f'Error eliminando registro: {e}'}

    # -------------------------
    # Utilidades
    # -------------------------
    def list_tables(self) -> Dict[str, Any]:
        return {'success': True, 'tables': list(self.tables.keys()), 'count': len(self.tables)}

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        if table_name not in self.tables:
            return {'success': False, 'error': f'Tabla "{table_name}" no existe'}
        table_info = self.tables.get(table_name)
        return {
            'success': True,
            'table_name': table_name,
            'fields': table_info.get('fields', []),
            'index_type': table_info.get('index_type'),
            'key_field': table_info.get('key_field'),
            'source': table_info.get('source') or table_info.get('source_file')
        }
    

    def _execute_multimedia_search(self, table_name: str, query_image_path: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Ejecuta búsqueda por similitud de imágenes usando los módulos multimedia.
        
        Esta función:
        1. Carga los modelos multimedia (codebook, histogramas, índice invertido)
        2. Realiza la búsqueda por similitud usando SIFT + BOVW
        3. Retorna los resultados en formato de diccionarios compatibles con el sistema
        
        Args:
            table_name: Nombre de la tabla Multimedia
            query_image_path: Ruta a la imagen query (puede ser absoluta o relativa)
            limit: Número máximo de resultados a retornar
            
        Returns:
            Lista de diccionarios con los resultados de la búsqueda
        """
        print(f"\n{'='*60}")
        print(f"DEBUG _execute_multimedia_search: Iniciando búsqueda multimedia")
        print(f"DEBUG Tabla: {table_name}")
        print(f"DEBUG Query image path: {query_image_path}")
        print(f"DEBUG Limit: {limit}")
        print(f"{'='*60}\n")
        
        try:
            # Importar módulos multimedia
            import sys
            import os
            multimedia_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'multimedia')
            if multimedia_path not in sys.path:
                sys.path.insert(0, multimedia_path)
            
            from multimedia.sift_features import get_image_paths
            from multimedia.bovw import load_codebook, load_histograms
            from multimedia.sift_inverted_index import SIFTInvertedIndex
            from multimedia.sequential_search import SequentialSIFTSearch
            
            # Asegurar que base_dir apunte a la raíz del proyecto, no a backend/
            base_dir = self.base_dir
            # Convertir a ruta absoluta si es relativa
            if not os.path.isabs(base_dir):
                base_dir = os.path.abspath(base_dir)
            
            # Si base_dir termina en 'backend', subir un nivel
            if base_dir.endswith('backend') or base_dir.endswith('backend/'):
                base_dir = os.path.dirname(base_dir)
            
            # Rutas a los archivos multimedia (absolutas)
            # Las imágenes ahora están en data/imagenes/
            IMAGE_DIR = os.path.abspath(os.path.join(base_dir, 'data', 'imagenes'))
            CODEBOOK_PATH = os.path.abspath(os.path.join(base_dir, 'multimedia', 'database', 'codebook.pkl'))
            HISTOGRAMS_PATH = os.path.abspath(os.path.join(base_dir, 'multimedia', 'database', 'histograms.npz'))
            INVERTED_INDEX_PATH = os.path.abspath(os.path.join(base_dir, 'multimedia', 'database', 'inverted_index.pkl'))
            
            print(f"DEBUG base_dir original: {self.base_dir}")
            print(f"DEBUG base_dir corregido: {base_dir}")
            
            print(f"DEBUG IMAGE_DIR: {IMAGE_DIR}")
            print(f"DEBUG CODEBOOK_PATH: {CODEBOOK_PATH}")
            print(f"DEBUG HISTOGRAMS_PATH: {HISTOGRAMS_PATH}")
            print(f"DEBUG INVERTED_INDEX_PATH: {INVERTED_INDEX_PATH}")
            
            # Verificar que existan los archivos necesarios
            if not os.path.exists(CODEBOOK_PATH):
                error_msg = f"Codebook no encontrado en {CODEBOOK_PATH}. Ejecute primero la construcción del índice multimedia."
                print(f"ERROR: {error_msg}")
                return [{'error': error_msg}]
            
            if not os.path.exists(HISTOGRAMS_PATH):
                error_msg = f"Histogramas no encontrados en {HISTOGRAMS_PATH}. Ejecute primero la construcción del índice multimedia."
                print(f"ERROR: {error_msg}")
                return [{'error': error_msg}]
            
            # Verificar que la imagen query exista
            if not os.path.exists(query_image_path):
                # Intentar con ruta relativa desde base_dir
                query_image_path_alt = os.path.join(self.base_dir, query_image_path)
                if os.path.exists(query_image_path_alt):
                    query_image_path = query_image_path_alt
                    print(f"DEBUG Usando ruta alternativa: {query_image_path}")
                else:
                    error_msg = f"Imagen query no encontrada: {query_image_path}"
                    print(f"ERROR: {error_msg}")
                    return [{'error': error_msg}]
            
            print(f"DEBUG Imagen query encontrada: {query_image_path}")
            
            # Cargar modelos multimedia
            print(f"DEBUG Cargando codebook desde {CODEBOOK_PATH}...")
            kmeans_model, _ = load_codebook(CODEBOOK_PATH)
            vocab_size = kmeans_model.n_clusters
            print(f"DEBUG Codebook cargado: vocab_size={vocab_size}")
            
            print(f"DEBUG Cargando histogramas desde {HISTOGRAMS_PATH}...")
            histograms_tfidf, idf_weights = load_histograms(HISTOGRAMS_PATH)
            print(f"DEBUG Histogramas cargados: shape={histograms_tfidf.shape}")
            
            # Obtener rutas de imágenes en la base de datos
            print(f"DEBUG Obteniendo rutas de imágenes desde {IMAGE_DIR}...")
            image_paths = get_image_paths(IMAGE_DIR)
            print(f"DEBUG Encontradas {len(image_paths)} imágenes en la base de datos")
            
            if len(image_paths) == 0:
                error_msg = f"No se encontraron imágenes en {IMAGE_DIR}"
                print(f"ERROR: {error_msg}")
                return [{'error': error_msg}]
            
            # Verificar que el número de imágenes coincida con el número de histogramas
            if len(image_paths) != len(histograms_tfidf):
                print(f"WARN: Número de imágenes ({len(image_paths)}) no coincide con número de histogramas ({len(histograms_tfidf)})")
                # Usar el mínimo para evitar errores
                min_len = min(len(image_paths), len(histograms_tfidf))
                image_paths = image_paths[:min_len]
                histograms_tfidf = histograms_tfidf[:min_len]
                print(f"DEBUG Usando {min_len} imágenes/histogramas")
            
            # Intentar usar índice invertido si existe, sino usar búsqueda secuencial
            if os.path.exists(INVERTED_INDEX_PATH):
                print(f"DEBUG Usando índice invertido desde {INVERTED_INDEX_PATH}...")
                inverted_index = SIFTInvertedIndex(vocab_size)
                inverted_index.load(INVERTED_INDEX_PATH)
                inverted_index.set_idf(idf_weights)
                
                # Realizar búsqueda
                print(f"DEBUG Realizando búsqueda con índice invertido...")
                search_results = inverted_index.search_by_image_path(query_image_path, kmeans_model, k=limit)
            else:
                print(f"DEBUG Índice invertido no encontrado, usando búsqueda secuencial...")
                searcher = SequentialSIFTSearch(histograms_tfidf, image_paths, kmeans_model, idf_weights)
                
                # Realizar búsqueda
                print(f"DEBUG Realizando búsqueda secuencial...")
                search_results = searcher.search(query_image_path, k=limit)
            
            print(f"DEBUG Búsqueda completada: {len(search_results)} resultados encontrados")
            
            # Convertir resultados al formato esperado por el sistema
            # Los resultados vienen como [(image_path, score), ...]
            # Necesitamos convertirlos a diccionarios con los campos de la tabla
            results = []
            
            # Obtener información de la tabla para saber qué campos tiene
            table_info = self.tables.get(table_name, {})
            fields_info = table_info.get('fields', [])
            field_names = [f['name'] for f in fields_info]
            
            print(f"DEBUG Campos de la tabla: {field_names}")
            
            for img_path, score in search_results:
                # Extraer nombre del archivo de la ruta
                img_filename = os.path.basename(img_path)
                
                # Crear diccionario con los resultados
                # Asumimos que la tabla tiene campos como 'id', 'title', 'image_path', etc.
                result_dict = {}
                
                # Intentar mapear campos comunes
                if 'id' in field_names:
                    # Usar el índice o nombre del archivo como ID
                    result_dict['id'] = img_filename
                
                if 'title' in field_names:
                    # Usar el nombre del archivo sin extensión como título
                    title = os.path.splitext(img_filename)[0]
                    result_dict['title'] = title
                
                if 'image_path' in field_names or 'path' in field_names:
                    result_dict['image_path' if 'image_path' in field_names else 'path'] = img_path
                
                if 'score' in field_names or 'similarity' in field_names:
                    result_dict['score' if 'score' in field_names else 'similarity'] = float(score)
                
                # Agregar todos los campos que faltan con valores None o vacíos
                for field_name in field_names:
                    if field_name not in result_dict:
                        result_dict[field_name] = None
                
                results.append(result_dict)
                print(f"DEBUG Resultado: {img_filename} -> score={score:.4f}")
            
            print(f"DEBUG Total de resultados formateados: {len(results)}")
            print(f"{'='*60}\n")
            
            return results
            
        except ImportError as e:
            error_msg = f"Error importando módulos multimedia: {e}"
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            return [{'error': error_msg}]
        except Exception as e:
            error_msg = f"Error ejecutando búsqueda multimedia: {e}"
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            return [{'error': error_msg}]