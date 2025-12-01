#!/usr/bin/env python3
"""
Executor SQL que toma ExecutionPlan y los ejecuta sobre las estructuras de datos.
"""

import os
import sys
import csv
import json
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
from indexes.isam import ISAMIndex
from indexes.sequential_file import SequentialIndex



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
        """Carga metadatos de tablas desde archivo JSON."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    self.tables = json.load(f)
                print(f"DEBUG Metadatos cargados: {list(self.tables.keys())}")
                
                # Recargar estructuras de índices
                for table_name, table_info in self.tables.items():
                    self._reload_structure(table_name, table_info)
            except Exception as e:
                print(f"ERROR cargando metadatos: {e}")
    
    def _reload_structure(self, table_name, table_info):
        """Recarga estructura de índice desde archivos persistidos."""
        index_type = table_info['index_type']
        fields = table_info['fields']
        key_field = table_info['key_field']
        
        try:
            print(f"DEBUG Recargando estructura: {table_name} ({index_type})")
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
        """Guarda metadatos de tablas en archivo JSON."""
        try:
            os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
            with open(self.metadata_file, 'w') as f:
                json.dump(self.tables, f, indent=2)
            print(f"DEBUG Metadatos guardados: {list(self.tables.keys())}")
        except Exception as e:
            print(f"ERROR guardando metadatos: {e}")
    
    def execute(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """
        Ejecuta un ExecutionPlan - VERIFICAR ENLACE DELETE.
        """
        print(f" DEBUG execute: {plan.operation if plan else 'None'}")
        
        try:
            if not plan or not hasattr(plan, 'operation'):
                return {'success': False, 'error': 'Plan de ejecución inválido'}
            
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
                result = self._execute_delete(plan)  # ← ¿Se está llamando?
            else:
                result = {'success': False, 'error': f'Operación no soportada: {operation}'}
            
            print(f" Resultado de {operation}: {result.get('success')}")
            
            # Asegurar que siempre tenga 'success'
            if 'success' not in result:
                result['success'] = False
                if 'error' not in result:
                    result['error'] = 'Error desconocido'
            
            return result
            
        except Exception as e:
            print(f" EXCEPCIÓN en execute: {e}")
            return {'success': False, 'error': f'Error ejecutando operación: {str(e)}'}
    
    def _execute_delete(self, plan: ExecutionPlan) -> Dict[str, Any]:
        table_name = plan.data['table_name']
        where_clause = plan.data.get('where_clause')
        
        if table_name not in self.tables:
            return {'success': False, 'error': f'Tabla "{table_name}" no existe'}
        
        try:
            structure = self.structures[table_name]
            
            print(f"DEBUG Estructura real: {type(structure)}")
            
            if not where_clause:
                return {'success': False, 'error': 'DELETE sin WHERE no implementado'}
            
            if where_clause.get('type') == 'comparison':
                field = where_clause['field']
                value = where_clause['value']
                operator = where_clause['operator']
                
                if operator == '=':
                    # Buscar primero para verificar existencia
                    existing = structure.search(value)
                    print(f"DEBUG Búsqueda previa: {existing}")
                    
                    if existing:
                        result = structure.delete(value)
                        print(f"DEBUG Resultado delete: {result}")
                        return {
                            'success': True,
                            'message': f'Registro con clave {value} eliminado de "{table_name}"'
                        }
                    else:
                        return {'success': False, 'error': f'Clave {value} no encontrada'}
            
            return {'success': False, 'error': 'Tipo de condición no soportado'}
            
        except Exception as e:
            return {'success': False, 'error': f'Error eliminando registro: {str(e)}'}
    
    def _create_table_from_file(self, table_name: str, plan: ExecutionPlan) -> Dict[str, Any]:
        """Crea tabla desde archivo CSV - VERSIÓN CORREGIDA."""
        file_path = plan.data['source']
        index_type = plan.data['index_type'].upper()
        key_field = plan.data['key_field']
        
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
        
        try:
            # Leer CSV para inferir esquema
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                field_names = reader.fieldnames or []
                first_row = next(reader, None)
            
            if not field_names:
                return {'success': False, 'error': f'Archivo CSV vacío o sin encabezados: {file_path}'}
            
            # Crear campos basados en los encabezados del CSV
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
                'source_file': file_path
            }
            
            structure = self._create_structure(table_name, index_type, fields, key_field)
            self.structures[table_name] = structure
            
            # Cargar datos del CSV
            record_count = self._load_data_from_csv(table_name, file_path, fields, structure, index_type, key_field)
            
            return {
                'success': True,
                'message': f'Tabla "{table_name}" creada exitosamente desde "{file_path}"',
                'rows_loaded': record_count,
                'fields': len(fields)
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Error creando tabla desde archivo: {str(e)}'}
    
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
                    'type': type_class,
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
            
            
            return {
                'success': True,
                'message': f'Tabla "{table_name}" creada exitosamente con esquema',
                'fields': len(fields),
                'index_type': index_type
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Error creando tabla desde esquema: {e}'}
    
    def _create_structure(self, table_name: str, index_type: str, fields: List, key_field: str):
        """Crea estructura de datos REAL"""
        index_type = index_type.upper()
        
        print(f"DEBUG Creando estructura REAL: {index_type} para {table_name}")
        print(f"DEBUG Campos recibidos: {fields}")
        print(f"DEBUG Key field: {key_field}")
        
        try:
            if index_type == 'SEQ' or index_type == 'SEQUENTIAL':
                # Crear objeto Table con los campos
                table_fields = []
                for field_info in fields:
                    # Convertir tipos string a clases Python
                    field_type = field_info.get('type', 'VARCHAR')
                    
                    if field_type == 'INT' or field_type == int:
                        data_type = int
                    elif field_type == 'FLOAT' or field_type == float:
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
                os.makedirs('data', exist_ok=True)
                
                structure = ExtendibleHashing(
                    bucketSize=3, 
                    index_filename=f"data/{table_name}_hash.idx",
                    table=table_obj  # ✅ Pasar la tabla al constructor
                )
                print(f"OK Sequential File creado: {type(structure)}")
                
            elif index_type == 'BTREE':
                os.makedirs('data', exist_ok=True)
                structure = BPlusTree(order=4, index_filename=f"data/{table_name}_btree.idx")
                print(f"OK B+ Tree creado: {type(structure)}")
                
            elif index_type == 'ISAM':
                # Crear objeto Table con los campos
                table_fields = []
                for field_info in fields:
                    # Convertir tipos string a clases Python
                    field_type = field_info.get('type', 'VARCHAR')
                    
                    if field_type == 'INT' or field_type == int:
                        data_type = int
                    elif field_type == 'FLOAT' or field_type == float:
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
                
                os.makedirs('data', exist_ok=True)
                structure = ISAMIndex(f"data/{table_name}_isam.dat", table=table_obj)
                print(f"OK ISAM creado: {type(structure)}")
                
            elif index_type == 'EXTENDIBLEHASH':
                os.makedirs('data', exist_ok=True)
                structure = ExtendibleHashing(bucketSize=3, index_filename=f"data/{table_name}_hash.idx")
                print(f"OK Extendible Hashing creado: {type(structure)}")
                
            elif index_type == 'RTREE':
                # Para R-tree necesitamos identificar campos espaciales
                spatial_fields = []
                for field_info in fields:
                    field_type = field_info.get('type', '')
                    if field_type == 'ARRAY[FLOAT]' or 'ubicacion' in field_info['name'].lower() or 'lat' in field_info['name'].lower() or 'lon' in field_info['name'].lower():
                        spatial_fields.append(field_info)
                
                if len(spatial_fields) < 2:
                    # Si no hay campos espaciales explícitos, usar los primeros dos campos numéricos
                    numeric_fields = [f for f in fields if f.get('type') in ['FLOAT', 'INT', float, int]]
                    if len(numeric_fields) >= 2:
                        spatial_fields = numeric_fields[:2]
                        print(f"DEBUG Usando campos numéricos para R-tree: {[f['name'] for f in spatial_fields]}")
                    else:
                        raise ValueError("R-tree requiere al menos 2 campos numéricos para coordenadas")
                
                # CREAR OBJETOS Field a partir de los diccionarios
                spatial_field_objects = []
                for field_info in spatial_fields[:2]:  # Solo necesitamos 2 campos para coordenadas
                    # Convertir tipo string a clase Python
                    field_type = field_info.get('type', 'FLOAT')
                    if field_type == 'INT' or field_type == int:
                        data_type = int
                    elif field_type == 'FLOAT' or field_type == float:
                        data_type = float
                    else:
                        data_type = str
                    
                    spatial_field_objects.append(Field(
                        name=field_info['name'],
                        data_type=data_type,
                        size=field_info.get('size', 0)
                    ))
                    print(f"DEBUG Campo R-tree: {field_info['name']} -> {data_type}")
                
                os.makedirs('data', exist_ok=True)
                structure = RTreeIndex(
                    index_filename=f"data/{table_name}_rtree.idx",
                    fields=spatial_field_objects,
                    max_children=4
                )
                print(f"OK R-tree creado exitosamente: {type(structure)}")
        
            else:
                raise ValueError(f"Tipo de índice no soportado: {index_type}")
            
            #  VERIFICAR que structure NO sea None
            if structure is None:
                raise RuntimeError(f"La estructura {index_type} no se creó correctamente")
            
            print(f"OK Estructura creada exitosamente: {type(structure)}")
            return structure
            
        except Exception as e:
            print(f"ERROR creando estructura real: {e}")
            import traceback
            traceback.print_exc()
            raise  # LANZAR excepción en lugar de retornar None
    
    def _load_data_from_csv(self, table_name, file_path, fields, structure, index_type, key_field):
        """Carga datos desde CSV Y construye el índice."""
        import csv
        
        print(f"DEBUG Cargando datos desde {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            count = 0
            
            for row in reader:
                try:
                    # Convertir valores según tipo de campo
                    values = []
                    for field_info in fields:
                        field_name = field_info['name']
                        field_type = field_info['type']
                        raw_value = row.get(field_name, '')
                        
                        # Conversión de tipos
                        if field_type == 'INT':
                            value = int(raw_value) if raw_value else 0
                        elif field_type == 'FLOAT':
                            value = float(raw_value) if raw_value else 0.0
                        else:
                            value = str(raw_value)
                        
                        values.append(value)
                    
                    # **INSERTAR en la estructura de índice**
                    key = values[0]  # Asumiendo que la primera columna es la clave
                    
                    if index_type in ['SEQ', 'SEQUENTIAL']:
                        from core.models import Record, Table, Field
                        
                        # CREAR CAMPOS CON LOS TIPOS CORRECTOS (no todo str)
                        table_fields = []
                        for field_info in fields:
                            field_type = field_info['type']
                            
                            # Mapear tipo string a clase Python
                            if field_type == 'INT':
                                python_type = int
                            elif field_type == 'FLOAT':
                                python_type = float
                            else:
                                python_type = str
                            

                            table_fields.append(Field(
                                name=field_info['name'],
                                data_type=python_type,  # Usar tipo correcto
                                size=field_info.get('size', 50)
                            ))
                        
                        table_obj = Table(table_name, table_fields, key_field)
                        
                        # CREAR RECORD CON VALORES EN SU TIPO ORIGINAL
                        record = Record(table_obj, values)  # values ya tiene int, float, str
                        
                        print(f"DEBUG Insertando registro {count}: {values} (tipos: {[type(v).__name__ for v in values]})")
                        structure.add(record)  # add() debe manejar diferentes tipos
                        
                    elif index_type == 'BTREE':
                        # Insertar en B+ Tree
                        structure.insert(key, values)  # values completos como payload
                        
                    elif index_type == 'ISAM':
                        structure.insert(key, values)
                        
                    elif index_type == 'EXTENDIBLEHASH':
                        structure.insert(key, values)
                        
                    elif index_type == 'RTREE':
                        # Para R-tree necesitamos coordenadas
                        coords = []
                        for v in values:
                            if isinstance(v, (int, float)):
                                coords.append(float(v))
                                if len(coords) == 2:
                                    break
                        
                        if len(coords) >= 2:
                            structure.insert(coords, values)
                        else:
                            print(f"WARN: No se encontraron coordenadas en fila {count}")
                    
                    count += 1
                    
                except Exception as e:
                    print(f"ERROR cargando fila {count}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        print(f"DEBUG Cargados {count} registros en {table_name}")
        return count
    
    def _execute_select(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Ejecuta SELECT usando las estructuras de índices."""
        table_name = plan.data['table_name']
        select_list = plan.data['select_list']
        where_clause = plan.data.get('where_clause')
        
        print(f"DEBUG _execute_select: tabla={table_name}, select={select_list}")
        
        if table_name not in self.tables:
            return {'success': False, 'error': f'Tabla {table_name} no existe'}
        
        table_info = self.tables[table_name]
        
        # ✅ VERIFICAR estructura
        if table_name not in self.structures:
            print(f"ERROR: Estructura de {table_name} no está en self.structures")
            print(f"DEBUG Estructuras disponibles: {list(self.structures.keys())}")
            return {'success': False, 'error': f'Estructura de {table_name} no cargada. Tablas disponibles: {list(self.structures.keys())}'}
        
        structure = self.structures[table_name]
        
        # ✅ VERIFICAR que no sea None
        if structure is None:
            print(f"ERROR: La estructura de {table_name} es None")
            return {'success': False, 'error': f'La estructura de {table_name} no se cargó correctamente'}
        
        print(f"DEBUG Estructura obtenida: {type(structure).__name__}")
        
        index_type = table_info['index_type']
        
        try:
            # Ejecutar WHERE usando índices
            limit_value = plan.data.get('limit')
            if where_clause:
                print(f"DEBUG Ejecutando WHERE: {where_clause}")
                results = self._execute_where_clause(structure, where_clause, index_type, limit_value)
            else:
                print(f"DEBUG Ejecutando SELECT * sobre {type(structure).__name__}")
                results = self._select_all(structure, index_type)
            
            print(f"DEBUG Resultados obtenidos: {len(results) if results else 0}")
            
            # Aplicar LIMIT si está presente
            limit_value = plan.data.get('limit')
            if limit_value and isinstance(limit_value, int) and limit_value > 0:
                print(f"DEBUG Aplicando LIMIT {limit_value}")
                results = results[:limit_value]
                print(f"DEBUG Resultados después de LIMIT: {len(results)}")
            
            # Aplicar proyección (select_list)
            if select_list != ['*'] and results:
                if isinstance(results[0], dict):
                    results = [{k: r[k] for k in select_list if k in r} for r in results]

            return {
                'success': True,
                'results': results,
                'count': len(results),
                'table_name': table_name,
                'index_type': index_type
            }
        except Exception as e:
            print(f"ERROR en _execute_select: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}


    def _select_all(self, structure, index_type):
        """Selecciona todos los registros USANDO get_all o similar."""
        print(f"DEBUG _select_all: tipo={index_type}, estructura={type(structure).__name__}")
        
        try:
            if index_type in ['SEQ', 'SEQUENTIAL']:
                if not hasattr(structure, 'get_all'):
                    print(f"ERROR: Sequential File no tiene método get_all")
                    return []  # etornar lista vacía en lugar de dict
                
                records = structure.get_all()
                print(f"DEBUG get_all() retornó: {type(records)} con {len(records) if records else 0} elementos")
                
                #  VERIFICAR que sea lista
                if not isinstance(records, list):
                    print(f"WARN: get_all() no retornó lista: {type(records)}")
                    return []
                
                if not records:
                    return []
                
                # Convertir Record a diccionarios
                results = []
                for record in records:
                    if hasattr(record, 'values'):
                        if isinstance(record.values, dict):
                            results.append(record.values)
                        elif isinstance(record.values, (list, tuple)):
                            # Crear diccionario con nombres de campos
                            field_names = [f.name for f in record.table.fields]
                            results.append(dict(zip(field_names, record.values)))
                        else:
                            results.append({'data': str(record.values)})
                    else:
                        # Si no tiene .values, usar el objeto directamente
                        results.append({'data': str(record)})
                
                print(f"DEBUG Resultados convertidos: {len(results)} registros")
                return results
                
            elif index_type == 'BTREE':
                if hasattr(structure, 'get_all_records'):
                    records = structure.get_all_records()
                    if not records:
                        return []
                    return [r if isinstance(r, dict) else {'data': str(r)} for r in records]
                else:
                    print("WARN: B+ Tree no tiene método get_all_records")
                    return []
                
            elif index_type == 'ISAM':
                if hasattr(structure, 'get_all'):
                    records = structure.get_all()
                    if not records:
                        return []
                    
                    # Convertir Record a diccionarios
                    results = []
                    for record in records:
                        if hasattr(record, 'values'):
                            if isinstance(record.values, dict):
                                results.append(record.values)
                            elif isinstance(record.values, (list, tuple)):
                                # Crear diccionario con nombres de campos
                                field_names = [f.name for f in record.table.fields]
                                results.append(dict(zip(field_names, record.values)))
                            else:
                                results.append({'data': str(record.values)})
                        else:
                            # Si no tiene .values, usar el objeto directamente
                            results.append({'data': str(record)})
                    
                    return results
                else:
                    print("WARN: ISAM no tiene método get_all")
                    return []
            
            elif index_type == 'EXTENDIBLEHASH':
                if hasattr(structure, 'get_all'):
                    records = structure.get_all()
                    if not records:
                        return []
                    return [r if isinstance(r, dict) else {'data': str(r)} for r in records]
                else:
                    print("WARN: Extendible Hash no tiene método get_all")
                    return []
            
            print(f"WARN: SELECT * no implementado para índice {index_type}")
            return []  # iempre retornar lista
            
        except Exception as e:
            print(f"ERROR en _select_all: {e}")
            import traceback
            traceback.print_exc()
            return []  #  Retornar lista vacía en caso de error
    
    def _execute_insert(self, plan: ExecutionPlan) -> Dict[str, Any]:
        table_name = plan.data['table_name']
        values = plan.data['values']
        
        if table_name not in self.tables:
            return {'success': False, 'error': f'Tabla "{table_name}" no existe'}
        
        try:
            table_info = self.tables[table_name]
            structure = self.structures[table_name]
            
            # Encontrar clave primaria
            key_field = table_info['key_field']
            key_index = next((i for i, f in enumerate(table_info['fields']) 
                            if f['name'] == key_field), 0)
            key_value = values[key_index] if key_index < len(values) else None
            
            if key_value is None:
                return {'success': False, 'error': 'No se pudo determinar clave primaria'}
            
            # Insertar en estructura REAL
            structure.insert(key_value, values)
            
            return {
                'success': True, 
                'message': f'Registro insertado en "{table_name}" con clave {key_value}',
                'values': values
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Error insertando registro: {str(e)}'}
    
    def _execute_update(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Ejecuta UPDATE."""
        table_name = plan.data['table_name']
        assignments = plan.data['assignments']
        where_clause = plan.data.get('where_clause')
        
        if table_name not in self.tables:
            return {'error': f'Tabla "{table_name}" no existe'}
        
        # TODO: Implementar UPDATE
        return {'error': 'UPDATE no implementado aún'}
    
    
    def list_tables(self) -> Dict[str, Any]:
        """Lista todas las tablas creadas."""
        return {
            'success': True,
            'tables': list(self.tables.keys()),
            'count': len(self.tables)
        }
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Obtiene información de una tabla - VERSIÓN CORREGIDA."""
        if table_name not in self.tables:
            return {'success': False, 'error': f'Tabla "{table_name}" no existe'}
        
        table_info = self.tables[table_name]
        
        # Usar la nueva estructura de metadatos
        return {
            'success': True,
            'table_name': table_name,
            'index_type': table_info['index_type'],
            'key_field': table_info['key_field'],
            'fields': len(table_info['fields'])  # Corregido: usar 'fields' en lugar de 'table.fields'
        }


    def _execute_create_table(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Ejecuta CREATE TABLE."""
        result = super()._execute_create_table(plan) if hasattr(super(), '_execute_create_table') else None
        
        table_name = plan.data['table_name']
        
        if plan.data.get('source'):  # CREATE TABLE FROM FILE
            result = self._create_table_from_file(table_name, plan)
        else:  # CREATE TABLE con esquema
            result = self._create_table_from_schema(table_name, plan)
        
        # Guardar metadatos después de crear tabla
        if result.get('success'):
            self._save_metadata()
        
        return result
    

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
        field = where_clause['field']
        
        # Obtener información de la tabla para saber cuál es el key_field
        table_name = None
        for tbl_name, tbl_info in self.tables.items():
            if self.structures.get(tbl_name) == structure:
                table_name = tbl_name
                break
        
        if not table_name:
            print("ERROR: No se encontró tabla para la estructura")
            return []
        
        table_info = self.tables[table_name]
        key_field = table_info['key_field']
        fields_info = table_info['fields']
        
        # BÚSQUEDA POR IGUALDAD
        if condition_type == 'comparison':
            value = where_clause['value']
            operator = where_clause['operator']
            
            if operator == '=':
                # ✅ VERIFICAR SI ES BÚSQUEDA POR CLAVE PRIMARIA
                if field == key_field:
                    # **USA EL ÍNDICE** para búsqueda rápida por clave
                    print(f"DEBUG Búsqueda por clave primaria: {field} = {value}")
                    
                    if index_type in ['BTREE', 'EXTENDIBLEHASH']:
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
                        
                    elif index_type == 'ISAM':
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
                        
                    elif index_type in ['SEQ', 'SEQUENTIAL']:
                        record = structure.search(value)
                        print(f"DEBUG Resultado de search: {type(record)}, valor: {record}")
                        
                        if record:
                            # Verificar el tipo de 'record'
                            if hasattr(record, 'values') and hasattr(record, 'table'):
                                field_names = [f.name for f in record.table.fields]
                                return [dict(zip(field_names, record.values))]
                            elif isinstance(record, (list, tuple)):
                                field_names = [f['name'] for f in fields_info]
                                return [dict(zip(field_names, record))]
                            elif isinstance(record, dict):
                                return [record]
                            else:
                                print(f"WARN: Tipo de record desconocido: {type(record)}")
                                return [{'data': str(record)}]
                        return []
                        
                    elif index_type == 'RTREE':
                        pos = structure.search(value)
                        if pos is not None:
                            if isinstance(pos, dict):
                                return [pos]
                            elif isinstance(pos, (list, tuple)):
                                field_names = [f['name'] for f in fields_info]
                                return [dict(zip(field_names, pos))]
                            else:
                                return [{'data': str(pos)}]
                        return []
                else:
                    # ✅ BÚSQUEDA POR CAMPO NO CLAVE - SCAN COMPLETO
                    print(f"WARN: Búsqueda por campo NO clave ({field}), requiere scan completo")
                    return self._scan_with_field_condition(structure, field, operator, value, index_type)
            else:
                # Para otros operadores (>, <, >=, <=), scan completo
                return self._scan_with_field_condition(structure, field, operator, value, index_type)
        
        # BÚSQUEDA POR RANGO
        elif condition_type == 'between':
            start, end = where_clause['start'], where_clause['end']
            
            # Solo usar range_search si es sobre la clave primaria
            if field == key_field:
                if index_type == 'BTREE':
                    positions = structure.range_search(start, end)
                    results = []
                    """
                    for key, pos in positions:
                        if isinstance(pos, dict):
                            results.append(pos)
                        elif isinstance(pos, (list, tuple)):
                            field_names = [f['name'] for f in fields_info]
                            results.append(dict(zip(field_names, pos)))
                    """  
                    results = positions;       
                    print(f"LOS RESULTADOS-PARSER: {results}")
                    return results
                    
                elif index_type == 'ISAM':
                    records = structure.range_search(start, end)
                    results = []
                    for record in records:
                        if hasattr(record, 'values'):
                            if isinstance(record.values, dict):
                                results.append(record.values)
                            elif isinstance(record.values, (list, tuple)):
                                field_names = [f['name'] for f in fields_info]
                                results.append(dict(zip(field_names, record.values)))
                            else:
                                results.append({'data': str(record.values)})
                        else:
                            results.append({'data': str(record)})
                    return results
                    
                elif index_type in ['SEQ', 'SEQUENTIAL']:
                    records = structure.rangeSearch(start, end)
                    if records:
                        results = []
                        for r in records:
                            if hasattr(r, 'values') and hasattr(r, 'table'):
                                field_names = [f.name for f in r.table.fields]
                                results.append(dict(zip(field_names, r.values)))
                            elif isinstance(r, dict):
                                results.append(r)
                            elif isinstance(r, (list, tuple)):
                                field_names = [f['name'] for f in fields_info]
                                results.append(dict(zip(field_names, r)))
                        return results
                    return []
            else:
                print(f"WARN: BETWEEN en campo NO clave ({field}), requiere scan completo")
                return self._scan_with_range_condition(structure, field, start, end, index_type)
        
        # BÚSQUEDA ESPACIAL
        elif condition_type == 'spatial':
            point = where_clause['point']
            radius_or_k = where_clause.get('radius') or where_clause.get('k', 10)
            
            if index_type == 'RTREE':
                ids = structure.spatial_search(point, radius_or_k)
                results = []
                for item in ids:
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
        
        return []
    
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
                elif operator == '!=' and record_value != value:
                    results.append(record)
        
        print(f"DEBUG Scan completado: {len(results)} registros encontrados")
        return results
    
    def _scan_with_range_condition(self, structure, field, start, end, index_type):
        """Realiza un scan completo para BETWEEN en campo NO clave."""
        print(f"DEBUG Realizando scan completo para rango: {field} BETWEEN {start} AND {end}")
        
        all_records = self._select_all(structure, index_type)
        
        results = []
        for record in all_records:
            if isinstance(record, dict) and field in record:
                record_value = record[field]
                if start <= record_value <= end:
                    results.append(record)
        
        print(f"DEBUG Scan de rango completado: {len(results)} registros encontrados")
        return results
    
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