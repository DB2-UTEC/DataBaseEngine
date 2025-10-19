#!/usr/bin/env python3
"""
Executor SQL que toma ExecutionPlan y los ejecuta sobre las estructuras de datos.
"""

import os
import sys
import csv
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
        self.tables = {}  # Almacena metadatos de las tablas
        self.structures = {}  # Almacena las estructuras de datos activas
    
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
                if col_name.lower() in ['id', 'codigo', 'numero']:
                    data_type = 'INT'
                    size = 0
                elif col_name.lower() in ['precio', 'valor', 'costo', 'rating']:
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
        
        try:
            print(f"DEBUG Creando estructura REAL: {index_type} para {table_name}")
            
            if index_type == 'SEQ':
                # Crear objeto Table con los campos
                table_fields = []
                for field_info in fields:
                    # Convertir tipos string a clases Python
                    if field_info['type'] == 'INT':
                        data_type = int
                    elif field_info['type'] == 'FLOAT':
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
                structure = SequentialIndex(f"data/{table_name}.dat", table_obj)
                print(f"DEBUG Sequential File creado: {type(structure)}")
                
            elif index_type == 'BTREE':
                structure = BPlusTree(order=4, index_filename=f"data/{table_name}_btree.idx")
                print(f"DEBUG B+ Tree creado: {type(structure)}")
                
            elif index_type == 'ISAM':
                structure = ISAMIndex(f"data/{table_name}_isam.dat")
                print(f"DEBUG ISAM creado: {type(structure)}")
                
            elif index_type == 'EXTENDIBLEHASH':
                structure = ExtendibleHashing(bucketSize=3, index_filename=f"data/{table_name}_hash.idx")
                print(f"DEBUG Extendible Hashing creado: {type(structure)}")
                
            elif index_type == 'RTREE':
                # Para R-tree necesitamos identificar campos espaciales
                spatial_fields = []
                for field_info in fields:
                    if field_info['type'] == 'ARRAY[FLOAT]' or 'ubicacion' in field_info['name'].lower():
                        spatial_fields.append(field_info)
                
                if len(spatial_fields) < 2:
                    # Si no hay campos espaciales explícitos, usar los primeros dos campos numéricos
                    numeric_fields = [f for f in fields if f['type'] in ['FLOAT', 'INT']]
                    if len(numeric_fields) >= 2:
                        spatial_fields = numeric_fields[:2]
                        print(f"DEBUG Usando campos numéricos para R-tree: {[f['name'] for f in spatial_fields]}")
                    else:
                        raise ValueError("R-tree requiere al menos 2 campos numéricos para coordenadas")
                
                # CREAR OBJETOS Field a partir de los diccionarios
                from core.models import Field
                spatial_field_objects = []
                for field_info in spatial_fields[:2]:  # Solo necesitamos 2 campos para coordenadas
                    # Convertir tipo string a clase Python
                    if field_info['type'] == 'INT':
                        data_type = int
                    elif field_info['type'] == 'FLOAT':
                        data_type = float
                    else:
                        data_type = str
                    
                    spatial_field_objects.append(Field(
                        name=field_info['name'],
                        data_type=data_type,
                        size=field_info.get('size', 0)
                    ))
                    print(f"DEBUG Campo R-tree: {field_info['name']} -> {data_type}")
                
                structure = RTreeIndex(
                    index_filename=f"data/{table_name}_rtree.idx",
                    fields=spatial_field_objects,
                    max_children=4
                )
                print(f"DEBUG R-tree creado exitosamente: {type(structure)}")
                
            
            print(f"DEBUG Estructura creada: {structure}")
            return structure
            
        except Exception as e:
            print(f"ERROR creando estructura real: {e}")
            import traceback
            traceback.print_exc()
            return None  # ← Asegurar que nunca retorne None silenciosamente
            
    
    def _load_data_from_csv(self, table_name, file_path, fields, structure, index_type, key_field):
        """Carga datos reales desde CSV a estructuras reales"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                record_count = 0
                
                for row in reader:
                    values = []
                    for field in fields:
                        value = row.get(field['name'], '')
                        
                        if field['type'] == 'INT':
                            values.append(int(value) if value.strip() else 0)
                        elif field['type'] == 'FLOAT':
                            values.append(float(value) if value.strip() else 0.0)
                        else:  # VARCHAR
                            values.append(str(value))
                    
                    # Encontrar valor de la clave
                    key_index = next((i for i, f in enumerate(fields) if f['name'] == key_field), 0)
                    key_value = values[key_index] if key_index < len(values) else record_count
                    
                    # Insertar en estructura REAL (manejo especial para R-tree)
                    if index_type == 'RTREE':
                        # Para R-tree, crear un diccionario con los nombres de campo y valores
                        record_dict = {}
                        for i, field in enumerate(fields):
                            record_dict[field['name']] = values[i]
                        
                        # También agregar 'key' para compatibilidad
                        record_dict['key'] = key_value
                        record_dict['id'] = key_value
                        
                        structure.insert(record_dict, record_count)
                        print(f"DEBUG Insertado en R-tree: {key_value} -> {record_dict}")
                    else:
                        structure.insert(key_value, values)
                    
                    record_count += 1
                
                return record_count
                
        except Exception as e:
            print(f"Error cargando datos CSV: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def _execute_select(self, plan: ExecutionPlan) -> Dict[str, Any]:
        table_name = plan.data['table_name']
        where_clause = plan.data.get('where_clause')
        
        if table_name not in self.tables:
            return {'success': False, 'error': f'Tabla "{table_name}" no existe'}
        
        try:
            table_info = self.tables[table_name]
            structure = self.structures[table_name]
            
            # Si no hay WHERE, simular SELECT * (para testing)
            if not where_clause:
                return {
                    'success': True,
                    'results': [f"SELECT * para {table_name} (implementar lectura completa)"],
                    'count': 1,
                    'message': 'SELECT * ejecutado (modo simulación)'
                }
            
            # Búsqueda con WHERE
            if where_clause.get('type') == 'comparison':
                field = where_clause['field']
                value = where_clause['value']
                operator = where_clause['operator']
                
                if operator == '=':
                    result = structure.search(value)
                    return {
                        'success': True,
                        'results': [result] if result else [],
                        'count': 1 if result else 0,
                        'message': f'Encontrado: {result}' if result else 'No encontrado'
                    }
            
            # Búsqueda espacial para R-tree
            elif where_clause.get('type') == 'spatial':
                field = where_clause['field']
                point = where_clause['point']
                radius = where_clause['radius']
                
                if hasattr(structure, 'spatial_search'):
                    results = structure.spatial_search(point, radius)
                    return {
                        'success': True,
                        'results': results,
                        'count': len(results),
                        'message': f'Encontrados {len(results)} registros en el radio especificado'
                    }
                else:
                    return {'success': False, 'error': 'Búsqueda espacial no soportada para este índice'}
            
            return {'success': False, 'error': 'Tipo de WHERE no implementado'}
            
        except Exception as e:
            return {'success': False, 'error': f'Error ejecutando SELECT: {str(e)}'}
    
    def _execute_where_clause(self, structure, where_clause, index_type):
        """Ejecuta cláusula WHERE."""
        condition_type = where_clause['type']
        field = where_clause['field']
        
        if condition_type == 'comparison':
            value = where_clause['value']
            operator = where_clause['operator']
            
            if index_type in ['SEQ', 'SEQUENTIAL']:
                if operator == '=':
                    result = structure.get_record(value)
                    return [result.values] if result else []
                else:
                    return [f'Operador {operator} no soportado para índice secuencial']
            else:
                if operator == '=':
                    result = structure.search(value)
                    return [result] if result else []
                else:
                    return [f'Operador {operator} no soportado para índice {index_type}']
        
        elif condition_type == 'between':
            start, end = where_clause['start'], where_clause['end']
            
            if index_type in ['SEQ', 'SEQUENTIAL']:
                results = structure.range_search(start, end)
                return [r.values for r in results]
            else:
                results = structure.range_search(start, end)
                return [r[1] for r in results]
        
        elif condition_type == 'spatial':
            point = where_clause['point']
            radius = where_clause['radius']
            
            if hasattr(structure, 'search_radius'):
                return structure.search_radius(point, radius)
            else:
                return [f'Búsqueda espacial no soportada para índice {index_type}']
        
        return []
    
    def _select_all(self, structure, index_type):
        """Selecciona todos los registros."""
        if index_type in ['SEQ', 'SEQUENTIAL']:
            records = structure.get_all()
            return [r.values for r in records]
        else:
            return ['SELECT * no implementado para este tipo de índice']
    
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
        table_name = plan.data['table_name']
        
        if plan.data.get('source'):  # CREATE TABLE FROM FILE
            return self._create_table_from_file(table_name, plan)
        else:  # CREATE TABLE con esquema
            return self._create_table_from_schema(table_name, plan)

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
            
            # Inicializar estructura de datos vacía
            structure = self._create_structure(table_name, index_type, fields, key_field)
            self.structures[table_name] = structure
            
            return {
                'success': True,
                'message': f'Tabla "{table_name}" creada exitosamente con esquema',
                'fields': len(fields),
                'index_type': index_type
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Error creando tabla desde esquema: {e}'}


    
