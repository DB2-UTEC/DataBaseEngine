#!/usr/bin/env python3
"""
SQL REPL (Read-Eval-Print Loop) con manejo robusto de errores y logging.
"""

import sys
import os
import traceback
from typing import Dict, Any, List
from sql_parser import SQLParser
from sql_executor import SQLExecutor
from lark.exceptions import LarkError

class SQLError(Exception):
    """Excepción personalizada para errores SQL."""
    
    def __init__(self, message: str, position: int = None, line: int = None, column: int = None):
        super().__init__(message)
        self.message = message
        self.position = position
        self.line = line
        self.column = column
    
    def __str__(self):
        if self.line and self.column:
            return f"Error en línea {self.line}, columna {self.column}: {self.message}"
        elif self.position:
            return f"Error en posición {self.position}: {self.message}"
        else:
            return self.message

class SQLLogger:
    """Sistema de logging para el SQL REPL."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.command_count = 0
    
    def log_command(self, command: str):
        """Log de comando ejecutado."""
        self.command_count += 1
        if self.verbose:
            print(f"[CMD {self.command_count}] {command}")
    
    def log_success(self, message: str):
        """Log de operación exitosa."""
        if self.verbose:
            print(f"[OK] {message}")
        else:
            print(message)
    
    def log_error(self, error: Exception):
        """Log de error con detalles."""
        print(f"[ERROR] {error}")
        
        if self.verbose and isinstance(error, SQLError):
            if error.line and error.column:
                print(f"  Posición: línea {error.line}, columna {error.column}")
    
    def log_info(self, message: str):
        """Log de información."""
        if self.verbose:
            print(f"[INFO] {message}")

class SQLREPL:
    """REPL principal para comandos SQL."""
    
    def __init__(self, verbose: bool = False, base_dir: str = None):
        """
        Inicializa el REPL.
        
        Args:
            verbose: Modo verbose para logging
            base_dir: Directorio base del proyecto (opcional, se detecta automáticamente si no se proporciona)
        """
        self.parser = SQLParser()
        # Si no se proporciona base_dir, intentar detectarlo
        if base_dir is None:
            # Intentar detectar desde el directorio actual
            current_dir = os.path.abspath(os.getcwd())
            # Si estamos en backend/, subir un nivel
            if current_dir.endswith('backend') or current_dir.endswith('backend/'):
                base_dir = os.path.dirname(current_dir)
            else:
                base_dir = current_dir
        self.executor = SQLExecutor(base_dir=base_dir)
        self.logger = SQLLogger(verbose)
        self.verbose = verbose
    
    def execute_command(self, command: str) -> Dict[str, Any]:
        """
        Ejecuta un comando SQL completo.
        
        Args:
            command: Comando SQL a ejecutar
            
        Returns:
            Diccionario con el resultado
        """
        try:
            # Log del comando
            self.logger.log_command(command)
            
            # DEBUG: Verificar si es una consulta multimedia (empieza con M o m)
            command_stripped = command.strip()
            if command_stripped and (command_stripped[0].upper() == 'M'):
                print(f"DEBUG execute_command: Detectada consulta multimedia")
                # Es una consulta multimedia, no pasar por el parser SQL normal
                return self._execute_multimedia_query(command_stripped)
            
            # Parsear comando SQL normal
            plan = self.parser.parse(command)
            
            if plan is None:
                return {'success': True, 'message': 'Comando vacío'}
            
            # Ejecutar plan
            result = self.executor.execute(plan)
            
            # Log del resultado
            if result.get('success'):
                self.logger.log_success(result.get('message', 'Operación exitosa'))
            else:
                self.logger.log_error(SQLError(result.get('error', 'Error desconocido')))
            
            return result
            
        except LarkError as e:
            error = SQLError(f"Error de sintaxis: {e}")
            self.logger.log_error(error)
            return {'success': False, 'error': str(error)}
            
        except Exception as e:
            error = SQLError(f"Error interno: {e}")
            self.logger.log_error(error)
            
            if self.verbose:
                traceback.print_exc()
            
            return {'success': False, 'error': str(error)}
    
    def _execute_multimedia_query(self, command: str) -> Dict[str, Any]:
        """
        Ejecuta una consulta multimedia en formato especial.
        
        Formato esperado: M SELECT * <NOMBRE-DE-LA-CARPETA> WHERE image-sim <-> <ruta-del-archivo>
        
        Args:
            command: Comando multimedia completo
            
        Returns:
            Diccionario con el resultado de la búsqueda
        """
        print(f"\n{'='*60}")
        print(f"DEBUG _execute_multimedia_query: Procesando consulta multimedia")
        print(f"DEBUG Comando completo: {command}")
        print(f"{'='*60}\n")
        
        try:
            # Remover el prefijo "M" o "m"
            command = command[1:].strip()
            print(f"DEBUG Comando sin prefijo M: {command}")
            
            # Parsear el formato: SELECT * <NOMBRE-DE-LA-CARPETA> WHERE image-sim <-> <ruta-del-archivo>
            # Usar expresión regular simple para extraer componentes
            import re
            
            # Patrón: SELECT * <carpeta> WHERE image-sim <-> <ruta>
            pattern = r'SELECT\s+\*\s+(\S+)\s+WHERE\s+image-sim\s+<->\s+(.+)'
            match = re.match(pattern, command, re.IGNORECASE)
            
            if not match:
                error_msg = f"Formato de consulta multimedia inválido. Formato esperado: M SELECT * <NOMBRE-CARPETA> WHERE image-sim <-> <ruta-archivo>"
                print(f"ERROR: {error_msg}")
                print(f"DEBUG Comando recibido: {command}")
                return {'success': False, 'error': error_msg}
            
            folder_name = match.group(1).strip()
            query_image_path = match.group(2).strip()
            
            # Remover comillas si las tiene
            if query_image_path.startswith(('"', "'")) and query_image_path.endswith(('"', "'")):
                query_image_path = query_image_path[1:-1]
            
            print(f"DEBUG Carpeta: {folder_name}")
            print(f"DEBUG Ruta imagen query: {query_image_path}")
            
            # Construir ruta completa a la carpeta de imágenes
            # Asegurar que base_dir apunte a la raíz del proyecto, no a backend/
            base_dir = self.executor.base_dir
            # Convertir a ruta absoluta si es relativa
            if not os.path.isabs(base_dir):
                base_dir = os.path.abspath(base_dir)
            
            # Si base_dir termina en 'backend', subir un nivel
            if base_dir.endswith('backend') or base_dir.endswith('backend/'):
                base_dir = os.path.dirname(base_dir)
            
            # Construir ruta absoluta a la carpeta de imágenes
            image_dir = os.path.join(base_dir, 'data', 'imagenes', folder_name)
            # Asegurar que sea ruta absoluta
            image_dir = os.path.abspath(image_dir)
            
            print(f"DEBUG base_dir original: {self.executor.base_dir}")
            print(f"DEBUG base_dir corregido: {base_dir}")
            print(f"DEBUG Directorio de imágenes (absoluto): {image_dir}")
            print(f"DEBUG ¿Existe?: {os.path.exists(image_dir)}")
            
            # Verificar que exista la carpeta
            if not os.path.exists(image_dir):
                # Intentar con diferentes variaciones de la ruta
                alt_paths = [
                    os.path.join('/app', 'data', 'imagenes', folder_name),  # Ruta absoluta Docker
                    os.path.join(base_dir, 'data', 'imagenes', folder_name),
                    image_dir
                ]
                
                found = False
                for alt_path in alt_paths:
                    alt_path = os.path.abspath(alt_path)
                    print(f"DEBUG Intentando ruta alternativa: {alt_path}")
                    if os.path.exists(alt_path):
                        image_dir = alt_path
                        found = True
                        print(f"DEBUG Ruta encontrada: {image_dir}")
                        break
                
                if not found:
                    error_msg = f"Carpeta de imágenes no encontrada: {image_dir}"
                    print(f"ERROR: {error_msg}")
                    print(f"DEBUG Rutas probadas:")
                    for alt_path in alt_paths:
                        print(f"  - {os.path.abspath(alt_path)} (existe: {os.path.exists(os.path.abspath(alt_path))})")
                    return {'success': False, 'error': error_msg}
            
            # Verificar que exista la imagen query
            if not os.path.exists(query_image_path):
                # Intentar con ruta relativa desde base_dir
                query_image_path_alt = os.path.join(self.executor.base_dir, query_image_path)
                if os.path.exists(query_image_path_alt):
                    query_image_path = query_image_path_alt
                    print(f"DEBUG Usando ruta alternativa: {query_image_path}")
                else:
                    error_msg = f"Imagen query no encontrada: {query_image_path}"
                    print(f"ERROR: {error_msg}")
                    return {'success': False, 'error': error_msg}
            
            print(f"DEBUG Imagen query encontrada: {query_image_path}")
            
            # Ejecutar búsqueda multimedia directamente usando las funciones de multimedia/
            result = self._execute_multimedia_search_direct(image_dir, query_image_path, limit=10)
            
            if result.get('success'):
                self.logger.log_success(f"Búsqueda multimedia completada: {result.get('count', 0)} resultados")
            else:
                self.logger.log_error(SQLError(result.get('error', 'Error en búsqueda multimedia')))
            
            return result
            
        except Exception as e:
            error_msg = f"Error procesando consulta multimedia: {e}"
            print(f"ERROR: {error_msg}")
            if self.verbose:
                traceback.print_exc()
            return {'success': False, 'error': error_msg}
    
    def _execute_multimedia_search_direct(self, image_dir: str, query_image_path: str, limit: int = 10) -> Dict[str, Any]:
        """
        Ejecuta búsqueda multimedia directamente usando las funciones de multimedia/.
        
        Args:
            image_dir: Directorio donde están las imágenes de la base de datos
            query_image_path: Ruta a la imagen query
            limit: Número máximo de resultados
            
        Returns:
            Diccionario con los resultados de la búsqueda
        """
        print(f"\n{'='*60}")
        print(f"DEBUG _execute_multimedia_search_direct: Iniciando búsqueda multimedia")
        print(f"DEBUG image_dir: {image_dir}")
        print(f"DEBUG query_image_path: {query_image_path}")
        print(f"DEBUG limit: {limit}")
        print(f"{'='*60}\n")
        
        try:
            # Importar módulos multimedia
            import sys
            multimedia_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'multimedia')
            if multimedia_path not in sys.path:
                sys.path.insert(0, multimedia_path)
            
            from multimedia.sift_features import get_image_paths
            from multimedia.bovw import load_codebook, load_histograms
            from multimedia.sift_inverted_index import SIFTInvertedIndex
            from multimedia.sequential_search import SequentialSIFTSearch
            
            # Asegurar que base_dir apunte a la raíz del proyecto
            base_dir = self.executor.base_dir
            # Convertir a ruta absoluta si es relativa
            if not os.path.isabs(base_dir):
                base_dir = os.path.abspath(base_dir)
            
            # Si base_dir termina en 'backend', subir un nivel
            if base_dir.endswith('backend') or base_dir.endswith('backend/'):
                base_dir = os.path.dirname(base_dir)
            
            # Rutas a los archivos multimedia (absolutas)
            CODEBOOK_PATH = os.path.abspath(os.path.join(base_dir, 'multimedia', 'database', 'codebook.pkl'))
            HISTOGRAMS_PATH = os.path.abspath(os.path.join(base_dir, 'multimedia', 'database', 'histograms.npz'))
            INVERTED_INDEX_PATH = os.path.abspath(os.path.join(base_dir, 'multimedia', 'database', 'inverted_index.pkl'))
            
            print(f"DEBUG base_dir para multimedia: {base_dir}")
            
            print(f"DEBUG CODEBOOK_PATH: {CODEBOOK_PATH}")
            print(f"DEBUG HISTOGRAMS_PATH: {HISTOGRAMS_PATH}")
            print(f"DEBUG INVERTED_INDEX_PATH: {INVERTED_INDEX_PATH}")
            
            # Verificar que existan los archivos necesarios
            if not os.path.exists(CODEBOOK_PATH):
                error_msg = f"Codebook no encontrado en {CODEBOOK_PATH}. Ejecute primero la construcción del índice multimedia."
                print(f"ERROR: {error_msg}")
                return {'success': False, 'error': error_msg}
            
            if not os.path.exists(HISTOGRAMS_PATH):
                error_msg = f"Histogramas no encontrados en {HISTOGRAMS_PATH}. Ejecute primero la construcción del índice multimedia."
                print(f"ERROR: {error_msg}")
                return {'success': False, 'error': error_msg}
            
            # Cargar modelos multimedia
            print(f"DEBUG Cargando codebook desde {CODEBOOK_PATH}...")
            kmeans_model, _ = load_codebook(CODEBOOK_PATH)
            vocab_size = kmeans_model.n_clusters
            print(f"DEBUG Codebook cargado: vocab_size={vocab_size}")
            
            print(f"DEBUG Cargando histogramas desde {HISTOGRAMS_PATH}...")
            histograms_tfidf, idf_weights = load_histograms(HISTOGRAMS_PATH)
            print(f"DEBUG Histogramas cargados: shape={histograms_tfidf.shape}")
            
            # Obtener rutas de imágenes en la carpeta especificada
            print(f"DEBUG Obteniendo rutas de imágenes desde {image_dir}...")
            image_paths = get_image_paths(image_dir)
            print(f"DEBUG Encontradas {len(image_paths)} imágenes en la carpeta")
            
            if len(image_paths) == 0:
                error_msg = f"No se encontraron imágenes en {image_dir}"
                print(f"ERROR: {error_msg}")
                return {'success': False, 'error': error_msg}
            
            # Verificar que el número de imágenes coincida con el número de histogramas
            # Nota: Esto asume que los histogramas corresponden a todas las imágenes
            # En una implementación completa, se debería filtrar por carpeta
            if len(image_paths) != len(histograms_tfidf):
                print(f"WARN: Número de imágenes ({len(image_paths)}) no coincide con número de histogramas ({len(histograms_tfidf)})")
                print(f"DEBUG Usando solo las imágenes de la carpeta especificada")
                # Por ahora, usamos todas las imágenes disponibles
                # En una implementación completa, se debería mapear correctamente
            
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
            
            # Convertir resultados al formato esperado
            results = []
            for img_path, score in search_results:
                # Extraer nombre del archivo de la ruta
                img_filename = os.path.basename(img_path)
                
                # Crear diccionario con los resultados
                result_dict = {
                    'id': img_filename,
                    'title': os.path.splitext(img_filename)[0],
                    'image_path': img_path,
                    'score': float(score),
                    'similarity': float(score)
                }
                
                results.append(result_dict)
                print(f"DEBUG Resultado: {img_filename} -> score={score:.4f}")
            
            print(f"DEBUG Total de resultados formateados: {len(results)}")
            print(f"{'='*60}\n")
            
            return {
                'success': True,
                'results': results,
                'count': len(results),
                'message': f'Búsqueda multimedia completada: {len(results)} resultados encontrados'
            }
            
        except ImportError as e:
            error_msg = f"Error importando módulos multimedia: {e}"
            print(f"ERROR: {error_msg}")
            if self.verbose:
                traceback.print_exc()
            return {'success': False, 'error': error_msg}
        except Exception as e:
            error_msg = f"Error ejecutando búsqueda multimedia: {e}"
            print(f"ERROR: {error_msg}")
            if self.verbose:
                traceback.print_exc()
            return {'success': False, 'error': error_msg}
    
    def execute_file(self, filename: str) -> List[Dict[str, Any]]:
        """
        Ejecuta comandos SQL desde un archivo.
        
        Args:
            filename: Ruta del archivo SQL
            
        Returns:
            Lista de resultados
        """
        if not os.path.exists(filename):
            error = SQLError(f"Archivo no encontrado: {filename}")
            self.logger.log_error(error)
            return [{'success': False, 'error': str(error)}]
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parsear todos los comandos
            plans = self.parser.parse_file_content(content)
            
            results = []
            for i, plan in enumerate(plans):
                self.logger.log_info(f"Ejecutando comando {i+1}/{len(plans)}")
                
                result = self.executor.execute(plan)
                results.append(result)
                
                if result.get('success'):
                    self.logger.log_success(result.get('message', f'Comando {i+1} ejecutado'))
                else:
                    self.logger.log_error(SQLError(result.get('error', f'Error en comando {i+1}')))
            
            return results
            
        except Exception as e:
            error = SQLError(f"Error procesando archivo: {e}")
            self.logger.log_error(error)
            return [{'success': False, 'error': str(error)}]
    
    def show_help(self):
        """Muestra la ayuda del sistema."""
        help_text = """
=== SQL REPL - Sistema de Base de Datos ===

Comandos SQL soportados:

1. CREATE TABLE:
   - CREATE TABLE nombre (campo1 TIPO, campo2 TIPO INDEX INDICE);
   - CREATE TABLE nombre FROM FILE "archivo.csv" USING INDEX tipo("campo");

2. SELECT:
   - SELECT * FROM tabla;
   - SELECT * FROM tabla WHERE campo = valor;
   - SELECT * FROM tabla WHERE campo BETWEEN inicio AND fin;

3. INSERT:
   - INSERT INTO tabla VALUES (valor1, valor2, ...);

4. DELETE:
   - DELETE FROM tabla WHERE campo = valor;

5. BÚSQUEDA MULTIMEDIA (por similitud de imágenes):
   - M SELECT * <NOMBRE-CARPETA> WHERE image-sim <-> "ruta/imagen.jpg"
   - Ejemplo: M SELECT * mi_carpeta WHERE image-sim <-> "D:\\imagenes\\query.jpg"
   - Las imágenes deben estar en: data/imagenes/<NOMBRE-CARPETA>/
   - Requiere que se haya construido el índice multimedia previamente

6. Comandos especiales:
   - .help - Mostrar esta ayuda
   - .tables - Listar tablas
   - .info tabla - Información de tabla
   - .verbose - Activar/desactivar modo verbose
   - .exit - Salir

Tipos de índices soportados:
- SEQ/SEQUENTIAL: Archivo secuencial
- BTree: Árbol B+
- ExtendibleHash: Hashing extensible
- ISAM: Índice secuencial de acceso múltiple
- RTree: Árbol R para datos espaciales

Ejemplos:
CREATE TABLE Restaurantes FROM FILE "datos.csv" USING INDEX BTree("id");
SELECT * FROM Restaurantes WHERE precio BETWEEN 20 AND 50;
INSERT INTO Restaurantes VALUES (100, "Nuevo", 25.50);
DELETE FROM Restaurantes WHERE id = 100;
M SELECT * mi_carpeta WHERE image-sim <-> "C:\\imagenes\\buscar.jpg";
        """
        print(help_text)
    
    def show_tables(self):
        """Muestra las tablas creadas."""
        result = self.executor.list_tables()
        if result['success']:
            if result['tables']:
                print(f"\nTablas creadas ({result['count']}):")
                for table in result['tables']:
                    print(f"  - {table}")
            else:
                print("\nNo hay tablas creadas.")
        else:
            print(f"\nError: {result['error']}")
    
    def show_table_info(self, table_name: str):
        """Muestra información de una tabla."""
        result = self.executor.get_table_info(table_name)
        if result['success']:
            print(f"\nInformación de la tabla '{table_name}':")
            print(f"  Tipo de índice: {result['index_type']}")
            print(f"  Campo clave: {result['key_field']}")
            print(f"  Número de campos: {result['fields']}")
        else:
            print(f"\nError: {result['error']}")
    
    def run_interactive(self):
        """Ejecuta el REPL en modo interactivo."""
        print("=== SQL REPL - Sistema de Base de Datos ===")
        print("Escriba comandos SQL o '.help' para ayuda")
        print("Escriba '.exit' para salir")
        print()
        
        while True:
            try:
                # Leer comando
                command = input("SQL> ").strip()
                
                if not command:
                    continue
                
                # Comandos especiales
                if command.lower() == '.exit':
                    print("¡Hasta luego!")
                    break
                elif command.lower() == '.help':
                    self.show_help()
                    continue
                elif command.lower() == '.tables':
                    self.show_tables()
                    continue
                elif command.lower().startswith('.info '):
                    table_name = command[6:].strip()
                    self.show_table_info(table_name)
                    continue
                elif command.lower() == '.verbose':
                    self.verbose = not self.verbose
                    self.logger.verbose = self.verbose
                    print(f"Modo verbose: {'activado' if self.verbose else 'desactivado'}")
                    continue
                
                # Ejecutar comando SQL
                result = self.execute_command(command)
                
                # Mostrar resultado detallado en modo verbose
                if self.verbose and result.get('success'):
                    if 'results' in result:
                        print(f"Resultados ({result.get('count', 0)} registros):")
                        for i, row in enumerate(result['results'][:5]):  # Mostrar solo los primeros 5
                            print(f"  {i+1}: {row}")
                        if result.get('count', 0) > 5:
                            print(f"  ... y {result.get('count', 0) - 5} más")
                
                print()  # Línea en blanco
                
            except KeyboardInterrupt:
                print("\n\nSaliendo...")
                break
            except EOFError:
                print("\n\nSaliendo...")
                break
            except Exception as e:
                print(f"\nError inesperado: {e}")
                if self.verbose:
                    traceback.print_exc()

def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SQL REPL con múltiples estructuras de datos')
    parser.add_argument('-v', '--verbose', action='store_true', help='Modo verbose')
    parser.add_argument('-f', '--file', help='Ejecutar archivo SQL')
    
    args = parser.parse_args()
    
    repl = SQLREPL(verbose=args.verbose)
    
    if args.file:
        # Ejecutar archivo
        results = repl.execute_file(args.file)
        
        # Mostrar resumen
        success_count = sum(1 for r in results if r.get('success'))
        total_count = len(results)
        
        print(f"\nResumen: {success_count}/{total_count} comandos ejecutados exitosamente")
        
        if success_count < total_count:
            print("Comandos con errores:")
            for i, result in enumerate(results):
                if not result.get('success'):
                    print(f"  {i+1}: {result.get('error', 'Error desconocido')}")
    else:
        # Modo interactivo
        repl.run_interactive()

if __name__ == "__main__":
    main()

