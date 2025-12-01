from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from pathlib import Path
import traceback
from werkzeug.utils import secure_filename

# Import parser
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'parser'))
from sql_parser import SQLParser, ExecutionPlan
from sql_executor import SQLExecutor

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data/imagenes/'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Extensiones de imagen permitidas (solo png, jpg, jpeg)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
CORS(app)

def allowed_file(filename):
    """Verifica si el archivo tiene una extensión de imagen permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_images():
    """Obtiene la lista de imágenes en el directorio data."""
    images = []
    for file in DATA_DIR.iterdir():
        if file.is_file() and allowed_file(file.name):
            images.append({
                'name': file.name,
                'file': str(file),
                'type': 'image'
            })
    return images

@app.route('/api/image', methods=['GET'])
def api_get_image():
    """Sirve una imagen desde una ruta especificada."""
    try:
        from flask import send_file
        import urllib.parse
        
        image_path = request.args.get('path', '')
        if not image_path:
            return jsonify({'error': 'Ruta de imagen no especificada'}), 400
        
        # Decodificar la ruta
        image_path = urllib.parse.unquote(image_path)
        
        print(f"DEBUG api_get_image: Ruta recibida: {image_path}")
        
        resolved_path = None
        base_images_path = BASE_DIR / 'data' / 'imagenes'
        
        # Estrategia 1: Si la ruta es absoluta tipo Docker (/app/...)
        if image_path.startswith('/app/'):
            # Mapear /app/ a BASE_DIR
            relative_to_app = image_path[5:]  # Remover '/app/'
            resolved_path = BASE_DIR / relative_to_app
            print(f"DEBUG Ruta Docker detectada. Mapeando a: {resolved_path}")
        
        # Estrategia 2: Si es una ruta absoluta del sistema
        elif os.path.isabs(image_path):
            resolved_path = Path(image_path)
            print(f"DEBUG Ruta absoluta del sistema: {resolved_path}")
        
        # Estrategia 3: Ruta relativa desde data/imagenes/
        else:
            resolved_path = base_images_path / image_path
            print(f"DEBUG Ruta relativa. Resolviendo a: {resolved_path}")
        
        # Verificar que la ruta exista
        if not resolved_path or not resolved_path.exists():
            print(f"ERROR Imagen no encontrada en: {resolved_path}")
            return jsonify({'error': f'Imagen no encontrada: {image_path}'}), 404
        
        # Verificar que sea un archivo
        if not resolved_path.is_file():
            print(f"ERROR La ruta no es un archivo: {resolved_path}")
            return jsonify({'error': f'La ruta no es un archivo: {image_path}'}), 404
        
        # Verificar que sea una imagen válida
        if not allowed_file(resolved_path.name):
            print(f"ERROR Tipo de archivo no permitido: {resolved_path.name}")
            return jsonify({'error': 'Tipo de archivo no permitido'}), 400
        
        # Verificar seguridad: que la ruta esté dentro de BASE_DIR
        try:
            resolved_path = resolved_path.resolve()
            base_dir_resolved = BASE_DIR.resolve()
            if not str(resolved_path).startswith(str(base_dir_resolved)):
                print(f"ERROR Intento de acceso fuera de BASE_DIR")
                return jsonify({'error': 'Acceso denegado'}), 403
        except Exception as e:
            print(f"ERROR al resolver ruta: {e}")
            return jsonify({'error': 'Error al resolver ruta'}), 500
        
        # Determinar el tipo MIME basado en la extensión
        ext = resolved_path.suffix.lower()
        mimetype = 'image/jpeg'  # Por defecto
        if ext == '.png':
            mimetype = 'image/png'
        elif ext in ['.jpg', '.jpeg']:
            mimetype = 'image/jpeg'
        
        print(f"DEBUG Sirviendo imagen: {resolved_path}")
        return send_file(str(resolved_path), mimetype=mimetype)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

parser = SQLParser()
executor = SQLExecutor(base_dir=str(BASE_DIR))


@app.route('/api/tables', methods=['GET'])
def api_tables():
    """Lista todas las tablas creadas usando el executor y las imágenes."""
    try:
        result = executor.list_tables()
        normalized = []
        
        # Agregar tablas
        if result.get('success'):
            tables = result.get('tables', [])
            for table in tables:
                normalized.append({
                    'name': table,
                    'file': str(DATA_DIR / f"{table}.csv"),
                    'columns': executor.get_table_info(table).get('fields', []),
                    'type': 'table'
                })
        
        # Agregar imágenes
        images = get_images()
        normalized.extend(images)
        
        return jsonify(normalized)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/tables/search')
def api_tables_search():
    """Busca tablas e imágenes por nombre."""
    q = request.args.get('q', '').lower()
    try:
        result = executor.list_tables()
        tables = result.get('tables', [])
        images = get_images()
        
        results = []
        
        # Buscar tablas
        if q:
            filtered_tables = [t for t in tables if q in t.lower()]
            for t in filtered_tables:
                results.append({
                    'name': t,
                    'file': str(DATA_DIR / f"{t}.csv"),
                    'type': 'table'
                })
        else:
            for t in tables:
                results.append({
                    'name': t,
                    'file': str(DATA_DIR / f"{t}.csv"),
                    'type': 'table'
                })
        
        # Buscar imágenes
        if q:
            filtered_images = [img for img in images if q in img['name'].lower()]
            results.extend(filtered_images)
        else:
            results.extend(images)
        
        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload-image', methods=['POST'])
def api_upload_image():
    """Sube una imagen al directorio data."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No se proporcionó ningún archivo'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Tipo de archivo no permitido. Solo se permiten PNG, JPG o JPEG.'}), 400
        
        # Asegurar nombre de archivo seguro
        filename = secure_filename(file.filename)
        filepath = DATA_DIR / filename
        
        # Si el archivo ya existe, agregar un número al final
        counter = 1
        original_filename = filename
        while filepath.exists():
            name, ext = original_filename.rsplit('.', 1)
            filename = f"{name}_{counter}.{ext}"
            filepath = DATA_DIR / filename
            counter += 1
        
        # Guardar el archivo
        file.save(str(filepath))
        
        return jsonify({
            'success': True,
            'message': 'Imagen subida exitosamente',
            'filename': filename,
            'filepath': str(filepath)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload-folder', methods=['POST'])
def api_upload_folder():
    """Sube múltiples imágenes de una carpeta al directorio data/imagenes/<nombre-carpeta>."""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No se proporcionaron archivos'}), 400
        
        files = request.files.getlist('images')
        
        if not files or len(files) == 0:
            return jsonify({'error': 'No se seleccionaron archivos'}), 400
        
        # Extraer el nombre de la carpeta desde la ruta del primer archivo
        # Los archivos vienen con rutas relativas como "carpeta/nombre.jpg"
        folder_name = None
        first_file_path = files[0].filename if files[0].filename else ''
        
        if '/' in first_file_path:
            folder_name = first_file_path.split('/')[0]
        elif '\\' in first_file_path:
            folder_name = first_file_path.split('\\')[0]
        else:
            # Si no hay separador, usar un nombre por defecto
            folder_name = 'uploaded_folder'
        
        # Limpiar el nombre de la carpeta
        folder_name = secure_filename(folder_name)
        if not folder_name:
            folder_name = 'uploaded_folder'
        
        # Crear la carpeta dentro de data/imagenes/
        target_folder = DATA_DIR / folder_name
        target_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"DEBUG Carpeta destino: {target_folder}")
        print(f"DEBUG Nombre de carpeta extraído: {folder_name}")
        
        uploaded_files = []
        skipped_files = []
        
        for file in files:
            if file.filename == '':
                continue
            
            # Validar formato
            if not allowed_file(file.filename):
                skipped_files.append(file.filename)
                continue
            
            # Extraer solo el nombre del archivo (sin la ruta de la carpeta)
            original_path = file.filename
            if '/' in original_path:
                filename = original_path.split('/')[-1]
            elif '\\' in original_path:
                filename = original_path.split('\\')[-1]
            else:
                filename = original_path
            
            # Asegurar nombre de archivo seguro
            filename = secure_filename(filename)
            filepath = target_folder / filename
            
            # Si el archivo ya existe, agregar un número al final
            counter = 1
            original_filename = filename
            while filepath.exists():
                name, ext = original_filename.rsplit('.', 1)
                filename = f"{name}_{counter}.{ext}"
                filepath = target_folder / filename
                counter += 1
            
            # Guardar el archivo en la carpeta creada
            file.save(str(filepath))
            uploaded_files.append(f"{folder_name}/{filename}")
            print(f"DEBUG Archivo guardado: {filepath}")
        
        return jsonify({
            'success': True,
            'message': f'{len(uploaded_files)} imagen(es) subida(s) exitosamente en la carpeta "{folder_name}"',
            'uploaded': uploaded_files,
            'folder_name': folder_name,
            'skipped': skipped_files,
            'total_uploaded': len(uploaded_files),
            'total_skipped': len(skipped_files)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/format', methods=['POST'])
def api_format():
    """Formatea query SQL (placeholder)."""
    data = request.get_json() or {}
    q = data.get('query', '')
    return jsonify({'formatted': q.strip()})


@app.route('/api/query', methods=['POST'])
def api_query():
    """
    Ejecuta query SQL usando el SQLExecutor.
    TODO pasa por las estructuras de índices.
    """
    try:
        body = request.get_json() or {}
        q = body.get('query', '')
        page = int(body.get('page', 1))
        limit = int(body.get('limit', 10))

        if not q.strip():
            return jsonify({'error': 'Query vacío'}), 400

        # DEBUG: Verificar si es una consulta multimedia (empieza con M o m)
        q_stripped = q.strip()
        if q_stripped and (q_stripped[0].upper() == 'M'):
            print(f"DEBUG api_query: Detectada consulta multimedia")
            # Es una consulta multimedia, usar el REPL para procesarla
            from parser.sql_repl import SQLREPL
            # Pasar el BASE_DIR correcto al REPL (ruta absoluta)
            repl = SQLREPL(verbose=False, base_dir=str(BASE_DIR))
            ex_res = repl._execute_multimedia_query(q_stripped)
            
            if not ex_res:
                return jsonify({'error': 'Error en consulta multimedia'}), 500
            
            # Mapear resultado al formato del frontend
            # Crear un plan dummy para compatibilidad
            from parser.sql_parser import ExecutionPlan
            dummy_plan = ExecutionPlan('SELECT', table_name='Multimedia', select_list=['*'], where_clause=None)
            return _map_executor_result_to_response(ex_res, dummy_plan, page, limit, is_multimedia=True)

        # Parsear la query SQL normal
        plan = parser.parse(q)
        
        if not plan:
            return jsonify({'error': 'No se pudo parsear la query'}), 400

        # Ejecutar usando el executor (que usa los índices)
        ex_res = executor.execute(plan)
        
        if not ex_res:
            return jsonify({'error': 'Executor no retornó resultado'}), 500

        # Mapear resultado al formato del frontend
        return _map_executor_result_to_response(ex_res, plan, page, limit)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


def _map_executor_result_to_response(ex_res, plan, page=1, limit=10, is_multimedia=False):
    """Mapea el resultado del SQLExecutor al formato esperado por el frontend."""
    
    if not ex_res.get('success'):
        return jsonify({'error': ex_res.get('error', 'Error desconocido')}), 400

    operation = plan.operation if isinstance(plan, ExecutionPlan) else plan.get('operation')
    
    # Para operaciones que no retornan datos (CREATE, INSERT, DELETE, UPDATE)
    if operation in ['CREATE_TABLE', 'INSERT', 'DELETE', 'UPDATE']:
        return jsonify({
            'data': {'columns': [], 'rows': []},
            'totalRows': 0,
            'message': ex_res.get('message', 'Operación exitosa'),
            'success': True,
            'isMultimedia': False
        })

    # Para SELECT (incluyendo multimedia)
    if operation == 'SELECT' or is_multimedia:
        rows = ex_res.get('results', [])
        total = ex_res.get('count', len(rows))
        
        # Para consultas multimedia, los resultados ya vienen en formato dict
        if is_multimedia:
            # Asegurar que los resultados estén ordenados por score descendente
            rows = sorted(rows, key=lambda x: x.get('score', x.get('similarity', 0)), reverse=True)
            cols = ['id', 'title', 'image_path', 'score', 'similarity'] if rows else []
        else:
            # Normalizar filas para consultas SQL normales
            if rows and isinstance(rows[0], (list, tuple)):
                # Si son listas, crear columnas genéricas
                cols = [f'col{i+1}' for i in range(len(rows[0]))]
                rows = [{cols[i]: v for i, v in enumerate(r)} for r in rows]
            elif rows and isinstance(rows[0], dict):
                cols = list(rows[0].keys())
            else:
                cols = []
        
        # Paginación (solo para consultas no multimedia, multimedia muestra todas)
        if not is_multimedia:
            start = (page - 1) * limit
            end = start + limit
            rows = rows[start:end]
        
        return jsonify({
            'data': {
                'columns': cols,
                'rows': rows
            },
            'totalRows': total,
            'success': True,
            'stats': ex_res.get('stats', {}),
            'isMultimedia': is_multimedia
        })

    return jsonify(ex_res)


if __name__ == '__main__':
    print("🚀 Backend iniciando en http://0.0.0.0:3001")
    app.run(host='0.0.0.0', port=3001, debug=True)

