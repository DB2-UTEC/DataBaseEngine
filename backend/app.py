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
DATA_DIR = BASE_DIR / 'data'
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
    """Sube múltiples imágenes de una carpeta al directorio data."""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No se proporcionaron archivos'}), 400
        
        files = request.files.getlist('images')
        
        if not files or len(files) == 0:
            return jsonify({'error': 'No se seleccionaron archivos'}), 400
        
        uploaded_files = []
        skipped_files = []
        
        for file in files:
            if file.filename == '':
                continue
            
            # Validar formato
            if not allowed_file(file.filename):
                skipped_files.append(file.filename)
                continue
            
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
            uploaded_files.append(filename)
        
        return jsonify({
            'success': True,
            'message': f'{len(uploaded_files)} imagen(es) subida(s) exitosamente',
            'uploaded': uploaded_files,
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

        # Parsear la query
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


def _map_executor_result_to_response(ex_res, plan, page=1, limit=10):
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
            'success': True
        })

    # Para SELECT
    if operation == 'SELECT':
        rows = ex_res.get('results', [])
        total = ex_res.get('count', len(rows))
        
        # Normalizar filas
        if rows and isinstance(rows[0], (list, tuple)):
            # Si son listas, crear columnas genéricas
            cols = [f'col{i+1}' for i in range(len(rows[0]))]
            rows = [{cols[i]: v for i, v in enumerate(r)} for r in rows]
        elif rows and isinstance(rows[0], dict):
            cols = list(rows[0].keys())
        else:
            cols = []
        
        # Paginación
        start = (page - 1) * limit
        end = start + limit
        
        return jsonify({
            'data': {
                'columns': cols,
                'rows': rows[start:end]
            },
            'totalRows': total,
            'success': True,
            'stats': ex_res.get('stats', {})
        })

    return jsonify(ex_res)


if __name__ == '__main__':
    print("🚀 Backend iniciando en http://0.0.0.0:3001")
    app.run(host='0.0.0.0', port=3001, debug=True)

