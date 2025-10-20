from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from pathlib import Path
import traceback

# Import parser
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'parser'))
from sql_parser import SQLParser, ExecutionPlan
from sql_executor import SQLExecutor

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app)

parser = SQLParser()
executor = SQLExecutor(base_dir=str(BASE_DIR))


@app.route('/api/tables', methods=['GET'])
def api_tables():
    """Lista todas las tablas creadas usando el executor."""
    try:
        result = executor.list_tables()
        if result.get('success'):
            tables = result.get('tables', [])
            # Normalizar formato para el frontend
            normalized = []
            for table in tables:
                normalized.append({
                    'name': table,
                    'file': str(DATA_DIR / f"{table}.csv"),
                    'columns': executor.get_table_info(table).get('fields', [])
                })
            return jsonify(normalized)
        return jsonify([])
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/tables/search')
def api_tables_search():
    """Busca tablas por nombre."""
    q = request.args.get('q', '').lower()
    try:
        result = executor.list_tables()
        tables = result.get('tables', [])
        if q:
            tables = [t for t in tables if q in t.lower()]
        return jsonify([{'name': t, 'file': str(DATA_DIR / f"{t}.csv")} for t in tables])
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
    app.run(port=3001, debug=True)
