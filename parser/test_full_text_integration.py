import sys
import os
import csv
import time
from pathlib import Path
import pytest

# Asegurar que el repo root esté en sys.path para importar paquetes internos
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parser.sql_parser import SQLParser
from parser.sql_executor import SQLExecutor

def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def test_fulltext_create_reload_query(tmp_path):
    """
    Integration test:
      - CREATE TABLE ... FROM FILE ... USING INDEX FULLTEXT (id)
      - Recreate executor (simular restart) que carga metadata y estructuras
      - SELECT ... WHERE text @@ '...' LIMIT N
    Verifica que el índice se crea, metadata se guarda y la query devuelve las filas esperadas.
    """
    base = tmp_path
    data_dir = base / "data"
    csv_path = data_dir / "test_ft.csv"

    rows = [
        {"id": "1", "title": "Amor eterno", "text": "amor en tiempos de guerra y paz"},
        {"id": "2", "title": "Guerra fría", "text": "tiempos de guerra sin amor"},
        {"id": "3", "title": "Vida", "text": "amor y esperanza"},
    ]
    write_csv(csv_path, rows, fieldnames=["id", "title", "text"])

    parser = SQLParser()
    # crear executor que use el tmp base (aislado)
    exec1 = SQLExecutor(base_dir=str(base))

    create_sql = f"CREATE TABLE TestFT FROM FILE '{csv_path}' USING INDEX FULLTEXT (id);"
    plan_create = parser.parse(create_sql)
    res_create = exec1.execute(plan_create)
    assert isinstance(res_create, dict) and res_create.get("success") is True

    # metadata debe existir en data/tables_metadata.json
    meta_file = data_dir / "tables_metadata.json"
    assert meta_file.exists()

    # simular reinicio creando nueva instancia (debe recargar metadata y SPIMI)
    exec2 = SQLExecutor(base_dir=str(base))

    # dar un pequeño margen por si hay I/O (indexación es síncrona en código actual)
    time.sleep(0.2)

    query_sql = "SELECT id, title FROM TestFT WHERE text @@ 'amor en tiempos de guerra' LIMIT 10;"
    plan_q = parser.parse(query_sql)
    res_q = exec2.execute(plan_q)
    assert isinstance(res_q, dict) and res_q.get("success") is True

    results = res_q.get("results", [])
    ids = {row["id"] for row in results}
    assert ids == {"1", "2", "3"}
    assert res_q.get("count") == 3

if __name__ == "__main__":
    # permite ejecutar el test directamente
    pytest.main([str(Path(__file__) )])