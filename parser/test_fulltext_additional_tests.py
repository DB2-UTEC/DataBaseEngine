# filepath: scripts/test_fulltext_additional_tests.py
import sys
import os
import csv
import time
import json
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

def create_and_index(tmp_path, rows):
    base = tmp_path
    data_dir = base / "data"
    csv_path = data_dir / "test_ft_many.csv"
    write_csv(csv_path, rows, fieldnames=list(rows[0].keys()))
    parser = SQLParser()
    exec1 = SQLExecutor(base_dir=str(base))
    create_sql = f"CREATE TABLE TestFT FROM FILE '{csv_path}' USING INDEX FULLTEXT (id);"
    plan_create = parser.parse(create_sql)
    res_create = exec1.execute(plan_create)
    assert res_create.get("success") is True
    # dar tiempo breve para indexación si aplica
    time.sleep(0.1)
    return base, parser, csv_path

def test_fulltext_limit(tmp_path):
    rows = [
        {"id":"1","text":"amor guerra"},
        {"id":"2","text":"amor paz"},
        {"id":"3","text":"amor vida"},
        {"id":"4","text":"guerra sin amor"},
        {"id":"5","text":"otra cosa"},
    ]
    base, parser, _ = create_and_index(tmp_path, rows)
    execu = SQLExecutor(base_dir=str(base))
    q = "SELECT id FROM TestFT WHERE text @@ 'amor' LIMIT 2;"
    plan = parser.parse(q)
    res = execu.execute(plan)
    assert res.get("success") is True
    assert res.get("count") == 2
    assert len(res.get("results", [])) == 2

def test_fulltext_no_results(tmp_path):
    rows = [
        {"id":"1","text":"hola mundo"},
        {"id":"2","text":"otra linea"},
    ]
    base, parser, _ = create_and_index(tmp_path, rows)
    execu = SQLExecutor(base_dir=str(base))
    q = "SELECT id FROM TestFT WHERE text @@ 'inexistente' LIMIT 10;"
    plan = parser.parse(q)
    res = execu.execute(plan)
    assert res.get("success") is True
    assert res.get("count") == 0
    assert res.get("results") == []

def test_metadata_saved_and_reload(tmp_path):
    rows = [
        {"id":"1","text":"amor en la guerra"},
        {"id":"2","text":"guerra y paz"},
    ]
    base, parser, csv_path = create_and_index(tmp_path, rows)
    # metadata file
    meta_file = Path(base) / "data" / "tables_metadata.json"
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert "TestFT" in meta
    info = meta["TestFT"]
    assert info.get("source") is not None
    assert info.get("text_col") is not None or info.get("id_col") is not None

    # simular reinicio y comprobar misma respuesta
    exec2 = SQLExecutor(base_dir=str(base))
    time.sleep(0.1)
    q = "SELECT id FROM TestFT WHERE text @@ 'guerra' LIMIT 10;"
    plan = parser.parse(q)
    res = exec2.execute(plan)
    assert res.get("success") is True
    ids = {r["id"] for r in res.get("results", [])}
    assert ids == {"1","2"}

if __name__ == "__main__":
    pytest.main([str(Path(__file__))])
# para ejecutar: python3 -m pytest -q parser/test_fulltext_additional_tests.py