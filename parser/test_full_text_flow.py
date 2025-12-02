# filepath: scripts/test_fulltext_flow.py
import os, csv, time, sys

# añadir la raíz del proyecto al path para que "parser" y "indexes" se importen
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from parser.sql_parser import SQLParser
from parser.sql_executor import SQLExecutor

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
csv_path = os.path.join(DATA_DIR, 'test_ft.csv')

# Crear CSV de prueba
rows = [
    {'id': '1', 'title': 'Amor eterno', 'text': 'amor en tiempos de guerra y paz'},
    {'id': '2', 'title': 'Guerra fría', 'text': 'tiempos de guerra sin amor'},
    {'id': '3', 'title': 'Vida', 'text': 'amor y esperanza'}
]
with open(csv_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id','title','text'])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

# Crear tabla + index
sql_create = f"CREATE TABLE TestFT FROM FILE '{csv_path}' USING INDEX FULLTEXT (text);"
parser = SQLParser()
plan = parser.parse(sql_create)
execu = SQLExecutor(base_dir=BASE)
print("Running CREATE...")
res = execu.execute(plan)
print(res)

# Esperar brevemente si el índice se construye asincrónicamente (no en este código)
time.sleep(1)

# Ejecutar consulta fulltext
sql_q = "SELECT id, title FROM TestFT WHERE text @@ 'amor en tiempos de guerra' LIMIT 10;"
plan_q = parser.parse(sql_q)
print("Running QUERY...")
res_q = execu.execute(plan_q)
print(res_q)
