import json, shutil
meta_path = "data/tables_metadata.json"
shutil.copy(meta_path, meta_path + ".bak")
meta = json.load(open(meta_path, "r", encoding="utf-8"))

meta["TestFT"] = {
  "table_name": "TestFT",
  "fields": [
    {"name": "id", "type": "VARCHAR", "size": 0, "index": None},
    {"name": "title", "type": "VARCHAR", "size": 0, "index": None},
    {"name": "text", "type": "VARCHAR", "size": 0, "index": None}
  ],
  "index_type": "FULLTEXT",
  "key_field": "id",
  "source": "/app/data/test_ft.csv",
  "text_col": "text",
  "id_col": "id"
}

open(meta_path, "w", encoding="utf-8").write(json.dumps(meta, indent=2, ensure_ascii=False))
print("Patched", meta_path, "-> backup at", meta_path + ".bak")