import json, csv, os
meta_path = "data/tables_metadata.json"
bak = meta_path + ".bak"
if not os.path.exists(meta_path):
    raise SystemExit("No tables_metadata.json encontrado")
import shutil
shutil.copy(meta_path, bak)

meta = json.load(open(meta_path, "r", encoding="utf-8"))
changed = False
for tname, info in meta.items():
    if info.get("fields") is None:
        src = info.get("source") or info.get("source_file")
        if not src:
            continue
        # normalizar ruta relativa a ./data si aplica
        if not os.path.isabs(src) and os.path.exists(os.path.join("data", src)):
            src = os.path.join("data", src)
        if os.path.exists(src):
            with open(src, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
            if header:
                info['fields'] = [{"name": h, "type": "VARCHAR", "size": 0, "index": None} for h in header]
                # if id_col exists, set as key_field if key_field is invalid
                if not info.get('key_field') or info.get('key_field') not in header:
                    info['key_field'] = header[0]
                changed = True
                print(f"Filled fields for {tname}: {header}")
        else:
            print(f"Source CSV missing for {tname}: {src}")

if changed:
    with open(meta_path, "w", encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("Updated", meta_path, "backup at", bak)
else:
    print("No changes made (no null fields found or CSVs missing).")