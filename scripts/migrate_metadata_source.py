# filepath: scripts/migrate_metadata_source.py
import json, os, sys
meta_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'tables_metadata.json')
if not os.path.exists(meta_path):
    print("No existe metadata:", meta_path); sys.exit(0)
with open(meta_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
changed = False
for t, info in data.items():
    if isinstance(info, dict) and 'source_file' in info:
        if 'source' not in info:
            info['source'] = info.pop('source_file')
            changed = True
        else:
            # If both, prefer canonical 'source'
            info.pop('source_file', None)
            changed = True
if changed:
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Metadata migrada: source_file -> source")
else:
    print("No cambios necesarios")