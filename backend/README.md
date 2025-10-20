Backend for DataBaseEngine

Quick start

- Create a Python venv and install requirements

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

- Run the backend

```powershell
python app.py
```

Notes
- The backend exposes endpoints expected by the frontend at http://localhost:3001/api
- `/api/tables` - GET: lists CSV files in `../data/` with header columns
- `/api/query` - POST: accepts JSON {query, page, limit} and uses the project's parser to create an ExecutionPlan and execute basic operations (CREATE FROM FILE, SELECT, INSERT, DELETE)
- `/api/tables/search?q=...` - GET: filter table names
- `/api/format` - POST: returns a simple 'formatted' string (placeholder)

Limitations
- This is an initial minimal implementation. Index types are stored as metadata sidecars but not implemented.
- The parser is used to produce ExecutionPlan objects; complex queries may need additional mapping.
