# DataBase Engine

Este es un motor de base de datos desarrollado para el curso de Base de Datos 2, que implementa índices B+ con persistencia en memoria secundaria.
video: https://drive.google.com/file/d/19dgbM95GYSWHEaix2SkzLYBn6xnmU9xD/view?usp=sharing

## Características

- **Índices B+**: Implementación completa con inserción, búsqueda, actualización y eliminación
- **Persistencia**: Los índices y registros se guardan automáticamente en memoria secundaria
- **Gestión de archivos**: Manejo eficiente de registros con lista de espacios libres
- **Búsquedas por rango**: Soporte para consultas de rango en los índices B+
- **Sincronización**: Los índices se mantienen sincronizados con los registros

## Estructura del Proyecto

```
DataBaseEngine/
├── bplus.py                    # Implementación del árbol B+ con persistencia
├── core/
│   ├── databasemanager.py     # Gestor principal que integra B+ con archivos
│   ├── file_manager.py        # Gestor de archivos para registros
│   └── models.py              # Modelos de datos (Table, Record, Field)
├── example_persistence.py     # Ejemplo de uso del sistema
└── README.md                  # Este archivo
```

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/DB2-UTEC/DataBaseEngine.git
cd DataBaseEngine
```

## Uso

### Ejemplo básico

```python
from core.models import Table, Field, Record
from core.databasemanager import DatabaseManager

# Definir estructura de tabla
fields = [
    Field("id", int),
    Field("nombre", str, 20),
    Field("edad", int)
]
table = Table("personas", fields, "id")

# Crear gestor de base de datos
db = DatabaseManager(table, "personas.dat", order=3)

# Agregar registros
record = Record(table, [1, "Juan Pérez", 25])
db.add_record(record)

# Buscar registro
found = db.get_record(1)
print(found.values)  # [1, "Juan Pérez", 25]

# Búsqueda por rango
range_results = db.range_search(1, 10)

# Eliminar registro
db.remove_record(1)
```

### Ejecutar ejemplo completo

```bash
python example_persistence.py
```

## Archivos generados

El sistema crea los siguientes archivos:

- `nombre.dat`: Archivo de datos con los registros
- `nombre.header`: Archivo de cabecera con metadatos
- `nombre.idx`: Archivo de índice B+ serializado

## API Principal

### DatabaseManager

- `add_record(record)`: Agrega un nuevo registro
- `get_record(key)`: Busca un registro por clave
- `update_record(key, new_values)`: Actualiza un registro existente
- `remove_record(key)`: Elimina un registro
- `range_search(start_key, end_key)`: Búsqueda por rango
- `get_all()`: Obtiene todos los registros
- `save_all()`: Fuerza el guardado de datos
- `get_index_info()`: Información sobre el estado del índice

### BPlusTree

- `insert(key, pos)`: Inserta una clave con su posición
- `search(key)`: Busca una clave
- `update(key, pos)`: Actualiza la posición de una clave
- `delete(key)`: Elimina una clave
- `range_search(start, end)`: Búsqueda por rango
- `load_from_file()`: Carga el árbol desde archivo
- `save_to_file()`: Guarda el árbol en archivo

## Persistencia

- **Automática**: Los cambios se guardan automáticamente después de cada operación
- **Recuperación**: Al inicializar, el sistema intenta cargar el índice desde archivo
- **Reconstrucción**: Si no existe el índice, se reconstruye desde los registros existentes
- **Sincronización**: Los índices se mantienen siempre sincronizados con los datos

## Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agrega nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request
