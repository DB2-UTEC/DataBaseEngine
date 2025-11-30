// Usa la variable de entorno si está disponible (Docker), sino usa localhost (desarrollo local)
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001/api';

export const apiService = {
  // Obtener lista de tablas
  async getTables() {
    const response = await fetch(`${API_BASE_URL}/tables`);
    if (!response.ok) throw new Error('Error al obtener tablas');
    return response.json();
  },

  // Ejecutar consulta SQL
  async executeQuery(query, page = 1, limit = 10) {
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query, page, limit }),
    });
    if (!response.ok) throw new Error('Error al ejecutar consulta');
    return response.json();
  },

  // Buscar tablas
  async searchTables(searchTerm) {
    const response = await fetch(`${API_BASE_URL}/tables/search?q=${encodeURIComponent(searchTerm)}`);
    if (!response.ok) throw new Error('Error al buscar tablas');
    return response.json();
  },

  // Formatear consulta SQL
  async formatQuery(query) {
    const response = await fetch(`${API_BASE_URL}/format`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query }),
    });
    if (!response.ok) throw new Error('Error al formatear consulta');
    return response.json();
  },

  // Subir imagen
  async uploadImage(file) {
    const formData = new FormData();
    formData.append('image', file);
    const response = await fetch(`${API_BASE_URL}/upload-image`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) throw new Error('Error al subir imagen');
    return response.json();
  },

  // Subir carpeta con múltiples imágenes
  async uploadFolder(files) {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('images', file);
    });
    const response = await fetch(`${API_BASE_URL}/upload-folder`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) throw new Error('Error al subir carpeta');
    return response.json();
  },
};