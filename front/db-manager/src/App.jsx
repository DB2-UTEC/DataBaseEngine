import React, { useState, useEffect } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import MainContent from './components/MainContent';
import { apiService } from './services/api';
import './App.css';

const initialQuery = `SELECT
  u.id,
  u.nombre,
  u.email,
  COUNT(p.id) as total_pedidos,
  SUM(p.total) as monto_total
FROM usuarios u
JOIN pedidos p ON u.id = p.usuario_id
GROUP BY u.id, u.nombre, u.email
LIMIT 10;`;

export default function App() {
  const [query, setQuery] = useState(initialQuery);
  const [results, setResults] = useState({ columns: [], rows: [] });
  const [stats, setStats] = useState('');
  const [tables, setTables] = useState([]);
  const [filteredTables, setFilteredTables] = useState([]);
  const [loading, setLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalRows, setTotalRows] = useState(0);
  const [isMultimedia, setIsMultimedia] = useState(false);

  // Cargar tablas al iniciar
  useEffect(() => {
    loadTables();
  }, []);

  const loadTables = async () => {
    try {
      const tablesData = await apiService.getTables();
      setTables(tablesData);
      setFilteredTables(tablesData);
    } catch (error) {
      console.error('Error cargando tablas:', error);
      // Si falla la llamada al backend, no mostrar tablas por defecto.
      setTables([]);
      setFilteredTables([]);
    }
  };

  const handleExecuteQuery = async (page = 1) => {
    if (!query.trim()) return;
    
    setLoading(true);
    try {
      const startTime = Date.now();
      const response = await apiService.executeQuery(query, page, 10);
      const executionTime = ((Date.now() - startTime) / 1000).toFixed(3);
      
      // Detectar si es una consulta multimedia
      const isMultimediaQuery = response.isMultimedia || false;
      setIsMultimedia(isMultimediaQuery);
      
      setResults(response.data);
      setTotalRows(response.totalRows);
      setCurrentPage(page);
      
      if (isMultimediaQuery) {
        setStats(`${response.totalRows} imagen(es) encontrada(s) • Tiempo: ${executionTime}s`);
      } else {
        setStats(`${response.totalRows} filas encontradas • Tiempo: ${executionTime}s`);
      }
    } catch (error) {
      console.error('Error ejecutando consulta:', error);
      setStats('Error al ejecutar la consulta');
      setIsMultimedia(false);
    } finally {
      setLoading(false);
    }
  };

  const handleFormatQuery = () => {
    // FORMATEAR debe limpiar el editor SQL (dejarlo en blanco)
    setQuery('');
  };

  const handleClearResults = () => {
    // LIMPIAR debe limpiar solo los resultados de la consulta
    setResults({ columns: [], rows: [] });
    setStats('');
    setCurrentPage(1);
    setTotalRows(0);
    setIsMultimedia(false);
  };

  const handleSearchTables = async (searchTerm) => {
    if (!searchTerm.trim()) {
      setFilteredTables(tables);
      return;
    }
    
    try {
      const searchResults = await apiService.searchTables(searchTerm);
      setFilteredTables(searchResults);
    } catch (error) {
      console.error('Error buscando tablas:', error);
      // Fallback a búsqueda local soportando tanto strings como objetos {name}
      const filtered = tables.filter(table => {
        const name = typeof table === 'string' ? table : (table && table.name ? table.name : String(table));
        return name.toLowerCase().includes(searchTerm.toLowerCase());
      });
      setFilteredTables(filtered);
    }
  };

  const handleUploadImage = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validar formato
    const validFormats = ['png', 'jpg', 'jpeg'];
    const fileExtension = file.name.split('.').pop()?.toLowerCase();
    if (!fileExtension || !validFormats.includes(fileExtension)) {
      alert('Solo se permiten archivos PNG, JPG o JPEG');
      return;
    }

    try {
      setLoading(true);
      await apiService.uploadImage(file);
      // Recargar tablas e imágenes después de subir
      await loadTables();
      alert('Imagen subida exitosamente');
    } catch (error) {
      console.error('Error subiendo imagen:', error);
      alert('Error al subir la imagen: ' + (error.message || 'Error desconocido'));
    } finally {
      setLoading(false);
    }
  };

  const handleUploadFolder = async (event) => {
    const files = Array.from(event.target.files || []);
    if (files.length === 0) return;

    // Filtrar solo imágenes válidas (png, jpg, jpeg)
    const validFormats = ['png', 'jpg', 'jpeg'];
    const imageFiles = files.filter(file => {
      const extension = file.name.split('.').pop()?.toLowerCase();
      return extension && validFormats.includes(extension);
    });

    if (imageFiles.length === 0) {
      alert('No se encontraron imágenes válidas (PNG, JPG, JPEG) en la carpeta seleccionada');
      return;
    }

    try {
      setLoading(true);
      await apiService.uploadFolder(imageFiles);
      // Recargar tablas e imágenes después de subir
      await loadTables();
      alert(`${imageFiles.length} imagen(es) subida(s) exitosamente`);
    } catch (error) {
      console.error('Error subiendo carpeta:', error);
      alert('Error al subir la carpeta: ' + (error.message || 'Error desconocido'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Header onSearchTables={handleSearchTables} />
      <div className="content-body">
        <PanelGroup direction="horizontal">
          <Panel defaultSize={20} minSize={15} maxSize={30}>
            <Sidebar tables={filteredTables} />
          </Panel>
          <PanelResizeHandle className="resize-handle" />
          <Panel minSize={30}>
            <MainContent
              query={query}
              setQuery={setQuery}
              results={results}
              stats={stats}
              loading={loading}
              currentPage={currentPage}
              totalRows={totalRows}
              onExecute={handleExecuteQuery}
              onFormat={handleFormatQuery}
              onClear={handleClearResults}
              onUploadImage={handleUploadImage}
              onUploadFolder={handleUploadFolder}
              isMultimedia={isMultimedia}
            />
          </Panel>
        </PanelGroup>
      </div>
    </div>
  );
}