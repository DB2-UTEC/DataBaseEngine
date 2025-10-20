import React from 'react';
import { Box, Typography } from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';

export default function ResultsTable({ results, stats, currentPage, totalRows, onPageChange, loading = false }) {
  const handlePageChange = (newPage) => {
    onPageChange(newPage + 1);
  };

  // Normalize results
  let cols = [];
  let rows = [];
  if (results) {
    if (results.data) {
      cols = results.data.columns || [];
      rows = results.data.rows || [];
    } else {
      cols = results.columns || [];
      rows = results.rows || [];
    }
  }

  // Si no hay datos, mostrar mensaje
  if (!cols || cols.length === 0) {
    return (
      <Box sx={{ padding: 2, textAlign: 'center' }}>
        <Typography variant="body2" color="text.secondary">
          No hay resultados. Ejecuta una consulta SQL.
        </Typography>
      </Box>
    );
  }

  // Convert columns
  const dgColumns = cols.map(c => ({ 
    field: c, 
    headerName: String(c), 
    flex: 1 
  }));

  // Ensure each row has an 'id'
  const dgRows = rows.map((r, i) => ({ 
    id: r.id ?? i, 
    ...r 
  }));

  // ✅ Crear key único basado en los datos
  const tableKey = `${cols.join('-')}-${rows.length}-${Date.now()}`;

  return (
    <Box sx={{ flexGrow: 1, height: '400px' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="h6" sx={{ marginBottom: '8px', color: 'text.primary' }}>
          Resultados de la consulta
        </Typography>
        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
          {stats || `${rows.length} filas`}
        </Typography>
      </Box>
      <DataGrid
        key={tableKey}
        rows={dgRows}
        columns={dgColumns}
        pagination
        page={currentPage - 1}
        pageSize={10}
        rowCount={totalRows}
        paginationMode="server"
        onPageChange={handlePageChange}
        loading={loading}
        density="compact"
        autoHeight={false}
      />
    </Box>
  );
}