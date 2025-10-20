import React from 'react';
import SqlEditorComponent from './SqlEditor';
import ResultsTable from './ResultsTable';
import { Button, Box, Tabs, Tab, Typography, CircularProgress } from '@mui/material';

export default function MainContent({ 
  query, 
  setQuery, 
  results, 
  stats, 
  loading,
  currentPage,
  totalRows,
  onExecute, 
  onFormat, 
  onClear 
}) {
  const [tabValue, setTabValue] = React.useState(0);

  return (
    <Box sx={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ borderBottom: 1, borderColor: 'divider', padding: '0 16px' }}>
        <Tabs value={tabValue} onChange={(e, newValue) => setTabValue(newValue)}>
          <Tab label="Consulta SQL" />
        </Tabs>
      </Box>

      {tabValue === 0 && (
        <Box sx={{ padding: '16px', display: 'flex', flexDirection: 'column', height: '100%' }}>
          <Typography variant="h6" sx={{ marginBottom: '8px', color: 'text.primary' }}>
            Editor SQL
          </Typography>
          <Box sx={{ height: '200px', border: '1px solid #ddd', borderRadius: '4px' }}>
            <SqlEditorComponent value={query} onChange={setQuery} />
          </Box>
          <Box sx={{ margin: '16px 0', display: 'flex', justifyContent: 'flex-end', gap: 1 }}>
            <Button 
              variant="contained" 
              onClick={() => onExecute(1)}
              disabled={loading || !query.trim()}
              startIcon={loading ? <CircularProgress size={16} /> : null}
            >
              {loading ? 'Ejecutando...' : 'Ejecutar'}
            </Button>
            <Button 
              variant="outlined" 
              onClick={onFormat}
              disabled={loading || !query.trim()}
            >
              Formatear
            </Button>
            <Button 
              variant="outlined" 
              onClick={onClear}
              disabled={loading}
            >
              Limpiar
            </Button>
          </Box>
          
          <ResultsTable 
            results={results} 
            stats={stats}
            loading={loading}
            currentPage={currentPage}
            totalRows={totalRows}
            onPageChange={onExecute}
          />
        </Box>
      )}
    </Box>
  );
}