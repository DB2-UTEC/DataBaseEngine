import React from 'react';
import SqlEditorComponent from './SqlEditor';
import ResultsTable from './ResultsTable';
import { Button, Box, Tabs, Tab, Typography, CircularProgress, Menu, MenuItem } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import ImageIcon from '@mui/icons-material/Image';
import FolderIcon from '@mui/icons-material/Folder';

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
  onClear,
  onUploadImage,
  onUploadFolder
}) {
  const [tabValue, setTabValue] = React.useState(0);
  const [anchorEl, setAnchorEl] = React.useState(null);
  const open = Boolean(anchorEl);

  const handleClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleUploadImageClick = () => {
    handleClose();
    // Trigger file input for single image
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/png,image/jpeg,image/jpg';
    input.onchange = (e) => {
      const file = e.target.files?.[0];
      if (file && onUploadImage) {
        onUploadImage({ target: { files: [file] } });
      }
    };
    input.click();
  };

  const handleUploadFolderClick = () => {
    handleClose();
    // Trigger directory input for folder
    const input = document.createElement('input');
    input.type = 'file';
    input.webkitdirectory = true;
    input.directory = true;
    input.multiple = true;
    input.accept = 'image/png,image/jpeg,image/jpg';
    input.onchange = (e) => {
      const files = Array.from(e.target.files || []);
      if (files.length > 0 && onUploadFolder) {
        onUploadFolder({ target: { files } });
      }
    };
    input.click();
  };

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
            <Button 
              variant="outlined" 
              startIcon={<CloudUploadIcon />}
              disabled={loading}
              onClick={handleClick}
            >
              Subir Imagen
            </Button>
            <Menu
              anchorEl={anchorEl}
              open={open}
              onClose={handleClose}
            >
              <MenuItem onClick={handleUploadImageClick}>
                <ImageIcon sx={{ mr: 1 }} />
                Subir Imagen
              </MenuItem>
              <MenuItem onClick={handleUploadFolderClick}>
                <FolderIcon sx={{ mr: 1 }} />
                Subir Carpeta
              </MenuItem>
            </Menu>
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