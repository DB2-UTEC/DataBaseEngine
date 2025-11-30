import React from 'react';
import { Box, Typography, List, ListItemButton, ListItemIcon, ListItemText, Divider } from '@mui/material';
import TableRowsIcon from '@mui/icons-material/TableRows';
import ImageIcon from '@mui/icons-material/Image';

export default function Sidebar({ tables = [] }) {
  // Separar tablas e imágenes
  const normalized = tables.map(t => (typeof t === 'string' ? { name: t, type: 'table' } : (t && t.name ? t : { name: String(t), type: 'table' })));
  
  const tablas = normalized.filter(item => item.type !== 'image');
  const imagenes = normalized.filter(item => item.type === 'image');

  return (
    <Box sx={{ 
      width: '100%', 
      height: '100%', 
      backgroundColor: '#2d3748',
      color: 'white',
      padding: '8px',
      overflowY: 'auto'
    }}>
      <Typography variant="h6" sx={{ padding: '16px 16px 8px' }}>
        Tablas ({tablas.length})
      </Typography>
      <List dense>
        {tablas.map(tabla => (
          <ListItemButton key={tabla.name}>
            <ListItemIcon>
              <TableRowsIcon sx={{ color: '#9f7aea' }} />
            </ListItemIcon>
            <ListItemText primary={tabla.name} />
          </ListItemButton>
        ))}
      </List>
      {tablas.length === 0 && (
        <Typography variant="caption" sx={{ padding: '16px', color: '#a0aec0' }}>
          No se encontraron tablas
        </Typography>
      )}
      
      <Divider sx={{ margin: '16px 0', backgroundColor: '#4a5568' }} />
      
      <Typography variant="h6" sx={{ padding: '16px 16px 8px' }}>
        Imágenes ({imagenes.length})
      </Typography>
      <List dense>
        {imagenes.map(imagen => (
          <ListItemButton key={imagen.name}>
            <ListItemIcon>
              <ImageIcon sx={{ color: '#48bb78' }} />
            </ListItemIcon>
            <ListItemText primary={imagen.name} />
          </ListItemButton>
        ))}
      </List>
      {imagenes.length === 0 && (
        <Typography variant="caption" sx={{ padding: '16px', color: '#a0aec0' }}>
          No se encontraron imágenes
        </Typography>
      )}
    </Box>
  );
}