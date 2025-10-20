import React from 'react';
import { Box, Typography, List, ListItemButton, ListItemIcon, ListItemText } from '@mui/material';
import TableRowsIcon from '@mui/icons-material/TableRows';

export default function Sidebar({ tables = [] }) {
  // tables may be array of strings or array of objects { name, file, columns }
  const normalized = tables.map(t => (typeof t === 'string' ? { name: t } : (t && t.name ? t : { name: String(t) })));

  return (
    <Box sx={{ 
      width: '100%', 
      height: '100%', 
      backgroundColor: '#2d3748',
      color: 'white',
      padding: '8px' 
    }}>
      <Typography variant="h6" sx={{ padding: '16px 16px 8px' }}>
        Tablas ({normalized.length})
      </Typography>
      <List dense>
        {normalized.map(tabla => (
          <ListItemButton key={tabla.name}>
            <ListItemIcon>
              <TableRowsIcon sx={{ color: '#9f7aea' }} />
            </ListItemIcon>
            <ListItemText primary={tabla.name} />
          </ListItemButton>
        ))}
      </List>
      {normalized.length === 0 && (
        <Typography variant="caption" sx={{ padding: '16px', color: '#a0aec0' }}>
          No se encontraron tablas
        </Typography>
      )}
    </Box>
  );
}