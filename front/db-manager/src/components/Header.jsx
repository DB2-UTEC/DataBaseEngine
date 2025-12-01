import React, { useState } from 'react';
import { Box, Button, TextField, InputAdornment } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import UserManual from './UserManual';

export default function Header({ onSearchTables }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [manualOpen, setManualOpen] = useState(false);

  const handleSearchChange = (event) => {
    const value = event.target.value;
    setSearchTerm(value);
    onSearchTables(value);
  };

  const handleOpenManual = () => {
    setManualOpen(true);
  };

  const handleCloseManual = () => {
    setManualOpen(false);
  };

  return (
    <Box sx={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '8px 16px',
      borderBottom: '1px solid #ddd',
      backgroundColor: 'white'
    }}>
      <TextField
        size="small"
        variant="outlined"
        placeholder="Buscar Tabla o Imagen..."
        value={searchTerm}
        onChange={handleSearchChange}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon />
            </InputAdornment>
          ),
        }}
      />
      <Box>
        <Button 
          variant="outlined" 
          sx={{ mr: 1 }}
          startIcon={<HelpOutlineIcon />}
          onClick={handleOpenManual}
        >
          Manual
        </Button>
        <Button variant="contained" sx={{ mr: 1 }}>Nueva Consulta</Button>
        <Button variant="outlined">Exportar</Button>
      </Box>
      <UserManual open={manualOpen} onClose={handleCloseManual} />
    </Box>
  );
}