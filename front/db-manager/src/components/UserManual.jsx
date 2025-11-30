import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  Close as CloseIcon,
  TableChart as TableIcon,
  Image as ImageIcon,
  CloudUpload as UploadIcon,
  Search as SearchIcon,
  Code as CodeIcon,
  PlayArrow as PlayIcon,
  FormatAlignLeft as FormatIcon,
  Clear as ClearIcon,
  Folder as FolderIcon,
} from '@mui/icons-material';

export default function UserManual({ open, onClose }) {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: { maxHeight: '90vh' }
      }}
    >
      <DialogTitle sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        backgroundColor: '#1976d2',
        color: 'white'
      }}>
        <Typography variant="h5" component="div">
          Manual del Usuario
        </Typography>
        <Button
          onClick={onClose}
          sx={{ color: 'white', minWidth: 'auto' }}
        >
          <CloseIcon />
        </Button>
      </DialogTitle>
      
      <DialogContent dividers sx={{ padding: '24px' }}>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          
          {/* Introducción */}
          <Box>
            <Typography variant="h6" gutterBottom sx={{ color: '#1976d2', fontWeight: 'bold' }}>
              Bienvenido al Sistema de Gestión de Base de Datos
            </Typography>
            <Typography variant="body1" paragraph>
              Este manual te guiará a través de todas las funcionalidades disponibles en la aplicación.
              Aprenderás a ejecutar consultas SQL, gestionar tablas, subir imágenes y mucho más.
            </Typography>
          </Box>

          <Divider />

          {/* Búsqueda */}
          <Box>
            <Typography variant="h6" gutterBottom sx={{ color: '#1976d2', fontWeight: 'bold' }}>
              <SearchIcon sx={{ verticalAlign: 'middle', mr: 1 }} />
              Búsqueda de Tablas e Imágenes
            </Typography>
            <Typography variant="body2" paragraph>
              Utiliza la barra de búsqueda en el header para encontrar rápidamente tablas o imágenes por nombre.
            </Typography>
            <List dense>
              <ListItem>
                <ListItemIcon>
                  <SearchIcon color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="Búsqueda en tiempo real"
                  secondary="Escribe el nombre de la tabla o imagen y los resultados se filtrarán automáticamente"
                />
              </ListItem>
            </List>
          </Box>

          <Divider />

          {/* Editor SQL */}
          <Box>
            <Typography variant="h6" gutterBottom sx={{ color: '#1976d2', fontWeight: 'bold' }}>
              <CodeIcon sx={{ verticalAlign: 'middle', mr: 1 }} />
              Editor SQL
            </Typography>
            <Typography variant="body2" paragraph>
              El editor SQL te permite escribir y ejecutar consultas SQL para interactuar con tu base de datos.
            </Typography>
            <List dense>
              <ListItem>
                <ListItemIcon>
                  <PlayIcon color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="Ejecutar Consulta"
                  secondary="Haz clic en 'Ejecutar' para ejecutar tu consulta SQL. Los resultados se mostrarán en la tabla inferior."
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <FormatIcon color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="Formatear"
                  secondary="El botón 'Formatear' limpia el editor SQL para que puedas escribir una nueva consulta."
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <ClearIcon color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="Limpiar"
                  secondary="El botón 'Limpiar' elimina los resultados de la consulta anterior, manteniendo tu consulta SQL."
                />
              </ListItem>
            </List>
          </Box>

          <Divider />

          {/* Ejemplos de Consultas SQL */}
          <Box>
            <Typography variant="h6" gutterBottom sx={{ color: '#1976d2', fontWeight: 'bold' }}>
              Ejemplos de Consultas SQL
            </Typography>
            <Box sx={{ backgroundColor: '#f5f5f5', padding: 2, borderRadius: 1, mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'bold' }}>
                Consulta básica SELECT:
              </Typography>
              <Typography variant="body2" component="pre" sx={{ fontFamily: 'monospace', margin: 0 }}>
{`SELECT * FROM usuarios LIMIT 10;`}
              </Typography>
            </Box>
            <Box sx={{ backgroundColor: '#f5f5f5', padding: 2, borderRadius: 1, mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'bold' }}>
                Consulta con JOIN:
              </Typography>
              <Typography variant="body2" component="pre" sx={{ fontFamily: 'monospace', margin: 0 }}>
{`SELECT u.nombre, p.total
FROM usuarios u
JOIN pedidos p ON u.id = p.usuario_id;`}
              </Typography>
            </Box>
            <Box sx={{ backgroundColor: '#f5f5f5', padding: 2, borderRadius: 1 }}>
              <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'bold' }}>
                Crear una nueva tabla:
              </Typography>
              <Typography variant="body2" component="pre" sx={{ fontFamily: 'monospace', margin: 0 }}>
{`CREATE TABLE productos (
  id INT PRIMARY KEY,
  nombre VARCHAR(100),
  precio DECIMAL(10,2)
);`}
              </Typography>
            </Box>
          </Box>

          <Divider />

          {/* Sidebar */}
          <Box>
            <Typography variant="h6" gutterBottom sx={{ color: '#1976d2', fontWeight: 'bold' }}>
              <TableIcon sx={{ verticalAlign: 'middle', mr: 1 }} />
              Panel Lateral (Sidebar)
            </Typography>
            <Typography variant="body2" paragraph>
              El panel lateral muestra todas las tablas e imágenes disponibles en tu base de datos.
            </Typography>
            <List dense>
              <ListItem>
                <ListItemIcon>
                  <TableIcon color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="Tablas"
                  secondary="Lista todas las tablas creadas en la base de datos. Haz clic en una tabla para ver más detalles."
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <ImageIcon color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="Imágenes"
                  secondary="Muestra todas las imágenes que has subido a la aplicación. Las imágenes están organizadas en una sección separada."
                />
              </ListItem>
            </List>
          </Box>

          <Divider />

          {/* Subida de Imágenes */}
          <Box>
            <Typography variant="h6" gutterBottom sx={{ color: '#1976d2', fontWeight: 'bold' }}>
              <UploadIcon sx={{ verticalAlign: 'middle', mr: 1 }} />
              Subir Imágenes
            </Typography>
            <Typography variant="body2" paragraph>
              Puedes subir imágenes individuales o carpetas completas con múltiples imágenes.
            </Typography>
            <List dense>
              <ListItem>
                <ListItemIcon>
                  <ImageIcon color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="Subir Imagen Individual"
                  secondary="Selecciona 'Subir Imagen' en el menú para subir una sola imagen. Formatos permitidos: PNG, JPG, JPEG."
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <FolderIcon color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="Subir Carpeta"
                  secondary="Selecciona 'Subir Carpeta' para subir todas las imágenes válidas de una carpeta. Solo se procesarán archivos PNG, JPG y JPEG."
                />
              </ListItem>
            </List>
            <Box sx={{ mt: 2, p: 2, backgroundColor: '#fff3cd', borderRadius: 1 }}>
              <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 1 }}>
                ⚠️ Nota importante:
              </Typography>
              <Typography variant="body2">
                Solo se permiten archivos de imagen con extensiones PNG, JPG o JPEG. 
                Si intentas subir otros formatos, serán ignorados automáticamente.
              </Typography>
            </Box>
          </Box>

          <Divider />

          {/* Resultados */}
          <Box>
            <Typography variant="h6" gutterBottom sx={{ color: '#1976d2', fontWeight: 'bold' }}>
              Visualización de Resultados
            </Typography>
            <Typography variant="body2" paragraph>
              Los resultados de tus consultas SQL se muestran en una tabla interactiva con las siguientes características:
            </Typography>
            <List dense>
              <ListItem>
                <ListItemText 
                  primary="Paginación"
                  secondary="Los resultados se muestran en páginas de 10 filas. Usa los controles de paginación para navegar entre páginas."
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Estadísticas"
                  secondary="Se muestra el número total de filas encontradas y el tiempo de ejecución de la consulta."
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Columnas dinámicas"
                  secondary="Las columnas se generan automáticamente según los campos seleccionados en tu consulta SELECT."
                />
              </ListItem>
            </List>
          </Box>

          <Divider />

          {/* Consejos */}
          <Box>
            <Typography variant="h6" gutterBottom sx={{ color: '#1976d2', fontWeight: 'bold' }}>
              💡 Consejos y Mejores Prácticas
            </Typography>
            <List dense>
              <ListItem>
                <ListItemText 
                  primary="Usa LIMIT en tus consultas"
                  secondary="Para consultas que pueden retornar muchos resultados, siempre incluye LIMIT para mejorar el rendimiento."
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Nombres descriptivos"
                  secondary="Al crear tablas, usa nombres descriptivos y consistentes para facilitar la búsqueda y organización."
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Organiza tus imágenes"
                  secondary="Sube imágenes con nombres descriptivos para encontrarlas fácilmente usando la barra de búsqueda."
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Revisa los resultados"
                  secondary="Siempre revisa las estadísticas de tus consultas para entender el impacto y rendimiento de tus operaciones."
                />
              </ListItem>
            </List>
          </Box>

        </Box>
      </DialogContent>
      
      <DialogActions sx={{ padding: '16px 24px', backgroundColor: '#f5f5f5' }}>
        <Button onClick={onClose} variant="contained" color="primary">
          Cerrar
        </Button>
      </DialogActions>
    </Dialog>
  );
}

