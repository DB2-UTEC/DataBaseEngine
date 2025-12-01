import React from 'react';
import { Box, Typography, Card, CardMedia, CardContent, Grid, Chip, CircularProgress } from '@mui/material';
import { styled } from '@mui/material/styles';

const StyledCard = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  transition: 'transform 0.2s, box-shadow 0.2s',
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: theme.shadows[8],
  },
}));

const ImageContainer = styled(Box)({
  position: 'relative',
  width: '100%',
  paddingTop: '75%', // Aspect ratio 4:3
  overflow: 'hidden',
  backgroundColor: '#f5f5f5',
});

const StyledCardMedia = styled(CardMedia)({
  position: 'absolute',
  top: 0,
  left: 0,
  width: '100%',
  height: '100%',
  objectFit: 'cover',
});

const ScoreChip = styled(Chip)(({ theme, score }) => {
  // Color basado en el score: verde para scores altos, amarillo para medios, rojo para bajos
  let color = '#4caf50'; // Verde por defecto
  if (score < 0.5) {
    color = '#f44336'; // Rojo
  } else if (score < 0.7) {
    color = '#ff9800'; // Naranja
  } else if (score < 0.85) {
    color = '#ffc107'; // Amarillo
  }
  
  return {
    backgroundColor: color,
    color: 'white',
    fontWeight: 'bold',
    position: 'absolute',
    top: 8,
    right: 8,
    zIndex: 1,
  };
});

export default function ImageResults({ results, stats, loading = false }) {
  // Extraer imágenes de los resultados (puede venir en results.rows o results.data.rows)
  let images = [];
  if (results) {
    if (results.rows && Array.isArray(results.rows)) {
      images = results.rows;
    } else if (results.data && results.data.rows && Array.isArray(results.data.rows)) {
      images = results.data.rows;
    }
  }
  
  // Si no hay resultados, mostrar mensaje
  if (!images || images.length === 0) {
    return (
      <Box sx={{ padding: 2, textAlign: 'center' }}>
        <Typography variant="body2" color="text.secondary">
          {loading ? 'Buscando imágenes similares...' : 'No hay resultados. Ejecuta una consulta multimedia.'}
        </Typography>
        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
            <CircularProgress />
          </Box>
        )}
      </Box>
    );
  }
  
  // Ordenar por score descendente (mayor a menor) si no están ordenadas
  const sortedImages = [...images].sort((a, b) => {
    const scoreA = a.score || a.similarity || 0;
    const scoreB = b.score || b.similarity || 0;
    return scoreB - scoreA;
  });

  // Función para obtener la URL de la imagen
  const getImageUrl = (imagePath) => {
    if (!imagePath) return '';
    
    // Si es una ruta absoluta de Windows, convertirla a URL del servidor
    if (imagePath.startsWith('http://') || imagePath.startsWith('https://')) {
      return imagePath;
    }
    
    // Si es una ruta de archivo, usar el endpoint del backend para servir la imagen
    // El backend debe tener un endpoint para servir imágenes
    const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001/api';
    const encodedPath = encodeURIComponent(imagePath);
    return `${API_BASE_URL}/image?path=${encodedPath}`;
  };

  return (
    <Box sx={{ flexGrow: 1, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ color: 'text.primary' }}>
          Resultados de búsqueda por similitud
        </Typography>
        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
          {stats || `${sortedImages.length} imagen(es) encontrada(s)`}
        </Typography>
      </Box>
      
      <Box sx={{ 
        flexGrow: 1, 
        overflowY: 'auto', 
        overflowX: 'hidden',
        padding: 1,
        '&::-webkit-scrollbar': {
          width: '8px',
        },
        '&::-webkit-scrollbar-track': {
          backgroundColor: '#f1f1f1',
        },
        '&::-webkit-scrollbar-thumb': {
          backgroundColor: '#888',
          borderRadius: '4px',
          '&:hover': {
            backgroundColor: '#555',
          },
        },
      }}>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '400px' }}>
            <CircularProgress />
          </Box>
        ) : (
          <Grid container spacing={2}>
            {sortedImages.map((image, index) => {
              const score = image.score || image.similarity || 0;
              const imagePath = image.image_path || image.path || '';
              const title = image.title || image.id || `Imagen ${index + 1}`;
              const imageUrl = getImageUrl(imagePath);
              
              return (
                <Grid item xs={12} sm={6} md={4} lg={3} key={index}>
                  <StyledCard>
                    <ImageContainer>
                      {imageUrl ? (
                        <>
                          <StyledCardMedia
                            component="img"
                            image={imageUrl}
                            alt={title}
                            onError={(e) => {
                              // Si falla cargar la imagen, mostrar placeholder
                              e.target.style.display = 'none';
                              e.target.nextSibling.style.display = 'flex';
                            }}
                          />
                          <Box
                            sx={{
                              position: 'absolute',
                              top: 0,
                              left: 0,
                              width: '100%',
                              height: '100%',
                              display: 'none',
                              alignItems: 'center',
                              justifyContent: 'center',
                              backgroundColor: '#f5f5f5',
                              color: '#999',
                            }}
                          >
                            <Typography variant="body2">Imagen no disponible</Typography>
                          </Box>
                        </>
                      ) : (
                        <Box
                          sx={{
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            width: '100%',
                            height: '100%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            backgroundColor: '#f5f5f5',
                            color: '#999',
                          }}
                        >
                          <Typography variant="body2">Sin imagen</Typography>
                        </Box>
                      )}
                      <ScoreChip
                        label={`${(score * 100).toFixed(1)}%`}
                        score={score}
                        size="small"
                      />
                    </ImageContainer>
                    <CardContent sx={{ flexGrow: 1, padding: 1.5 }}>
                      <Typography 
                        variant="body2" 
                        sx={{ 
                          fontWeight: 500,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                        title={title}
                      >
                        {title}
                      </Typography>
                      <Typography 
                        variant="caption" 
                        color="text.secondary"
                        sx={{
                          display: 'block',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          mt: 0.5,
                        }}
                        title={imagePath}
                      >
                        {imagePath.split(/[/\\]/).pop()}
                      </Typography>
                    </CardContent>
                  </StyledCard>
                </Grid>
              );
            })}
          </Grid>
        )}
      </Box>
    </Box>
  );
}

