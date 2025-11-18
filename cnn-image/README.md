# CNN Service

## Propósito

Servicio REST que clasifica imágenes (3 clases: airplane, automobile, bird) usando una red neuronal convolucional.  
Aplica filtros de convolución (blur, edge, sharpen) antes de la clasificación.  
Devuelve la clase predicha, confianza y advertencia de limitaciones.  
Ideal para integración en sistemas MLOps con visión por computadora.

## Endpoints

### `POST /classify`
Sube una imagen y recibe la clasificación.

**Request:**
- `multipart/form-data` con campo `file` (imagen JPG/PNG)

**Response:**
```json
{
  "predicted_class": "airplane",
  "confidence": 0.92,
  "applied_filters": ["blur", "edge", "sharpen"],
  "limitations": "Este modelo solo reconoce: airplane, automobile, bird...",
  "error": null
}