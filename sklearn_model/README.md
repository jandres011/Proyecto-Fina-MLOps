# ML Classic Service

## Propósito

Servicio REST que expone un modelo de clasificación de vinos (dataset `wine`) entrenado con scikit-learn.  
Incluye pipeline de preprocesamiento (StandardScaler) y RandomForestClassifier.  
Registra métricas, parámetros y artefactos en MLflow.  
Ideal para integración en pipelines MLOps con observabilidad completa.

## Variables de entorno

| Variable | Valor por defecto | Descripción |
|---------|------------------|-------------|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | URL del servidor MLflow (usar `http://mlflow:5000` en Docker) |

## Endpoints

### `POST /predict`
Envía una lista de 13 características de vino y recibe la predicción.

**Request:**
```json
{
  "features": [13.74, 1.67, 2.25, 18.6, 103.0, 2.6, 2.8, 0.26, 1.28, 4.18, 1.06, 3.4, 980]
}