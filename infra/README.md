# Infraestructura MLOps

## Propósito

Archivos de orquestación para desplegar todos los servicios del pipeline inteligente:  
- LLM Connector  
- ML Clásico  
- CNN  
- Frontend Gradio  
- MLflow    

Incluye configuraciones para desarrollo local (Docker Compose) y producción (Docker Swarm).

## Contenido

- `docker-compose.yml`: Configuración para entorno de desarrollo.
- `stack.yml`: Configuración para despliegue en Docker Swarm (producción).

## Cómo usar en desarrollo

### 1. Asegúrate de tener Docker y Docker Compose instalados
```bash
docker --version
docker-compose --version