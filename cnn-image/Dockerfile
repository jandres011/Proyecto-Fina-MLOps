FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY app/ .

# Crear directorio para modelos
RUN mkdir -p /app/models

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV MODEL_PATH=/app/models/cnn_model.h5


# Exponer puerto
EXPOSE 8003

