# 🧠 Pipeline Inteligente: LLM + ML Clásico + CNN + Despliegue Profesional

Proyecto final de MLOps que integra tres servicios de inteligencia artificial en un pipeline modular, escalable y desplegable.

## 🎯 Objetivo

Desarrollar una aplicación modular en Python que integre:

- **Modelo de Lenguaje (LLM)**: Conversación con Ollama
- **Modelo de Machine Learning Clásico**: Clasificación de vinos con scikit-learn
- **Red Neuronal Convolucional (CNN)**: Clasificación de imágenes con filtros
- **Frontend Gradio**: Interfaz web para interactuar con todos los servicios
- **Infraestructura**: Docker Compose + Docker Swarm
- **Observabilidad**: MLflow + Logs en JSON

## 🧩 Arquitectura

```text
project-root/
├─ infra/ # 📦 Docker Compose & Swarm
├─ llm_connector/ # 🧠 LLM con Ollama
├─ ml_model/ # 📊 ML clásico con scikit-learn
├─ cnn/ # 👁️ CNN con TensorFlow
├─ frontend_gradio/ # 🎨 Interfaz Gradio
├─ .github/workflows/ # 🔄 CI/CD
├─ .gitignore
├─ README.md
└─ Makefile
```

## 🚀 Cómo ejecutar localmente

### 1. Requisitos

- Docker
- Docker Compose
- Python 3.10+

### 2. Clonar y preparar

```bash
git clone <repo-url>
cd proyecto-mlops-inteligente
```

### 3. Entrenar modelos (una sola vez)

```bash
# Entrenar modelo ML
cd ml_model
python app/train.py

# Entrenar modelo CNN
cd ../cnn
python app/train.py
```

### 4. Levantar todos los servicios

```bash
cd infra
docker-compose up --build
```

### 5. Acceder a las interfaces

**Frontend Gradio**: <http://localhost:7860>  
**MLflow**: <http://localhost:5000>  
**Ollama**: <http://localhost:11434>

**Servicios API**:

- LLM: <http://localhost:8001>
- ML: <http://localhost:8002>
- CNN: <http://localhost:8003>

## 🧪 Pruebas

Ejecutar pruebas para cada servicio:

```bash
cd llm_connector && python -m pytest tests/
cd ml_model && python -m pytest tests/
cd cnn && python -m pytest tests/
cd frontend_gradio && python -m pytest tests/
```

## ⚙️ Despliegue en Producción (Docker Swarm)

```bash
docker swarm init
docker build -t llm_connector llm_connector/
docker build -t ml_model ml_model/
docker build -t cnn cnn/
docker build -t frontend_gradio frontend_gradio/
docker stack deploy -c infra/stack.yml mlops-stack
```

## 📊 Observabilidad

- **MLflow**: Registra métricas, parámetros y modelos de ML
- **Logs JSON**: Todos los servicios generan logs estructurados
- **Monitoreo**: Endpoints `/health` en todos los servicios

## 🧾 Registro de métricas

- **ML Clásico**: Accuracy, F1-Score, classification report
- **CNN**: Accuracy, loss, confusión matrix
- **MLflow**: Accesible en <http://localhost:5000>

## 🧼 Clean Code

- **Nomenclatura clara**: Nombres descriptivos
- **Modularidad**: Separación de lógica, entrada/salida y configuración
- **Documentación**: Pydantic, comentarios y READMEs
- **Type hints**: En todas las funciones
- **Manejo de errores**: Controlado y amigable

## 🧪 Integración Continua

- Tests automáticos para cada servicio
- Validación de estructura y estilo de código
- Build de imágenes Docker
- Prueba de docker-compose up

## 📚 Recursos

- [MLflow](https://mlflow.org/)
- [Gradio](https://gradio.app/)
- [Docker](https://www.docker.com/)

## 🏁 Entrega

- Repositorio con acceso al docente
- README con instrucciones completas
- Evidencia visual de los 4 componentes funcionando
- Todo reproducible con `docker-compose up`

---

**Autor**: Juan Mosquera, Anderson Bornachera  
**Fecha**: Noviembre 2025  
**Curso**: MLOps Avanzado
