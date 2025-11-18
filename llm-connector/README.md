# LLM Service

## Propósito

Servicio REST que actúa como conector entre la aplicación y un modelo de lenguaje local (Ollama).  
Recibe preguntas en texto y devuelve respuestas generadas por un modelo LLM (por defecto: `llama3`).  
Ideal para integración en sistemas MLOps con inteligencia conversacional.

## Variables de entorno

| Variable | 
|---------|
| `GEMINI_API_KEY` 
| `GEMINI_MODEL`

## Endpoints

### `POST /query`
Envía una pregunta y recibe una respuesta del LLM.
