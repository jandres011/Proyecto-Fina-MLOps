import os
import requests
from typing import Dict, Any
from app.schemas import QueryResponse
import google.generativeai as genai

# Configuración para Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Inicializar la API de Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
else:
    raise ValueError("GEMINI_API_KEY no está configurada en las variables de entorno")


def query_gemini(prompt: str) -> QueryResponse:
    try:
        # Generar contenido usando el modelo de Gemini
        response = model.generate_content(prompt)

        if response.text:
            gemini_response = response.text.strip()
            return QueryResponse(response=gemini_response, model=GEMINI_MODEL)
        else:
            return QueryResponse(
                response="",
                model=GEMINI_MODEL,
                error="El modelo no generó una respuesta válida.",
            )

    except Exception as e:
        return QueryResponse(
            response="", model=GEMINI_MODEL, error=f"Error interno: {str(e)}"
        )
