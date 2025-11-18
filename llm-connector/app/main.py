import logging.config
import json
from fastapi import FastAPI, HTTPException
from app.schemas import QueryRequest, QueryResponse, HealthResponse
from app.llm_client import query_gemini, GEMINI_MODEL

with open("app/logging_config.json") as f:
    config = json.load(f)
logging.config.dictConfig(config)

logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Connector Service", version="1.0.0")


@app.post("/query")
async def query_llm(request: QueryRequest):
    if not request.prompt.strip():
        logger.warning("Prompt vacío recibido")
        return QueryResponse(
            response="", model=GEMINI_MODEL, error="El prompt no puede estar vacío."
        )

    result = query_gemini(request.prompt)

    if result.error:
        logger.error(f"Error en LLM: {result.error}")
        raise HTTPException(status_code=500, detail=result.error)

    logger.info(f"LLM response generado para prompt: {request.prompt[:30]}...")
    return result


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="healthy", model=GEMINI_MODEL, api_url="Gemini API")
