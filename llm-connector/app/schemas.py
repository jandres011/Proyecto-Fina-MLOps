from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    prompt: str


class QueryResponse(BaseModel):
    response: str
    model: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    api_url: str
