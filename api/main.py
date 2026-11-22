"""
FastAPI backend for the NER model.

Run from the project root:
    pip install fastapi uvicorn
    uvicorn api.main:app --reload

Interactive docs: http://127.0.0.1:8000/docs

Environment variables:
    NER_MODEL_DIR  Folder with the saved model (config.json inside).
    CORS_ORIGINS   Comma-separated origins allowed to call the API,
                   for example "http://127.0.0.1:5500". Defaults to "*"
                   (any origin), which is fine for local development only.
"""

import os
import string
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.infer import (
    DEFAULT_PLACEHOLDER,
    get_model_dir,
    load_ner_pipeline,
    predict,
    redact_text,
)

MAX_TEXT_CHARS = 100_000
ALLOWED_PLACEHOLDER_FIELDS = {"type", "score"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once when the server starts, not on every request.
    # Pass None explicitly so the cache key matches the one predict() uses.
    load_ner_pipeline(None)
    yield


app = FastAPI(title="NER Extraction API", lifespan=lifespan)

# allow cross-origin-resource sharing 
origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ---------- Request / response schemas ----------

class ExtractRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_CHARS)
    min_score: float = Field(0.0, ge=0.0, le=1.0)


class Entity(BaseModel):
    entity: str
    type: str
    score: float
    start: int
    end: int


class ExtractResponse(BaseModel):
    entities: list[Entity]


class RedactRequest(ExtractRequest):
    style: Literal["label", "mask"] = "label"
    placeholder: str = Field(DEFAULT_PLACEHOLDER, max_length=100)


class RedactResponse(BaseModel):
    redacted_text: str
    entities: list[Entity]


# ---------- Helpers ----------

def validate_placeholder(placeholder: str):
    """
    Allow only {type} and {score} fields (with an optional number format).
    Blocks things like {type.__class__} that str.format would otherwise accept.
    """
    try:
        parsed = list(string.Formatter().parse(placeholder))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=f"Invalid placeholder: {error}")

    for _, field_name, format_spec, _ in parsed:
        if field_name is None:
            continue
        if field_name not in ALLOWED_PLACEHOLDER_FIELDS or "{" in (format_spec or ""):
            raise HTTPException(
                status_code=422,
                detail="Placeholder may only use {type} and {score}.",
            )

    # Catch bad number formats (for example {score:abc}) before using them
    try:
        placeholder.format(type="PER", score=0.5)
    except (ValueError, KeyError, IndexError) as error:
        raise HTTPException(status_code=422, detail=f"Invalid placeholder: {error}")


def find_entities(text: str, min_score: float):
    entities = predict(text)
    return [e for e in entities if e["score"] >= min_score]


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(get_model_dir())}


@app.post("/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest):
    """Return the entities found in the text."""
    return {"entities": find_entities(request.text, request.min_score)}


@app.post("/redact", response_model=RedactResponse)
def redact(request: RedactRequest):
    """Return the text with each entity replaced, plus the entities redacted."""
    if request.style == "label":
        validate_placeholder(request.placeholder)

    entities = find_entities(request.text, request.min_score)
    redacted = redact_text(
        request.text, entities, request.placeholder, 0.0, request.style
    )
    return {"redacted_text": redacted, "entities": entities}