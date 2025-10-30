from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from openai import OpenAI

from open_webui.env import (
    SRC_LOG_LEVELS,
)
from open_webui.utils import logger

# initialize global logger once
logger.start_logger()
log = logger.logger


router = APIRouter(prefix="/tradeberg", tags=["tradeberg"])


_cached_models: List[str] | None = None


def get_openai_client() -> OpenAI:
    # The official OpenAI client will auto-read OPENAI_API_KEY
    return OpenAI()


def list_openai_models(client: OpenAI) -> List[str]:
    global _cached_models
    if _cached_models is not None:
        return _cached_models
    try:
        models = client.models.list()
        ids = sorted([m.id for m in models.data])
        _cached_models = ids
        log.info(f"Detected OpenAI models: {', '.join(ids[:8])}{'...' if len(ids) > 8 else ''}")
        return ids
    except Exception as e:
        log.exception(e)
        _cached_models = []
        return _cached_models


def pick_model(models: List[str], *, has_image: bool, wants_image_gen: bool, deep: bool) -> str:
    # Force GPT‑5 family if available. Only fall back to 4o when GPT‑5 is missing.
    priorities: List[str] = []

    if wants_image_gen:
        # image generation
        if any(m.startswith("gpt-image-") for m in models):
            return next(m for m in models if m.startswith("gpt-image-"))
        if "gpt-image-1" in models:
            return "gpt-image-1"

    if has_image:
        # Vision-capable first: GPT‑5 vision if present, else plain GPT‑5 if it supports vision, else 4o
        for p in [
            "gpt-5-vision",
            "gpt-5",
        ]:
            for m in models:
                if m.startswith(p):
                    return m
        # Fallback to 4o family for images
        for p in ["gpt-4o", "gpt-4o-mini"]:
            for m in models:
                if m.startswith(p):
                    return m
        return models[0] if models else "gpt-4o"

    # Non-vision: prefer GPT‑5, then GPT‑5-mini, then 4o
    if deep:
        priorities = ["gpt-5", "gpt-5-mini", "gpt-4o", "gpt-4.1"]
    else:
        priorities = ["gpt-5", "gpt-5-mini", "gpt-4o", "gpt-4o-mini"]

    for p in priorities:
        for m in models:
            if m.startswith(p):
                return m

    # Final fallback
    return models[0] if models else "gpt-4o"


def infer_capabilities(body: Dict[str, Any]) -> Dict[str, bool]:
    messages: List[Dict[str, Any]] = body.get("messages", [])
    wants_image_gen = False
    has_image_input = False
    deep = False

    text = " ".join(
        [
            (m.get("content") if isinstance(m.get("content"), str) else "")
            for m in messages
            if m.get("role") == "user"
        ]
    ).lower()

    # Naive intent checks
    for kw in ["create image", "generate image", "make an image", "dalle", "picture"]:
        if kw in text:
            wants_image_gen = True
            break

    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            # OpenAI JSON content blocks; check image parts
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = str(part.get("type", ""))
                if ptype in {"image_url", "input_image", "image"}:
                    has_image_input = True
                # also detect data-url in text blocks
                if ptype == "text" and isinstance(part.get("text"), str) and part["text"].startswith("data:image"):
                    has_image_input = True

    # Heuristic for deep/long
    deep = len(text) > 1500 or any(k in text for k in ["proof", "derive", "multi-step", "chain of thought", "deep reasoning"])  # noqa: E501

    return {
        "wants_image_gen": wants_image_gen,
        "has_image_input": has_image_input,
        "deep": deep,
    }


@router.post("/chat/completions")
async def tradeberg_chat_completions(request: Request):
    body = await request.json()
    client = get_openai_client()

    models = list_openai_models(client)
    caps = infer_capabilities(body)
    model = pick_model(
        models,
        has_image=caps["has_image_input"],
        wants_image_gen=caps["wants_image_gen"],
        deep=caps["deep"],
    )

    # Enforce our chosen model
    body["model"] = model

    # Streaming passthrough
    stream = bool(body.get("stream", True))

    if stream:
        try:
            completion = client.chat.completions.create(**body, stream=True)

            async def gen():
                for chunk in completion:
                    yield chunk.to_json() + "\n"

            return StreamingResponse(gen(), media_type="application/x-ndjson")
        except Exception as e:
            log.exception(e)
            raise HTTPException(status_code=400, detail=str(e))
    else:
        try:
            completion = client.chat.completions.create(**body)
            return completion.to_dict()
        except Exception as e:
            log.exception(e)
            raise HTTPException(status_code=400, detail=str(e))


