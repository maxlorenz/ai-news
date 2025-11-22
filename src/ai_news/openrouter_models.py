from __future__ import annotations

from datetime import datetime, timezone

import httpx
from loguru import logger
from pydantic import BaseModel


class ModelPricing(BaseModel):
    prompt: str
    completion: str
    request: str | None = None
    image: str | None = None


class ModelArchitecture(BaseModel):
    modality: str | None = None
    tokenizer: str | None = None
    instruct_type: str | None = None
    input_modalities: list[str] | None = None
    output_modalities: list[str] | None = None


class OpenRouterModel(BaseModel):
    id: str
    name: str
    pricing: ModelPricing
    context_length: int
    architecture: ModelArchitecture | None = None
    created: int | None = None


class OpenRouterModelsResponse(BaseModel):
    data: list[OpenRouterModel]


def fetch_openrouter_models(timeout: int = 30) -> list[OpenRouterModel]:
    """Fetch all models from OpenRouter API."""
    url = "https://openrouter.ai/api/v1/models"
    logger.info("Fetching models from OpenRouter API: {}", url)

    try:
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        parsed = OpenRouterModelsResponse(**data)
        logger.info("Fetched {} models from OpenRouter", len(parsed.data))
        return parsed.data
    except httpx.HTTPError as e:
        logger.error("Failed to fetch OpenRouter models: {}", e)
        raise
    except Exception as e:
        logger.error("Error parsing OpenRouter models response: {}", e)
        raise


def filter_free_text_models(models: list[OpenRouterModel]) -> list[OpenRouterModel]:
    """Filter for free models that support text input, have >=12B parameters (if specified), and are not code-specific."""
    import re

    free_text_models = []

    for model in models:
        # Check if pricing is free (prompt = "0")
        is_free = model.pricing.prompt == "0"

        # Check if model supports text input
        supports_text = False
        if model.architecture and model.architecture.input_modalities:
            supports_text = "text" in model.architecture.input_modalities
        else:
            # If no architecture info, assume text support for backward compatibility
            supports_text = True

        if not (is_free and supports_text):
            continue

        # Exclude code-specific models (models with "code" or "coder" in the name)
        if "code" in model.id.lower() or "coder" in model.id.lower():
            logger.debug("Excluding code-specific model {}", model.id)
            continue

        # Check for parameter size in model ID
        # Match patterns like: 7b, 8b, 9b, 11b (should be excluded)
        # Keep models without parameter info or with >=12b
        param_match = re.search(r"(\d+)b", model.id.lower())
        if param_match:
            param_size = int(param_match.group(1))
            if param_size < 12:
                logger.debug(
                    "Excluding model {} ({}B parameters < 12B)", model.id, param_size
                )
                continue

        free_text_models.append(model)

    logger.info(
        "Filtered to {} free text-supporting models with >=12B params, excluding code models (from {} total)",
        len(free_text_models),
        len(models),
    )
    return free_text_models


def get_model_data_for_db(
    models: list[OpenRouterModel],
) -> list[tuple[str, str, str, int, int | None, datetime]]:
    """Convert OpenRouterModel objects to database tuples."""
    timestamp = datetime.now(timezone.utc)
    return [
        (
            model.id,
            model.name,
            model.pricing.prompt,
            model.context_length,
            model.created,
            timestamp,
        )
        for model in models
    ]
