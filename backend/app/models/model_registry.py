# backend/app/models/model_registry.py

# Defines specifications for supported LLMs
MODEL_SPECS = {
    "gemini-2.0-flash": {
        "provider": "google",
        "context_length": 1048576, # According to recent Google announcements for 1.5 Flash
        "default_params": {"temperature": 0}
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "context_length": 128000, # Standard context window for GPT-4o mini
        "default_params": {"temperature": 0}
    },
    # Add other models here if needed in the future
    # "gpt-4o": {
    #     "provider": "openai",
    #     "context_length": 128000,
    #     "default_params": {"temperature": 0}
    # }
}

def get_model_spec(model_name: str) -> dict:
    """Retrieves the specification for a given model name."""
    spec = MODEL_SPECS.get(model_name)
    if spec is None:
        # Fallback or error handling if model name is not found
        print(f"Warning: Model spec not found for '{model_name}'. Returning empty spec.")
        # Or raise ValueError(f"Model spec not found for '{model_name}'")
        return {}
    return spec

def get_model_provider(model_name: str) -> str | None:
     """Returns the provider ('google' or 'openai') for a model."""
     spec = get_model_spec(model_name)
     return spec.get("provider")

def get_context_length(model_name: str) -> int:
    """Returns the context length (in tokens) for a model."""
    spec = get_model_spec(model_name)
    # Return a default or fallback if not specified
    return spec.get("context_length", 4096) # Default fallback
