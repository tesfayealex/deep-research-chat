from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: langchain_anthropic not available. Anthropic models will not work.")

from .model_registry import get_model_spec
from ..config import settings # Import the validated settings

# Cache for instantiated models to avoid redundant creation
_model_cache = {}

def create_model(model_name: str, **kwargs) -> BaseChatModel:
    """
    Factory function to create and return a LangChain chat model instance 
    based on the model name and provider specified in the registry.
    Uses cached instances if available.
    
    Args:
        model_name: The name of the model (e.g., 'gemini-flash', 'gpt-4o-mini').
        **kwargs: Additional parameters to pass to the model constructor 
                  (e.g., temperature, max_tokens), overriding defaults.
                  Note: API keys are handled automatically via environment variables 
                  expected by LangChain (OPENAI_API_KEY, GOOGLE_API_KEY).

    Returns:
        An instance of a LangChain BaseChatModel.
        
    Raises:
        ValueError: If the model name or provider is not supported.
    """
    cache_key = (model_name, tuple(sorted(kwargs.items())))
    if cache_key in _model_cache:
        print(f"Using cached model instance for: {model_name} with kwargs: {kwargs}")
        return _model_cache[cache_key]

    print(f"Creating new model instance for: {model_name} with kwargs: {kwargs}")
    spec = get_model_spec(model_name)
    if not spec:
        raise ValueError(f"Model '{model_name}' not found in registry.")

    provider = spec.get("provider")
    default_params = spec.get("default_params", {}).copy()
    
    # Override defaults with any provided kwargs
    final_params = {**default_params, **kwargs}

    model_instance: BaseChatModel

    try:
        if provider == "openai":
            # OPENAI_API_KEY is automatically picked up by ChatOpenAI from env/settings
            if not settings.OPENAI_API_KEY:
                 raise ValueError("OPENAI_API_KEY not found in settings for OpenAI model.")
            model_instance = ChatOpenAI(model=model_name, **final_params)
        
        elif provider == "google":
            # GOOGLE_API_KEY needs to be set in the environment for ChatGoogleGenerativeAI
            if not settings.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY not found in settings for Google model.")
            # Ensure the key is available in the environment where ChatGoogleGenerativeAI looks for it
            import os
            if "GOOGLE_API_KEY" not in os.environ and settings.GOOGLE_API_KEY:
                 os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
                 print("Set GOOGLE_API_KEY in environment for ChatGoogleGenerativeAI")
                 
            model_instance = ChatGoogleGenerativeAI(model=model_name, **final_params)
        
        elif provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ValueError("langchain_anthropic package is required for Anthropic models. Install with: pip install langchain-anthropic")
            
            if not settings.ANTHROPIC_API_KEY or settings.ANTHROPIC_API_KEY == "YOUR_ANTHROPIC_API_KEY_HERE":
                raise ValueError("ANTHROPIC_API_KEY not found in settings for Anthropic model.")
            
            # Ensure the key is available in the environment where ChatAnthropic looks for it
            import os
            if "ANTHROPIC_API_KEY" not in os.environ and settings.ANTHROPIC_API_KEY:
                os.environ["ANTHROPIC_API_KEY"] = settings.ANTHROPIC_API_KEY
                print("Set ANTHROPIC_API_KEY in environment for ChatAnthropic")
            
            model_instance = ChatAnthropic(model=model_name, **final_params)
        
        else:
            raise ValueError(f"Unsupported provider '{provider}' for model '{model_name}'.")
            
        # Cache the newly created model
        _model_cache[cache_key] = model_instance
        return model_instance

    except ImportError as e:
        print(f"Error importing library for {provider}: {e}. Ensure necessary packages are installed.")
        raise ValueError(f"Missing dependency for provider '{provider}'. Please install required packages.") from e
    except Exception as e:
        print(f"Error creating model '{model_name}': {e}")
        raise # Re-raise the exception after logging

# --- Helper functions to get specific models based on config --- 

def get_main_model(**kwargs) -> BaseChatModel:
    """Gets the main model instance specified in settings."""
    return create_model(settings.MAIN_MODEL_NAME, **kwargs)

def get_extraction_model(**kwargs) -> BaseChatModel:
    """Gets the extraction model instance specified in settings.
       Defaults to FAST_MODEL_NAME if defined, otherwise MAIN_MODEL_NAME.
    """
    model_name = settings.FAST_MODEL_NAME if hasattr(settings, 'FAST_MODEL_NAME') and settings.FAST_MODEL_NAME else settings.MAIN_MODEL_NAME
    return create_model(model_name, **kwargs)

def get_report_model(**kwargs) -> BaseChatModel:
    """Gets the report model instance specified in settings.
       Defaults to MAIN_MODEL_NAME.
       Add REPORT_MODEL_NAME to config if a separate model is desired.
    """
    model_name = settings.REPORT_MODEL_NAME if hasattr(settings, 'REPORT_MODEL_NAME') and settings.REPORT_MODEL_NAME else settings.MAIN_MODEL_NAME
    return create_model(model_name, **kwargs)

def get_reflection_model(**kwargs) -> BaseChatModel:
    """Gets the reflection model instance specified in settings.
       Defaults to MAIN_MODEL_NAME.
       Uses REFLECTION_MODEL_NAME from config if specified.
    """
    model_name = settings.REFLECTION_MODEL_NAME if hasattr(settings, 'REFLECTION_MODEL_NAME') and settings.REFLECTION_MODEL_NAME else settings.MAIN_MODEL_NAME
    return create_model(model_name, **kwargs)

def get_plan_extension_model(**kwargs) -> BaseChatModel:
    """Gets the plan extension model instance specified in settings.
       Defaults to MAIN_MODEL_NAME.
       Uses PLAN_EXTENSION_MODEL_NAME from config if specified.
    """
    model_name = settings.PLAN_EXTENSION_MODEL_NAME if hasattr(settings, 'PLAN_EXTENSION_MODEL_NAME') and settings.PLAN_EXTENSION_MODEL_NAME else settings.MAIN_MODEL_NAME
    return create_model(model_name, **kwargs)
