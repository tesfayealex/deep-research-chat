import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # --- API Keys ---
    # It's recommended to load sensitive keys from environment variables
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "YOUR_TAVILY_API_KEY_HERE") 
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_HERE")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY_HERE")
    # Add other API keys as needed (e.g., Anthropic)
    # ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY_HERE")

    # --- Model Selection ---
    # Specify the main LLM and embedding model names
    # Examples: "gpt-4o", "gpt-3.5-turbo", "claude-3-opus-20240229", "text-embedding-3-small"
    MAIN_MODEL_NAME: str = os.getenv("MAIN_MODEL_NAME", "gpt-4o-mini") 
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
    # Optionally specify a separate, potentially faster/cheaper model for specific tasks
    FAST_MODEL_NAME: str = os.getenv("FAST_MODEL_NAME", MAIN_MODEL_NAME) # Default to main model
    EXTRACTION_MODEL_NAME: str = os.getenv("EXTRACTION_MODEL_NAME", "gpt-4o-mini")
    PLAN_EXTENSION_MODEL_NAME: str = os.getenv("PLAN_EXTENSION_MODEL_NAME", MAIN_MODEL_NAME)  # Model for plan extension

    # --- Agent Configuration ---
    MAX_STEPS: int = int(os.getenv("MAX_STEPS", "10")) # Maximum planning steps
    MAX_SEARCH_RESULTS_PER_STEP: int = int(os.getenv("MAX_SEARCH_RESULTS_PER_STEP", 3))
    MAX_TOOL_CALLS_PER_STEP: int = int(os.getenv("MAX_TOOL_CALLS_PER_STEP", "5"))
    TAVILY_MAX_RESULTS: int = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
    MAX_URLS_TO_SCRAPE: int = int(os.getenv("MAX_URLS_TO_SCRAPE", 3)) # Max URLs to fetch per extraction step
    MAX_TOTAL_REQUEST_TOKENS: int = int(os.getenv("MAX_TOTAL_REQUEST_TOKENS", 4000)) # Safety limit for LLM input tokens
    MAX_STEP_REPETITIONS: int = int(os.getenv("MAX_STEP_REPETITIONS", "2")) # Max internal retries/refinements within executor step
    SEARCH_METHOD: str = os.getenv("SEARCH_METHOD", "direct")  # "subquery" or "direct"

    # --- URL Extraction / Scraping ---
    SCRAPE_TIMEOUT: int = int(os.getenv("SCRAPE_TIMEOUT", 10)) # Timeout in seconds for fetching URL content
    USER_AGENT: str = os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    REPORT_MODE: str = os.getenv("REPORT_MODE", "unified")
    # --- Vector Store / Memory (Weaviate Config) ---
    WEAVIATE_URL: str = os.getenv("WEAVIATE_URL", "") # e.g., "https://your-cluster-url.weaviate.network"
    WEAVIATE_API_KEY: str = os.getenv("WEAVIATE_API_KEY", "") # Weaviate Cloud API Key
    WEAVIATE_HISTORY_CLASS: str = os.getenv("WEAVIATE_HISTORY_CLASS", "ChatHistory")
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", 3))
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

    # --- Reflection Configuration ---
    MIN_CALLS_FOR_EXTENSION: int = int(os.getenv("MIN_CALLS_FOR_EXTENSION", 3))  # Minimum calls needed to justify extension
    QUALITY_THRESHOLD: int = int(os.getenv("QUALITY_THRESHOLD", 70))  # Quality score threshold for completion
    MAX_REFLECTION_EXTENSIONS: int = int(os.getenv("MAX_REFLECTION_EXTENSIONS", 2))  # Max times we can extend after reflection
    REFLECTION_MODEL_NAME: str = os.getenv("REFLECTION_MODEL_NAME", MAIN_MODEL_NAME)  # Model for reflection (can be cheaper)
    ENABLE_MANDATORY_REFLECTION: bool = os.getenv("ENABLE_MANDATORY_REFLECTION", "true").lower() == "true"  # Toggle reflection
    MAX_TOOL_CALLS_GLOBAL: int = int(os.getenv("MAX_TOOL_CALLS_GLOBAL", 30))  # Global tool call limit

    # --- Reporting ---
    REPORT_FORMAT: str = os.getenv("REPORT_FORMAT", "markdown") # e.g., "markdown", "json"
    REPORT_MODEL_NAME: str = os.getenv("REPORT_MODEL_NAME", MAIN_MODEL_NAME)  # Model for report generation
    
    # --- Logging ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    class Config:
        # If you have a .env file, pydantic-settings will load it automatically.
        # Specify the .env file path if it's not in the default location.
        env_file = '.env' 
        env_file_encoding = 'utf-8'
        extra = 'ignore' # Ignore extra fields from environment

# Create a single instance of the settings to be imported across the application
settings = Settings()

# --- Optional: Add validation or logging after loading ---
# Example: Check if critical API keys are set
# if not settings.OPENAI_API_KEY or "YOUR_" in settings.OPENAI_API_KEY:
#     print("Warning: OPENAI_API_KEY is not set or is using a placeholder.")
# if not settings.TAVILY_API_KEY or "YOUR_" in settings.TAVILY_API_KEY:
#     print("Warning: TAVILY_API_KEY is not set or is using a placeholder.")

# print(f"Loaded settings: Main Model='{settings.MAIN_MODEL_NAME}', Max Steps='{settings.MAX_STEPS}'")

# --- Validation --- (Optional)
if not settings.WEAVIATE_URL or not settings.WEAVIATE_API_KEY:
    print("Warning: WEAVIATE_URL or WEAVIATE_API_KEY is not set in environment/config.")
    print("Chat history persistence will be disabled.") 