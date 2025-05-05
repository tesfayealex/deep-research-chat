import weaviate
import weaviate.classes as wvc # New way to import classes in v4
import asyncio
import time
from weaviate.exceptions import WeaviateQueryException, WeaviateBaseError
from weaviate.classes.init import AdditionalConfig, Timeout

from ..config import settings

_client = None
_history_class_name = settings.WEAVIATE_HISTORY_CLASS

def get_weaviate_client() -> weaviate.WeaviateClient:
    """Initializes and returns a Weaviate client instance."""
    global _client
    if _client is not None:
        return _client

    if not settings.WEAVIATE_URL or not settings.WEAVIATE_API_KEY:
        print("Weaviate connection skipped: URL or API Key not configured.")
        return None # Return None if not configured

    try:
        print(f"Connecting to Weaviate at {settings.WEAVIATE_URL}...")
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=settings.WEAVIATE_URL,
            auth_credentials=weaviate.auth.AuthApiKey(settings.WEAVIATE_API_KEY),
            additional_config=AdditionalConfig(
                                    timeout=Timeout(init=30, query=60, insert=120)  # Values in seconds
                                )
            # Optional: Add headers if needed, e.g., for OpenAI keys
            # headers={
            #     "X-OpenAI-Api-Key": settings.OPENAI_API_KEY
            # }
        )
        
        # .connect_to_wcs(
        #     cluster_url=settings.WEAVIATE_URL,
        #     auth_credentials=weaviate.auth.AuthApiKey(settings.WEAVIATE_API_KEY),
        #     # Optional: Add headers if needed, e.g., for OpenAI keys
        #     # headers={
        #     #     "X-OpenAI-Api-Key": settings.OPENAI_API_KEY
        #     # }
        # )
        client.connect()
        _client = client
        print("Weaviate client connected successfully.")
        ensure_history_schema(_client) # Ensure schema exists on connect
        return _client
    except Exception as e:
        print(f"Error connecting to Weaviate: {e}")
        # Potentially raise the error or handle it based on application needs
        _client = None # Ensure client is None on failure
        return None

def ensure_history_schema(client: weaviate.WeaviateClient):
    """Ensures the ChatHistory class exists in Weaviate."""
    if not client or not client.is_connected():
        print("Skipping schema check: Weaviate client not connected.")
        return
        
    history_class = _history_class_name
    try:
        if not client.collections.exists(history_class):
            print(f"Creating Weaviate class: {history_class}")
            client.collections.create(
                name=history_class,
                # Define properties for storing event data
                properties=[
                    wvc.config.Property(name="conversation_id", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="timestamp", data_type=wvc.config.DataType.DATE),
                    wvc.config.Property(name="event_type", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="sender", data_type=wvc.config.DataType.TEXT), # 'user' or 'agent'
                    wvc.config.Property(name="data_json", data_type=wvc.config.DataType.TEXT), # Store full event data as JSON string
                ],
                # Optional: Configure vectorizer if needed for semantic search later
                # vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(), 
                # generative_config=wvc.config.Configure.Generative.openai()
            )
            print(f"Class '{history_class}' created successfully.")
        # else:
            # print(f"Weaviate class '{history_class}' already exists.")
            
    except Exception as e:
        print(f"Error ensuring Weaviate schema for {history_class}: {e}")

async def save_event_to_weaviate(event_data: dict):
    """Saves a single chat event to Weaviate with retry logic."""
    client = get_weaviate_client()
    collection = client.collections.get(_history_class_name)
    if not client:
        # Silently ignore if Weaviate is not configured/connected
        return 
        
    max_retries = 3
    retry_delay = 1.0 # Initial delay in seconds
    
    properties_to_save = {
        "conversation_id": event_data.get("conversation_id"),
        "timestamp": event_data.get("timestamp"),
        "event_type": event_data.get("type"), # Use 'type' field from event
        "sender": event_data.get("sender"), # May not exist for all events
        "name": event_data.get("name"), # May not exist for all events
        "data_json": event_data.get("data_json") # The stringified original data
    }
    
    # Filter out None values, Weaviate might error on null required fields if schema changes
    properties_to_save = {k: v for k, v in properties_to_save.items() if v is not None}

    for attempt in range(max_retries):
        try:
            # Use client.data.insert which handles single objects
            # Batching can be configured at the client level or used explicitly if needed
            uuid = collection.data.insert(properties_to_save)
            
            # print(f"Saved event to Weaviate (ConvID: {event_data.get('conversation_id')}, Type: {event_data.get('type')}, UUID: {uuid})")
            return # Success
        except WeaviateBaseError as e: # Catch specific Weaviate errors
            print(f"Weaviate save error (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay:.2f} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2 # Exponential backoff
            else:
                print(f"Failed to save event to Weaviate after {max_retries} attempts.")
                # Log persistently or raise alert if needed
                break # Exit loop after final failure
        except Exception as e:
            # Catch unexpected errors
            print(f"Unexpected error saving to Weaviate (Attempt {attempt + 1}): {e}")
            # Decide if retry makes sense for general errors
            break # Exit loop on unexpected error for now 