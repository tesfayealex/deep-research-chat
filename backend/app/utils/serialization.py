import json
import traceback
from langchain_core.messages import BaseMessage, AIMessageChunk
from langchain_core.load import dumpd

class LangChainObjectEncoder(json.JSONEncoder):
    """
    Custom JSON encoder specifically for LangChain objects,
    ensuring BaseMessages and related types are serialized predictably.
    """
    def default(self, obj):
        if isinstance(obj, AIMessageChunk):
            # Handle AIMessageChunk specifically: Extract key attributes
            # Note: getattr is used for safety if attributes are optional
            return {
                "lc_class": ["langchain_core", "messages", "AIMessageChunk"],
                "data": {
                    "content": obj.content,
                    "id": obj.id,
                    # Add other relevant chunk attributes if needed (e.g., tool_call_chunks)
                    "tool_call_chunks": getattr(obj, 'tool_call_chunks', None), 
                    "response_metadata": getattr(obj, 'response_metadata', None),
                    "usage_metadata": getattr(obj, 'usage_metadata', None),
                }
            }
        elif isinstance(obj, BaseMessage):
            # For other BaseMessages, use LangChain's dumpd for standard representation
            try:
                return dumpd(obj)
            except Exception as e:
                print(f"Warning: dumpd failed for {type(obj)}. Error: {e}. Falling back to basic dict.")
                # Fallback to .dict() if available, otherwise string representation
                if hasattr(obj, 'dict'):
                    return obj.dict()
                else:
                    return str(obj)
        
        # Let the base class default method raise the TypeError for other types initially
        try:
            return super().default(obj)
        except TypeError:
            # Broad fallback for any other non-serializable types encountered
            print(f"Warning: Cannot serialize object of type {type(obj)} using default JSONEncoder. Returning string representation.")
            return str(obj)

def safe_serialize(data: object) -> str:
    """
    Serializes data to a JSON string using the LangChainObjectEncoder,
    with robust error handling.
    """
    try:
        return json.dumps(data, cls=LangChainObjectEncoder)
    except Exception as e:
        print(f"--- ERROR: Could not serialize data. Error: {e} ---")
        traceback.print_exc()
        # Return a JSON string indicating the error
        return json.dumps({
            "error": "Data serialization failed",
            "exception_type": type(e).__name__,
            "exception_message": str(e)
        }) 