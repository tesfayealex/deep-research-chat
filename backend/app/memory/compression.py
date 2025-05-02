from ..models.model_registry import get_context_length
# from ..models.model_factory import create_model # Needed for actual LLM compression

# Rough estimate: 1 token ~= 4 characters in English text
# This is highly approximate and varies by model/tokenizer.
# Consider using a proper tokenizer like tiktoken for OpenAI models later.
TOKEN_CHAR_RATIO = 4

def estimate_tokens(text: str) -> int:
    """Estimates the number of tokens in a string based on character count."""
    if not text:
        return 0
    return (len(text) + TOKEN_CHAR_RATIO - 1) // TOKEN_CHAR_RATIO

async def compress_text_to_fit_context(
    text: str, 
    model_name: str, 
    prompt_buffer: int = 1000 # Reserve tokens for the rest of the prompt
) -> str:
    """
    Checks if text likely fits within the model's context window and truncates if not.
    
    Args:
        text: The text content to potentially compress.
        model_name: The target LLM name (to get context length).
        prompt_buffer: Estimated tokens needed for the surrounding prompt.

    Returns:
        The original text if it fits, or a truncated version if it exceeds the limit.
        (Future implementation could use LLM summarization instead of truncation).
    """
    model_context_limit = get_context_length(model_name)
    if model_context_limit <= 0:
        print(f"Warning: Invalid context length ({model_context_limit}) for model '{model_name}'. Cannot check size.")
        return text # Return original if limit is unknown
        
    max_allowed_tokens = model_context_limit - prompt_buffer
    estimated_text_tokens = estimate_tokens(text)

    if estimated_text_tokens > max_allowed_tokens:
        print(f"Warning: Estimated text tokens ({estimated_text_tokens}) exceed allowed limit ({max_allowed_tokens}) for model {model_name}. Truncating.")
        # Truncate based on character ratio
        max_chars = max_allowed_tokens * TOKEN_CHAR_RATIO
        truncated_text = text[:max_chars]
        
        # TODO: Implement LLM-based summarization for better compression
        # Example (requires importing create_model and handling async):
        # compression_llm = create_model("gpt-4o-mini") # Or another fast model
        # summary_prompt = f"Summarize the following text very concisely to capture the main points:\n\n{text}"
        # try:
        #     summary_response = await compression_llm.ainvoke(summary_prompt)
        #     truncated_text = summary_response.content
        #     print("Used LLM summarization for compression.")
        # except Exception as e:
        #     print(f"LLM compression failed ({e}), falling back to truncation.")
        #     truncated_text = text[:max_chars] # Fallback to simple truncation
            
        return truncated_text
    else:
        # Text fits within limits
        return text
