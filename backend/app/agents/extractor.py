from typing import Dict, List
import asyncio
import re # For parsing URLs
from langchain_core.messages import AIMessage # Import AIMessage

from ..schemas import AgentState
from ..config import settings
from ..tools.web_search import perform_web_search
from ..tools.url_extractor import extract_text_from_url
from ..models.model_factory import get_extraction_model
from ..models.model_registry import get_context_length # Needed for context checking
from ..memory.compression import compress_text_to_fit_context, estimate_tokens # Import compression utils

# Simple regex to find potential URLs
URL_REGEX = r"https?://[\w\./\-?=&%#]+"

def parse_urls_from_search(search_result: str, max_urls: int = 5) -> List[str]:
    """Extracts URLs from raw search result text."""
    # This is a basic implementation. Assumes URLs are plain text.
    # If Tavily returns structured data, parsing should be adapted.
    urls = re.findall(URL_REGEX, search_result)
    # Simple deduplication and limit
    unique_urls = list(dict.fromkeys(urls))
    print(f"Parsed {len(unique_urls)} unique URLs from search results.")
    return unique_urls[:max_urls]

async def execute_step(state: AgentState) -> Dict:
    """
    Executes the current step of the plan.
    This involves: 
    1. Performing web search based on step_detail.
    2. Extracting URLs from search results.
    3. Fetching and extracting text from relevant URLs.
    4. Using an LLM to process extracted text against the step goal.
    5. Enforcing step-level tool call limits.
    
    Args:
        state: The current AgentState.
        
    Returns:
        A dictionary containing updates for the AgentState.
    """
    print("--- Entering Extractor Agent --- ")
    plan = state.get('plan')
    current_index = state.get('current_step_index', 0)
    history = state.get('messages', [])
    global_tool_calls = state.get('global_tool_calls', 0)
    step_results = state.get('step_results', [])
    
    while len(step_results) <= current_index:
        step_results.append({}) 
        
    step_tool_calls = 0 # Reset for this step

    if not plan or current_index >= len(plan):
        print("Extractor skipped: No plan or index out of bounds.")
        return {"current_step_index": len(plan) if plan else 0}
        
    current_step = plan[current_index]
    step_name = current_step.get("step_name", f"Step {current_index + 1}")
    step_detail = current_step.get("step_detail", "")
    
    print(f"Extractor executing: \'{step_name}\' - \'{step_detail}\'")
    
    updates = {
        "messages": list(history), # Ensure it's a mutable list
        "error": None
    }
    step_data = step_results[current_index] # Get the dict for the current step
    step_data.update({"step_name": step_name, "step_detail": step_detail}) # Ensure base info is set
    
    search_result_str = ""
    # --- 1. Perform Web Search --- 
    if step_tool_calls < settings.MAX_TOOL_CALLS_PER_STEP:
        print(f"Performing web search (Step Tool Call #{step_tool_calls + 1})...")
        search_result_str = await perform_web_search(step_detail)
        step_tool_calls += 1
        global_tool_calls += 1
        # Use AIMessage for internal updates
        updates["messages"].append(AIMessage(content=f"Performed web search for '{step_detail}'. Results obtained."))
        step_data["search_results_raw"] = search_result_str
    else:
        print(f"Skipping web search for step {current_index} due to step tool limit.")
        step_data["error"] = f"Skipped search due to step tool limit ({settings.MAX_TOOL_CALLS_PER_STEP})"
        updates["error"] = step_data["error"]

    # --- 2. Extract URLs --- 
    urls_to_process = []
    if search_result_str and not updates["error"]:
        urls_to_process = parse_urls_from_search(search_result_str, max_urls=settings.TAVILY_MAX_RESULTS)
        print(f"Attempting to process up to {len(urls_to_process)} URLs.")
    
    # --- 3. Fetch & Extract Text from URLs --- 
    extracted_contents = [] # Store successful extractions: {"text": ..., "source": url}
    if urls_to_process:
        tasks = []
        for url in urls_to_process:
            # if step_tool_calls < settings.MAX_TOOL_CALLS_PER_STEP:
                print(f"Scheduling text extraction for {url} (Step Tool Call #{step_tool_calls + 1})...")
                tasks.append(extract_text_from_url(url))
                step_tool_calls += 1 # Increment per attempt
                global_tool_calls += 1
            # else:
            #     print(f"Skipping URL extraction for remaining URLs due to step tool limit.")
            #     break
        
        if tasks:
            extraction_results = await asyncio.gather(*tasks)
            for result in extraction_results:
                if result["text"] and not result["error"]:
                    extracted_contents.append(result)
                elif result["error"]:
                    print(f"Failed extraction from {result['source']}: {result['error']}")
                    # Log error to step data maybe?
                    if "extraction_errors" not in step_data:
                        step_data["extraction_errors"] = []
                    step_data["extraction_errors"].append(f"{result['source']}: {result['error']}")
            # Use AIMessage for internal updates
            updates["messages"].append(AIMessage(content=f"Attempted to extract text from {len(tasks)} URLs."))

    # --- 4. LLM Processing of Extracted Text --- 
    final_findings = "No actionable content extracted." # Default findings
    final_sources = step_data.get("sources", []) # Keep initial search source if nothing else

    if extracted_contents:
        print(f"Processing extracted text from {len(extracted_contents)} sources using LLM...")
        # Corrected formatting for the combined text
        combined_text_parts = [
            f"---\nSource: {et['source']}\n{et['text']}\n---" 
            for et in extracted_contents
        ]
        combined_text_str = "\n\n".join(combined_text_parts)
        initial_tokens = estimate_tokens(combined_text_str)
        step_data["tokens_before_compression"] = initial_tokens
        
        # Check and compress text if needed before sending to LLM
        try:
            compressed_text = await compress_text_to_fit_context(
                combined_text_str, 
                settings.EXTRACTION_MODEL_NAME,
                prompt_buffer=5000 # Buffer for the prompt itself
            )
            final_compressed_tokens = estimate_tokens(compressed_text)
            step_data["tokens_after_compression"] = final_compressed_tokens
            if initial_tokens > final_compressed_tokens:
                 print(f"Compressed context from ~{initial_tokens} to ~{final_compressed_tokens} tokens.")
            
            extraction_llm = get_extraction_model()
            prompt = f"Based ONLY on the following text extracted from various sources, provide a concise answer addressing the query: '{step_detail}'. Synthesize the information, do not just list summaries. If the text doesn't answer the query, state that clearly.\n\nExtracted Content:\n{compressed_text}"
            
            processed_response = await extraction_llm.ainvoke(prompt)
            final_findings = processed_response.content
            final_sources = [et["source"] for et in extracted_contents] # Update sources to successful extractions
            # Use AIMessage for internal updates
            updates["messages"].append(AIMessage(content=f"Synthesized findings from {len(extracted_contents)} sources."))
            print(f"LLM processing complete. Findings: {final_findings[:200]}...")

        except Exception as e:
             error_msg = f"Error during LLM processing of extracted text: {e}"
             print(error_msg)
             # Keep raw search results as findings if LLM fails?
             final_findings = step_data.get("findings", f"Error processing extracted text: {e}") 
             updates["error"] = error_msg # Add error to step updates
             # Also add an error message to history
             updates["messages"].append(AIMessage(content=f"Encountered error during text synthesis: {e}"))

    elif not extracted_contents and search_result_str:
        # If only search results are available (no successful extractions)
        print("No text extracted from URLs, using raw search results as findings.")
        # Estimate tokens for raw search results if needed for logging
        step_data["tokens_before_compression"] = estimate_tokens(search_result_str)
        step_data["tokens_after_compression"] = step_data["tokens_before_compression"]
        final_findings = step_data.get("findings", "Could not extract details from search results.") # Use placeholder findings
        final_sources = step_data.get("sources", [])
    else: # No search results and no extracted content
        step_data["tokens_before_compression"] = 0
        step_data["tokens_after_compression"] = 0
    
    # Update step_results with the final findings and sources for this step
    step_data["findings"] = final_findings
    step_data["sources"] = final_sources
        
    # Update final state counters and results
    updates["step_results"] = step_results # The list containing the updated step_data
    updates["current_step_index"] = current_index + 1
    updates["global_tool_calls"] = global_tool_calls
    updates["step_tool_calls"] = step_tool_calls # Pass the final count for this step

    print(f"--- Exiting Extractor Agent for step {current_index} --- ")
    return updates
