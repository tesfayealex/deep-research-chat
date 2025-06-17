from typing import Dict, List, Tuple, Any
import asyncio
import re
import json # Added for parsing LLM output
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage # Import AIMessage
from langchain_core.output_parsers import StrOutputParser # Added for simple parsing
from langchain_core.exceptions import OutputParserException # Handle parsing errors

from ..schemas import AgentState
from ..config import settings # Import settings
from ..tools.web_search import perform_web_search
from ..tools.url_extractor import extract_text_from_url
from ..models.model_factory import get_extraction_model, get_main_model # Use main model for query generation
from ..models.model_registry import get_context_length # Needed for context checking
from ..memory.compression import compress_text_to_fit_context, estimate_tokens # Import compression utils
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
    Executes the current step using LLM-generated sub-queries.
    This involves: 
    1. Using an LLM to analyze the step and generate targeted sub-queries.
    2. Prioritizing and executing sub-queries based on the tool call budget.
    3. Performing web searches, extracting content, and handling potential retries.
    4. Using an LLM to synthesize findings from all executed queries.
    
    Args:
        state: The current AgentState.
        
    Returns:
        A dictionary containing updates for the AgentState.
    """
    print("--- Entering Agentic Extractor --- ")
    plan = state.get('plan')
    current_index = state.get('current_step_index', 0)
    previous_step_results = state.get('step_results', [])[:current_index]
    overall_goal = state.get('query', "")
    max_repetitions = settings.MAX_STEP_REPETITIONS # Keep for retry logic
    print(f"(Max step repetitions configured: {max_repetitions})" )
    
    global_tool_calls = state.get('global_tool_calls', 0)
    step_results = list(state.get('step_results', []))
    
    while len(step_results) <= current_index:
        step_results.append({}) 
        
    step_tool_calls = 0 # Reset for this step
    
    if not plan or current_index >= len(plan):
        print("Extractor skipped: No plan or index out of bounds.")
        return {"current_step_index": len(plan) if plan else 0}
        
    current_step = plan[current_index]
    step_name = current_step.get("step_name", f"Step {current_index + 1}")
    step_detail = current_step.get("step_detail", "")
    
    print(f"Extractor executing: '{step_name}' - '{step_detail}'")
    
    updates = {"error": None}
    step_data = step_results[current_index]
    step_data.update({"step_name": step_name, "step_detail": step_detail})
    
    # --- 1. Build Context --- 
    context_summary = build_context_from_previous_steps(previous_step_results)
    
    # Check search method from configuration
    search_method = settings.SEARCH_METHOD.lower()
    print(f"Using search method: {search_method}")
    
    if search_method == "direct":
        # --- Direct Search Method: Single comprehensive search using Gemini ---
        available_calls = settings.MAX_TOOL_CALLS_PER_STEP
        print(f"Executing direct search for step: '{step_detail}'")
        
        try:
            # Perform single comprehensive search
            search_result = await perform_web_search(step_detail)
            step_tool_calls += 1
            global_tool_calls += 1
            
            if search_result:
                print(f"Direct search successful for '{step_detail}'")
                # For direct search, the result is already synthesized by Gemini
                final_findings = search_result
                final_sources = ["Gemini Google Search API - Comprehensive Research"]
                
                step_data['search_method'] = 'direct'
                step_data['direct_search_query'] = step_detail
                step_data['queries_attempted_count'] = 1
                step_data['queries_successful_count'] = 1
                
            else:
                print(f"Direct search returned no results for '{step_detail}'")
                final_findings = "No results found from direct search."
                final_sources = []
            available_calls -= 1
                
        except Exception as e:
            error_msg = f"Error during direct search: {e}"
            print(error_msg)
            final_findings = f"Error performing direct search: {e}"
            final_sources = []
            updates["error"] = error_msg
    
    else:
        # --- Sub-Query Search Method: Original multi-step approach ---
        print(f"Executing sub-query search method")
        
        # --- 2. Generate Sub-Queries via LLM --- 
        available_calls = settings.MAX_TOOL_CALLS_PER_STEP
        print(f"Generating agentic sub-queries (Budget: {available_calls} calls)")
        
        try:
            sub_queries = await generate_agentic_sub_queries(
                step_detail, 
                overall_goal, 
                context_summary, 
                # Pass budget hint to LLM (optional, but can help guide generation)
                max_queries_hint=available_calls 
            )
            if not sub_queries:
                print("LLM did not generate sub-queries. Falling back to step detail.")
                sub_queries = [step_detail] # Fallback
            else:
                 print(f"LLM generated {len(sub_queries)} sub-queries.")
                 step_data['llm_generated_queries'] = sub_queries
                 
        except Exception as e:
            print(f"Error generating agentic sub-queries: {e}. Falling back to step detail.")
            sub_queries = [step_detail] # Fallback on error
            updates["error"] = f"Failed to generate sub-queries: {e}"
        
        # --- 3. Execute Sub-Queries within Budget ---
        print(f"Executing queries within budget: {available_calls} calls total for step")
        
        all_search_results = []
        all_extracted_contents = []
        attempted_queries = []
        successful_queries = []
        
        # Simple budget allocation: Prioritize searches, then URL extractions
        # Let's reserve ~1 call per potential URL extraction (assume 2 URLs per search avg)
        search_call_limit = max(1, available_calls // 3) # Min 1 search, use up to 1/3 budget
        queries_to_attempt = sub_queries[:search_call_limit] # Prioritize first N queries from LLM
        
        print(f"Attempting up to {len(queries_to_attempt)} searches (Search call limit: {search_call_limit})")

        for i, query in enumerate(queries_to_attempt):
            if step_tool_calls >= available_calls:
                print(f"Stopping searches early: Budget limit reached ({step_tool_calls}/{available_calls})")
                break
            
            # Validate query before processing
            if not query or not query.strip() or len(query.strip()) < 3:
                print(f"⚠️ Skipping invalid/empty query: '{query}'")
                attempted_queries.append(query)  # Track as attempted for logging
                continue
                
            clean_query = query.strip()
            print(f"Executing sub-query {i+1}/{len(queries_to_attempt)}: '{clean_query}'")
            attempted_queries.append(clean_query)
            
            try:
                search_result = await perform_web_search(clean_query)
                step_tool_calls += 1
                global_tool_calls += 1
                
                if search_result:
                    all_search_results.append({"query": clean_query, "result": search_result})
                    successful_queries.append(clean_query)
                    print(f"Search successful for '{clean_query}'")
                else:
                    print(f"No results found for '{clean_query}'")
            except Exception as e:
                print(f"Error searching for '{clean_query}': {e}")
                # Continue with next query on error, don't increment step_tool_calls if search tool failed internally

        step_data['queries_attempted_count'] = len(attempted_queries)
        step_data['queries_successful_count'] = len(successful_queries)

        # --- 4. Retry Failed Queries (Optional & Simple) ---
        # Simple retry logic if budget allows (can be removed if LLM generation is robust)
        failed_queries = [q for q in attempted_queries if q not in successful_queries]
        if failed_queries and max_repetitions > 0 and step_tool_calls < available_calls:
            print(f"Attempting simple retry for {len(failed_queries)} failed queries...")
            retry_budget = min(len(failed_queries), available_calls - step_tool_calls) # How many calls left for retries?
            
            for i, failed_query in enumerate(failed_queries[:retry_budget]):
                refined_query = refine_failed_query(failed_query) # Use simple refinement
                print(f"Retrying {i+1}/{retry_budget} with: '{refined_query}'")
                try:
                    search_result = await perform_web_search(refined_query)
                    step_tool_calls += 1
                    global_tool_calls += 1
                    if search_result:
                        all_search_results.append({"query": refined_query, "result": search_result}) # Add retry results
                        successful_queries.append(refined_query) # Mark as successful
                        print(f"Retry successful for '{refined_query}'")
                except Exception as e:
                     print(f"Error during retry search for '{refined_query}': {e}")

        # --- 5. Extract URLs and Content ---
        if all_search_results:
            # Extract URLs from all successful search results
            all_urls_info = []
            for result_info in all_search_results:
                urls = parse_urls_from_search(result_info["result"], max_urls=settings.TAVILY_MAX_RESULTS)
                all_urls_info.extend([{"url": url, "source_query": result_info["query"]} for url in urls])
            
            unique_urls_info = []
            seen_urls = set()
            for url_info in all_urls_info:
                if url_info["url"] not in seen_urls:
                    unique_urls_info.append(url_info)
                    seen_urls.add(url_info["url"])
            
            print(f"Found {len(unique_urls_info)} unique URLs across successful searches.")
            
            # Extract content from ALL URLs found (not limited by step tool calls)
            # URL extraction is crucial for research quality and should not be artificially limited
            if unique_urls_info:
                print(f"Extracting content from ALL {len(unique_urls_info)} URLs (unlimited extraction)")
                extraction_tasks = []
                for url_info in unique_urls_info:
                    extraction_tasks.append(extract_text_from_url(url_info["url"]))
                    # Note: We still count tool calls for tracking but don't limit extraction
                    step_tool_calls += 1
                    global_tool_calls += 1

                if extraction_tasks:
                    extraction_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
                    
                    for i, res in enumerate(extraction_results):
                        url_info = unique_urls_info[i]
                        url = url_info["url"]
                        source_query = url_info["source_query"]
                        
                        if isinstance(res, Exception):
                            print(f"Failed extraction from {url}: {res}")
                            if "extraction_errors" not in step_data: step_data["extraction_errors"] = []
                            step_data["extraction_errors"].append(f"{url}: {str(res)}")
                        elif isinstance(res, dict) and res.get("text") and not res.get("error"):
                            res["source_query"] = source_query # Add originating query info
                            all_extracted_contents.append(res)
                            print(f"Extracted content from {url} (from query: '{source_query}')")
                        elif isinstance(res, dict) and res.get("error"):
                            print(f"Error extracting from {url}: {res.get('error')}")
                            if "extraction_errors" not in step_data: step_data["extraction_errors"] = []
                            step_data["extraction_errors"].append(f"{url}: {res.get('error')}")
        
        # --- 6. Synthesize Findings with LLM ---
        final_findings = "No actionable content found or processed for this step."
        final_sources = []

        if all_extracted_contents:
            print(f"Synthesizing findings from {len(all_extracted_contents)} extracted contents...")
            # Organize content by the query that led to it
            content_by_query = {}
            for content in all_extracted_contents:
                query = content.get("source_query", "Unknown Query")
                if query not in content_by_query: content_by_query[query] = []
                content_by_query[query].append(content)

            # Format combined text for LLM synthesis
            combined_text_parts = [f"=== Content related to query: '{query}' ===" + 
                                   "\n".join([f"---\nSource: {c['source']}\n{c['text']}\n---" 
                                             for c in contents]) 
                                   for query, contents in content_by_query.items()]
            combined_text_str = "\n\n".join(combined_text_parts)
            
            try:
                extraction_llm = get_extraction_model() # Use extraction model for synthesis
                
                synthesis_prompt_template = ChatPromptTemplate.from_messages([
                    SystemMessage(content=f"""You are an AI assistant synthesizing information gathered from multiple targeted web searches to answer a specific research step.
                    Overall Research Goal: {overall_goal}
                    Current Step Goal: {step_detail}
                    Context from Previous Steps: {context_summary}

                    Analyze the provided content, which is grouped by the search query that found it. 
                    Synthesize a concise answer to the 'Current Step Goal', drawing ONLY from the provided text. Highlight key findings and mention any significant gaps based on the content provided. Structure the output clearly."""),
                    HumanMessage(content="Extracted Content:\n\n{compressed_text}")
                ])

                prompt_base = synthesis_prompt_template.format(compressed_text="")
                prompt_tokens = estimate_tokens(str(prompt_base))
                
                compressed_text = await compress_text_to_fit_context(
                    combined_text_str,
                    settings.EXTRACTION_MODEL_NAME, # Model used for synthesis
                    prompt_buffer=prompt_tokens + 500
                )
                
                final_prompt = synthesis_prompt_template.format(compressed_text=compressed_text)
                processed_response = await extraction_llm.ainvoke(final_prompt)
                
                final_findings = processed_response.content
                final_sources = [content["source"] for content in all_extracted_contents]
                
                step_data["tokens_before_compression"] = estimate_tokens(combined_text_str)
                step_data["tokens_after_compression"] = estimate_tokens(compressed_text)
                print(f"Synthesis complete. Findings preview: {final_findings[:200]}...")

            except Exception as e:
                error_msg = f"Error during synthesis: {e}"
                print(error_msg)
                final_findings = f"Error synthesizing results: {e}. See raw extracted content."
                # Keep extracted sources even if synthesis fails
                final_sources = [content["source"] for content in all_extracted_contents] 
                step_data["synthesis_error"] = error_msg # Log synthesis error

        elif all_search_results: # Fallback if only search results, no content
            print("Using raw search results as findings due to lack of extracted content.")
            # print(res_info)
            # print("#####################################")
            print(all_search_results)
            final_findings = "Could not extract content from URLs. Raw search result snippets:\n\n"
            for res_info in all_search_results:
                final_findings += f"Query: '{res_info['query']}'\nResult: {res_info['result'][:250]}...\n\n"
            final_sources = [f"Search result for: {res_info['query']}" for res_info in all_search_results]
        
    # --- 7. Update State ---
    step_data["tool_calls_used"] = step_tool_calls
    step_data["findings"] = final_findings
    step_data["sources"] = final_sources
    
    if step_tool_calls >= available_calls:
        step_data["budget_exhausted"] = True
        step_data["budget_limit"] = available_calls
        print("Step budget exhausted.")
    
    updates["step_results"] = step_results
    updates["current_step_index"] = current_index + 1
    updates["global_tool_calls"] = global_tool_calls
    
    print(f"--- Exiting Agentic Extractor (used {step_tool_calls}/{available_calls} tool calls) ---")
    return updates

# --- Helper Functions ---

def build_context_from_previous_steps(previous_step_results: List[Dict]) -> str:
    """Build a concise summary of previous step results for context."""
    if not previous_step_results: return "None"
    
    context_summary = "Relevant previous findings:\n"
    for i, res in enumerate(previous_step_results[-3:]): # Limit context to last 3 steps
        findings = res.get('findings', 'N/A')
        prev_step_name = res.get('step_name', f'Step {i+1}')
        # Make summary very brief
        context_summary += f"- {prev_step_name}: {str(findings)[:100]}...\n" 
    return context_summary

async def generate_agentic_sub_queries(step_detail: str, overall_goal: str, context_summary: str, max_queries_hint: int) -> List[str]:
    """
    Uses an LLM to generate targeted search queries based on the step goal and context.
    """
    print("Using LLM to generate targeted sub-queries...")
    
    # Validate inputs
    if not step_detail or not step_detail.strip():
        print("⚠️ Empty or invalid step detail provided")
        return []
    
    if not overall_goal or not overall_goal.strip():
        print("⚠️ Empty or invalid overall goal provided")
        return []
    
    # Use the main model for potentially more complex reasoning for query generation
    query_generation_llm = get_main_model() 
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content=f"""You are an expert research assistant. Your task is to break down a research step into specific, effective search engine queries.
            
            Overall Research Goal: {overall_goal}
            Current Step Goal: {step_detail}
            Context from Previous Steps: {context_summary}
            Tool Budget Hint: You should aim to generate queries that can be reasonably executed within {max_queries_hint} total tool calls (searches + URL reads). Generate essential queries first.

            Based on the Current Step Goal and the overall context, generate a list of 3-5 precise search queries that will help achieve the step goal.
            Focus on queries that directly target the required information. Avoid overly broad or vague queries.
            
            IMPORTANT: Each query must be:
            - At least 3 words long
            - Specific and searchable
            - Related to the step goal
            - Not empty or just whitespace
            
            Output ONLY a JSON list of strings, where each string is a search query. Example:
            ["query 1", "query 2", "query 3"]
            
            If the step goal is simple and needs only one query, return a list with one query.
            If the step goal seems un-searchable or makes no sense, return an empty list [].
            """
        ),
        HumanMessage(content=f"Generate the search queries for the step: '{step_detail}'")
    ])

    # Chain the prompt with the LLM and a JSON parser
    # Adding basic retry or error handling for LLM call
    chain = prompt | query_generation_llm | StrOutputParser()
    
    try:
        response = await chain.ainvoke({})
        print(f"LLM response for queries: {response}")
        
        # Attempt to parse the JSON list
        try:
            # Basic cleaning: Find the list within the response
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                json_str = match.group(0)
                queries = json.loads(json_str)
                if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                    # Validate each query
                    valid_queries = []
                    for query in queries[:max_queries_hint]:
                        if query and query.strip() and len(query.strip()) >= 3:
                            valid_queries.append(query.strip())
                        else:
                            print(f"⚠️ Skipping invalid query: '{query}'")
                    
                    if not valid_queries:
                        print("⚠️ No valid queries generated, creating fallback query")
                        return [f"information about {step_detail}"]
                    
                    return valid_queries
                else:
                    print("LLM output was not a valid list of strings.")
                    return [f"information about {step_detail}"]
            else:
                 print("Could not find JSON list in LLM output.")
                 return [f"information about {step_detail}"]
        except json.JSONDecodeError as json_err:
            print(f"Failed to parse LLM query output as JSON: {json_err}")
            # Fallback: maybe the LLM just gave a single query string?
            if isinstance(response, str) and len(response.strip()) > 5 and '\n' not in response:
                 print("Treating LLM output as single query.")
                 return [response.strip()] # Treat as single query if reasonable
            return [f"information about {step_detail}"] # Fallback query
            
    except Exception as e:
        print(f"LLM call failed during query generation: {e}")
        # Return fallback query instead of raising exception
        print("Using fallback query due to LLM failure")
        return [f"information about {step_detail}"]

# Simple refinement for retrying failed searches (kept from previous logic)
def refine_failed_query(query: str) -> str:
    """Create an alternative version of a failed query."""
    # Simple refinement strategies
    if len(query) > 60: # Shorten very long queries
        return ' '.join(query.split()[:8]) + " summary"
    
    # Add different phrasing
    if not query.lower().startswith("details about"):
        return f"details about {query}"
    if not query.lower().startswith("overview of"):
        return f"overview of {query}"
        
    # Remove potentially problematic terms (example)
    for term in ["latest", "comprehensive", "full"]:
        if term in query.lower():
            return query.lower().replace(term, "").strip()
            
    # Default: append "key facts"
    return f"{query} key facts"
