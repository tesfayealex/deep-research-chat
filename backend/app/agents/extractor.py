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

async def execute_step(state: AgentState) -> Dict:
    """
    Executes the current step using either a direct search or LLM-generated sub-queries.
    
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
    
    context_summary = build_context_from_previous_steps(previous_step_results)
    search_method = settings.SEARCH_METHOD.lower()
    available_calls = settings.MAX_TOOL_CALLS_PER_STEP
    print(f"Using search method: {search_method} with budget: {available_calls} calls")
    
    final_findings = "No results found for this step."
    final_sources = []

    if search_method == "direct":
        print(f"Executing direct search for step: '{step_detail}'")
        try:
            search_findings, search_sources = await perform_web_search(
                step_detail, None, state.get('current_step_index', 0)
            )
            step_tool_calls += 1
            global_tool_calls += 1
            
            if search_findings:
                print(f"Direct search successful for '{step_detail}'")
                final_findings = search_findings
                final_sources = search_sources
                step_data.update({
                    'search_method': 'direct',
                    'direct_search_query': step_detail,
                    'queries_attempted_count': 1,
                    'queries_successful_count': 1
                })
            else:
                print(f"Direct search returned no results for '{step_detail}'")
        except Exception as e:
            error_msg = f"Error during direct search: {e}"
            print(error_msg)
            final_findings = f"Error performing direct search: {e}"
            updates["error"] = error_msg
    
    else: # Sub-Query Search Method
        print("Executing sub-query search method")
        try:
            sub_queries = await generate_agentic_sub_queries(
                state, max_queries_hint=available_calls
            )
            if not sub_queries:
                sub_queries = [step_detail]
            print(f"Generated {len(sub_queries)} sub-queries.")
            step_data['llm_generated_queries'] = sub_queries
        except Exception as e:
            print(f"Error generating sub-queries: {e}. Falling back to step detail.")
            sub_queries = [step_detail]
            updates["error"] = f"Failed to generate sub-queries: {e}"
        
        all_findings = []
        all_sources = []
        attempted_queries = []
        successful_queries = []
        
        search_call_limit = max(1, available_calls // 2)
        queries_to_attempt = sub_queries[:search_call_limit]

        for query in queries_to_attempt:
            if step_tool_calls >= available_calls:
                print("Stopping searches: Budget limit reached.")
                break
            
            clean_query = query.strip()
            if not clean_query or len(clean_query) < 3:
                continue
                
            print(f"Executing sub-query: '{clean_query}'")
            attempted_queries.append(clean_query)
            try:
                search_findings, search_sources = await perform_web_search(
                    clean_query, None, state.get('current_step_index', 0)
                )
                step_tool_calls += 1
                global_tool_calls += 1
                
                if search_findings:
                    all_findings.append(search_findings)
                    all_sources.extend(search_sources)
                    successful_queries.append(clean_query)
                    print(f"Search and synthesis successful for '{clean_query}'")
            except Exception as e:
                print(f"Error processing sub-query '{clean_query}': {e}")

        step_data['queries_attempted_count'] = len(attempted_queries)
        step_data['queries_successful_count'] = len(successful_queries)

        if all_findings:
            final_findings = "\n\n---\n\n".join(all_findings)
            seen_urls = set()
            unique_sources = []
            for source in all_sources:
                if source['original_url'] not in seen_urls:
                    unique_sources.append(source)
                    seen_urls.add(source['original_url'])
            final_sources = unique_sources
        
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

async def generate_agentic_sub_queries(state: AgentState, max_queries_hint: int) -> List[str]:
    """
    Uses an LLM to generate targeted search queries based on the step goal and context.
    """
    print("Using LLM to generate targeted sub-queries...")
    
    current_index = state.get('current_step_index', 0)
    plan = state.get('plan', [])
    step_detail = plan[current_index].get("step_detail", "") if plan and current_index < len(plan) else ""
    overall_goal = state.get('query', "")
    previous_step_results = state.get('step_results', [])[:current_index]
    context_summary = build_context_from_previous_steps(previous_step_results)

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
