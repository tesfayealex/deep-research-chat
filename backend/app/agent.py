"""
Core Agent Logic using LangGraph - Main graph definition
"""
import os
import operator
from typing import TypedDict, Annotated, Sequence, List, Dict, Optional, AsyncGenerator, Any
import asyncio
import json
import time
from fastapi import HTTPException
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
import traceback

# LangChain/LangGraph Imports
from langchain_openai import ChatOpenAI # Keep for now, factory will replace direct use later
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# ToolNode might not be needed directly if tools are called within agents
# from langgraph.prebuilt.tool_node import ToolNode 

# Local Imports
from .config import settings # Import the settings instance
from .schemas import AgentState, create_initial_state # Import AgentState from schemas
# from .models.model_factory import get_main_model, get_extraction_model, get_report_model # Will use later
# from .agents.planner import plan_research # Will use later
# from .agents.extractor import execute_step # Will use later
# from .agents.reporter import generate_final_report # Will use later
# Import agent functions
from .agents.planner import plan_research
from .agents.extractor import execute_step
from .agents.reporter import generate_final_report, evaluate_results
# from .memory.compressor import compress_context # Optional: Context compression
# from .utils.llm_token_estimator import estimate_token_count # For cost estimation

# --- Configuration --- #
# Replace hardcoded values with settings
# OPENAI_MODEL_NAME = settings.OPENAI_MODEL_NAME # Keep for now if Tavily tool relies on it? Or refactor tool init.
# TAVILY_MAX_RESULTS = settings.TAVILY_MAX_RESULTS
# MAX_ITERATIONS = settings.MAX_ITERATIONS
# MAX_TOOL_CALLS = settings.MAX_TOOL_CALLS # Global max tool calls
# AGENT_TIMEOUT_SECONDS = settings.AGENT_TIMEOUT_SECONDS
# LANGGRAPH_RECURSION_LIMIT = settings.LANGGRAPH_RECURSION_LIMIT
# Using MAX_STEPS from settings as a replacement for MAX_ITERATIONS for now
MAX_ITERATIONS = settings.MAX_STEPS
# MAX_TOOL_CALLS might not be defined; using a default or adding to config
MAX_TOOL_CALLS = 10 # Example default
# LANGGRAPH_RECURSION_LIMIT might not be defined; using a default
LANGGRAPH_RECURSION_LIMIT = 15 # Example default

# --- Tools --- #
# TODO: Move tool initialization to tools/ directory later
# tavily_tool = TavilySearchResults(max_results=TAVILY_MAX_RESULTS)
# tools = [tavily_tool]
# tool_node = ToolNode(tools) # May not be used directly in graph

# --- LLM Setup --- #
# TODO: Replace this with calls to model_factory later
# Using OpenAI temporarily, ensure OPENAI_API_KEY is set in settings
# Use MAIN_MODEL_NAME from settings
llm = ChatOpenAI(model=settings.MAIN_MODEL_NAME, temperature=0)

# --- Graph Node Wrappers --- 
# These wrappers adapt the agent functions to the input/output expected by LangGraph nodes
# They take the state, call the agent function, and merge the result back into the state.

async def plan_step_node(state: AgentState) -> AgentState:
    result = await plan_research(state)
    # Merge the updates from the planner into the state
    return {**state, **result}

async def execute_step_node(state: AgentState) -> AgentState:
    result = await execute_step(state)
    return {**state, **result}

async def refine_or_report_node(state: AgentState) -> AgentState:
    result = await evaluate_results(state) # Evaluation decides refinement
    return {**state, **result}

async def generate_report_node(state: AgentState) -> AgentState:
    result = await generate_final_report(state)
    return {**state, **result}

async def handle_error_node(state: AgentState) -> AgentState:
    print(f"--- ERROR HANDLER NODE ---")
    error = state.get("error", "Unknown error")
    print(f"Error recorded in state: {error}")
    # Error is already in the state, just return it. 
    # The generate_report node can check for it, or we add a final error formatting step.
    # For now, the error handler acts as a terminal state via edge connection.
    # We can enhance it later to produce a formatted error report.
    return {**state, "report": f"Agent stopped due to error: {error}"} # Overwrite report with error

# --- Graph Definition --- #

# Conditional edge logic remains similar, checking state fields like 'error', 'iterations', 'refinement_needed' etc.
def should_continue(state: AgentState) -> str:
    """Determines the next node after the refine_or_report_node evaluates the state."""
    iterations = state.get('iterations', 0) + 1 # Increment happens conceptually entering this check
    state['iterations'] = iterations # Persist increment
    global_tool_calls = state.get('global_tool_calls', 0)

    print(f"--- Should Continue Check (Iter: {iterations}, Tools: {global_tool_calls}, Refine?: {state.get('refinement_needed')}) --- ")

    if state.get("error"):
        print("Routing to error handler.")
        return "error_handler_node"

    if iterations > MAX_ITERATIONS:
        print(f"Max iterations ({MAX_ITERATIONS}) reached. Routing to generate report.")
        return "generate_report_node"
    if global_tool_calls >= MAX_TOOL_CALLS:
        print(f"Max global tool calls ({MAX_TOOL_CALLS}) reached. Routing to generate report.")
        return "generate_report_node"

    if state.get("refinement_needed"):
        print("Refinement needed. Routing back to planner.")
        # Reset refinement flag before replanning
        state['refinement_needed'] = False
        return "plan_step_node"

    # If no refinement needed and limits not hit, proceed based on plan completion
    plan = state.get('plan')
    current_index = state.get('current_step_index', 0)
    if plan and current_index < len(plan):
        print("Plan steps remaining. Routing to execute next step.")
        return "execute_step_node"
    else:
        # Plan is complete, and refinement wasn't triggered by evaluation
        print("Plan complete and results evaluated as sufficient. Routing to generate report.")
        return "generate_report_node"

# Build the graph instance
workflow = StateGraph(AgentState)

# Add nodes using the wrapper functions
workflow.add_node("plan_step_node", plan_step_node)
workflow.add_node("execute_step_node", execute_step_node)
workflow.add_node("refine_or_report_node", refine_or_report_node)
workflow.add_node("generate_report_node", generate_report_node)
workflow.add_node("error_handler_node", handle_error_node)

# Define entry point and standard edges
workflow.set_entry_point("plan_step_node")
workflow.add_edge("plan_step_node", "execute_step_node")
workflow.add_edge("execute_step_node", "refine_or_report_node") # Always evaluate after execution
workflow.add_edge("generate_report_node", END)
workflow.add_edge("error_handler_node", END)

# Add conditional edges originating from the evaluation node
workflow.add_conditional_edges(
    "refine_or_report_node",
    should_continue,
    {
        "plan_step_node": "plan_step_node",       # Refinement needed
        "execute_step_node": "execute_step_node",  # Plan steps remain
        "generate_report_node": "generate_report_node", # Plan complete or limits reached
        "error_handler_node": "error_handler_node"  # Error detected
    }
)

# Compile the graph
agent_graph = workflow.compile()
print("--- Agent Graph Compiled Successfully ---")

# --- Agent Runner Async Generator (Streamer) --- #

async def stream_agent_research(
    query: str,
    max_steps: Optional[int] = None,
    include_summaries: bool = True # This arg might be less relevant with event streaming
) -> AsyncGenerator[str, None]: # Changed return type hint to str
    """
    Streams the research process step-by-step using graph events.

    Args:
        query: The initial research query.
        max_steps: Override maximum steps from settings.
        include_summaries: (Currently unused with event streaming)

    Yields:
        A JSON string for each step/update in the research process, ending with a newline.
        Format: {"type": "status" | "plan" | "step_result" | "report" | "error" | "heartbeat", "data": ...}
    """
    config: RunnableConfig = {"recursion_limit": LANGGRAPH_RECURSION_LIMIT} # Use constant defined earlier
    initial_state = create_initial_state(query, max_steps or settings.MAX_STEPS)
    # Use astream_events V2 for detailed events
    stream_config = {"version": "v2"}

    HEARTBEAT_INTERVAL = 5 # seconds
    last_heartbeat = time.time()

    try:
        # Use astream_events to get granular start/end events
        async for event in agent_graph.astream_events(initial_state, config=config, **stream_config):
            kind = event["event"]
            name = event.get("name")
            data = event.get("data", {})
            tags = event.get("tags", [])
            metadata = event.get("metadata", {})
            
            current_time = time.time()
            if current_time - last_heartbeat > HEARTBEAT_INTERVAL:
                 yield json.dumps({"type": "heartbeat", "data": "alive"}) + "\n"
                 last_heartbeat = current_time

            # --- Handle Node Start Events (Status Updates) ---
            if kind == "on_chain_start":
                status_message = None
                if name == "plan_step_node": status_message = "Planning research steps..."
                elif name == "execute_step_node": status_message = "Executing research step..."
                elif name == "refine_or_report_node": status_message = "Evaluating results..."
                elif name == "generate_report_node": status_message = "Generating final report..."
                elif name == "error_handler_node": status_message = "Handling error..."
                
                if status_message:
                    print(f"[Stream] Yielding: status - {status_message}")
                    yield json.dumps({"type": "status", "data": status_message}) + "\n"
                    last_heartbeat = current_time # Reset after yielding

            # --- Handle Node End Events (Data Updates) ---
            elif kind == "on_chain_end":
                output_state = data.get("output") 
                input_state = data.get("input") # Get input state as well
                
                if not isinstance(output_state, dict):
                    print(f"[Stream] Warning: Node '{name}' output is not a dict: {type(output_state)}. Skipping data yield.")
                    continue
                if not isinstance(input_state, dict): # Should also be a dict
                     input_state = {} # Gracefully handle if missing
                    
                node_error = output_state.get("error")
                if node_error:
                    # Error might be handled by error_handler_node, or occurred within another node
                    print(f"[Stream] Yielding: error - {node_error}")
                    yield json.dumps({"type": "error", "data": node_error}) + "\n"
                    # Decide if we should break here or let the graph proceed to the error handler node
                    # If the error handler node itself ends, it will yield the final error report below.
                    # Let's not break here to allow the graph's error handling edge to work.
                    last_heartbeat = current_time

                # Yield specific data based on which node finished
                if name == "plan_step_node":
                    plan = output_state.get("plan")
                    if plan and not node_error and not input_state.get("plan"):
                        print(f"[Stream] Yielding: plan ({len(plan)} steps) - First Generation")
                        yield json.dumps({"type": "plan", "data": plan}) + "\n"
                        last_heartbeat = current_time
                    elif plan and not node_error and input_state.get("plan"):
                         print(f"[Stream] Skipping plan yield - Refinement loop.")
                        
                elif name == "execute_step_node":
                    step_results = output_state.get("step_results")
                    current_idx = output_state.get("current_step_index", 0)
                    # Yield result for the step that just finished (index is updated *after* execution)
                    finished_step_idx = current_idx - 1 
                    if step_results and finished_step_idx >= 0 and len(step_results) > finished_step_idx and not node_error:
                        recent_step_result = step_results[finished_step_idx]
                        print(f"[Stream] Yielding: step_result - Index {finished_step_idx}")
                        # Limit findings preview length for streaming
                        findings_preview = str(recent_step_result.get("findings", ""))
                        if len(findings_preview) > 500:
                            findings_preview = findings_preview[:500] + "..."
                        step_data = {
                            "step_index": finished_step_idx,
                            "step_name": recent_step_result.get("step_name", "?"),
                            "findings_preview": findings_preview, # Send preview
                            "sources": recent_step_result.get("sources", [])
                            # Optionally include tokens: "tokens_before": ..., "tokens_after": ...
                        }
                        yield json.dumps({"type": "step_result", "data": step_data}) + "\n"
                        last_heartbeat = current_time

                elif name == "refine_or_report_node":
                    # This node evaluates, yield evaluation status
                    refinement_needed = output_state.get('refinement_needed')
                    status = "Evaluation complete. Refinement needed." if refinement_needed else "Evaluation complete. Proceeding to report."
                    print(f"[Stream] Yielding: evaluation - {status}")
                    yield json.dumps({"type": "evaluation", "data": {"status": status, "refinement_needed": refinement_needed}}) + "\n"
                    last_heartbeat = current_time
                    
                elif name == "generate_report_node":
                    report = output_state.get("report")
                    if report and not node_error:
                        print(f"[Stream] Yielding: report")
                        yield json.dumps({"type": "report", "data": report}) + "\n"
                        # Report generation is usually the final step before END
                        # No need to break, graph flow will handle termination
                        last_heartbeat = current_time
                        
                elif name == "error_handler_node":
                     # The error handler node might put a formatted error into the 'report' field
                    error_report = output_state.get("report") 
                    if error_report:
                        print(f"[Stream] Yielding: error_report - {error_report}")
                        yield json.dumps({"type": "error", "data": error_report}) + "\n"
                    else: # Fallback if error handler didn't produce a report
                        yield json.dumps({"type": "error", "data": output_state.get("error", "Unhandled error occurred")}) + "\n"
                    # Error handler leads to END, so graph will terminate.
                    last_heartbeat = current_time

    except Exception as e:
        error_message = f"Critical error in agent stream orchestration: {str(e)}"
        print(f"[Stream] Orchestration Error: {error_message}")
        try:
            yield json.dumps({"type": "error", "data": error_message}) + "\n"
        except Exception as yield_err:
             print(f"[Stream] Error yielding orchestration error message: {yield_err}")

    finally:
        # Optional: Cleanup resources if needed
        print("--- Research Stream Ended (Finally Block) ---")
        # Send a final completion message if the stream ends normally
        yield json.dumps({"type": "complete", "data": "Stream ended"}) + "\n"

# --- Example Usage (for testing) ---
async def main():
    query = "What are the latest advancements in AI for drug discovery?"
    print(f"Starting research for query: \'{query}\'")

    async for step_update in stream_agent_research(query):
        print(f"---")
        print(f"Update Type: {step_update.get('type')}")
        print(json.dumps(step_update.get('data'), indent=2))
        print(f"---")

if __name__ == "__main__":
    asyncio.run(main()) 