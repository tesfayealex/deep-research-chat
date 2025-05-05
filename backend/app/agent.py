"""
Core Agent Logic using LangGraph - Main graph definition
"""
import os
import operator
from typing import TypedDict, Annotated, Sequence, List, Dict, Optional, AsyncGenerator, Any
import asyncio
import json
import time
from datetime import datetime, timezone # For timestamping
import uuid # For generating conversation IDs

from fastapi import HTTPException
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
import traceback
from datetime import datetime, timezone # For timestamping
import uuid # For generating conversation IDs

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
# Import the Weaviate saving function
from .vectorstore.weaviate_client import save_event_to_weaviate, get_weaviate_client
# Import the custom serialization utility
from .utils.serialization import safe_serialize

# --- Initialize Weaviate Client on startup (optional but recommended) ---
get_weaviate_client() 

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

# --- Custom JSON Encoder for LangChain Objects ---
class LangChainObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseMessage):
            # Use LangChain's serialization helper or .dict()
            try:
                # dumpd is generally preferred for LC objects
                return dumpd(obj) 
            except Exception:
                # Fallback if dumpd fails for some reason
                print(f"Warning: dumpd failed for {type(obj)}. Falling back to basic dict.")
                return obj.dict() 
        # Let the base class default method raise the TypeError for other types
        try:
            return super().default(obj)
        except TypeError:
             # Fallback for other potentially unserializable objects
             print(f"Warning: Cannot serialize object of type {type(obj)}. Returning string representation.")
             return str(obj) 

# --- Graph Node Wrappers --- 
# These wrappers adapt the agent functions to the input/output expected by LangGraph nodes
# They take the state, call the agent function, and merge the result back into the state.

async def plan_step_node(state: AgentState) -> AgentState:
    print("--- Entering plan_step_node ---")
    try:
        # print(f"Current state before planning: {state}")
        result = await plan_research(state)
        # print(f"--- plan_research completed successfully. Result: {result} ---")
        # Merge the updates from the planner into the state
        return {**state, **result}
    except Exception as e:
        # print(f"--- ERROR in plan_step_node: {e} ---")
        traceback.print_exc() # Print full traceback
        # Propagate error into state for the graph to handle
        return {**state, "error": f"Failed during planning: {str(e)}"}

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
    include_summaries: bool = True 
) -> AsyncGenerator[str, None]: 
    """
    Streams the research process step-by-step using graph events,
    saves distinct agent_yield and agent_internal_state events to Weaviate.
    """
    config: RunnableConfig = {"recursion_limit": LANGGRAPH_RECURSION_LIMIT}
    initial_state = create_initial_state(query, max_steps or settings.MAX_STEPS)
    stream_config = {"version": "v2"}

    HEARTBEAT_INTERVAL = 10
    last_heartbeat = time.time()
    
    # --- Generate Conversation ID & Save Initial Message --- 
    conversation_id = str(uuid.uuid4())
    print(f"[Stream] Starting new conversation with ID: {conversation_id}")
    initial_user_event = {
        "conversation_id": conversation_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "user_message",
        "sender": "user",
        "name": "user_input", # Fixed name for user input
        "tags": [],
        "data_json": json.dumps({"content": query})
    }
    await save_event_to_weaviate(initial_user_event)
    print(f"Initial user event saved for conv {conversation_id}")

    try:
        print(f"--- Starting agent event stream for conv {conversation_id} ---")
        async for event in agent_graph.astream_events(initial_state, config=config, **stream_config):
            kind = event["event"]
            name = event.get("name")
            event_payload = event.get("data", {}) 
            tags = event.get("tags", [])
            current_timestamp = datetime.now(timezone.utc).isoformat()

            # --- Heartbeat --- 
            current_time = time.time()
            if current_time - last_heartbeat > HEARTBEAT_INTERVAL:
                yield json.dumps({"type": "heartbeat", "data": "alive"}) + "\n"
                last_heartbeat = current_time

            # --- Prepare Data for Frontend Yield --- 
            # Initialize with guaranteed defaults to prevent None values
            yield_type = "debug" # Default event type
            yield_data = None    # Default data
            save_event = False   # Flag indicating whether to save this event

            # --- Determine yield_type and yield_data based on event kind and node --- 
            if kind == "on_chain_start":
                # Chain start yields a status update
                status_message = None
                if name == "plan_step_node": status_message = "Planning research steps..."
                elif name == "execute_step_node": status_message = f"Executing Step {initial_state.get('current_step_index', 0) + 1}..."
                elif name == "refine_or_report_node": status_message = "Evaluating results..."
                elif name == "generate_report_node": status_message = "Generating final report..."
                elif name == "error_handler_node": status_message = "Handling error..."
                
                if status_message:
                    yield_type = "status"  # Status is a valid yield type
                    yield_data = status_message
                    save_event = True      # We'll save status events

            elif kind == "on_chain_end":
                # Chain end events contain the actual results
                output_state = event_payload.get("output")
                if not isinstance(output_state, dict):
                    output_state = {} # Ensure output_state is a dict for safety

                # Check for errors first
                node_error = output_state.get("error")
                if node_error:
                    yield_type = "error"
                    yield_data = node_error
                    save_event = True
                else:
                    # For non-error cases, map to specific yield types based on node name
                    if name == "plan_step_node":
                        plan = output_state.get("plan")
                        if plan is not None:
                            yield_type = "plan"
                            yield_data = plan
                            save_event = True
                    elif name == "execute_step_node":
                        step_results = output_state.get("step_results")
                        finished_step_idx = output_state.get("current_step_index", 0) - 1
                        if step_results and finished_step_idx >= 0 and len(step_results) > finished_step_idx:
                            recent_step = step_results[finished_step_idx]
                            yield_type = "step_result"
                            yield_data = {
                                "step_index": finished_step_idx,
                                "step_name": recent_step.get("step_name", "?"),
                                "findings_preview": str(recent_step.get("findings", ""))[:500] + "...",
                                "sources": recent_step.get("sources", [])
                            }
                            save_event = True
                    elif name == "refine_or_report_node":
                        refinement_needed = output_state.get('refinement_needed')
                        if refinement_needed is not None:  # Make explicit check for None
                            status = "Evaluation complete. Refinement needed." if refinement_needed else "Evaluation complete. Proceeding."
                            yield_type = "evaluation"
                            yield_data = {"status": status, "refinement_needed": refinement_needed}
                            save_event = True
                    elif name == "generate_report_node":
                        report = output_state.get("report")
                        if report is not None:
                            yield_type = "report"
                            yield_data = report
                            save_event = True
                    elif name == "error_handler_node":
                        # This is redundant with error check above, but for clarity
                        if node_error:
                            yield_type = "error"
                            yield_data = f"Error handled: {node_error}"
                            save_event = True
                        else:
                            # Something went to error handler but no node_error field?
                            yield_type = "error"
                            yield_data = "Unknown error occurred in handler"
                            save_event = True
                    # else:
                    #    // Other node types would be handled here

            # --- Yield to Frontend --- 
            # Only yield non-debug events that have data
            if yield_type != "debug" and yield_data is not None:
                yield_json = json.dumps({"type": yield_type, "data": yield_data})
                yield yield_json + "\n"
                last_heartbeat = time.time() # Reset heartbeat countdown

            # --- Save Events to Weaviate (Asynchronously) --- 
            # Save agent_yield if we should and have a valid type and data
            # print(f"[SAVE_DEBUG] Event type: {yield_type}, Save flag: {save_event}, Node: {name}")
            if save_event and yield_type != "debug" and yield_data is not None:
                # Ensure yield_type is a string (defensive programming)
                agent_yield_name = str(yield_type) if yield_type is not None else "unknown"
                
                # Create the agent_yield event (what the frontend receives)
                agent_yield_event = {
                    "conversation_id": conversation_id,
                    "timestamp": current_timestamp,
                    "type": "agent_yield", 
                    "sender": "agent", 
                    "name": agent_yield_name,  # Guaranteed to be a valid string
                    "tags": tags,
                    "data_json": safe_serialize(yield_data)
                }
                print(f"[SAVE_DEBUG] Saving agent_yield with name='{agent_yield_name}'")
                asyncio.create_task(save_event_to_weaviate(agent_yield_event))

            # Save agent_internal_state for chain_end events with valid output state
            if kind == "on_chain_end" and isinstance(event_payload.get("output"), dict):
                output_state_to_save = event_payload.get("output")
                # Ensure we have a valid name for the internal state
                internal_state_name = str(name) if name is not None else "unknown_node"
                
                # Create the agent_internal_state event (for potential continuation)
                internal_state_event = {
                    "conversation_id": conversation_id,
                    "timestamp": current_timestamp, 
                    "type": "agent_internal_state", 
                    "sender": "agent", 
                    "name": internal_state_name,  # Always store the node name
                    "tags": tags,
                    "data_json": safe_serialize(output_state_to_save)
                }
                print(f"[SAVE_DEBUG] Saving agent_internal_state for node='{internal_state_name}'")
                asyncio.create_task(save_event_to_weaviate(internal_state_event))

    except Exception as e:
        print(f"--- ERROR in agent stream main loop (conv {conversation_id}): {e} ---")
        traceback.print_exc()
        error_message = f"Agent stream error: {str(e)}"
        # Yield error to frontend
        yield json.dumps({"type": "error", "data": error_message}) + "\n"
        # Save the critical orchestration error
        error_event = {
            "conversation_id": conversation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "orchestration_error",
            "sender": "system",
            "name": "stream_loop_exception", # Guaranteed string
            "tags": [],
            "data_json": safe_serialize({"message": error_message, "traceback": traceback.format_exc()})
        }
        await save_event_to_weaviate(error_event) 

    finally:
        print(f"--- Stream ended for conv {conversation_id} ---")
        final_timestamp = datetime.now(timezone.utc).isoformat()
        # Yield completion
        yield json.dumps({"type": "complete", "data": "Stream ended"}) + "\n"
        # Save completion event (as agent_yield with explicit name)
        complete_event = {
            "conversation_id": conversation_id,
            "timestamp": final_timestamp,
            "type": "agent_yield", 
            "sender": "agent",
            "name": "complete",  # This is the correct, explicit value
            "tags": [],
            "data_json": safe_serialize("Stream ended")
        }
        print(f"[SAVE_DEBUG] Saving final complete event with name='complete'")
        await save_event_to_weaviate(complete_event)

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