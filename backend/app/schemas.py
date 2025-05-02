from typing import TypedDict, Annotated, Sequence, List, Dict, Optional
from langchain_core.messages import BaseMessage, HumanMessage
import operator

# Represents the state passed between nodes in the graph
class AgentState(TypedDict):
    # Core Query & Plan
    initial_query: str
    plan: List[Dict[str, str]] | None # Changed to list of dicts: {"step_name": ..., "step_detail": ...}
    current_step_index: int

    # Execution Tracking
    # Store results per step, including source info
    # Example: step_results[0] = {"step_name": ..., "findings": ..., "sources": [url1, url2]}
    step_results: List[Dict] | None 
    
    # Control Flow & Limits
    refinement_needed: bool
    error: str | None
    iterations: int # Global iteration count
    global_tool_calls: int # Total tool calls across all steps
    step_tool_calls: int # Tool calls within the current step

    # Final Output
    report: str | None

    # LangGraph Specific
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Configuration (Optional, could be passed differently)
    # config: dict | None 

def create_initial_state(query: str, max_steps: int) -> AgentState:
    """
    Creates an initial state for the agent workflow.
    
    Args:
        query: The initial research query.
        max_steps: The maximum number of steps allowed.
    
    Returns:
        An AgentState dictionary with initial values.
    """
    return AgentState(
        initial_query=query,
        plan=None,
        current_step_index=0,
        step_results=[],
        refinement_needed=False,
        error=None,
        iterations=0,
        global_tool_calls=0,
        step_tool_calls=0,
        report=None,
        messages=[HumanMessage(content=query)]
    )
