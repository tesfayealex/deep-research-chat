from typing import TypedDict, Annotated, Sequence, List, Dict, Optional, Any
from langchain_core.messages import BaseMessage, HumanMessage
import operator
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

# Represents the state passed between nodes in the graph
class StepResult(BaseModel):
    """Represents the result of a single research step."""
    step_index: int
    query: str
    search_results: List[Dict[str, Any]] = []
    extracted_content: List[Dict[str, Any]] = []
    summary: str = ""
    sources: List[str] = []
    error: Optional[str] = None
    tool_calls_made: int = 0

class AgentState(TypedDict):
    # Core Query & Plan
    initial_query: str
    plan: List[Dict[str, Any]]  # Each item: {"step": int, "description": str, "query": str, "status": str}
    current_step_index: int

    # Execution Tracking
    # Store results per step, including source info
    # Example: step_results[0] = {"step_name": ..., "findings": ..., "sources": [url1, url2]}
    step_results: List[StepResult]
    
    # Control Flow & Limits
    refinement_needed: bool
    error: Optional[str]
    iterations: int # Global iteration count
    global_tool_calls: int # Total tool calls across all steps
    step_tool_calls: int # Tool calls within the current step

    # Final Output
    report: str

    # LangGraph Specific
    messages: Annotated[List[Dict[str, Any]], add_messages]

    # Configuration (Optional, could be passed differently)
    # config: dict | None 

    # New fields
    step_status: Dict[int, str]  # Maps step index to status: "pending", "in_progress", "completed", "failed"

    # Reflection system fields
    original_plan: Optional[List[Dict[str, Any]]]  # Preserve original plan before extensions
    reflection_results: List[Dict[str, Any]]  # Store reflection analysis results
    knowledge_gaps: List[Dict[str, Any]]  # Identified gaps from reflection
    plan_extensions: List[Dict[str, Any]]  # Track what was added via reflection
    reflection_count: int  # Number of reflection cycles performed
    quality_score: Optional[int]  # Overall quality assessment (0-100)
    completion_reason: str  # Why the research concluded: "quality_met", "budget_exhausted", "max_iterations", "error"

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
        plan=[],
        current_step_index=0,
        step_results=[],
        refinement_needed=False,
        error=None,
        iterations=0,
        global_tool_calls=0,
        step_tool_calls=0,
        report="",
        messages=[HumanMessage(content=query)],
        step_status={},
        original_plan=None,
        reflection_results=[],
        knowledge_gaps=[],
        plan_extensions=[],
        reflection_count=0,
        quality_score=None,
        completion_reason=""
    )
