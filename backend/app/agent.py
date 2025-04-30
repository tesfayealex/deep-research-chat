"""
Core Agent Logic using LangGraph
"""
import os
import operator
from typing import TypedDict, Annotated, Sequence, List, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_node import ToolNode # May need ToolInvocation instead depending on exact setup
from dotenv import load_dotenv
import asyncio
import json # For streaming
import time

# Load environment variables (ensure .env file exists with API keys)
load_dotenv()

# --- Configuration --- #
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
TAVILY_MAX_RESULTS = 2
MAX_ITERATIONS = 2 # Max refinement/planning loops
MAX_TOOL_CALLS = 2 # Max calls to Tavily search
AGENT_TIMEOUT_SECONDS = 10
# Give LangGraph more room than our explicit limits
LANGGRAPH_RECURSION_LIMIT = MAX_ITERATIONS * 3 

# --- Tools --- #
# Only using Tavily search for now, as per PRD and r1-reasoning inspiration
tavily_tool = TavilySearchResults(max_results=TAVILY_MAX_RESULTS)
tools = [tavily_tool]
tool_node = ToolNode(tools)

# --- LLM Setup --- #
# Using OpenAI, ensure OPENAI_API_KEY is set in .env
llm = ChatOpenAI(model=OPENAI_MODEL_NAME, temperature=0)

# --- Agent State --- #
# Represents the state passed between nodes in the graph
class AgentState(TypedDict):
    initial_query: str # The original user query
    plan: List[str] | None # The current research plan steps
    executed_steps: List[Dict] | None # Results from executed steps { "step": str, "result": str }
    current_step_index: int # Index of the plan step being executed
    refinement_needed: bool # Flag if refinement is required
    error: str | None # Store errors if they occur
    report: str | None # Final generated report
    messages: Annotated[Sequence[BaseMessage], operator.add] # Conversation history
    iterations: int # Tracks planning/refinement cycles
    tool_calls: int # Tracks tool execution count

# --- Graph Nodes --- #

async def plan_step(state: AgentState) -> AgentState:
    """Node to generate the initial research plan or refine an existing one."""
    print("--- ENTERING: plan_step ---")
    print("--- PLAN STEP ---")
    query = state['initial_query']
    history = state.get('messages', [])
    executed = state.get('executed_steps')

    prompt_messages = [
        SystemMessage(
            content=f"""You are a research planning assistant. Your goal is to create a step-by-step plan to answer the user's query thoroughly. 
            Query: {query}
            Break the query down into logical search steps using the available tool: 'tavily_search_results_json'.
            Focus on gathering specific information needed to construct a comprehensive answer.
            Keep the plan concise, typically 3-5 steps.
            Output the plan as a numbered list.
            """
        ),
    ]
    # If refining, add context
    if executed:
        prompt_messages.append(SystemMessage(content="Refining plan based on previous results:"))
        for step_result in executed:
            prompt_messages.append(SystemMessage(content=f"- {step_result['step']}: {step_result['result'][:200]}...")) # Show partial previous results

    prompt_messages.append(HumanMessage(content=f"Create a plan for this query: {query}"))

    response = await llm.ainvoke(prompt_messages)
    plan_text = response.content
    # Simple parsing of numbered list
    plan_list = [line.strip().split('. ', 1)[1] for line in plan_text.split('\n') if line.strip() and '.' in line]

    print(f"Generated Plan: {plan_list}")
    return {
        "plan": plan_list,
        "current_step_index": 0,
        "executed_steps": state.get('executed_steps', []), # Keep existing results if refining
        "refinement_needed": False, # Reset refinement flag
        "messages": history + [AIMessage(content=f"Okay, I have a plan:\n{plan_text}")]
    }

async def execute_tools(state: AgentState) -> AgentState:
    """Node to execute the tool (Tavily search) for the current plan step."""
    print("--- ENTERING: execute_tools ---")
    print("--- EXECUTE TOOLS ---")
    plan = state['plan']
    current_index = state['current_step_index']
    history = state.get('messages', [])
    tool_calls = state.get('tool_calls', 0)

    if not plan or current_index >= len(plan):
        print("Execution skipped: No plan or index out of bounds.")
        return {"refinement_needed": True} # Signal something went wrong, maybe refine

    if tool_calls >= MAX_TOOL_CALLS:
        print(f"Max tool calls ({MAX_TOOL_CALLS}) reached. Skipping execution.")
        # Signal to proceed to report generation
        return {"current_step_index": len(state.get('plan', []))} # Mark plan as 'done'
    
    step_query = plan[current_index]
    print(f"Executing Step {current_index + 1} (Tool Call #{tool_calls + 1}): {step_query}")
    tool_input = {"query": step_query}

    try:
        tool_result = await tavily_tool.ainvoke(tool_input)
        print(f"Tool Result (shortened): {str(tool_result)[:500]}...")
        executed_steps = state.get('executed_steps', [])
        executed_steps.append({"step": step_query, "result": str(tool_result)})

        return {
            "executed_steps": executed_steps,
            "current_step_index": current_index + 1,
            "tool_calls": tool_calls + 1, # Increment tool call counter
            "messages": history + [AIMessage(f"Executing search for: {step_query}"), HumanMessage(f"Search results obtained.")]
        }
    except Exception as e:
        print(f"Error executing tool: {e}")
        return {"error": f"Failed to execute step: {step_query}. Error: {e}"}

async def process_results(state: AgentState) -> AgentState:
    """Node to process the results of the executed steps using an LLM."""
    # This node might not be strictly necessary if refinement/reporting handles processing,
    # but included for potential step-by-step analysis as per PRD.
    print("--- PROCESS RESULTS ---")
    # For simplicity, we'll let the refine/report nodes handle synthesis for now.
    # This could be expanded to summarize each step's findings.
    return {}

async def refine_or_report(state: AgentState) -> AgentState:
    """Node to decide whether to refine the plan or generate the final report."""
    print("--- ENTERING: refine_or_report ---")
    print("--- REFINE OR REPORT DECISION ---")
    plan = state.get('plan', [])
    current_index = state.get('current_step_index', 0)
    executed_steps = state.get('executed_steps', [])
    history = state.get('messages', [])
    query = state['initial_query']

    # Check if plan execution is complete
    if current_index < len(plan):
        # Plan not finished, maybe prompt refinement based on last result? Not implemented yet.
        # For now, assume we continue if plan steps remain.
        print("Plan not complete, continuing execution.")
        return { "refinement_needed": False }

    # Plan is complete, decide if results are sufficient for a report
    prompt_messages = [
        SystemMessage(
            content="""You are a research evaluation assistant. Review the research plan, the original query, and the gathered results. 
            Decide if the information is sufficient to generate a comprehensive final report, or if the plan needs refinement (e.g., new search steps, modified queries). 
            Respond with only 'REFINEMENT_NEEDED' or 'REPORT_READY'."""
        ),
        HumanMessage(content=f"Original Query: {query}"),
        HumanMessage(content=f"Plan: {plan}"),
    ]
    results_summary = "\n".join([f"- Step '{res['step']}': {res['result'][:200]}..." for res in executed_steps])
    prompt_messages.append(HumanMessage(content=f"Results Gathered:\n{results_summary}"))
    prompt_messages.append(HumanMessage(content="Decision (REFINEMENT_NEEDED or REPORT_READY): "))

    response = await llm.ainvoke(prompt_messages)
    decision = response.content.strip()

    print(f"Refinement Decision: {decision}")

    if "REFINEMENT_NEEDED" in decision:
        return {"refinement_needed": True, "messages": history + [AIMessage(content="The results seem insufficient. I need to refine the plan.")]}
    else:
        return {"refinement_needed": False, "messages": history + [AIMessage(content="The results look promising. Proceeding to generate the report.")]}

async def generate_report(state: AgentState) -> AgentState:
    """Node to generate the final report based on executed steps."""
    print("--- ENTERING: generate_report ---")
    print("--- GENERATE REPORT ---")
    query = state['initial_query']
    executed_steps = state.get('executed_steps', [])
    history = state.get('messages', [])

    prompt_messages = [
        SystemMessage(
            content="""You are a research reporting assistant. Synthesize the information gathered from the research steps into a comprehensive and coherent report that directly answers the user's original query. 
            Structure the report clearly. Use markdown for formatting if appropriate.
            Base the report *only* on the provided results."""
        ),
        HumanMessage(content=f"Original User Query: {query}"),
    ]
    results_summary = "\n".join([f"### Step: {res['step']}\nResult: {res['result']}\n--- " for res in executed_steps])
    prompt_messages.append(SystemMessage(content=f"Research Findings:\n{results_summary}"))
    prompt_messages.append(HumanMessage(content="Generate the final research report:"))

    response = await llm.ainvoke(prompt_messages)
    report_text = response.content

    print(f"Generated Report (preview): {report_text[:500]}...")
    return {"report": report_text, "messages": history + [AIMessage(content=report_text)]}

# --- Graph Definition --- # 

# Define conditional edges
def should_continue(state: AgentState) -> str:
    """Determines the next step based on the current state."""
    iterations = state.get('iterations', 0)
    tool_calls = state.get('tool_calls', 0)
    state['iterations'] = iterations + 1 # Increment iteration count here or in planner

    print(f"--- Should Continue Check (Iteration: {state['iterations']}, Tool Calls: {tool_calls}) ---")

    if state.get("error"): 
        print("Routing to error handler.")
        return "error_handler"
    
    # Check limits before checking refinement needs
    if state['iterations'] > MAX_ITERATIONS:
        print(f"Max iterations ({MAX_ITERATIONS}) reached. Routing to generate report.")
        return "generate_report"
    if tool_calls >= MAX_TOOL_CALLS:
        print(f"Max tool calls ({MAX_TOOL_CALLS}) reached. Routing to generate report.")
        return "generate_report"
    
    if state.get("refinement_needed"): 
        print("Refinement needed. Routing to plan step.")
        return "plan_step"

    # If no report generated yet, continue plan or evaluate
    plan = state.get('plan')
    current_index = state.get('current_step_index', 0)
    if plan and current_index < len(plan):
        print("Plan steps remaining. Routing to execute tools.")
        return "execute_tools"
    else:
        print("Plan complete or empty. Routing to refine or report.")
        return "refine_or_report"

# Handle errors (simple version)
def handle_error(state: AgentState) -> AgentState:
    print(f"--- ERROR HANDLER ---")
    error = state.get("error")
    print(f"Error encountered: {error}")
    # For now, just end and report the error
    return {"report": f"An error occurred during processing: {error}"} 

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("plan_step", plan_step)
workflow.add_node("execute_tools", execute_tools)
# workflow.add_node("process_results", process_results) # Keep it simple for now
workflow.add_node("refine_or_report", refine_or_report)
workflow.add_node("generate_report", generate_report)
workflow.add_node("error_handler", handle_error)

# Define entry and edges
workflow.set_entry_point("plan_step")

workflow.add_edge("plan_step", "execute_tools") # Always execute after planning
workflow.add_edge("execute_tools", "refine_or_report") # Decide after execution
workflow.add_edge("generate_report", END)
workflow.add_edge("error_handler", END) # End after handling error

# Conditional edges based on the decider function
workflow.add_conditional_edges(
    "refine_or_report",
    should_continue,
    {
        "plan_step": "plan_step", # Refinement needed
        "execute_tools": "execute_tools", # Continue plan (shouldn't happen if logic is right, but safe)
        "generate_report": "generate_report", # Report ready
        END: END
    }
)

# Compile the graph
agent_graph = workflow.compile()

# --- Agent Runner Async Generator --- #
async def stream_agent_research(query: str):
    """Runs the agent graph and yields structured status updates."""
    print(f"[Stream] Starting research for: {query}")
    
    # Initialize state
    initial_state: AgentState = {
        "initial_query": query,
        "plan": None,
        "executed_steps": [],
        "current_step_index": 0,
        "refinement_needed": False,
        "error": None,
        "report": None,
        "messages": [HumanMessage(content=query)],
        "iterations": 0,
        "tool_calls": 0
    }

    config = {"recursion_limit": LANGGRAPH_RECURSION_LIMIT}
    
    # Heartbeat interval (seconds)
    HEARTBEAT_INTERVAL = 5
    last_heartbeat = 0
    
    try:
        async for event in agent_graph.astream_events(initial_state, config=config, version="v2"):
            # Less verbose raw log for critical events
            if event["event"] in ["on_chain_start", "on_chain_end"]:
                print(f"\n--- Relevant Event --- Type: {event['event']}, Name: {event.get('name')}, Tags: {event.get('tags')} ---")
                # print(f"--- RAW ---:\n{event}\n----------") # Uncomment for full detail if needed
            
            current_time = time.time()
            
            # Send heartbeat if needed
            if current_time - last_heartbeat > HEARTBEAT_INTERVAL:
                yield json.dumps({"type": "heartbeat", "data": "alive"}) + "\n"
                last_heartbeat = current_time
            
            # Process event
            kind = event["event"]
            name = event.get("name", "")
            tags = event.get("tags", [])
            data = event.get("data", {})
            
            # Always yield the raw event for debugging
            yield json.dumps({"type": "debug", "data": {"event": kind, "tags": tags, "name": name}}) + "\n"
            
            # Yield meaningful updates - Use the 'name' field instead of tags
            if kind == "on_chain_start":
                if name == "plan_step":
                    print("[Stream] Yielding: status - Planning research...") # Log yield
                    yield json.dumps({"type": "status", "data": "Planning research..."}) + "\n"
                elif name == "execute_tools":
                    print("[Stream] Yielding: status - Executing search...") # Log yield
                    yield json.dumps({"type": "status", "data": "Executing search..."}) + "\n"
                elif name == "refine_or_report":
                    print("[Stream] Yielding: status - Evaluating results...") # Log yield
                    yield json.dumps({"type": "status", "data": "Evaluating results..."}) + "\n"
                elif name == "generate_report":
                    print("[Stream] Yielding: status - Generating report...") # Log yield
                    yield json.dumps({"type": "status", "data": "Generating report..."}) + "\n"
            
            elif kind == "on_chain_end":
                if name == "plan_step":
                    plan = data.get("output", {}).get("plan")
                    if plan:
                        print("[Stream] Yielding: plan") # Log yield
                        yield json.dumps({"type": "plan", "data": plan}) + "\n"
                elif name == "execute_tools":
                    # Get the state *after* the node executed from the output
                    output_state = data.get("output", {}) 
                    executed_steps = output_state.get("executed_steps", [])
                    current_step_index = output_state.get("current_step_index", 0)
                    
                    if executed_steps and current_step_index > 0:
                        # Get the most recent step result
                        recent_step = executed_steps[-1]
                        print(f"[Stream] Yielding: step_result - Index {current_step_index - 1}") # Log yield
                        step_data = {
                            "step_index": current_step_index - 1,
                            "step": recent_step["step"],
                            "result": recent_step["result"][:500] + ("..." if len(str(recent_step["result"])) > 500 else "")
                        }
                        yield json.dumps({"type": "step_result", "data": step_data}) + "\n"
                elif name == "generate_report":
                    report = data.get("output", {}).get("report")
                    if report:
                        print("[Stream] Yielding: report") # Log yield
                        yield json.dumps({"type": "report", "data": report}) + "\n"
                        return
    
    except Exception as e:
        print(f"[Stream] Error during stream: {e}") # Log exceptions
        yield json.dumps({"type": "error", "data": str(e)}) + "\n"
    finally:
        print("[Stream] Yielding: complete") # Log yield
        yield json.dumps({"type": "complete", "data": "Stream ended"}) + "\n"

# Example usage (for testing directly)
# if __name__ == "__main__":
#     import asyncio
#     async def main():
#         test_query = "What are the main differences between LangGraph and LangChain Agents?"
#         result = await run_agent_research(test_query)
#         print("\n--- FINAL REPORT ---")
#         print(result)
#     asyncio.run(main()) 