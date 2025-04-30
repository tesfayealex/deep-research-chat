# PRD: Deep Research Agent - Backend

## 1. Overview

The backend service will act as the brain of the Deep Research Agent. It will receive requests from the frontend, orchestrate the multi-step research process using LLMs and external tools (like Tavily via Langflow), and send the results back. Inspiration for tool integration and flow comes from `deansaco/r1-reasoning-rag`.

## 2. Key Components & Features

1.  **API Endpoint(s):**
    *   `/api/research` (POST): Receives the user's research query from the frontend. Initiates the research process. Returns the final report.
    *   **(Optional) `/api/status` (GET or WebSocket):** Provides updates on the agent's current step/status for the frontend to display (e.g., "Planning", "Executing Step 1: Search", "Processing Results"). This enhances user experience for long-running tasks.

2.  **Agent Orchestrator:**
    *   The core logic that implements the "Plan-Execute-Process-Refine-Report" cycle.
    *   This could be implemented using:
        *   **Langflow:** Visually design the agent's workflow, connecting LLM calls, search tools (Tavily), parsers, and custom logic nodes. The backend would then execute this Langflow graph.
        *   **LangChain/LangGraph:** Programmatically define the agent's state machine or sequence of operations.
        *   **Custom Python Logic:** Build the orchestration flow manually.

3.  **Planning Module:**
    *   Takes the user query and uses an LLM to generate a step-by-step research plan.
    *   Example Plan Steps:
        1.  Define key concepts in the query.
        2.  Search for definition of concept X.
        3.  Search for applications of concept Y.
        4.  Compare X and Y based on findings.
        5.  Summarize the comparison.

4.  **Execution Module:**
    *   Iterates through the plan steps.
    *   For search steps, calls the Tavily API with optimized queries.
    *   For processing/analysis steps, calls an LLM.

5.  **Processing/Synthesis Module:**
    *   Uses an LLM to analyze search results, extract key information, and synthesize findings relevant to the current step or overall query.

6.  **Refinement Module:**
    *   Evaluates the results of a step or the overall progress.
    *   Uses an LLM or predefined logic to decide if the plan needs adjustment (e.g., refine search terms, add new steps, discard irrelevant information). If refinement is needed, it loops back to Planning or Execution.

7.  **Reporting Module:**
    *   Compiles the synthesized information from all successful steps into a final, coherent report or answer.
    *   Formats the report for display in the frontend (e.g., Markdown).

8.  **Tool Integration:**
    *   **LLM:** Interface with one or more Large Language Models (e.g., via OpenAI API, Anthropic API, Hugging Face). Configuration should allow swapping models.
    *   **Tavily:** Integrate the Tavily Search API client for executing research-focused web searches. Requires API key management.
    *   **Langflow (if used):** The backend needs to load and execute flows defined in Langflow.

## 3. Technology Stack (Recommendation)

*   **Language:** Python (common for AI/ML and web backends)
*   **Framework:** FastAPI or Flask (FastAPI is modern, async-first, good for API development)
*   **LLM Interaction:** LangChain, LiteLLM, or direct API client libraries (OpenAI, Anthropic, etc.)
*   **Agent Orchestration:** Langflow, LangChain/LangGraph, or custom implementation.
*   **Search:** Tavily Python client library.
*   **Environment Management:** Docker, Poetry, or `venv`.

## 4. Sample Code Snippets (Conceptual)

**API Endpoint (FastAPI):**

```python name=main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Assume agent_orchestrator handles the core logic
from agent import agent_orchestrator, AgentStatusTracker # Hypothetical modules

app = FastAPI()

class ResearchQuery(BaseModel):
    query: str
    session_id: str | None = None # Optional: To track conversation state

class ResearchResponse(BaseModel):
    report: str
    session_id: str # Return session_id for stateful conversation

@app.post("/api/research", response_model=ResearchResponse)
async def research_endpoint(query: ResearchQuery):
    """
    Receives a research query and returns the agent's report.
    """
    try:
        # Optional: Initialize or retrieve status tracker for this session
        # status_tracker = AgentStatusTracker(query.session_id)

        # agent_orchestrator would contain the Plan-Execute-Process-Refine-Report logic
        # It would use LLMs, Tavily, etc. internally
        # It might accept the status_tracker to report progress
        final_report = await agent_orchestrator.run_research(query.query) # Add status_tracker if used

        # Create a new session_id if none provided, or reuse
        response_session_id = query.session_id or agent_orchestrator.create_session_id()

        return ResearchResponse(report=final_report, session_id=response_session_id)

    except Exception as e:
        print(f"Error during research: {e}") # Log the error
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Optional: WebSocket endpoint for status updates
# @app.websocket("/ws/status/{session_id}")
# async def websocket_status_endpoint(websocket: WebSocket, session_id: str):
#     await AgentStatusTracker.connect(websocket, session_id)
#     try:
#         while True:
#             # Keep connection open or handle client messages
#             await websocket.receive_text()
#     except WebSocketDisconnect:
#         AgentStatusTracker.disconnect(websocket, session_id)

```

**Conceptual Agent Orchestrator Snippet (using pseudo-Langflow/LangChain ideas):**

```python name=agent/agent_orchestrator.py
# --- This is highly conceptual ---
# Assume existence of LLM wrappers, Tavily tool, etc.

# from langchain.agents import AgentExecutor, create_react_agent # Example
# from langchain_community.tools.tavily_search import TavilySearchResults # Example
# from some_llm_library import llm # Example

# tavily_tool = TavilySearchResults(max_results=5)
# tools = [tavily_tool]

# Define the core prompt including the cycle description
# agent_prompt = """
# You are a research agent. Follow these steps:
# 1. Plan: Break down the user query '{input}' into steps.
# 2. Execute: Use available tools ({tools}) to execute each step.
# 3. Process: Analyze the results.
# 4. Refine: If results are insufficient, refine the plan or execution. Repeat steps if needed.
# 5. Report: Compile a final report based on your findings.
# Thought: ... (Chain-of-thought reasoning)
# Action: ... (Tool to use)
# Action Input: ... (Input for the tool)
# Observation: ... (Result from the tool)
# ... (Repeat Thought/Action/Action Input/Observation cycle)
# Final Answer: [The final compiled report]
# """

# In a real Langflow/LangGraph setup, this would be nodes/edges
# For simplicity, a hypothetical function:
async def run_research(user_query: str) -> str:
    # 1. Plan Phase (LLM call)
    plan = await generate_plan(user_query) # llm.invoke(...)

    # 2. Execution Loop (Iterate plan, use tools like Tavily, process with LLM)
    results = []
    for step in plan:
        if step.type == "search":
            # Execute Search (Tavily)
            search_results = await execute_search(step.query) # tavily_tool.invoke(...)
            # Process Results (LLM call)
            processed_info = await process_search_results(search_results, step.objective) # llm.invoke(...)
            results.append(processed_info)
            # Refine (Optional LLM call or logic to check quality)
            if not check_quality(processed_info):
                 # Modify plan or retry step - complex logic here
                 pass
        elif step.type == "analysis":
             # Process/Analyze existing results (LLM call)
             analysis_result = await analyze_data(results, step.objective) # llm.invoke(...)
             results.append(analysis_result)
             # Refine...

    # 5. Report Phase (LLM call to synthesize results)
    final_report = await generate_report(user_query, results) # llm.invoke(...)
    return final_report

# --- Helper function placeholders ---
async def generate_plan(query): # Call LLM
    # ...
    return [{"type": "search", "query": "...", "objective": "..."}, ...] # Example structure

async def execute_search(query): # Call Tavily
    # tavily_tool.invoke(...)
    return "Search results..."

async def process_search_results(results, objective): # Call LLM
    # ...
    return "Processed info..."

async def analyze_data(current_results, objective): # Call LLM
    # ...
    return "Analysis result..."

def check_quality(info): # Simple logic or LLM call
    # ...
    return True # or False

async def generate_report(original_query, all_results): # Call LLM
    # ...
    return "Final compiled report..."

def create_session_id():
    import uuid
    return str(uuid.uuid4())

```