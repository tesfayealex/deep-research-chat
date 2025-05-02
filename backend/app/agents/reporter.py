from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from ..schemas import AgentState
from ..config import settings
from ..models.model_factory import get_report_model, get_main_model # May need main model for evaluation

async def evaluate_results(state: AgentState) -> Dict:
    """
    Evaluates if the current results are sufficient or if refinement is needed.
    This acts as the decision logic for the 'refine_or_report' node.
    
    Args:
        state: The current AgentState.
        
    Returns:
        A dictionary containing {"refinement_needed": bool, "messages": UpdatedMessages}.
    """
    print("--- Entering Reporter Agent (Evaluation) --- ")
    plan = state.get('plan', [])
    step_results = state.get('step_results', [])
    history = state.get('messages', [])
    query = state['initial_query']

    # Simple heuristic: If we have results for all planned steps, assume report ready.
    # A more complex evaluation could involve an LLM call.
    if len(step_results) >= len(plan):
        print("Evaluation: All planned steps have results. Proceeding to report.")
        refinement_needed = False
    else:
        # Could add LLM call here to check result quality even if incomplete
        print("Evaluation: Not all planned steps have results. Assuming refinement needed for now.")
        refinement_needed = True # Default to refinement if plan not fully executed
        # Example LLM evaluation (more robust):
        # try:
        #     llm = get_main_model() # Or a specific evaluation model
        #     prompt = [ ... format prompt asking if results are sufficient ... ]
        #     response = await llm.ainvoke(prompt)
        #     refinement_needed = "REFINEMENT_NEEDED" in response.content.strip()
        # except Exception as e:
        #     print(f"Error during LLM evaluation: {e}. Defaulting to refinement_needed=True")
        #     refinement_needed = True

    ai_message = "The results seem insufficient. I need to refine the plan." if refinement_needed else "The results look promising. Proceeding to generate the report."
    
    return {
        "refinement_needed": refinement_needed,
        "messages": history + [AIMessage(content=ai_message)]
    }

async def generate_final_report(state: AgentState) -> Dict:
    """
    Generates the final report based on the accumulated step results.
    Supports different reporting modes based on settings.
    
    Args:
        state: The current AgentState.
        
    Returns:
        A dictionary containing {"report": str, "messages": UpdatedMessages}.
    """
    print(f"--- Entering Reporter Agent (Generation - Mode: {settings.REPORT_MODE}) --- ")
    query = state['initial_query']
    step_results = state.get('step_results', [])
    history = state.get('messages', [])

    if not step_results:
         report_text = "No research results were gathered to generate a report."
         return {"report": report_text, "messages": history + [AIMessage(content=report_text)]}
         
    try:
        llm = get_report_model()
    except ValueError as e:
        print(f"Error getting report model: {e}")
        report_text = f"Error: Could not load the reporting model ({settings.REPORT_MODEL})."
        return {"report": report_text, "messages": history + [AIMessage(content=report_text)]}

    prompt_messages = [
        SystemMessage(
            content="""You are a research reporting assistant. Synthesize the gathered information into a comprehensive report answering the user's query.
            Structure the report clearly using Markdown.
            Focus on directly answering the original query based *only* on the provided step findings.
            Include citations/sources if available within the findings text or list them at the end."""
        ),
        HumanMessage(content=f"Original User Query: {query}"),
    ]

    # Prepare context based on reporting mode
    if settings.REPORT_MODE == "stepwise":
        # TODO: Implement stepwise reporting (generate summary per step first?)
        # For now, fall back to unified context for stepwise as well.
        print("Stepwise reporting not fully implemented, using unified context.")
        context = "\n".join([
            f"### Step: {res.get('step_name','?')}\nFindings: {res.get('findings','N/A')}\nSources: {res.get('sources', [])}\n--- " 
            for res in step_results if res # Ensure step result exists
        ])
    else: # Unified mode (default)
        context = "\n".join([
            # Focus more on findings and sources, less on step structure in unified
            f"Findings from step '{res.get('step_name','?')}': {res.get('findings','N/A')}\nSources: {res.get('sources', [])}\n--- " 
            for res in step_results if res
        ])
        
    prompt_messages.append(SystemMessage(content=f"Research Findings Context:\n{context}"))
    prompt_messages.append(HumanMessage(content="Generate the final research report based on the context provided."))

    try:
        response = await llm.ainvoke(prompt_messages)
        report_text = response.content
        print(f"Reporter generated report (preview): {report_text[:500]}...")
        return {"report": report_text, "messages": history + [AIMessage(content=report_text)]}
    except Exception as e:
        error_msg = f"Error during final report generation: {e}"
        print(error_msg)
        # Return error message as the report
        return {"report": f"Failed to generate report: {error_msg}", "messages": history}
