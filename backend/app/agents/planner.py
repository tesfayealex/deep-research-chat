import json
from typing import List, Dict, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from ..schemas import AgentState
from ..models.model_factory import get_main_model
from ..config import settings

MAX_PLAN_LENGTH = settings.MAX_STEPS # Use setting

async def plan_research(state: AgentState) -> Dict:
    """
    Generates or refines a research plan based on the initial query and previous results.
    
    Args:
        state: The current AgentState.
        
    Returns:
        A dictionary containing the updated plan and messages, suitable for merging back into the state.
        Example: {"plan": [...], "messages": [...], "error": None}
    """
    print("--- Entering Planner Agent --- ")
    query = state['initial_query']
    history = state.get('messages', [])
    step_results = state.get('step_results')
    
    # Get the appropriate LLM using the factory
    try:
        llm = get_main_model()
    except ValueError as e:
        print(f"Error getting main model: {e}")
        return {"error": f"Failed to get planning model: {e}", "plan": None, "messages": history}

    prompt_messages = [
        SystemMessage(
            content=f"""You are a research planning assistant. Your goal is to create a step-by-step plan to answer the user's query thoroughly.
            Query: {query}
            Break the query down into logical search steps.
            Focus on gathering specific information needed to construct a comprehensive answer.
            Keep the plan concise, typically 3-5 steps.
            Output the plan ONLY as a JSON list of objects, each with 'step_name' (a short title) and 'step_detail' (the specific query or task for that step).
            Example: [{{"step_name": "Define X", "step_detail": "Find formal definition and key characteristics of X"}}, ...]
            Ensure the output is valid JSON.
            """
        ),
    ]
    if step_results:
        prompt_messages.append(SystemMessage(content="Refining plan based on previous results:"))
        for i, res in enumerate(step_results):
            summary = str(res.get('findings', ''))[:200]
            prompt_messages.append(SystemMessage(content=f"- Result from Step {i+1} ('{res.get('step_name', 'Unknown')}'): {summary}..." ))

    prompt_messages.append(HumanMessage(content=f"Create the JSON plan for the query: {query}"))

    try:
        response = await llm.ainvoke(prompt_messages)
        plan_text = response.content.strip()
        
        # Basic cleanup: remove potential markdown code fences
        if plan_text.startswith("```json"):
            plan_text = plan_text[7:]
        if plan_text.endswith("```"):
            plan_text = plan_text[:-3]
            
        plan_list = json.loads(plan_text)
        
        # --- Add Step Validation --- 
        if len(plan_list) > MAX_PLAN_LENGTH:
            print(f"Warning: Generated plan ({len(plan_list)} steps) exceeds max length ({MAX_PLAN_LENGTH}). Truncating.")
            plan_list = plan_list[:MAX_PLAN_LENGTH]
            
        validated_plan = []
        for i, step in enumerate(plan_list):
            name = step.get("step_name", "").strip()
            detail = step.get("step_detail", "").strip()
            if not detail:
                 print(f"Warning: Step {i+1} ('{name}') has empty detail. Skipping.")
                 continue # Skip steps with no detail/query
            validated_plan.append({"step_name": name if name else f"Step {i+1}", "step_detail": detail})
        
        if not validated_plan:
             raise ValueError("Plan generation resulted in no valid steps after validation.")
             
        print(f"Planner generated plan (validated): {validated_plan}")
        print(llm.model_dump_json())
        ai_message = f"Generated a plan with {len(validated_plan)} steps."
        # --- End Step Validation ---
         
        return {
            "plan": validated_plan,
            "messages": history + [AIMessage(content=ai_message)],
            "error": None
        }

    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse plan JSON: {e}. Raw response: {plan_text}"
        print(error_msg)
        return {"error": error_msg, "plan": None, "messages": history}
    except ValueError as e:
        error_msg = f"Plan validation failed: {e}. Raw response: {plan_text}"
        print(error_msg)
        return {"error": error_msg, "plan": None, "messages": history}
    except Exception as e:
        error_msg = f"Unexpected error during planning: {e}"
        print(error_msg)
        return {"error": error_msg, "plan": None, "messages": history}
