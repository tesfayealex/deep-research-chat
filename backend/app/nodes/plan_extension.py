"""
Plan Extension Node for Deep Research Agent

This node extends the research plan based on reflection analysis and identified knowledge gaps.
"""

import json
import logging
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage

from ..schemas import AgentState
from ..config import settings
from ..models.model_factory import get_plan_extension_model

# Set up logging
logger = logging.getLogger(__name__)


async def plan_extension_node(state: AgentState) -> Dict[str, Any]:
    """
    Extend the research plan based on reflection analysis and identified knowledge gaps.
    
    This node:
    1. Analyzes reflection results and knowledge gaps
    2. Generates new research steps to address gaps
    3. Extends the existing plan while preserving completed work
    4. Updates step status and tracking
    
    Args:
        state: Current agent state with reflection results
        
    Returns:
        Updated state with extended plan
    """
    logger.info("📋 Starting plan extension based on reflection analysis...")
    
    # Determine which model is being used for plan extension
    plan_extension_model_name = getattr(settings, 'PLAN_EXTENSION_MODEL_NAME', settings.MAIN_MODEL_NAME)
    print(f"Using plan extension model: {plan_extension_model_name}")
    
    try:
        # Get reflection results
        reflection_results = state.get("reflection_results", [])
        if not reflection_results:
            logger.warning("⚠️ No reflection results found - cannot extend plan")
            return {"error": "No reflection results available for plan extension"}
        
        latest_reflection = reflection_results[-1]
        knowledge_gaps = latest_reflection.get("knowledge_gaps", [])
        proposed_steps = latest_reflection.get("proposed_steps", [])
        
        if not knowledge_gaps and not proposed_steps:
            logger.info("ℹ️ No knowledge gaps or proposed steps - plan extension not needed")
            return {"refinement_needed": False}
        
        # Initialize extension model using model factory
        extension_model = get_plan_extension_model(
            temperature=0.3  # Slightly higher temperature for creative planning
        )
        
        # Create extension prompt
        extension_prompt = _create_extension_prompt(state, latest_reflection)
        
        # Generate plan extension
        response = await extension_model.ainvoke([
            SystemMessage(content=_get_extension_system_prompt()),
            HumanMessage(content=extension_prompt)
        ])
        
        # Parse extension response
        extension_data = _parse_extension_response(response.content)
        
        # Apply plan extension
        updated_state = _apply_plan_extension(state, extension_data)
        
        # Log extension results
        new_steps_count = len(extension_data.get("new_steps", []))
        logger.info(f"📈 Plan extended with {new_steps_count} new research steps")
        logger.info(f"🎯 Total plan steps: {len(updated_state['plan'])}")
        
        return updated_state
        
    except Exception as e:
        logger.error(f"❌ Plan extension failed: {str(e)}")
        return {
            "error": f"Plan extension failed: {str(e)}",
            "refinement_needed": False,
            "completion_reason": "extension_error"
        }


def _create_extension_prompt(state: AgentState, reflection: Dict[str, Any]) -> str:
    """Create a prompt for plan extension based on reflection results."""
    
    initial_query = state.get("initial_query", "")
    current_plan = state.get("plan", [])
    step_results = state.get("step_results", [])
    knowledge_gaps = reflection.get("knowledge_gaps", [])
    proposed_steps = reflection.get("proposed_steps", [])
    quality_score = reflection.get("quality_score", 0)
    
    # Calculate remaining budget
    total_tool_calls = state.get("global_tool_calls", 0)
    max_total_calls = settings.MAX_TOOL_CALLS_PER_STEP * settings.MAX_STEPS
    remaining_calls = max_total_calls - total_tool_calls
    
    prompt = f"""
# RESEARCH PLAN EXTENSION

## Original Research Query
{initial_query}

## Current Research Plan
{json.dumps(current_plan, indent=2)}

## Completed Research Summary
Steps completed: {len(step_results)}
Current quality score: {quality_score}/100

## Reflection Analysis Results
### Knowledge Gaps Identified:
{json.dumps(knowledge_gaps, indent=2)}

### Proposed Research Steps:
{json.dumps(proposed_steps, indent=2)}

## Extension Constraints
- Remaining tool call budget: {remaining_calls}
- Maximum new steps recommended: {min(3, remaining_calls // settings.MIN_CALLS_FOR_EXTENSION)}
- Focus on highest priority gaps only

## Extension Requirements
Please generate new research steps that:

1. **Address Critical Gaps**: Focus on high-priority knowledge gaps that significantly impact research quality
2. **Complement Existing Work**: Build upon completed research without duplicating efforts
3. **Optimize Resource Usage**: Design efficient steps that maximize information gain within budget constraints
4. **Maintain Coherence**: Ensure new steps integrate logically with the existing research flow

## Step Design Guidelines
Each new step should:
- Have a clear, specific research objective (NOT generic like "Research more about X")
- Include a targeted, searchable query (NOT vague like "additional information")
- Specify expected information types
- Estimate tool calls required (typically 2-3 per step)
- Address specific knowledge gaps identified in the reflection

## Examples of GOOD vs BAD steps:

### GOOD Examples:
- description: "Analyze competitive pricing strategies for enterprise AI platforms"
  query: "enterprise AI platform pricing models comparison 2024 competitors"
  
- description: "Research regulatory compliance requirements for healthcare AI applications"
  query: "healthcare AI regulatory compliance FDA HIPAA requirements 2024"

### BAD Examples (DO NOT CREATE):
- description: "Research more about AI" (too generic)
  query: "AI information" (too vague)
  
- description: "Additional research step" (not specific)
  query: "find more details" (not searchable)

Please provide the extension plan in the specified JSON format with specific, actionable steps.
"""
    
    return prompt


def _get_extension_system_prompt() -> str:
    """Get the system prompt for plan extension."""
    
    return f"""
You are an expert research planner specializing in extending research plans based on quality analysis.

Your role is to:
1. Analyze identified knowledge gaps and their impact on research objectives
2. Design targeted research steps that efficiently address the most critical gaps
3. Optimize for information gain within available resource constraints
4. Ensure new steps complement rather than duplicate existing research

## Extension Principles:
- **Priority-Driven**: Focus on high-impact gaps that significantly improve research quality
- **Resource-Efficient**: Design steps that maximize information gain per tool call
- **Complementary**: Build upon existing research without redundancy
- **Specific**: Create targeted, actionable research objectives

## Budget Considerations:
- Available tool calls: Limited remaining budget
- Typical step cost: {settings.MAX_TOOL_CALLS_PER_STEP} tool calls
- Minimum viable extension: {settings.MIN_CALLS_FOR_EXTENSION} calls

## Step Requirements:
Each new step MUST have:
- **description**: A clear, specific research objective (e.g., "Analyze market competition for AI tools", "Research regulatory requirements for data privacy")
- **query**: A targeted, searchable query (e.g., "AI tool market competition analysis 2024", "GDPR data privacy compliance requirements")
- **expected_info**: What specific information you expect to find
- **priority**: high/medium/low based on impact
- **estimated_calls**: 1-3 tool calls needed
- **addresses_gaps**: Which specific gaps this step addresses

## Quality Guidelines:
- Descriptions should be actionable and specific (NOT generic like "Research more about X")
- Queries should be searchable and focused (NOT vague like "additional information")
- Each step should target a specific knowledge gap or research angle
- Avoid duplicate research already covered in existing steps

## Response Format:
```json
{{
    "extension_rationale": "detailed explanation of why these specific extensions address the most critical gaps and improve research quality",
    "new_steps": [
        {{
            "step": <next_step_number>,
            "description": "specific, actionable research objective (e.g., 'Analyze competitive landscape for enterprise AI solutions')",
            "query": "targeted search query (e.g., 'enterprise AI solutions market analysis competitors 2024')",
            "expected_info": "specific types of information expected (e.g., 'competitor names, market share, pricing models')",
            "priority": "high/medium/low",
            "estimated_calls": <1-3>,
            "addresses_gaps": ["specific gap 1", "specific gap 2"]
        }}
    ],
    "integration_notes": "how new steps connect with existing research and build upon completed work",
    "expected_improvement": "specific anticipated improvements to research quality and completeness"
}}
```

Focus on actionable, high-impact extensions with specific descriptions and targeted queries that will meaningfully improve the research outcome.
"""


def _parse_extension_response(response_content: str) -> Dict[str, Any]:
    """Parse the extension model response into structured data."""
    
    try:
        # Extract JSON from response
        start_idx = response_content.find('{')
        end_idx = response_content.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON found in response")
            
        json_str = response_content[start_idx:end_idx]
        extension_data = json.loads(json_str)
        
        # Validate and clean new steps
        new_steps = extension_data.get("new_steps", [])
        validated_steps = []
        
        for step in new_steps:
            if not isinstance(step, dict):
                continue
                
            description = step.get("description", "").strip()
            query = step.get("query", "").strip()
            
            # Validate description quality
            invalid_descriptions = [
                "extended research step", "additional research", "research more", 
                "step", "extended step", "more research", "further research"
            ]
            if not description or len(description) < 10:
                logger.warning(f"⚠️ Skipping step with too short description: '{description}'")
                continue
            if any(invalid_desc in description.lower() for invalid_desc in invalid_descriptions):
                logger.warning(f"⚠️ Skipping step with generic description: '{description}'")
                continue
                
            # Validate query quality
            invalid_queries = [
                "additional information", "more info", "research", "find", 
                "search", "additional research", "further details"
            ]
            if not query or len(query) < 5:
                logger.warning(f"⚠️ Skipping step with too short query: '{query}'")
                continue
            if any(invalid_q in query.lower() for invalid_q in invalid_queries):
                logger.warning(f"⚠️ Skipping step with generic query: '{query}'")
                continue
            
            # Create validated step
            validated_step = {
                "step": step.get("step", len(validated_steps) + 1),
                "description": description,
                "query": query,
                "status": "pending",
                "priority": step.get("priority", "medium"),
                "estimated_calls": min(step.get("estimated_calls", 2), settings.MAX_TOOL_CALLS_PER_STEP),
                "addresses_gaps": step.get("addresses_gaps", []),
                "expected_info": step.get("expected_info", "")
            }
            validated_steps.append(validated_step)
            logger.info(f"✅ Validated extension step: '{description}' with query: '{query}'")
        
        extension_data["new_steps"] = validated_steps
        
        if not validated_steps:
            logger.warning("⚠️ No valid steps found in extension response after validation")
        
        return extension_data
        
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"⚠️ Failed to parse extension response: {e}")
        
        # Return minimal extension
        return {
            "extension_rationale": "Failed to parse extension response",
            "new_steps": [],
            "integration_notes": "Extension parsing failed",
            "expected_improvement": "Unknown"
        }


def _apply_plan_extension(state: AgentState, extension_data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the plan extension to the current state."""
    
    # Preserve original plan if not already saved
    original_plan = state.get("original_plan")
    if original_plan is None:
        original_plan = state.get("plan", []).copy()
    
    # Get current plan and new steps
    current_plan = state.get("plan", [])
    new_steps = extension_data.get("new_steps", [])
    
    if not new_steps:
        logger.warning("⚠️ No new steps to add in extension")
        return {"refinement_needed": False}
    
    # Calculate next step numbers based on existing step numbers in the plan
    max_current_step = 0
    if current_plan:
        # Get the maximum step number from the current plan
        for step in current_plan:
            step_number = step.get("step", 0)
            if step_number > max_current_step:
                max_current_step = step_number
    
    # Add new steps to plan with meaningful names
    extended_plan = current_plan.copy()
    plan_extensions = []
    
    for i, new_step in enumerate(new_steps):
        step_number = max_current_step + i + 1
        
        # Extract meaningful step name and detailed query
        step_description = new_step.get("description", "")
        step_query = new_step.get("query", "")
        
        # Generate meaningful step name from description or query
        if step_description and not step_description.startswith("Extended step"):
            step_name = step_description
        elif step_query:
            # Extract key topic from query for meaningful name
            topic = _extract_topic_from_query(step_query)
            step_name = f"Research {topic}"
        else:
            step_name = f"Additional Research - Step {step_number}"
        
        # Ensure we have a detailed query
        if not step_query or len(step_query.strip()) < 3:
            if step_description:
                step_query = step_description
            else:
                step_query = f"Research additional information for step {step_number}"
        
        # Create step with correct field names expected by extractor
        extended_step = {
            "step": step_number,
            "step_name": step_name,  # This is what extractor expects
            "step_detail": step_query,  # This is what extractor expects
            "status": "pending",
            "priority": new_step.get("priority", "medium"),
            "source": "reflection_extension",
            "addresses_gaps": new_step.get("addresses_gaps", []),
            "expected_info": new_step.get("expected_info", "Additional research findings"),
            "estimated_calls": new_step.get("estimated_calls", 2)
        }
        
        extended_plan.append(extended_step)
        plan_extensions.append(extended_step)
    
    # Update step status for new steps using step numbers
    current_step_status = state.get("step_status", {})
    for step in plan_extensions:
        step_number = step["step"]
        current_step_status[step_number] = "pending"
        # Also log for debugging
        logger.info(f"   Setting step {step_number} status to 'pending'")
    
    # Prepare updated state
    updated_state = {
        "plan": extended_plan,
        "original_plan": original_plan,
        "plan_extensions": state.get("plan_extensions", []) + plan_extensions,
        "step_status": current_step_status,
        "refinement_needed": False,  # Extension complete, ready to continue execution
        "completion_reason": ""  # Clear any previous completion reason
    }
    
    logger.info(f"✅ Plan extended: {len(current_plan)} → {len(extended_plan)} steps")
    for ext_step in plan_extensions:
        logger.info(f"  📝 Added step {ext_step['step']}: {ext_step['step_name']}")
        logger.info(f"          Query: {ext_step['step_detail']}")
    
    return updated_state


def _extract_topic_from_query(query: str) -> str:
    """Extract a meaningful topic from a search query for step naming."""
    
    if not query:
        return "Additional Information"
    
    # Remove common search terms and focus on key topics
    stop_words = {
        "what", "how", "when", "where", "why", "who", "is", "are", "the", "of", "in", "on", "at", 
        "to", "for", "with", "by", "and", "or", "but", "a", "an", "this", "that", "these", "those",
        "find", "search", "research", "analysis", "information", "about", "regarding", "concerning"
    }
    
    # Clean and split query
    words = query.lower().replace("?", "").replace(",", "").replace(".", "").split()
    meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    if meaningful_words:
        # Take first 2-3 meaningful words to create topic
        topic_words = meaningful_words[:3]
        topic = " ".join(topic_words).title()
        
        # Add context if it's too short
        if len(topic) < 10 and len(meaningful_words) > 3:
            topic_words = meaningful_words[:4]
            topic = " ".join(topic_words).title()
            
        return topic
    
    # Fallback: use first few words of original query, cleaned up
    cleaned_query = query.strip()
    if len(cleaned_query) > 30:
        # Truncate long queries for step names
        return cleaned_query[:27] + "..."
    
    return cleaned_query.title()


def get_extension_summary(state: AgentState) -> Dict[str, Any]:
    """Get a summary of plan extensions performed."""
    
    original_plan = state.get("original_plan", [])
    current_plan = state.get("plan", [])
    plan_extensions = state.get("plan_extensions", [])
    reflection_count = state.get("reflection_count", 0)
    
    return {
        "original_steps": len(original_plan),
        "current_steps": len(current_plan),
        "extensions_added": len(plan_extensions),
        "reflection_cycles": reflection_count,
        "extension_details": plan_extensions
    } 