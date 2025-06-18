"""
Reflection Node for Deep Research Agent

This node analyzes completed research, identifies knowledge gaps, and determines 
if plan extension is needed based on quality assessment.
"""

import json
import logging
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage

from ..schemas import AgentState
from ..config import settings
from ..models.model_factory import get_reflection_model

# Set up logging
logger = logging.getLogger(__name__)


async def reflection_node(state: AgentState) -> Dict[str, Any]:
    """
    Mandatory reflection step that analyzes completed research and determines next actions.
    
    This node:
    1. Analyzes all completed step results
    2. Identifies knowledge gaps and quality issues
    3. Determines if plan extension is needed
    4. Provides quality assessment
    
    Args:
        state: Current agent state with completed step results
        
    Returns:
        Updated state with reflection results and recommendations
    """
    logger.info("🔍 Starting mandatory reflection analysis...")
    print(f"Using reflection model: {settings.REFLECTION_MODEL_NAME}")
    
    try:
        # Initialize reflection model using model factory
        reflection_model = get_reflection_model(
            temperature=0.1  # Lower temperature for more consistent analysis
        )
        
        # Prepare reflection analysis
        reflection_prompt = _create_reflection_prompt(state)
        
        # Perform reflection analysis
        response = await reflection_model.ainvoke([
            SystemMessage(content=_get_reflection_system_prompt()),
            HumanMessage(content=reflection_prompt)
        ])
        
        # Parse reflection results
        reflection_data = _parse_reflection_response(response.content)
        
        # Update state with reflection results
        updated_state = {
            "reflection_results": state.get("reflection_results", []) + [reflection_data],
            "knowledge_gaps": reflection_data.get("knowledge_gaps", []),
            "quality_score": reflection_data.get("quality_score", 0),
            "reflection_count": state.get("reflection_count", 0) + 1,
            "refinement_needed": reflection_data.get("needs_extension", False)
        }
        
        # Log reflection results
        logger.info(f"📊 Reflection completed - Quality Score: {reflection_data.get('quality_score', 0)}/100")
        logger.info(f"🔍 Knowledge gaps identified: {len(reflection_data.get('knowledge_gaps', []))}")
        logger.info(f"📈 Extension needed: {reflection_data.get('needs_extension', False)}")
        
        return updated_state
        
    except Exception as e:
        logger.error(f"❌ Reflection analysis failed: {str(e)}")
        return {
            "error": f"Reflection failed: {str(e)}",
            "refinement_needed": False,
            "quality_score": 0,
            "completion_reason": "reflection_error"
        }


def _create_reflection_prompt(state: AgentState) -> str:
    """Create a comprehensive reflection prompt based on current state."""
    
    # Gather completed research data
    initial_query = state.get("initial_query", "")
    plan = state.get("plan", [])
    step_results = state.get("step_results", [])
    current_iteration = state.get("iterations", 0)
    total_tool_calls = state.get("global_tool_calls", 0)
    
    # Build research summary
    research_summary = []
    for result in step_results:
        if hasattr(result, 'step_index'):
            research_summary.append({
                "step": result.step_index,
                "query": result.query,
                "summary": result.summary,
                "sources_count": len(result.sources),
                "tool_calls": result.tool_calls_made,
                "error": result.error
            })
        else:
            # Handle dict format for backward compatibility
            research_summary.append(result)
    
    prompt = f"""
# RESEARCH REFLECTION ANALYSIS

## Original Research Query
{initial_query}

## Research Plan Executed
{json.dumps(plan, indent=2)}

## Research Results Summary
{json.dumps(research_summary, indent=2)}

## Current Status
- Iteration: {current_iteration}
- Total Tool Calls Used: {total_tool_calls}
- Steps Completed: {len(step_results)}
- Remaining Tool Budget: {settings.MAX_TOOL_CALLS_PER_STEP * settings.MAX_STEPS - total_tool_calls}

## Analysis Required
Please analyze the completed research and provide:

1. **Quality Assessment**: Rate the overall research quality (0-100) based on:
   - Comprehensiveness of coverage
   - Depth of information gathered
   - Relevance to original query
   - Source diversity and reliability

2. **Knowledge Gap Analysis**: Identify specific areas where information is:
   - Missing or incomplete
   - Contradictory or unclear
   - Requiring deeper investigation
   - Needing additional perspectives

3. **Extension Recommendation**: Determine if plan extension is justified based on:
   - Significant knowledge gaps that impact query fulfillment
   - Available tool budget for additional research
   - Potential for meaningful improvement

4. **Proposed Extensions**: If extension is recommended, suggest specific new research steps.

Please respond in the specified JSON format.
"""
    
    return prompt


def _get_reflection_system_prompt() -> str:
    """Get the system prompt for research reflection."""
    
    return f"""
You are an expert research analyst specializing in comprehensive research quality assessment.

Your role is to conduct thorough analysis of research outcomes, focusing on:

## Core Analysis Areas:

### 1. Question-Answer Alignment Analysis
- **Coverage Assessment**: How well do the research findings address each original question?
- **Depth Analysis**: Are answers superficial or comprehensive?
- **Specificity Check**: Are answers specific enough or too general?
- **Evidence Quality**: Is there sufficient evidence supporting each answer?

### 2. Knowledge Gap Identification
- **Missing Information**: What critical information is absent?
- **Incomplete Areas**: Which questions received partial answers?
- **Unexplored Angles**: What perspectives or approaches weren't considered?
- **Data Limitations**: Where is more data needed for complete understanding?

### 3. Research Quality Evaluation
- **Source Diversity**: Are sources varied and authoritative?
- **Information Recency**: Is the information current and relevant?
- **Methodological Soundness**: Are research approaches appropriate?
- **Bias Assessment**: Are there potential biases in sources or analysis?

### 4. Improvement Recommendations
- **Priority Gaps**: Which gaps are most critical to address?
- **Research Strategies**: What approaches would fill important gaps?
- **Resource Optimization**: How to maximize information gain efficiently?

## Quality Scoring Guidelines:
- **90-100**: Exceptional - Comprehensive coverage, high-quality sources, minimal gaps
- **80-89**: Strong - Good coverage with minor gaps, solid evidence base
- **70-79**: Adequate - Covers main points but has notable gaps or limitations
- **60-69**: Weak - Significant gaps, limited coverage, or quality issues
- **Below 60**: Poor - Major deficiencies requiring substantial additional research

## Response Format:
```json
{{
    "overall_quality_score": <0-100>,
    "question_answer_analysis": [
        {{
            "original_question": "the research question or objective",
            "coverage_assessment": "how well this was addressed",
            "answer_quality": "depth and specificity of findings",
            "evidence_strength": "quality and quantity of supporting evidence",
            "coverage_score": <0-100>,
            "identified_gaps": ["specific gap 1", "specific gap 2"],
            "improvement_needs": "what would make this answer better"
        }}
    ],
    "critical_knowledge_gaps": [
        {{
            "gap_description": "detailed description of what's missing",
            "impact_level": "high/medium/low",
            "why_important": "why this gap matters for the research objectives",
            "suggested_approach": "how to address this gap",
            "estimated_effort": "resource requirements to fill this gap"
        }}
    ],
    "research_quality_assessment": {{
        "source_diversity_score": <0-100>,
        "information_recency_score": <0-100>,
        "evidence_strength_score": <0-100>,
        "coverage_completeness_score": <0-100>,
        "overall_methodology_score": <0-100>
    }},
    "extension_recommendations": [
        {{
            "priority": "high/medium/low",
            "focus_area": "what to research next",
            "specific_questions": ["question 1", "question 2"],
            "expected_impact": "how this would improve research quality",
            "resource_estimate": "estimated tool calls needed"
        }}
    ],
    "summary_assessment": "overall evaluation of research completeness and quality"
}}
```

Provide detailed, actionable analysis that clearly identifies what's missing and why it matters.
"""


def _parse_reflection_response(response_content: str) -> Dict[str, Any]:
    """Parse the reflection model response into structured data."""
    
    try:
        # Extract JSON from response
        start_idx = response_content.find('{')
        end_idx = response_content.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON found in response")
            
        json_str = response_content[start_idx:end_idx]
        reflection_data = json.loads(json_str)
        
        # Extract key metrics with new format
        quality_score = reflection_data.get("overall_quality_score", 50)
        
        # Parse question-answer analysis
        qa_analysis = reflection_data.get("question_answer_analysis", [])
        
        # Parse critical knowledge gaps
        knowledge_gaps = reflection_data.get("critical_knowledge_gaps", [])
        
        # Parse quality assessment details
        quality_assessment = reflection_data.get("research_quality_assessment", {})
        
        # Parse extension recommendations
        extension_recommendations = reflection_data.get("extension_recommendations", [])
        
        # Determine if extension is needed based on quality score and gaps
        needs_extension = (
            quality_score < settings.QUALITY_THRESHOLD or
            any(gap.get("impact_level") == "high" for gap in knowledge_gaps) or
            any(rec.get("priority") == "high" for rec in extension_recommendations)
        )
        
        # Create structured knowledge gaps for backward compatibility
        structured_gaps = []
        for gap in knowledge_gaps:
            structured_gaps.append({
                "area": gap.get("gap_description", "Unknown gap")[:50],
                "description": gap.get("gap_description", "Gap not specified"),
                "priority": gap.get("impact_level", "medium"),
                "suggested_approach": gap.get("suggested_approach", "Additional research needed"),
                "why_important": gap.get("why_important", "Impact not specified"),
                "estimated_effort": gap.get("estimated_effort", "Unknown")
            })
        
        # Create proposed steps from extension recommendations
        proposed_steps = []
        for rec in extension_recommendations:
            if rec.get("priority") in ["high", "medium"]:
                for question in rec.get("specific_questions", []):
                    proposed_steps.append({
                        "description": f"Research: {rec.get('focus_area', 'Additional investigation')}",
                        "query": question,
                        "priority": rec.get("priority", "medium"),
                        "expected_impact": rec.get("expected_impact", "Improve research quality"),
                        "resource_estimate": rec.get("resource_estimate", "2-3 calls")
                    })
        
        # Return structured reflection data
        return {
            "quality_score": quality_score,
            "quality_assessment": reflection_data.get("summary_assessment", f"Research quality score: {quality_score}/100"),
            "question_answer_analysis": qa_analysis,
            "knowledge_gaps": structured_gaps,
            "research_quality_details": quality_assessment,
            "extension_recommendations": extension_recommendations,
            "needs_extension": needs_extension,
            "extension_justification": _create_extension_justification(quality_score, knowledge_gaps, qa_analysis),
            "proposed_steps": proposed_steps,
            "completion_recommendation": "extend" if needs_extension else "finalize",
            "detailed_analysis": reflection_data  # Keep full analysis for reference
        }
        
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"⚠️ Failed to parse reflection response: {e}")
        
        # Return fallback reflection
        return {
            "quality_score": 50,
            "quality_assessment": "Failed to parse reflection analysis",
            "question_answer_analysis": [],
            "knowledge_gaps": [{"area": "Analysis", "description": "Reflection parsing failed", "priority": "medium", "suggested_approach": "Retry analysis"}],
            "research_quality_details": {},
            "extension_recommendations": [],
            "needs_extension": True,
            "extension_justification": "Reflection analysis failed - extension recommended for safety",
            "proposed_steps": [],
            "completion_recommendation": "extend",
            "detailed_analysis": {}
        }


def _create_extension_justification(quality_score: int, knowledge_gaps: List[Dict], qa_analysis: List[Dict]) -> str:
    """Create a detailed justification for extension recommendation."""
    
    if quality_score >= settings.QUALITY_THRESHOLD:
        high_impact_gaps = [gap for gap in knowledge_gaps if gap.get("impact_level") == "high"]
        if not high_impact_gaps:
            return f"Quality score {quality_score}/100 meets threshold. No extension needed."
    
    reasons = []
    
    if quality_score < settings.QUALITY_THRESHOLD:
        reasons.append(f"Quality score {quality_score}/100 below threshold of {settings.QUALITY_THRESHOLD}")
    
    high_impact_gaps = [gap for gap in knowledge_gaps if gap.get("impact_level") == "high"]
    if high_impact_gaps:
        reasons.append(f"{len(high_impact_gaps)} high-impact knowledge gaps identified")
    
    low_coverage_questions = [qa for qa in qa_analysis if qa.get("coverage_score", 100) < 70]
    if low_coverage_questions:
        reasons.append(f"{len(low_coverage_questions)} questions have inadequate coverage")
    
    if not reasons:
        return "Extension recommended based on analysis"
    
    return "Extension needed: " + "; ".join(reasons)


def should_extend_plan(state: AgentState) -> bool:
    """
    Determine if the plan should be extended based on reflection results and constraints.
    
    Args:
        state: Current agent state with reflection results
        
    Returns:
        True if plan should be extended, False otherwise
    """
    
    # Check if reflection indicates extension is needed
    reflection_results = state.get("reflection_results", [])
    if not reflection_results:
        return False
    
    latest_reflection = reflection_results[-1]
    needs_extension = latest_reflection.get("needs_extension", False)
    
    if not needs_extension:
        return False
    
    # Check constraints
    current_iterations = state.get("iterations", 0)
    reflection_count = state.get("reflection_count", 0)
    total_tool_calls = state.get("global_tool_calls", 0)
    
    # Check maximum iterations
    if current_iterations >= settings.MAX_ITERATIONS:
        logger.info("🚫 Maximum iterations reached - cannot extend plan")
        return False
    
    # Check maximum reflection extensions
    if reflection_count > settings.MAX_REFLECTION_EXTENSIONS:
        logger.info("🚫 Maximum reflection extensions reached - cannot extend plan")
        return False
    
    # Check remaining tool budget
    max_total_calls = settings.MAX_TOOL_CALLS_PER_STEP * settings.MAX_STEPS
    remaining_calls = max_total_calls - total_tool_calls
    
    if remaining_calls < settings.MIN_CALLS_FOR_EXTENSION:
        logger.info(f"🚫 Insufficient tool budget remaining ({remaining_calls}) - cannot extend plan")
        return False
    
    logger.info("✅ Plan extension approved based on reflection analysis")
    return True 