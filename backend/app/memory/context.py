# backend/app/memory/context.py

# Currently, the AgentState in schemas.py manages the core memory components 
# (e.g., step_results list).

# If more complex memory management (like summarization of past steps, 
# or specific context objects per step) is needed later, 
# define classes like StepContext and ConversationMemory here.

# Example structure (if needed later):
# from typing import List, Dict, Any

# class StepContext:
#     def __init__(self, step_name: str, step_detail: str):
#         self.name = step_name
#         self.detail = step_detail
#         self.search_results_raw: str | None = None
#         self.extracted_contents: List[Dict[str, Any]] = [] # {"text": ..., "source": ...}
#         self.findings: str | None = None
#         self.sources: List[str] = []
#         self.errors: List[str] = []
#         self.token_usage: Dict[str, int] = {}

# class ConversationMemory:
#     def __init__(self, initial_query: str):
#         self.initial_query = initial_query
#         self.steps: List[StepContext] = []
#         self.global_token_usage: Dict[str, int] = {}

#     def add_step(self, step: StepContext):
#         self.steps.append(step)

#     def get_full_context_for_reporting(self) -> str:
#         # Logic to compile context from all steps
#         pass
