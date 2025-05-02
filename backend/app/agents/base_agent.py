from abc import ABC, abstractmethod
from typing import Dict

from ..schemas import AgentState

class BaseAgent(ABC):
    """Abstract base class for all agents in the research pipeline."""

    agent_name: str = "BaseAgent"

    @abstractmethod
    async def run(self, state: AgentState) -> Dict:
        """
        Executes the agent's logic based on the current state.

        Args:
            state: The current state of the research process.

        Returns:
            A dictionary containing the updates to be merged back into the state.
            Example: {"plan": [...], "messages": [...], "error": None}
                 or {"step_results": [...], "messages": [...], "error": None}
                 or {"report": "...", "messages": [...], "error": None}
        """
        pass

    def __str__(self):
        return self.agent_name

    def __repr__(self):
        return f"<{self.agent_name}>"
