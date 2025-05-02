from langchain_community.tools.tavily_search import TavilySearchResults
from ..config import settings

def get_tavily_tool() -> TavilySearchResults:
    """Initializes and returns the Tavily search tool based on settings."""
    # TODO: Add error handling for API key missing
    # TODO: Potentially allow overriding max_results via function args
    print(f"Initializing Tavily Search with max_results={settings.TAVILY_MAX_RESULTS}")
    return TavilySearchResults(max_results=settings.TAVILY_MAX_RESULTS)

# You might want a function to directly invoke the search as well
async def perform_web_search(query: str, max_results: int | None = None) -> str:
    """Performs a Tavily search for the given query."""
    search_tool = get_tavily_tool()
    if max_results:
        search_tool.max_results = max_results # Allow override
    
    try:
        result = await search_tool.ainvoke({"query": query})
        # Result format might vary, adjust extraction if needed.
        # Often it's a list of dicts or a string summary.
        print(f"Tavily search for '{query}' returned: {str(result)[:200]}...")
        # Ensure a consistent string output for the agent
        return str(result) 
    except Exception as e:
        print(f"Error during Tavily search for '{query}': {e}")
        return f"Error performing web search: {e}"
