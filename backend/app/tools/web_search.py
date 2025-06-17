from langchain_community.tools.tavily_search import TavilySearchResults
from ..config import settings
import datetime
import os

# Import for direct search method
try:
    from google.genai import Client
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google.genai not available. Direct search method will not work.")

def get_tavily_tool() -> TavilySearchResults:
    """Initializes and returns the Tavily search tool based on settings."""
    # TODO: Add error handling for API key missing
    # TODO: Potentially allow overriding max_results via function args
    print(f"Initializing Tavily Search with max_results={settings.TAVILY_MAX_RESULTS}")
    return TavilySearchResults(max_results=settings.TAVILY_MAX_RESULTS)

async def perform_web_search_tavily(query: str, max_results: int | None = None) -> str:
    """Performs a Tavily search for the given query (original subquery method)."""
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

def perform_web_search_direct(query: str) -> str:
    """Performs a direct search using Gemini's Google Search API tool."""
    if not GEMINI_AVAILABLE:
        raise ImportError("google.genai package is required for direct search method")
    
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        raise ValueError("GEMINI_API_KEY is not properly configured")
    
    try:
        # Initialize Gemini client
        genai_client = Client(api_key=settings.GEMINI_API_KEY)
        
        # Format the search prompt
        web_searcher_instructions = """Conduct targeted Google Searches to gather the most recent, credible information on "{research_topic}" and synthesize it into a verifiable text artifact.

Instructions:
- Query should ensure that the most current information is gathered. The current date is {current_date}.
- Conduct multiple, diverse searches to gather comprehensive information.
- Consolidate key findings while meticulously tracking the source(s) for each specific piece of information.
- The output should be a well-written summary or report based on your search findings. 
- Only include the information found in the search results, don't make up any information.

Research Topic:
{research_topic}
"""
        
        formatted_prompt = web_searcher_instructions.format(
            current_date=datetime.datetime.now().strftime("%B %d, %Y"),
            research_topic=query,
        )

        # Perform the search using Gemini with Google Search tool
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=formatted_prompt,
            config={
                "tools": [{"google_search": {}}],
                "temperature": 0,
            },
        )
        
        print(f"Direct search for '{query}' completed. Usage: {response.usage_metadata}")
        return response.text
        
    except Exception as e:
        print(f"Error during direct search for '{query}': {e}")
        return f"Error performing direct web search: {e}"

async def perform_web_search(query: str, max_results: int | None = None) -> str:
    """Unified search interface that switches between methods based on configuration."""
    search_method = settings.SEARCH_METHOD.lower()
    
    if search_method == "direct":
        print(f"Using direct search method for: '{query}'")
        try:
            # Direct search is synchronous, but we're in an async context
            result = perform_web_search_direct(query)
            return result
        except Exception as e:
            print(f"Direct search failed for '{query}': {e}")
            # Fallback to Tavily if direct search fails
            print("Falling back to Tavily search...")
            return await perform_web_search_tavily(query, max_results)
    
    elif search_method == "subquery":
        print(f"Using subquery search method for: '{query}'")
        return await perform_web_search_tavily(query, max_results)
    
    else:
        print(f"Unknown search method '{search_method}', defaulting to subquery method")
        return await perform_web_search_tavily(query, max_results)
