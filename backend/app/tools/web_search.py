from langchain_community.tools.tavily_search import TavilySearchResults
from ..config import settings
from typing import List, Any, Dict
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
        print(result)
        wefewfwefew
        return str(result) 
    except Exception as e:
        print(f"Error during Tavily search for '{query}': {e}")
        return f"Error performing web search: {e}"

def perform_web_search_direct(query: str, state_name: str) -> str:
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

        # print(response.text)

         # resolve the urls to short urls for saving tokens and time
        resolved_urls = resolve_urls(
            response.candidates[0].grounding_metadata.grounding_chunks, state_name
        )
        # Gets the citations and adds them to the generated text
        citations = get_citations(response, resolved_urls)
        modified_text = insert_citation_markers(response.text, citations)
        sources_gathered = [item for citation in citations for item in citation["segments"]]
        print(f"Direct search for '{query}' completed. Usage: {response.usage_metadata}")
        print(sources_gathered)
        print("########################################################")
        print(modified_text)
        print("########################################################")
        # print(citations)
        wefffewfe
        return response.text
        
    except Exception as e:
        print(f"Error during direct search for '{query}': {e}")
        return f"Error performing direct web search: {e}"

async def perform_web_search(query: str, max_results: int | None = None, state_name: str = "") -> str:
    """Unified search interface that switches between methods based on configuration."""
    search_method = settings.SEARCH_METHOD.lower()
    
    if search_method == "direct":
        print(f"Using direct search method for: '{query}'")
        try:
            # Direct search is synchronous, but we're in an async context
            result = perform_web_search_direct(query, state_name)
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
    
def resolve_urls(urls_to_resolve: List[Any], id: int) -> Dict[str, str]:
    """
    Create a map of the vertex ai search urls (very long) to a short url with a unique id for each url.
    Ensures each original URL gets a consistent shortened form while maintaining uniqueness.
    """
    prefix = f"https://vertexaisearch.cloud.google.com/id/"
    urls = [site.web.uri for site in urls_to_resolve]

    # Create a dictionary that maps each unique URL to its first occurrence index
    resolved_map = {}
    for idx, url in enumerate(urls):
        if url not in resolved_map:
            resolved_map[url] = f"{prefix}{id}-{idx}"

    return resolved_map

def get_citations(response, resolved_urls_map):
    """
    Extracts and formats citation information from a Gemini model's response.

    This function processes the grounding metadata provided in the response to
    construct a list of citation objects. Each citation object includes the
    start and end indices of the text segment it refers to, and a string
    containing formatted markdown links to the supporting web chunks.

    Args:
        response: The response object from the Gemini model, expected to have
                  a structure including `candidates[0].grounding_metadata`.
                  It also relies on a `resolved_map` being available in its
                  scope to map chunk URIs to resolved URLs.

    Returns:
        list: A list of dictionaries, where each dictionary represents a citation
              and has the following keys:
              - "start_index" (int): The starting character index of the cited
                                     segment in the original text. Defaults to 0
                                     if not specified.
              - "end_index" (int): The character index immediately after the
                                   end of the cited segment (exclusive).
              - "segments" (list[str]): A list of individual markdown-formatted
                                        links for each grounding chunk.
              - "segment_string" (str): A concatenated string of all markdown-
                                        formatted links for the citation.
              Returns an empty list if no valid candidates or grounding supports
              are found, or if essential data is missing.
    """
    citations = []

    # Ensure response and necessary nested structures are present
    if not response or not response.candidates:
        return citations

    candidate = response.candidates[0]
    if (
        not hasattr(candidate, "grounding_metadata")
        or not candidate.grounding_metadata
        or not hasattr(candidate.grounding_metadata, "grounding_supports")
    ):
        return citations

    for support in candidate.grounding_metadata.grounding_supports:
        citation = {}

        # Ensure segment information is present
        if not hasattr(support, "segment") or support.segment is None:
            continue  # Skip this support if segment info is missing

        start_index = (
            support.segment.start_index
            if support.segment.start_index is not None
            else 0
        )

        # Ensure end_index is present to form a valid segment
        if support.segment.end_index is None:
            continue  # Skip if end_index is missing, as it's crucial

        # Add 1 to end_index to make it an exclusive end for slicing/range purposes
        # (assuming the API provides an inclusive end_index)
        citation["start_index"] = start_index
        citation["end_index"] = support.segment.end_index

        citation["segments"] = []
        if (
            hasattr(support, "grounding_chunk_indices")
            and support.grounding_chunk_indices
        ):
            for ind in support.grounding_chunk_indices:
                try:
                    chunk = candidate.grounding_metadata.grounding_chunks[ind]
                    resolved_url = resolved_urls_map.get(chunk.web.uri, None)
                    citation["segments"].append(
                        {
                            "label": chunk.web.title.split(".")[:-1][0],
                            "short_url": resolved_url,
                            "value": chunk.web.uri,
                        }
                    )
                except (IndexError, AttributeError, NameError):
                    # Handle cases where chunk, web, uri, or resolved_map might be problematic
                    # For simplicity, we'll just skip adding this particular segment link
                    # In a production system, you might want to log this.
                    pass
        citations.append(citation)
    return citations

def insert_citation_markers(text, citations_list):
    """
    Inserts citation markers into a text string based on start and end indices.

    Args:
        text (str): The original text string.
        citations_list (list): A list of dictionaries, where each dictionary
                               contains 'start_index', 'end_index', and
                               'segment_string' (the marker to insert).
                               Indices are assumed to be for the original text.

    Returns:
        str: The text with citation markers inserted.
    """
    # Sort citations by end_index in descending order.
    # If end_index is the same, secondary sort by start_index descending.
    # This ensures that insertions at the end of the string don't affect
    # the indices of earlier parts of the string that still need to be processed.
    sorted_citations = sorted(
        citations_list, key=lambda c: (c["end_index"], c["start_index"]), reverse=True
    )

    modified_text = text
    for citation_info in sorted_citations:
        # These indices refer to positions in the *original* text,
        # but since we iterate from the end, they remain valid for insertion
        # relative to the parts of the string already processed.
        end_idx = citation_info["end_index"]
        marker_to_insert = ""
        for segment in citation_info["segments"]:
            marker_to_insert += f" [{segment['label']}]({segment['short_url']})"
        # Insert the citation marker at the original end_idx position
        modified_text = (
            modified_text[:end_idx] + marker_to_insert + modified_text[end_idx:]
        )

    return modified_text
