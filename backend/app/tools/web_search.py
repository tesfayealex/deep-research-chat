import asyncio
import datetime
from typing import Any, Dict, List, Tuple

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..config import settings
from ..models.model_factory import get_main_model
from ..tools.url_extractor import extract_text_from_url

# Import for direct search method
try:
    from google.genai import Client
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google.genai not available. Direct search method will not work.")

def get_tavily_tool() -> TavilySearchResults:
    """Initializes and returns the Tavily search tool."""
    return TavilySearchResults(
        api_key=settings.TAVILY_API_KEY,
        max_results=settings.TAVILY_MAX_RESULTS,
        search_depth="advanced",
    )

async def perform_web_search_tavily(
    query: str, max_results: int | None = None, state_name: str = ""
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Performs a Tavily search, extracts content from URLs, synthesizes it with citations,
    and returns the result and sources.
    """
    search_tool = get_tavily_tool()
    if max_results:
        search_tool.max_results = max_results

    try:
        tavily_results = await search_tool.ainvoke({"query": query})
        if not tavily_results:
            print(f"Tavily search for '{query}' returned no results.")
            return "No search results found.", []

        print(f"Tavily search for '{query}' returned {len(tavily_results)} results.")
        urls_to_process = [res["url"] for res in tavily_results if res.get("url")]
        url_metadata = {
            res["url"]: {"title": res.get("title", "No Title")}
            for res in tavily_results
            if res.get("url")
        }

        if not urls_to_process:
            return "Search results contained no usable URLs.", []

        resolved_urls_map = resolve_urls(urls_to_process, state_name, source="tavily")
        print(f"Extracting content from {len(urls_to_process)} URLs...")
        extraction_tasks = [extract_text_from_url(url) for url in urls_to_process]
        extraction_results = await asyncio.gather(
            *extraction_tasks, return_exceptions=True
        )

        combined_content = ""
        valid_extractions = []
        for i, res in enumerate(extraction_results):
            url = urls_to_process[i]
            if isinstance(res, dict) and res.get("text") and not res.get("error"):
                short_url = resolved_urls_map.get(url)
                title = url_metadata.get(url, {}).get("title", "Source")
                combined_content += f'Source: [{title}]({short_url})\nURL: {url}\nContent:\n{res["text"]}\n\n---\n\n'
                valid_extractions.append(
                    {
                        "url": url,
                        "short_url": short_url,
                        "title": title,
                        "content": res["text"],
                    }
                )
            else:
                print(f"Failed to extract content from {url}: {res}")

        if not combined_content:
            return "Could not extract any content from the search result URLs.", []

        synthesis_model = get_main_model()
        synthesis_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=f'You are an expert research analyst. Your task is to synthesize the provided text, which has been extracted from various web sources.\n\nHere is the user\'s original query: "{query}"\n\nInstructions:\n1.  Analyze the provided content, which includes the source URL (shortened) and the extracted text.\n2.  Create a comprehensive, synthesized response that directly answers the user\'s query.\n3.  **Crucially, you must cite your sources.** When you use information from a source, add a citation marker in markdown format, like this: `[Source Title](short_url)`.\n4.  Base your response ONLY on the information provided. Do not add any external knowledge.\n5.  If the provided content is insufficient to answer the query, state that clearly.\n6.  The output should be a well-structured and readable report.'
                ),
                HumanMessage(
                    content=f"Here is the content to synthesize:\n\n{combined_content}"
                ),
            ]
        )

        chain = synthesis_prompt | synthesis_model | StrOutputParser()
        synthesized_text = await chain.ainvoke({})
        sources_list = [
            {
                "label": ve["title"],
                "short_url": ve["short_url"],
                "original_url": ve["url"],
            }
            for ve in valid_extractions
        ]

        print(f"Tavily synthesis for '{query}' completed.")
        return synthesized_text, sources_list

    except Exception as e:
        print(f"Error during Tavily search and synthesis for '{query}': {e}")
        return f"An error occurred: {e}", []

def perform_web_search_direct(
    query: str, state_name: str
) -> Tuple[str, List[Dict[str, Any]]]:
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

        urls_to_resolve = [
            chunk.web.uri
            for chunk in response.candidates[0].grounding_metadata.grounding_chunks
        ]
        resolved_urls = resolve_urls(urls_to_resolve, state_name, source="gemini")
        citations = get_citations(response, resolved_urls)
        modified_text = insert_citation_markers(response.text, citations)
        sources_gathered = [
            {
                "label": item["label"],
                "short_url": item["short_url"],
                "original_url": item["value"],
            }
            for citation in citations
            for item in citation["segments"]
        ]

        unique_sources = []
        seen_urls = set()
        for source in sources_gathered:
            if source["original_url"] not in seen_urls:
                unique_sources.append(source)
                seen_urls.add(source["original_url"])

        print(f"Direct search for '{query}' completed. Usage: {response.usage_metadata}")
        print(sources_gathered)
        print("########################################################")
        print(modified_text)
        print("########################################################")
        return modified_text, unique_sources
        
    except Exception as e:
        print(f"Error during direct search for '{query}': {e}")
        return f"Error performing direct web search: {e}", []

async def perform_web_search(
    query: str, max_results: int | None = None, state_name: str = ""
) -> Tuple[str, List[Dict[str, Any]]]:
    """Unified search interface that switches between methods based on configuration."""
    search_method = settings.SEARCH_METHOD.lower()
    
    if search_method == "direct":
        print(f"Using direct search method for: '{query}'")
        try:
            loop = asyncio.get_running_loop()
            result_text, result_sources = await loop.run_in_executor(
                None, perform_web_search_direct, query, state_name
            )
            return result_text, result_sources
        except Exception as e:
            print(f"Direct search failed for '{query}': {e}")
            # Fallback to Tavily if direct search fails
            print("Falling back to Tavily search...")
            return await perform_web_search_tavily(query, max_results, state_name)
    
    elif search_method == "subquery":
        print(f"Using subquery search method for: '{query}'")
        return await perform_web_search_tavily(query, max_results, state_name)
    
    else:
        print(f"Unknown search method '{search_method}', defaulting to subquery method")
        return await perform_web_search_tavily(query, max_results, state_name)
    
def resolve_urls(
    urls_to_resolve: List[str], id: str, source: str = "gemini"
) -> Dict[str, str]:
    """
    Create a map of URLs to a short url with a unique id for each url.
    """
    prefix = (
        "https://tavily.com/id/"
        if source == "tavily"
        else "https://vertexaisearch.cloud.google.com/id/"
    )
    id_str = str(id)
    resolved_map = {}
    for idx, url in enumerate(
        list(dict.fromkeys(urls_to_resolve))
    ):  # Deduplicate URLs while preserving order
        if url not in resolved_map:
            resolved_map[url] = f"{prefix}{id_str}-{idx}"
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
