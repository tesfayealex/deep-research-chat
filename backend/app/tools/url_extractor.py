import httpx
from bs4 import BeautifulSoup
from typing import Dict, Optional
import asyncio

# Configure a shared async client for potential reuse and connection pooling
# Set appropriate timeouts
httpx_client = httpx.AsyncClient(follow_redirects=True, timeout=15.0)

async def extract_text_from_url(url: str) -> Dict[str, Optional[str]]:
    """
    Fetches content from a URL and extracts clean text using BeautifulSoup.
    
    Args:
        url: The URL to fetch and extract text from.
        
    Returns:
        A dictionary containing:
        - "text": The extracted text (str) or None if extraction failed.
        - "source": The original URL (str).
        - "error": An error message (str) if fetching or parsing failed, otherwise None.
    """
    print(f"Attempting to extract text from URL: {url}")
    try:
        response = await httpx_client.get(url)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        # Check content type - avoid parsing non-HTML content
        content_type = response.headers.get("content-type", "").lower()
        if "html" not in content_type:
            print(f"Skipping non-HTML content ({content_type}) at URL: {url}")
            return {"text": None, "source": url, "error": f"Skipped non-HTML content: {content_type}"}
            
        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Basic text extraction (can be refined)
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        
        # Get text, remove leading/trailing whitespace, separate paragraphs
        text = soup.get_text(separator="\n", strip=True)
        
        # Further cleaning (optional): remove excessive blank lines
        lines = (line.strip() for line in text.splitlines())
        clean_text = "\n".join(line for line in lines if line)
        
        print(f"Successfully extracted ~{len(clean_text)} chars from: {url}")
        return {"text": clean_text, "source": url, "error": None}
        
    except httpx.RequestError as e:
        error_msg = f"HTTP Request Error fetching {url}: {e.__class__.__name__} - {e}"
        print(error_msg)
        return {"text": None, "source": url, "error": error_msg}
    except Exception as e:
        # Catch other potential errors (e.g., BeautifulSoup issues)
        error_msg = f"Error processing URL {url}: {e.__class__.__name__} - {e}"
        print(error_msg)
        return {"text": None, "source": url, "error": error_msg}

# Optional: Add a function to close the client if needed on app shutdown
async def close_httpx_client():
    await httpx_client.aclose()
    print("HTTPX client closed.")

# Example usage (for testing)
# async def main():
#     result = await extract_text_from_url("https://example.com")
#     if result["error"]:
#         print(f"Error: {result['error']}")
#     else:
#         print(f"Extracted Text (first 500 chars):\n{result['text'][:500]}...")
#     await close_httpx_client()

# if __name__ == "__main__":
#     asyncio.run(main())
