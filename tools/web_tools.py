import requests
from ddgs import DDGS
from bs4 import BeautifulSoup
from deep_research import run_deep_research
from logger import get_logger

log = get_logger("tools")

def search_web(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            
        if not results:
            return f"No results found for query: '{query}'"
            
        formatted_results = [f"Search results for: '{query}'\n"]
        for i, res in enumerate(results, 1):
            formatted_results.append(f"{i}. {res.get('title', 'No Title')}")
            formatted_results.append(f"   URL: {res.get('href', 'No URL')}")
            formatted_results.append(f"   Snippet: {res.get('body', 'No Snippet')}\n")
            
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error searching the web: {e}"

def read_webpage(url: str, timeout: int = 15) -> str:
    """Read and extract text content from a webpage URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        
        max_chars = 15000
        if len(text) > max_chars:
            text = text[:max_chars] + "... [Content Truncated]"
            
        return f"Content of {url}:\n\n{text}"
    except Exception as e:
        return f"Error reading webpage '{url}': {e}"
