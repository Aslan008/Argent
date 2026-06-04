import time
import random
import requests
import urllib.parse
from ddgs import DDGS
from bs4 import BeautifulSoup
try:
    from markdownify import markdownify
except ImportError:
    markdownify = None
from deep_research import run_deep_research
from logger import get_logger

log = get_logger("tools")

# 1. Connection Pooling
_session = requests.Session()

# Modern User Agents for randomization
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0"
]

def search_web(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo with retry logic for rate limits."""
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
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
                
            formatted_results.append("\n[SYSTEM REMINDER: These are only short snippets. To get the actual answer, you MUST call `read_webpage(url)` on the most relevant URL above! Do not just search again.]")
            return "\n".join(formatted_results)
            
        except Exception as e:
            error_msg = str(e).lower()
            if "ratelimit" in error_msg or "202" in error_msg or attempt < max_retries - 1:
                log.warning(f"DDGS search failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {base_delay}s...")
                time.sleep(base_delay)
                base_delay *= 2  # Exponential backoff
            else:
                log.error(f"Error searching the web after {max_retries} attempts: {e}")
                return f"Error searching the web: {e}"
    
    return f"Error: Could not fetch results for '{query}' due to rate limits."

def read_webpage(url: str, timeout: int = 15, raw_mode: bool = False) -> str:
    """Read and extract meaningful text content from a webpage URL.
    If raw_mode is True, bypasses semantic extraction and returns all visible text."""
    try:
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        response = _session.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        def extract_content(element) -> str:
            if not element:
                return ""
            if markdownify:
                # Convert relative URLs to absolute first
                for a in element.find_all('a', href=True):
                    a['href'] = urllib.parse.urljoin(url, a['href'])
                return markdownify(str(element), heading_style="ATX").strip()
            return element.get_text(separator='\n', strip=True)

        if raw_mode:
            # In raw_mode, only remove scripts and styles, keep everything else
            for element in soup(["script", "style", "noscript"]):
                element.extract()
            text = extract_content(soup.body if soup.body else soup)
        else:
            # 1. Remove noise elements completely
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
                element.extract()
                
            # 2. Try to find the semantic core (article or main)
            text = ""
            main_content = soup.find('article') or soup.find('main')
            if not main_content:
                # Find all potential content divs and pick the one with the most text
                content_divs = soup.find_all('div', class_=lambda x: x and ('content' in x.lower() or 'article' in x.lower()))
                if content_divs:
                    main_content = max(content_divs, key=lambda d: len(d.get_text(strip=True)))
                    
            if main_content:
                text = extract_content(main_content)
                
            # Fallback to body if semantic tags failed or if the extracted text is suspiciously short 
            # compared to the overall body (e.g. it grabbed a small header wrapper).
            body = soup.find('body')
            if body:
                body_text = extract_content(body)
                if len(text) < 300 and len(body_text) > len(text) * 2:
                    text = body_text
            elif not text:
                text = extract_content(soup)
        
        # Clean up excessive newlines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)
        
        max_chars = 15000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [Content Truncated]"
            
        return f"Content of {url}:\n\n{text}"
    except Exception as e:
        log.error(f"Error reading webpage '{url}': {e}")
        return f"Error reading webpage '{url}': {e}"
