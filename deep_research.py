import sys
import json
import requests
import time
import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from ddgs import DDGS
from config import get_current_model
from providers import create_provider, ProviderError
from ui import console
from logger import get_logger

log = get_logger("research")

# New imports for advanced pipeline
try:
    from crawl4ai import AsyncWebCrawler
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

try:
    import numpy as np
except ImportError:
    pass

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def _call_llm_sync(prompt: str, json_format: bool = False, temperature: float = 0.3) -> str:
    """Synchronous internal call to the configured LLM provider."""
    model = get_current_model()
    if not model:
        model = "llama3.2"
    
    try:
        provider = create_provider()
        return provider.sync_chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            json_format=json_format,
        )
    except ProviderError as e:
        console.print(f"[bold red]Deep Research Error:[/bold red] {e}")
        return ""
    except Exception as e:
        console.print(f"[bold red]Deep Research Error:[/bold red] {e}")
        return ""

def _generate_queries(objective: str) -> List[str]:
    """Ask Ollama to brainstorm 3-5 distinct search queries for DuckDuckGo."""
    prompt = f"""You are an elite autonomous research agent. 
The user wants you to research the following topic deeply: '{objective}'.
Generate exactly 5 distinct, highly effective search queries that you would type into Google to find the best technical articles, forums, or documentation on this topic.
Focus on different aspects: technical docs, GitHub issues, stackoverflow, and deep-dive blogs.
Return ONLY a valid JSON array of strings. No markup, no explanations.
Example: ["query 1", "query 2", "query 3", "query 4", "query 5"]
"""
    result = _call_llm_sync(prompt, json_format=True, temperature=0.7)
    if not result:
        return [objective]
        
    try:
        queries = json.loads(result)
        if isinstance(queries, list) and len(queries) > 0:
            return [str(q) for q in queries][:5]
    except Exception:
        pass
        
    cleaned = result.replace('```json', '').replace('```', '').strip()
    try:
        queries = json.loads(cleaned)
        if isinstance(queries, list) and len(queries) > 0:
            return [str(q) for q in queries][:5]
    except Exception:
        pass
        
    return [objective]

def _scrape_url(url: str) -> str:
    """Read a webpage and extract clean Markdown/Text content."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return ""
            
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Remove garbage
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            element.extract()
            
        # Try to find main content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|article|post|main', re.I))
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)
            
        return text
    except Exception as e:
        console.print(f"  [dim red]Scraping failed for {url}: {e}[/dim red]")
        return ""

def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks."""
    if not text:
        return []
    
    # Simple character-based chunking for performance
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
        
    return chunks

def _rerank_chunks(objective: str, all_chunks: List[str], top_n: int = 10) -> List[str]:
    """Use sentence-transformers to find the most relevant chunks."""
    if not all_chunks:
        return []
        
    try:
        console.print(f"  [dim cyan]Reranking {len(all_chunks)} chunks using SentenceTransformer...[/dim cyan]")
        model = SentenceTransformer('all-MiniLM-L6-v2') # Light and fast
        
        obj_embedding = model.encode([objective])
        chunk_embeddings = model.encode(all_chunks)
        
        # Compute cosine similarity
        similarities = np.dot(chunk_embeddings, obj_embedding.T).flatten()
        
        # Get top-N indices
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        return [all_chunks[i] for i in top_indices]
    except Exception as e:
        console.print(f"  [dim yellow]Reranking failed: {e}. Using first chunks instead.[/dim yellow]")
        return all_chunks[:top_n]

def _extract_info(objective: str, chunks: List[str]) -> str:
    """Ask Ollama to synthesize information from multiple chunks."""
    combined_text = "\n---\n".join(chunks)
    prompt = f"""You are a research data extractor. 
Your overarching objective is: '{objective}'

Below are some fragments of text found on the internet. 
Extract and summarize any useful facts, code snippets, optimizations, or relevant details that help achieve the objective.
Synthesize the information into a cohesive set of notes. Omit irrelevant parts.
If NOTHING useful is found, reply with "NOTHING".

TEXT FRAGMENTS:
{combined_text[:12000]}
"""
    result = _call_llm_sync(prompt, temperature=0.2).strip()
    if result.upper() == "NOTHING" or result.upper() == '"NOTHING"':
        return ""
    return result

def run_deep_research(objective: str) -> str:
    """
    Executes an autonomous Deep Research loop:
    1. Generates 5 search queries.
    2. Searches DDG and gets top links.
    3. Scrapes content from top sites.
    4. Chunks text and Reranks most relevant fragments.
    5. Summarizes everything into a unified report.
    """
    console.print(f"\n[bold cyan]Starting Advanced Deep Research...[/bold cyan]")
    log.info("Deep research started: %s", objective[:100])
    console.print(f"Objective: {objective}")
    
    console.print("Thinking: Brainstorming search queries...")
    queries = _generate_queries(objective)
    for i, q in enumerate(queries, 1):
        console.print(f"  {i}. {q}")
    
    visited_urls = set()
    all_links = []
    
    # 1. Search
    console.print("Searching DuckDuckGo...")
    with DDGS() as ddgs:
        for q in queries:
            try:
                results = list(ddgs.text(q, max_results=3))
                for r in results:
                    url = r.get("href")
                    if url and url not in visited_urls:
                        if "youtube.com" not in url and "youtu.be" not in url:
                            visited_urls.add(url)
                            all_links.append(url)
            except Exception as e:
                console.print(f"[dim yellow]Search warning for '{q}': {e}[/dim yellow]")
                
    console.print(f"Found {len(all_links)} unique sources to analyze.")
    
    # 2. Scrape and Chunk
    all_chunks = []
    source_map = {} # Map chunks back to sources
    
    for url in all_links[:10]: # Analyze up to 10 sources
        console.print(f"Reading: {url}")
        content = _scrape_url(url)
        if len(content) < 200:
            continue
            
        chunks = _chunk_text(content)
        for c in chunks:
            all_chunks.append(c)
            source_map[c] = url

    if not all_chunks:
        return f"Deep Research failed to find any text content for: '{objective}'."
        
    # 3. Rerank
    relevant_chunks = _rerank_chunks(objective, all_chunks, top_n=15)
    
    # 4. Extract & Synthesize Finding per Chunk Group (batching to Ollama)
    console.print("  [dim]Synthesizing extracted knowledge via LLM...[/dim]")
    extracted_notes = _extract_info(objective, relevant_chunks)
    
    if not extracted_notes:
        return f"Research completed, but no relevant technical info was found in the {len(all_links)} sources."
        
    # 5. Final Synthesis
    console.print("[bold cyan]Building Final Report...[/bold cyan]")
    
    # Collect sources used in final chunks
    used_sources = sorted(list(set(source_map[c] for c in relevant_chunks if c in source_map)))
    sources_text = "\n".join([f"- {s}" for s in used_sources])
    
    synthesis_prompt = f"""You are an elite expert researcher.
Your overarching research objective was: '{objective}'

Here are the extracted findings from the most relevant web fragments:
{extracted_notes}

Synthesize this information into a MASSIVE, highly structured, cohesive, and deeply technical Markdown report.
Group similar concepts together. Include code blocks where applicable. Ensure no valuable technical details are lost. Use clear headers and bullet points.
Include a section 'SOURCES' at the bottom listing these URLs:
{sources_text}

Do not add conversational fluff. Output ONLY the markdown report.
"""
    final_report = _call_llm_sync(synthesis_prompt, temperature=0.3)
    
    if not final_report:
        return f"Research completed, but failed to synthesize final report.\nRaw findings:\n{extracted_notes}"
    
    console.print("[bold green]Deep Research Complete![/bold green]")
    return final_report
