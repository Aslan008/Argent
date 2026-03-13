import sys
import json
import requests
import time
from typing import List, Dict
from bs4 import BeautifulSoup
from ddgs import DDGS
from config import get_current_model
from ui import console

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def _call_ollama_sync(prompt: str, json_format: bool = False, temperature: float = 0.3) -> str:
    """Synchronous internal call to the local Ollama model."""
    model = get_current_model()
    if not model:
        model = "llama3.2" 
        
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    
    if json_format:
        payload["format"] = "json"
        
    try:
        # Увеличил таймаут до 600 секунд (10 минут) для медленных/тяжелых моделей
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    except Exception as e:
        console.print(f"[bold red]Deep Research Ollama Error:[/bold red] {e}")
        return ""

def _generate_queries(objective: str) -> List[str]:
    """Ask Ollama to brainstorm 3 distinct search queries for DuckDuckGo."""
    prompt = f"""You are an elite autonomous research agent. 
The user wants you to research the following topic deeply: '{objective}'.
Generate exactly 3 distinct, highly effective search queries that you would type into Google to find the best technical articles, forums, or documentation on this topic.
Return ONLY a valid JSON array of strings. No markup, no explanations.
Example: ["query 1", "query 2", "query 3"]
"""
    result = _call_ollama_sync(prompt, json_format=True, temperature=0.7)
    if not result:
        return [objective]
        
    try:
        queries = json.loads(result)
        if isinstance(queries, list) and len(queries) > 0:
            return [str(q) for q in queries][:3]
    except Exception:
        pass
        
    cleaned = result.replace('```json', '').replace('```', '').strip()
    try:
        queries = json.loads(cleaned)
        if isinstance(queries, list) and len(queries) > 0:
            return [str(q) for q in queries][:3]
    except Exception:
        pass
        
    return [objective]

def _extract_info(objective: str, chunk: str) -> str:
    """Ask Ollama to read a chunk of HTML text and extract relevant info."""
    prompt = f"""You are a research data extractor. 
Your overarching objective is: '{objective}'

Please read the following text extracted from a webpage. 
Extract and summarize any useful facts, code snippets, optimizations, or relevant details that help achieve the objective.
If the text does NOT contain anything genuinely useful for the objective, or if it is just SEO filler, reply with exactly the word "NOTHING".

TEXT:
{chunk[:6000]}
"""
    result = _call_ollama_sync(prompt, temperature=0.2).strip()
    if result.upper() == "NOTHING" or result.upper() == '"NOTHING"':
        return ""
    return result

def run_deep_research(objective: str) -> str:
    """
    Executes an autonomous Deep Research loop:
    1. Generates 3 search queries.
    2. Searches DDG and gets top 3 links per query.
    3. Scrapes HTML, filters garbage.
    4. Evaluates and extracts relevance using local LLM.
    5. Summarizes everything into a unified report.
    """
    console.print(f"\n[bold cyan]🧠 Starting Deep Research Sub-Agent...[/bold cyan]")
    console.print(f"Objective: {objective}")
    
    console.print("🤔 Brainstorming search queries...")
    queries = _generate_queries(objective)
    for i, q in enumerate(queries, 1):
        console.print(f"  {i}. {q}")
    
    visited_urls = set()
    all_links = []
    
    # 1. Search
    console.print("🔎 Searching DuckDuckGo...")
    with DDGS() as ddgs:
        for q in queries:
            try:
                results = list(ddgs.text(q, max_results=3))
                for r in results:
                    url = r.get("href")
                    if url and url not in visited_urls:
                        # Exclude YouTube videos as they are hard to scrape text from here
                        if "youtube.com" not in url and "youtu.be" not in url:
                            visited_urls.add(url)
                            all_links.append(url)
            except Exception as e:
                console.print(f"[dim yellow]Search warning for '{q}': {e}[/dim yellow]")
                
    console.print(f"📚 Found {len(all_links)} unique text sources to analyze.")
    
    # 2. Scrape and Extract
    extracted_knowledge = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36"
    }
    
    for url in all_links[:6]: # Cap at 6 to prevent script from taking 10+ minutes locally
        console.print(f"📖 Reading: {url}")
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                console.print(f"  [dim]Skipped ({resp.status_code})[/dim]")
                continue
                
            soup = BeautifulSoup(resp.text, 'html.parser')
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.extract()
                
            text = soup.get_text(separator=' ', strip=True)
            if len(text) < 200:
                console.print(f"  [dim]Skipped (Too little text)[/dim]")
                continue
                
            # Take the first ~6500 chars to evaluate (avoids hitting context window limit too hard for 8b models)
            chunk = text[:6500]
            console.print("  [dim]Evaluating relevance via LLM...[/dim]")
            info = _extract_info(objective, chunk)
            
            if info:
                console.print(f"  [green]✓ Relevant info extracted![/green]")
                extracted_knowledge.append(f"--- SOURCE: {url} ---\n{info}\n")
            else:
                console.print(f"  [dim]✗ No relevant technical info found.[/dim]")
                
        except Exception as e:
            console.print(f"  [dim]Failed to read: {e}[/dim]")
            
    if not extracted_knowledge:
        return f"Deep Research failed to find heavily relevant information for: '{objective}'. The agent tried {len(all_links)} sources but found nothing concrete."
        
    # 3. Final Synthesis
    console.print("[bold cyan]🔄 Synthesizing Final Report...[/bold cyan]")
    combined_notes = "\n".join(extracted_knowledge)
    
    # We send the final synthesis request. This might be quite large.
    synthesis_prompt = f"""You are an elite expert researcher.
Your overarching research objective was: '{objective}'

Here are the raw notes extracted from various internet sources by your sub-agents:
{combined_notes}

Synthesize this information into a MASSIVE, highly structured, cohesive, and deeply technical Markdown report.
Group similar concepts together. Include code blocks where applicable. Ensure no valuable technical details are lost. Use clear headers and bullet points.
Include a list of sources used at the bottom.
Do not add conversational fluff. Output ONLY the markdown report.
"""
    final_report = _call_ollama_sync(synthesis_prompt, temperature=0.3)
    
    if not final_report:
        return f"Research completed, but failed to synthesize final report.\nRaw findings:\n{combined_notes}"
    
    console.print("[bold green]✅ Deep Research Complete![/bold green]")
    return final_report
