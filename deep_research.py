from typing import List
from config import get_current_model
from providers import create_provider, ProviderError
from ui import console
from logger import get_logger
from src.research.chunking import chunk_document
from src.research.rerank import rerank
from src.research.synthesis import parse_query_list, number_sources

log = get_logger("research")


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

_LANGUAGE_NAMES = {
    "en": "English", "ru": "Russian", "de": "German", "fr": "French",
    "es": "Spanish", "it": "Italian", "pt": "Portuguese", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "tr": "Turkish", "pl": "Polish",
    "uk": "Ukrainian", "nl": "Dutch",
}


def _language_rule() -> str:
    """The LANGUAGE clause of the query prompt, from the configured languages."""
    from config import get_search_languages
    langs = get_search_languages()
    if langs == ["en"]:
        return ("write queries in ENGLISH even if the topic is stated in another "
                "language. Technical documentation, issues and answers are "
                "overwhelmingly in English; a translated query silently loses most "
                "of the good sources.")

    names = [_LANGUAGE_NAMES.get(l, l) for l in langs]
    others = [n for n in names if n != "English"]
    other_list = ", ".join(others) if others else "the local language"
    return (
        f"choose the language PER QUERY from: {', '.join(names)}. "
        f"Default to ENGLISH — that is where technical documentation, issues and "
        f"answers live. But write the query in {other_list} when the best sources "
        f"plainly are in it: regional services and products, local regulations, "
        f"country-specific practice, or a topic whose strongest community writes "
        f"in that language. When both could apply, spend one query on each."
    )


def _generate_queries(objective: str) -> List[str]:
    """Brainstorm distinct search queries for the objective.

    Query wording decides what the engines can possibly return — no reranker
    recovers from a query that never surfaced the right page — so the rules
    below encode what actually retrieves well rather than asking for "effective
    queries" and hoping.

    The language rule adapts to config.get_search_languages(): English-only by
    default (that is where technical sources live), but when other languages are
    enabled the model decides PER TOPIC — writing an English-only query for a
    regional service or a local-language community would miss the best sources
    entirely.
    """
    prompt = f"""You are a search strategist. Topic to research: '{objective}'.

Write exactly 5 search queries that will retrieve the best technical sources.

RULES — these decide whether the search finds anything:
1. LANGUAGE: {_language_rule()}
   Keep proper nouns (product, library, API names) exactly as they are.
2. ERROR MESSAGES: if the topic contains an error or exception, put the
   distinctive part in "double quotes" VERBATIM and drop the variable parts
   (paths, line numbers, GUIDs, timestamps). Quoted exact strings are the single
   highest-yield search technique.
3. VARY THE ANGLE, one per query — do not paraphrase the same question 5 times:
   - the official documentation (add site: for the vendor's docs domain when you
     know it, e.g. site:docs.unity3d.com or site:learn.microsoft.com);
   - a bug report / issue thread (add words like "issue", "regression", or the
     version number);
   - a practical Q&A phrasing (how someone actually asks it on StackOverflow);
   - the underlying concept or mechanism, for background;
   - a comparison / alternatives angle.
4. BE SPECIFIC: include version numbers, exact API/class names and the platform
   when they are known. Vague queries return vague pages.
5. NO natural-language sentences and no question marks — write keyword queries
   the way an experienced engineer types them.
6. OPERATORS are available and each engine is automatically given only the ones
   it understands, so use them freely:
   - "exact phrase" — the words must appear together, verbatim;
   - site:domain — restrict to a vendor's docs or one forum;
   - filetype:pdf — for specs, papers and manuals;
   - -word — exclude a term that keeps polluting the results (e.g. -tutorial
     when you need reference material, not beginner posts);
   - intitle:word — the term must be in the title, not just mentioned.

Return ONLY a valid JSON array of 5 strings. No markup, no explanations.
Example: ["\\"NullReferenceException\\" Unity Addressables LoadAssetAsync", "site:docs.unity3d.com Addressables memory management"]
"""
    result = _call_llm_sync(prompt, json_format=True, temperature=0.7)
    return parse_query_list(result, limit=5) or [objective]


def _find_gaps(objective: str, notes: str) -> List[str]:
    """Ask the model which important aspects are still missing, as follow-up
    search queries. Returns [] when coverage already looks sufficient."""
    prompt = f"""You are a meticulous research auditor.
Objective: '{objective}'

Notes gathered so far:
{notes[:6000]}

List up to 3 follow-up web search queries targeting IMPORTANT aspects of the
objective that are still MISSING or thin in the notes above. If coverage is
already sufficient, return an empty array.
Return ONLY a JSON array of strings (it may be empty)."""
    return parse_query_list(_call_llm_sync(prompt, json_format=True, temperature=0.5), limit=3)


def _gather(queries: List[str], visited_urls: set, all_chunks: list,
            source_map: dict, max_new_sources: int = 10) -> int:
    """Search the queries, fetch new pages and chunk them into all_chunks
    (mapping each chunk back to its URL). Mutates the passed collections and
    returns the number of new sources read."""
    from src.research.search import meta_search
    from src.research.fetch import fetch_page

    new_urls = []
    for q in queries:
        try:
            for r in meta_search(q, max_results=5):
                url = r.get("url")
                if (url and url not in visited_urls
                        and "youtube.com" not in url and "youtu.be" not in url):
                    visited_urls.add(url)
                    new_urls.append(url)
        except Exception as e:
            console.print(f"[dim yellow]Search failed for '{q}': {e}[/dim yellow]")

    read = 0
    for url in new_urls[:max_new_sources]:
        console.print(f"Reading: {url}")
        res = fetch_page(url)
        content = res["text"] if res.get("ok") else ""
        if len(content) < 200:
            continue
        for c in chunk_document(content):
            all_chunks.append(c)
            source_map[c] = url
        read += 1
    return read

def _extract_info(objective: str, combined_text: str) -> str:
    """Extract useful, source-tagged notes from the annotated fragments.

    ``combined_text`` is the chunks already prefixed with [n] source markers
    (see number_sources); the model is told to keep those markers so facts stay
    traceable through to the final report."""
    prompt = f"""You are a research data extractor.
Your overarching objective is: '{objective}'

Below are fragments found on the internet, each prefixed with a [n] source marker.
Extract useful facts, code snippets, optimizations, or relevant details that help
achieve the objective. Keep the [n] marker on each fact you retain, so it stays
traceable to its source. Omit irrelevant parts.
If NOTHING useful is found, reply with "NOTHING".

TEXT FRAGMENTS:
{combined_text[:12000]}
"""
    result = _call_llm_sync(prompt, temperature=0.2).strip()
    if result.upper().strip('"') == "NOTHING":
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
    
    visited_urls, all_chunks, source_map = set(), [], {}

    # Round 1 — gather from the initial queries (federated meta-search + fetch).
    console.print("Searching the web (federated)...")
    _gather(queries, visited_urls, all_chunks, source_map)

    if not all_chunks:
        return f"Deep Research failed to find any text content for: '{objective}'."

    def _rerank_and_extract():
        # Cross-encoder rerank (falls back to bi-encoder, then input order),
        # then extract source-tagged notes from the best fragments.
        console.print(f"  [dim cyan]Reranking {len(all_chunks)} chunks...[/dim cyan]")
        relevant = rerank(objective, all_chunks, top_n=15)
        annotated, sources = number_sources(relevant, source_map)
        console.print("  [dim]Synthesizing extracted knowledge via LLM...[/dim]")
        return _extract_info(objective, annotated), sources

    extracted_notes, sources = _rerank_and_extract()

    # Round 2 — one bounded gap-filling pass: find important missing aspects,
    # gather more sources for them, then re-rank and re-extract over the larger
    # corpus. Bounded to a single extra round so a model can't loop forever.
    gap_queries = _find_gaps(objective, extracted_notes) if extracted_notes else []
    if gap_queries:
        console.print("Filling coverage gaps with follow-up searches...")
        for i, q in enumerate(gap_queries, 1):
            console.print(f"  +{i}. {q}")
        if _gather(gap_queries, visited_urls, all_chunks, source_map):
            extracted_notes, sources = _rerank_and_extract()

    if not extracted_notes:
        return f"Research completed, but no relevant technical info was found in the {len(source_map)} fragments."

    # Final synthesis with numbered, inline-citable sources.
    console.print("[bold cyan]Building Final Report...[/bold cyan]")
    sources_text = "\n".join(sources)
    synthesis_prompt = f"""You are an elite expert researcher.
Your overarching research objective was: '{objective}'

Here are the extracted findings. Each fact carries a [n] marker identifying its source:
{extracted_notes}

Synthesize this into a structured, cohesive, deeply technical Markdown report.
Group similar concepts together. Include code blocks where applicable. Preserve
the [n] inline citations next to the claims they support, so every claim stays
traceable. End with a 'SOURCES' section listing exactly:
{sources_text}

Do not add conversational fluff. Output ONLY the markdown report.
"""
    final_report = _call_llm_sync(synthesis_prompt, temperature=0.3)

    if not final_report:
        return f"Research completed, but failed to synthesize final report.\nRaw findings:\n{extracted_notes}"

    console.print("[bold green]Deep Research Complete![/bold green]")
    return final_report
