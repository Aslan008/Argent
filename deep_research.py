from typing import List
from config import get_current_model
from providers import create_provider, ProviderError
from ui import console
from logger import get_logger
from src.research.chunking import chunk_document
from src.research.rerank import rerank
from src.research.synthesis import parse_query_list, number_sources, dedup_chunks

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


def _generate_queries(objective: str, focus: str = None,
                        context: str = None) -> List[str]:
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
    focus_clause = f"\nFocus particularly on: {focus}." if focus else ""
    context_clause = ""
    if context:
        context_clause = ("\n\nKNOWN CONTEXT — the following is already established; "
                          "do NOT search for it, search for what it does NOT cover:\n"
                          f"{context[:2000]}")
    prompt = f"""You are a search strategist. Topic to research: '{objective}'.{focus_clause}{context_clause}

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


def _find_gaps(objective: str, notes: str, focus: str = None,
               context: str = None) -> List[str]:
    """Ask the model which important aspects are still missing, as follow-up
    search queries. Returns [] when coverage already looks sufficient."""
    focus_clause = f"\nFocus area: {focus}." if focus else ""
    context_clause = ""
    if context:
        context_clause = ("\n\nAlready-known context (do NOT re-search these):\n"
                          f"{context[:1500]}")
    prompt = f"""You are a meticulous research auditor.
Objective: '{objective}'{focus_clause}{context_clause}

Notes gathered so far:
{notes[:6000]}

List up to 3 follow-up web search queries targeting IMPORTANT aspects of the
objective that are still MISSING or thin in the notes above. If coverage is
already sufficient, return an empty array.
Return ONLY a JSON array of strings (it may be empty)."""
    return parse_query_list(_call_llm_sync(prompt, json_format=True, temperature=0.5), limit=3)


def _fetch_with_retry(url: str, max_retries: int = 2) -> dict:
    """Fetch a page with retry on network errors (timeout, connection).
    HTTP 4xx/5xx are NOT retried — the page exists but is inaccessible."""
    import time
    from src.research.fetch import fetch_page, UnsafeURLError
    for attempt in range(max_retries + 1):
        res = fetch_page(url)
        if res.get("ok"):
            return res
        err = res.get("error", "")
        # Retry only on network/timeout errors, not on HTTP status or SSRF.
        retryable = any(k in err.lower() for k in
                        ("timeout", "timed out", "connection", "unreachable",
                         "reset", "broken pipe"))
        if attempt < max_retries and retryable:
            backoff = 1.0 * (2 ** attempt)
            console.print(f"  [dim yellow]Retry {attempt+1}/{max_retries} in {backoff:.0f}s: {url}[/dim yellow]")
            time.sleep(backoff)
            continue
        return res
    return res


def _gather(queries: List[str], visited_urls: set, all_chunks: list,
            source_map: dict, max_new_sources: int = 15) -> int:
    """Search the queries, fetch new pages and chunk them into all_chunks
    (mapping each chunk back to its URL). Mutates the passed collections and
    returns the number of new sources read."""
    from src.research.search import meta_search

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
        res = _fetch_with_retry(url)
        content = res["text"] if res.get("ok") else ""
        if len(content) < 200:
            continue
        for c in chunk_document(content):
            all_chunks.append(c)
            source_map[c] = url
        read += 1
    return read

def _smart_truncate(text: str, max_chars: int) -> str:
    """Truncate at max_chars without breaking mid-sentence when possible."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Try to break at the last sentence boundary within the limit.
    for sep in (".\n", ". ", "! ", "? ", "\n"):
        idx = cut.rfind(sep)
        if idx > max_chars * 0.6:
            return cut[:idx + len(sep)].rstrip()
    return cut.rstrip()


def _extract_info(objective: str, combined_text: str, focus: str = None,
                  context: str = None) -> str:
    """Extract useful, source-tagged notes from the annotated fragments.

    ``combined_text`` is the chunks already prefixed with [n] source markers
    (see number_sources); the model is told to keep those markers so facts stay
    traceable through to the final report."""
    focus_clause = f"\nPay special attention to: {focus}." if focus else ""
    context_clause = ""
    if context:
        context_clause = ("\n\nAlready-known facts (do NOT re-extract these, "
                          "focus on NEW information):\n"
                          f"{context[:1500]}")
    prompt = f"""You are a research data extractor.
Your overarching objective is: '{objective}'{focus_clause}{context_clause}

Below are fragments found on the internet, each prefixed with a [n] source marker.
Extract useful facts, code snippets, optimizations, or relevant details that help
achieve the objective. Keep the [n] marker on each fact you retain, so it stays
traceable to its source. Omit irrelevant parts.
If NOTHING useful is found, reply with "NOTHING".

TEXT FRAGMENTS:
{_smart_truncate(combined_text, 12000)}
"""
    result = _call_llm_sync(prompt, temperature=0.2).strip()
    if result.upper().strip('"') == "NOTHING":
        return ""
    return result

def run_deep_research(objective: str, focus: str = None,
                     context: str = None, max_rounds: int = 2,
                     max_sources: int = 15) -> str:
    """Execute an autonomous Deep Research loop with model-controlled parameters.

    Parameters
    ----------
    objective : str
        The research topic or question.
    focus : str, optional
        Aspect to prioritise — injected into query generation, extraction and
        synthesis so the report leans toward this angle.
    context : str, optional
        Already-known facts. The pipeline avoids re-searching these and the
        extractor focuses on NEW information beyond this context.
    max_rounds : int (default 2)
        Total rounds including the initial one. ``max_rounds=1`` disables
        gap-filling; ``max_rounds=3`` allows two gap-filling passes. Capped
        at 3 to prevent unbounded loops.
    max_sources : int (default 15)
        Maximum new sources per gather pass.

    Returns a Markdown report with [n] inline citations and a SOURCES section.
    """
    max_rounds = max(1, min(max_rounds, 3))

    console.print(f"\n[bold cyan]Starting Advanced Deep Research...[/bold cyan]")
    log.info("Deep research started: %s (focus=%s, rounds=%d, sources=%d)",
             objective[:100], focus, max_rounds, max_sources)
    console.print(f"Objective: {objective}")
    if focus:
        console.print(f"Focus: {focus}")
    if context:
        console.print(f"Context: {len(context)} chars of prior knowledge provided")

    console.print("Thinking: Brainstorming search queries...")
    queries = _generate_queries(objective, focus=focus, context=context)
    for i, q in enumerate(queries, 1):
        console.print(f"  {i}. {q}")

    visited_urls, all_chunks, source_map = set(), [], {}

    # Round 1 — gather from the initial queries (federated meta-search + fetch).
    console.print("Searching the web (federated)...")
    _gather(queries, visited_urls, all_chunks, source_map, max_new_sources=max_sources)

    if not all_chunks:
        return f"Deep Research failed to find any text content for: '{objective}'."

    def _rerank_and_extract():
        # Deduplicate near-identical chunks before reranking (mirrors, syndication).
        unique = dedup_chunks(all_chunks)
        if len(unique) < len(all_chunks):
            console.print(f"  [dim]Dedup: {len(all_chunks)} → {len(unique)} chunks[/dim]")
        # Cross-encoder rerank (falls back to bi-encoder, then input order),
        # then extract source-tagged notes from the best fragments.
        console.print(f"  [dim cyan]Reranking {len(unique)} chunks...[/dim cyan]")
        relevant = rerank(objective, unique, top_n=15)
        annotated, sources = number_sources(relevant, source_map)
        console.print("  [dim]Synthesizing extracted knowledge via LLM...[/dim]")
        return _extract_info(objective, annotated, focus=focus, context=context), sources

    extracted_notes, sources = _rerank_and_extract()

    # Gap-filling rounds — find important missing aspects, gather more sources,
    # re-rank and re-extract. Bounded to max_rounds-1 extra rounds.
    for round_num in range(2, max_rounds + 1):
        if not extracted_notes:
            break
        gap_queries = _find_gaps(objective, extracted_notes, focus=focus, context=context)
        if not gap_queries:
            console.print(f"[dim]Round {round_num}/{max_rounds}: coverage sufficient, stopping.[/dim]")
            break
        console.print(f"Round {round_num}/{max_rounds}: filling coverage gaps...")
        for i, q in enumerate(gap_queries, 1):
            console.print(f"  +{i}. {q}")
        if _gather(gap_queries, visited_urls, all_chunks, source_map,
                   max_new_sources=max_sources):
            extracted_notes, sources = _rerank_and_extract()
        else:
            console.print(f"[dim]No new sources found in round {round_num}.[/dim]")
            break

    if not extracted_notes:
        return f"Research completed, but no relevant technical info was found in the {len(source_map)} fragments."

    # Final synthesis with numbered, inline-citable sources.
    console.print("[bold cyan]Building Final Report...[/bold cyan]")
    sources_text = "\n".join(sources)
    focus_clause = f"\n\nFocus the report on: {focus}." if focus else ""
    synthesis_prompt = f"""You are an elite expert researcher.
Your overarching research objective was: '{objective}'{focus_clause}

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
