"""
stagehand_tools.py — High-level semantic AI browser automation tools using Stagehand v4.

Integrates with Argent's browser_engine side-by-side with existing Playwright tools.
Connects directly to the shared Chromium instance via CDP (Chrome DevTools Protocol).
"""

import json
import os
import re
import threading
from typing import Optional, Dict, Any
from pathlib import Path

from logger import get_logger
from browser_engine import browser_engine

log = get_logger("stagehand")

# Check if stagehand package is available
try:
    import stagehand
    from stagehand import Stagehand
    from stagehand.browser import local_browser, StagehandBrowser
    from stagehand.client_models import ClientLLM
    STAGEHAND_AVAILABLE = True
except ImportError:
    STAGEHAND_AVAILABLE = False
    Stagehand = None
    local_browser = None
    StagehandBrowser = None
    ClientLLM = None

_STAGEHAND_INSTANCE: Optional[Any] = None
_STAGEHAND_LOCK = threading.Lock()


def _get_llm_config() -> Dict[str, Any]:
    """Resolve LLM model configuration for Stagehand.
    
    Checks environment / Argent config for available providers:
    1. OpenAI API key
    2. Anthropic API key
    3. OpenRouter API key
    4. Argent native provider bridge fallback
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return {"model": "openai/gpt-4o", "model_api_key": openai_key}

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        return {"model": "anthropic/claude-3-7-sonnet", "model_api_key": anthropic_key}

    # For all other providers (OpenRouter, local models, etc.), use the ClientLLM callback
    # which cleanly bypasses Stagehand's restrictive ModelConfig regex validators.

    # Provider bridge callback.
    #
    # It never worked: it called stream_chat(model_name=...) — the parameter is
    # `model`, so every call raised and the callback quietly answered "Error
    # generating completion"; and it kept only chunks that were `str`, while the
    # providers yield dicts ({"content": ...}), so even a successful call came
    # back empty. It also ran the blocking stream on the event loop that drives
    # the browser, freezing every page action for the length of the completion,
    # and it handed Stagehand a bare function where a ClientLLM is required.
    def _complete(full_prompt: str) -> str:
        from config import get_current_model
        from providers import create_provider

        provider = create_provider()
        text = ""
        for chunk in provider.stream_chat(
            model=get_current_model(),
            messages=[{"role": "user", "content": full_prompt}],
            tools=None,
            context_window=None,
            temperature=0.0,
        ):
            if isinstance(chunk, dict):
                text += chunk.get("content") or ""
            elif isinstance(chunk, str):
                text += chunk
        return text

    async def argent_generate_callback(params):
        import asyncio
        from stagehand._generated.models import (
            LLMMessageGenerateResult, LLMStructuredGenerateResult, LLMRole,
            LLMMessageContentBlock, LLMStructuredGenerateParams,
        )

        # Build prompt from messages
        prompt_parts = []
        messages = getattr(params, "messages", [])
        for m in messages:
            role = getattr(m, "role", "user")
            content = getattr(m, "content", "")
            if isinstance(content, list):
                content = " ".join(getattr(b, "text", str(b)) for b in content)
            prompt_parts.append(f"{role}: {content}")
        full_prompt = "\n".join(prompt_parts)

        try:
            # Off the loop: the stream blocks, and this loop is the browser's.
            response_text = await asyncio.to_thread(_complete, full_prompt)
        except Exception as e:
            log.error("Stagehand provider bridge error: %s", e)
            response_text = f"Error generating completion: {e}"

        if isinstance(params, LLMStructuredGenerateParams):
            try:
                parsed_json = json.loads(response_text)
            except Exception:
                parsed_json = {"result": response_text}
            return LLMStructuredGenerateResult(
                role=LLMRole.assistant,
                content=[LLMMessageContentBlock(type="text", text=response_text)],
                output_format="json_schema",
                structured_content=parsed_json,
            )
        return LLMMessageGenerateResult(
            role=LLMRole.assistant,
            content=[LLMMessageContentBlock(type="text", text=response_text)],
            output_format="text",
        )

    return {"model": ClientLLM(generate=argent_generate_callback)}


async def _get_stagehand_instance_async(session: str = "default") -> Any:
    """Async helper to get or initialize the Stagehand singleton attached to Argent's browser."""
    global _STAGEHAND_INSTANCE

    if not STAGEHAND_AVAILABLE:
        raise RuntimeError(
            "Stagehand is not installed in the Python environment. "
            "Install it via: pip install stagehand, or use standard tools "
            "(browser_action click / type, browser_inspect)."
        )

    # Check if session exists in browser_engine FIRST
    if session not in browser_engine._sessions:
        raise KeyError(
            f"Session '{session}' does not exist or browser is not open. "
            f"Call browser_open(url='...') first."
        )

    # Check if Argent's browser is active and exposes CDP
    cdp_url = browser_engine.get_cdp_url()
    if not cdp_url:
        raise RuntimeError(
            f"Browser is running, but CDP endpoint is not available. "
            f"Ensure the browser was opened with remote debugging enabled."
        )

    # Reuse cached instance if still connected and healthy
    if _STAGEHAND_INSTANCE is not None:
        try:
            if not _STAGEHAND_INSTANCE.browser.closed:
                return _STAGEHAND_INSTANCE
        except Exception:
            _STAGEHAND_INSTANCE = None

    # Point Stagehand extension directory to clean folder without leading underscore
    clean_dir = Path.home() / ".argent" / "stagehand_extension"
    if clean_dir.is_dir() and (clean_dir / "manifest.json").is_file():
        try:
            import stagehand.extension_assets
            stagehand.extension_assets.extension_directory = lambda: clean_dir
            import stagehand.browser
            stagehand.browser.extension_directory = lambda: clean_dir
        except Exception:
            pass

    # Automatically discover preloaded Stagehand extension ID from CDP targets
    ext_id = None
    try:
        import urllib.request
        with urllib.request.urlopen(f"{cdp_url}/json/list", timeout=2) as resp:
            targets = json.loads(resp.read().decode())
            # Prioritize known Stagehand extension IDs
            known_stagehand_ids = [
                "kbllhomkfighlohagocmnjkhfehkombc",
                "kmgeemlmoiohlehomoligkpelnhjbahl",
                "knkbjekpciapkagiphkcfdmhgaggjilb",
                "hjkgacheefjckdpkhppjbhldfpdmbbmh",
            ]
            for t in targets:
                t_url = t.get("url", "")
                t_title = t.get("title", "").lower()
                for kid in known_stagehand_ids:
                    if kid in t_url:
                        ext_id = kid
                        break
                if ext_id:
                    break
                if "stagehand" in t_title or "stagehand" in t_url.lower():
                    if t_url.startswith("chrome-extension://"):
                        ext_id = t_url.split("/")[2]
                        break
                # In test mock environments, match dummy manifest.json target
                if t_url.startswith("chrome-extension://") and "manifest.json" in t_url:
                    ext_id = t_url.split("/")[2]
                    break
    except Exception as e:
        log.debug("Could not query CDP targets for extension_id: %s", e)

    connect_kwargs = {"cdp_url": cdp_url}
    if ext_id:
        connect_kwargs["extension_id"] = ext_id
        log.info("Discovered preloaded Stagehand extension ID: %s", ext_id)

    log.info("Connecting Stagehand v4 to browser CDP: %s (extension_id=%s)", cdp_url, ext_id)
    browser = await local_browser.connect(**connect_kwargs)
    llm_kwargs = _get_llm_config()

    stagehand_inst = await Stagehand.create(browser=browser, **llm_kwargs)
    _STAGEHAND_INSTANCE = stagehand_inst
    log.info("Stagehand v4 initialized and attached to session '%s'", session)

    # Immediately close any stray wake-service-worker tab so the user never sees it
    try:
        if browser_engine._browser:
            for ctx in browser_engine._browser.contexts:
                for p in ctx.pages:
                    url_lower = p.url.lower()
                    title_lower = ""
                    try:
                        title_lower = (await p.title()).lower()
                    except Exception:
                        pass
                    if (
                        "wake-service-worker" in url_lower
                        or "kmgeemlmoiohlehomoligkpelnhjbahl" in url_lower
                        or "kbllhomkfighlohagocmnjkhfehkombc" in url_lower
                        or "knkbjekpciapkagiphkcfdmhgaggjilb" in url_lower
                        or "hjkgacheefjckdpkhppjbhldfpdmbbmh" in url_lower
                        or "ihcjicgdanjaechkgeegckofjjedodee" in url_lower
                        or "bgnkhhnnamicmpeenaelnjfhikgbkllg" in url_lower
                        or "kmgeemlmoiohlehomoligkpelnhjbahl" in title_lower
                        or "ihcjicgdanjaechkgeegckofjjedodee" in title_lower
                        or "bgnkhhnnamicmpeenaelnjfhikgbkllg" in title_lower
                        or "файл недоступен" in title_lower
                        or "file not found" in title_lower
                    ):
                        await p.close()
    except Exception as e:
        log.debug("Auto-closing wake tab failed: %s", e)

    return _STAGEHAND_INSTANCE


def reset_stagehand() -> None:
    """Reset and invalidate the cached Stagehand instance (called on browser close/shutdown)."""
    global _STAGEHAND_INSTANCE
    with _STAGEHAND_LOCK:
        if _STAGEHAND_INSTANCE is not None:
            try:
                # Do not close the user's browser, just detach Stagehand
                log.info("Resetting cached Stagehand instance")
            except Exception as e:
                log.warning("Error releasing Stagehand instance: %s", e)
            _STAGEHAND_INSTANCE = None


# ---------------------------------------------------------------------------
# Public Tool Functions
# ---------------------------------------------------------------------------

def _extract_text_payload(instruction: str) -> Optional[str]:
    """Извлекает текст для ввода из инструкции (в кавычках или после глагола ввода)."""
    quote_match = re.search(r"['\"]([^'\"]+)['\"]", instruction)
    if quote_match:
        return quote_match.group(1).strip()
    m = re.search(
        r"(?:type|input|enter|введи|напиши|заполни|впечатай)\s+(.*?)\s+(?:in|into|to|в|внутрь)\b",
        instruction,
        re.IGNORECASE,
    )
    if m:
        val = m.group(1).strip()
        if val and not any(w in val.lower() for w in ("текст", "слово", "символы", "text")):
            return val
    return None


def _try_leeth_browser_act(instruction: str, session: str = "default") -> Optional[str]:
    """Fast-Path: быстрое исполнение действия в браузере за 10-15 мс через Leeth BrowserRouter."""
    try:
        active_sessions = getattr(browser_engine, "_sessions", {})
        if session not in active_sessions:
            return None
        sess_ctx = active_sessions[session]
        if not sess_ctx or not getattr(sess_ctx, "page", None) or sess_ctx.page.is_closed():
            return None

        elements = browser_engine.run(browser_engine.get_interactive_elements_raw(session))
        if not elements:
            return None

        from src.leeth.browser_router import get_browser_router
        router = get_browser_router()
        if not router.is_ready:
            return None

        route = router.route(instruction, elements)
        if route is None:
            return None

        target_el = elements[route.index]
        dom_idx = target_el.get("idx", route.index)
        target_label = target_el.get("label") or target_el.get("name") or target_el.get("text") or target_el.get("tag") or ""
        target_tag = (target_el.get("tag") or "").upper()

        if route.requires_confirmation:
            return (
                f"[Fast-Path Safety Gate]: Action on element [{dom_idx}] '{target_label}' requires user confirmation "
                f"({route.reason or 'potentially dangerous action'})."
            )

        text_payload = _extract_text_payload(instruction)
        is_input_elem = target_tag in ("INPUT", "TEXTAREA") or target_el.get("role") in ("textbox", "searchbox")
        has_type_intent = any(w in instruction.lower() for w in ("type", "input", "enter", "fill", "введи", "напиши", "заполни"))

        from tools.browser_tools import _reset_poll_tracker
        _reset_poll_tracker(session)

        if text_payload is not None:
            res = browser_engine.run(browser_engine.fill_input(dom_idx, text=text_payload, session=session, return_state=True))
            return f"Fast-Path Act Succeeded (score={route.score:.3f}, confidence={route.score*100:.1f}%): typed '{text_payload}' into element [{dom_idx}] '{target_label}'\n{res}"
        elif is_input_elem and has_type_intent:
            res = browser_engine.run(browser_engine.click(index=dom_idx, session=session, return_state=True))
            return f"Fast-Path Act Succeeded (score={route.score:.3f}, focused): focused input element [{dom_idx}] '{target_label}'\n{res}"
        else:
            res = browser_engine.run(browser_engine.click(index=dom_idx, session=session, return_state=True))
            return f"Fast-Path Act Succeeded (score={route.score:.3f}, confidence={route.score*100:.1f}%): clicked element [{dom_idx}] '{target_label}'\n{res}"

    except Exception as e:
        log.debug("Leeth Fast-Path browser execution skipped: %s", e)
        return None


def browser_act(instruction: str, session: str = "default") -> str:
    """Perform a high-level natural language semantic action on the web page using Stagehand v4 or Leeth Fast-Path.
    
    Examples:
      - browser_act("click on the sign in button")
      - browser_act("type 'python automation' into the search field and press Enter")
      - browser_act("select 'United States' from the country dropdown")
    """
    if not instruction or not isinstance(instruction, str) or not instruction.strip():
        return "Error: You must provide a non-empty instruction for browser_act (e.g. 'click on the sign in button')."

    # 1. Попытка мгновенного Leeth Fast-Path (10-15 мс)
    fast_res = _try_leeth_browser_act(instruction, session)
    if fast_res is not None:
        return fast_res

    # 2. Фолбэк на Stagehand v4
    if not STAGEHAND_AVAILABLE:
        return (
            "Stagehand v4 is not installed. To perform manual actions, "
            "use browser_inspect to inspect elements, then browser_action (click / type). "
            "(To enable Stagehand: pip install stagehand)"
        )

    async def _run():
        sh = await _get_stagehand_instance_async(session)
        result = await sh.act(instruction)
        return result

    try:
        res = browser_engine.run(_run())
        from tools.browser_tools import _reset_poll_tracker
        _reset_poll_tracker(session)
        success_msg = f"Stagehand Act Succeeded: '{instruction}'"
        if res:
            success_msg += f"\nResult: {res}"
        return success_msg
    except KeyError as ke:
        return str(ke.args[0] if ke.args else ke)
    except Exception as e:
        log.error("browser_act error: %s", e)
        # Если Stagehand упал, пробуем показать пользователю текущее состояние страницы
        try:
            from tools.browser_tools import browser_state
            st = browser_state(session=session)
            return (
                f"Notice: Stagehand action failed ({e}).\n"
                f"Page is active. Available interactive elements:\n{st}\n"
                f"Hint: Use browser_action(action=\"click\", index=N) or browser_action(action=\"type\", index=N, text=\"...\") directly."
            )
        except Exception:
            return (
                f"Error performing semantic action '{instruction}': {e}\n"
                f"(Hint: If Stagehand fails, fall back to browser_inspect + browser_action click / type)"
            )


def browser_extract(instruction: str, schema: dict = None, session: str = "default") -> str:
    """Extract structured data from the current web page using Stagehand v4.
    
    Args:
      instruction: Natural language description of what data to extract (e.g. 'extract all articles with title and link').
      schema: Optional dictionary describing the JSON schema of the output.
      session: Browser session name. Default: 'default'.
    """
    if not instruction or not isinstance(instruction, str) or not instruction.strip():
        return "Error: You must provide a non-empty instruction for browser_extract (e.g. 'extract all articles with title and link')."

    if schema is not None and not isinstance(schema, dict):
        return "Error: The 'schema' parameter for browser_extract must be a dictionary representing a JSON schema."

    if not STAGEHAND_AVAILABLE:
        return (
            "Stagehand v4 is not installed. Use browser_inspect(mode=\"text_content\") "
            "or browser_action(action=\"eval_js\") to inspect page contents."
        )

    async def _run():
        sh = await _get_stagehand_instance_async(session)
        result = await sh.extract(instruction)
        return result

    try:
        res = browser_engine.run(_run())
        if hasattr(res, "data"):
            return json.dumps(res.data, indent=2, ensure_ascii=False)
        return str(res)
    except KeyError as ke:
        return str(ke.args[0] if ke.args else ke)
    except Exception as e:
        log.error("browser_extract error: %s", e)
        # Resilient fallback: extract structured data via native DOM/Markdown + LLM
        log.info("Attempting native Argent extraction fallback for instruction: %s", instruction)
        try:
            markdown = browser_engine.run(browser_engine.get_markdown(session=session))
            if not markdown or not markdown.strip():
                return f"Error extracting data with Stagehand: {e}"
            from providers import create_provider
            provider = create_provider()
            prompt = (
                f"You are an expert data extractor. Based on the page content below, extract the requested data in JSON format.\n\n"
                f"Instruction: {instruction}\n"
            )
            if schema:
                prompt += f"Output JSON Schema: {json.dumps(schema, ensure_ascii=False)}\n"
            prompt += f"\nPage Content:\n{markdown[:25000]}\n\nReturn ONLY valid JSON (no markdown formatting, no code fences)."

            resp = provider.chat(messages=[{"role": "user", "content": prompt}])
            clean_resp = (resp or "").strip()
            if clean_resp.startswith("```"):
                clean_resp = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean_resp, flags=re.DOTALL).strip()
            return clean_resp
        except Exception as fallback_err:
            log.error("Native extraction fallback also failed: %s", fallback_err)
            return f"Error extracting data with Stagehand: {e}"


def browser_observe(instruction: str = None, session: str = "default") -> str:
    """Observe and discover actionable interactive elements on the web page using Stagehand v4.
    
    Args:
      instruction: Optional prompt focusing observation on specific tasks (e.g. 'find navigation links').
      session: Browser session name. Default: 'default'.
    """
    if instruction is not None and not isinstance(instruction, str):
        return "Error: The 'instruction' parameter for browser_observe must be a string if provided."
    if not STAGEHAND_AVAILABLE:
        return (
            "Stagehand v4 is not installed. Use browser_inspect to inspect interactive elements."
        )

    async def _run():
        sh = await _get_stagehand_instance_async(session)
        result = await sh.observe(instruction)
        return result

    try:
        res = browser_engine.run(_run())
        return f"Stagehand Observed Elements:\n{res}"
    except KeyError as ke:
        return str(ke.args[0] if ke.args else ke)
    except Exception as e:
        log.error("browser_observe error: %s", e)
        # Resilient Native Fallback via Playwright State
        try:
            from tools.browser_tools import browser_state
            st = browser_state(session=session, query=instruction)
            return (
                f"Stagehand Notice: Overlay observation unavailable ({e}).\n"
                f"Active Page Elements (via Native Fast-Path):\n{st}"
            )
        except Exception:
            pass
        err_str = str(e)
        if "Extensions.loadUnpacked" in err_str:
            return (
                f"Stagehand Notice: Visual element observation overlay is unavailable on this Chrome build ({e}).\n"
                f"Stagehand is active and prioritized: you can execute your desired action directly via "
                f"browser_action(action=\"act\", instruction=\"<your semantic instruction>\") or extract data via browser_inspect(mode=\"extract\", instruction=\"<data to extract>\"). "
                f"Stagehand will automatically locate and interact with the elements."
            )
        return (
            f"Error observing page with Stagehand: {e}\n"
            f"(Hint: Stagehand remains active. Proceed directly using browser_action(action=\"act\", instruction=\"<action>\") without observe)."
        )
