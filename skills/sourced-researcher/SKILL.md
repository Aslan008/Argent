---
name: sourced-researcher
description: Answer a factual or current-events question with evidence — search the web, read the best sources, then write an answer where every claim carries an inline citation and a URL. Use when the user says "research", "find out", "is it true that…", asks about recent or current facts, compares options, or wants sources instead of a guess.
version: 1.0.0
allowed-tools: search_web, read_webpage, read_file, write_file
when_to_use: A factual, comparative, or current-events question where being correct and citable matters more than speed. Not for opinions, code edits, or facts already present in the conversation.
---

# Sourced Researcher — the one rule

**Every factual sentence in the final answer ends with a citation to a page you actually opened. No source → do not claim it; say "couldn't verify".** A confident answer with no sources is a failure, even if it happens to be right.

This skill turns a question into a grounded, cited answer using `search_web` to find sources and `read_webpage` to read them. Follow the five phases in order — skipping or reordering them is how hallucinations get in.

---

## Phase 1 — Plan (≤5 lines, before searching)

- Restate the question as **2–4 concrete sub-claims** you must verify.
- Pick query languages (e.g. `[en]`, or `[ru, en]` for a Russian-context topic).
- Recency: if the answer can change over time (prices, versions, "latest", current office-holders, news), plan to add the **current year** to queries and prefer fresh pages.
- **Fetch budget: at most 6 `read_webpage` calls.** Stop early once the sub-claims are confirmed.

## Phase 2 — Search (3–6 queries)

Run `search_web` with varied queries — `key noun + qualifier + year` if time-sensitive. Aim at least 1–2 queries at official / primary sources. Collect candidate URLs into a table, tag each with a quality class, then **pick the top 3–6 by quality, not by search rank**:

- **primary** — the thing itself, official site, `.gov` / `.edu`, a standards body, original docs or paper.
- **secondary** — established news, encyclopedic references, well-known organizations.
- **tertiary** — blogs, forums, vendor marketing. Corroboration only — **never** the sole basis for a claim.

Dedupe by domain before choosing.

## Phase 3 — Read and take evidence notes

For each chosen URL call `read_webpage`, then **immediately** write a compact note and discard the raw page text (don't carry full pages forward — it bloats context and weak models drown in it):

```yaml
- url: <full URL>
  class: primary | secondary | tertiary
  says: <≤2 sentences — what it states about a sub-claim>
  quote: "<≤200 chars, verbatim, proving the point>"
```

Only note pages you actually opened. **Hard stop at the fetch budget.**

## Phase 4 — Cross-check

- A sub-claim is **CONFIRMED** only if ≥2 *independent* sources agree, **or** one primary/authoritative source states it directly.
- Exactly one non-primary source → mark the claim **UNVERIFIED**.
- Sources disagree → **report the disagreement**; do not silently pick a side.

## Phase 5 — Write the answer

```
<2–4 sentence direct answer. Every factual sentence ends with [n].>

Key points
- <claim> [n]
- <claim> [n]

Sources
[1] <page title> — <URL>
[2] <page title> — <URL>

Confidence & gaps
<one line: what is solid, what is single-source/UNVERIFIED, what you could not find>
```

A fuller worked example is in `references/example-report.md` — read it with `read_file` if you want the exact format. You may also save the report with `write_file` if the user asks for a file.

---

## Hard rules (non-negotiable)

1. **No fact without `[n]`.** Can't source it → write "couldn't verify", never guess.
2. **Never invent a URL or a quote.** Cite only pages opened with `read_webpage`.
3. **Respect the fetch budget.** If evidence is thin, ship with explicit gaps — do not pad with unrelated pages.
4. **Prefer primary/official sources.** A tertiary blog may corroborate but may never be the only basis.
5. **Time-sensitive question → include the current year** in queries and prefer recent pages.
6. **Compress every page into a note immediately;** do not carry raw page text into later phases.
