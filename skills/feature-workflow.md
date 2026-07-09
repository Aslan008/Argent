---
name: feature-workflow
description: End-to-end workflow for building a NEW feature — one-question-at-a-time interview, spec + prototype approval, implementation with a deviation journal, then a review document and a short comprehension quiz for the user. Use when the user asks to design or build a feature; not for small fixes.
---

# Feature Workflow — interview, prototype, journal, review

For NEW features and substantial changes — not for one-line fixes. Three
phases; every artifact lives in `.argent/artifacts/`, named after a short task
slug (one artifact set per task; overwrite drafts, don't multiply files).

## Phase 1 — Interview & prototype (BEFORE any implementation code)

1. **Explore first** (`read_skill("blind-spot-pass")` for the method): map the
   affected code, report FACTS / RISKS / ASSUMPTIONS. Never interview the user
   about what the code can answer.
2. **Brainstorm**: present 2–3 viable approaches with one-line trade-offs each,
   and recommend one. This is where bad architectures die cheaply.
3. **Interview — ONE question at a time.** This deliberately overrides
   blind-spot-pass batching: a feature earns a real interview.
   - Ask exactly one question per message (`ask_user_questions`, concrete
     options + free answer).
   - Start with the ambiguity whose answer most changes the ARCHITECTURE:
     data model, storage, dependencies, sync vs async, what must survive
     restarts, who else consumes the result.
   - Let each answer shape the next question — that's the point of asking
     singly instead of batching.
   - Stop when the remaining unknowns no longer change the design (usually
     ≤5 questions). Don't drag the interview out.
4. **Prototype before building** — the smallest artifact that makes the design
   concrete and criticizable:
   - an interface sketch: public signatures, data flow, list of files to
     create/modify → `create_artifact("<slug>_spec.md", ...)`; or
   - a runnable vertical slice that touches every layer once; or
   - a visual mockup (`create_svg_image`) when the feature is UI.
5. **Approval gate**: `request_user_approval` on the spec. No full
   implementation before an explicit yes.

## Phase 2 — Implementation with a deviation journal

At the first line of implementation, create
`create_artifact("<slug>_notes.md", "# Implementation notes: <task>\n")`.
Then `append_to_file` an entry EVERY time reality forces you off the approved
spec — an edge case, a project quirk, a hidden coupling:

    ## <short title> (<files touched>)
    - Planned: <what the spec said>
    - Found: <the edge case / quirk that got in the way>
    - Did instead: <the actual change>
    - Why: <one line>

Also journal: surprising couplings you discovered, deliberate workarounds,
TODOs you consciously left. Do NOT journal routine progress — only departures
and discoveries; bullets, not prose.

If a deviation invalidates a decision the user explicitly approved — STOP and
surface it before continuing. The journal records history; it does not replace
consent.

## Phase 3 — Review & comprehension check (AFTER the task is done)

1. **Assemble the review**: `create_artifact("<slug>_review.md", ...)` with:
   - the as-built spec (what actually exists now);
   - "changed vs approved spec" — distilled from the deviation journal;
   - file-by-file summary of changes;
   - how to verify: exact commands / steps / what to click;
   - open TODOs and follow-ups.
2. Give the user the artifact path and a 5–8 line summary in chat.
3. **Mini-quiz** — walk the user through the changes with 3–4 single-choice
   questions (`ask_user_questions`), each about a decision that matters for
   future work: "Where do new X handlers go?", "What happens when Y is
   empty?", "Which file owns the Z config now?". After the answers, confirm
   the correct ones and gently correct the wrong ones with the file/line
   reference — the goal is that the user OWNS the change, not that they pass.
4. The quiz is a comprehension check, not an exam: skip it for tiny changes or
   when the user asks to skip.

## Economy

- Small fix ≠ feature. For ordinary tasks, blind-spot-pass rules (batched
  questions, max 3) apply — don't ceremonialize a rename.
- Spec ≤ 1 page; review ≤ 1.5 pages; journal entries are 4 bullets each.
- One interview question must be able to change the design; if it can't, it
  isn't worth the user's time — assume and note it.
