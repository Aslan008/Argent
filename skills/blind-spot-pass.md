---
name: blind-spot-pass
description: Shrink the unknowns in a task BEFORE building — classify what's stated, what's openly unclear, what the user will only recognize on sight, and what nobody thought of. Run it for ambiguous or large tasks, unfamiliar code, and hard-to-reverse changes; then refine the prompt into a working contract.
---

# Blind Spot Pass — shrink the unknowns before you build

User prompts have gaps. Your job is to close them WITH the user, cheaply — not
to guess silently and build the wrong thing, and not to interrogate them about
trivia. Every task splits into four quadrants; each has its own correct move.

## The four quadrants

1. **Known knowns** — requirements stated in the prompt.
   Move: restate them in 2–4 bullets — this is the contract. Pin it with
   `set_goal(objective=..., current_task=...)` so it survives long sessions.

2. **Known unknowns** — things the user KNOWS they haven't decided
   ("which library?", "какой дизайн?"). Move: make each explicit, then route it:
   - answerable from the code/docs → investigate yourself, report the answer;
   - a genuine user decision → structured question (see Rules of economy);
   - low impact → proceed under a NAMED assumption (see Assumption protocol).

3. **Unknown knowns** — criteria the user can't articulate but will instantly
   recognize ("сделай красиво", game feel, tone of a text). Abstract questions
   are useless here ("what style do you want?" gets you nothing).
   Move: SHOW, don't ask — offer 2–3 concrete variants with trade-offs via
   `ask_user_questions`, or build the smallest vertical slice / mockup first
   and iterate on the reaction.

4. **Unknown unknowns** — risks, constraints and options nobody mentioned.
   No question can reach them; only exploration can. Move: the pass below.

## The pass

Run it for: ambiguous or large tasks, unfamiliar parts of the codebase,
destructive or hard-to-reverse changes, integrations with external systems.

1. **Explore first**: outline / grep / read what the change touches. Map entry
   points, callers, adjacent tests, configs, data formats. Don't ask the user
   anything you can learn here.
2. **List risks**: what can break (hidden couplings, shared state), environment
   and platform constraints, data migrations, performance hot paths, security
   surface, concurrent users of the same code.
3. **List open decisions** the prompt doesn't cover: edge cases, failure
   behaviour, backwards compatibility, naming/UX.
4. **Report compactly**, in exactly this shape:
   - FACTS — established from the prompt + the code (quadrant 1 + investigated 2)
   - RISKS — top 3–5, each with its blast radius
   - ASSUMPTIONS — what you will assume unless corrected (explicit!)
   - QUESTIONS — max 3, highest-impact first, each with concrete options
5. **Ask** the questions with `ask_user_questions` (structured options + free
   answer), one batch — never a wall of prose, never one question per message.
6. **Refine the task**: restate the improved formulation ("Уточнённая
   постановка: …") — this replaces the original prompt as the working
   contract. Pin it with `set_goal`.

## Rules of economy

- A question must CHANGE what you build. If every answer leads to the same
  action — don't ask; assume and note it.
- Max 3 questions per round, batched. Follow-ups only after real progress.
- Trivial, well-specified tasks: skip the pass entirely. Fixing a typo needs
  no interview.
- Never block on style questions you can resolve by mirroring the existing
  codebase — mirroring IS the answer.
- Once the user answers, that decision is settled; don't re-ask variants of it
  later in the session.

## Assumption protocol

State assumptions where the user will see them, AT decision time, in one line:
"Assuming X (because Y) — скажи, если не так." An explicit wrong assumption is
corrected in one message; a silent one is discovered after the build, at the
price of a rework.
