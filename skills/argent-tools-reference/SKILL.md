---
name: argent-tools-reference
description: Full reference for Argent tools whose schemas keep only a short line — rooms, wait_heartbeat, search_web, run_subagent, manage_automations, todo_write, ultracode. Read this when a tool's one-line description is not enough to make the call.
version: 1.0.0
---

# Argent tools — the long form

A tool description is paid for in **every request**, so the schemas keep one line
each: what it does, when not to use it, what comes back. The detail lives here and is
read only when it is actually needed.

## Which tool when

The overlaps, resolved once, so nobody has to guess from the descriptions:

**Two browsers, one page.** `browser_open/action/inspect/close` own the page: opening,
navigation, clicking by selector, reading its state. `leeth_scan_page` / `leeth_click_many` /
`leeth_fill_form` / `leeth_map` work through the SAME `browser_engine` session and exist for
volume — "scan every post / bookmark and keep the ones about X" — because one batched local
pass costs milliseconds where a per-element click costs a model round trip. Never open or
navigate with `leeth_*`.

**Three ways to find things.** `grep_search` searches INSIDE files (regex, plus LSP for
definitions/references). `search_files` finds files BY NAME or glob. `list_directory` shows
one level. None of them replaces another.

**Two screenshots.** `capture_desktop_window` for one window — it reports the window's
rectangle, so you can crop and know what you got. `capture_desktop_screen` for the whole
screen or a fullscreen game, which has no window to target.

**Two commands.** `run_command(admin=true)` is the elevated variant of the same tool; there
is no separate admin tool to choose between any more.

**Five ways to delegate.** `run_subagent` — one self-contained part, returns a report.
`subagents` — manage those (list/result/apply/discard). `rooms` — a repeatable recipe with
checks and hard limits. `ultracode` — a big job that needs a plan, workers and a critic.
`delegate_to_leeth` — routine local actions (read, list, grep, run a command), no model
cost. One part → subagent. Repeats → room. Large and needs review → ultracode. Mechanical
→ leeth.


Run a reusable, bounded RECIPE (a "room") instead of improvising a procedure.

Built-in rooms: `fix_bug`, `debug_loop` (red tests → fix → re-run until green),
`code_review` (the uncommitted diff), `implement_feature` (plan → the person approves →
code → tests → review), `release_check`, `research` (web question with sources),
`explain_repo`, `analyze`.

Rooms have hard step/token/time limits, verdicts come from processes (the test command's
exit code), risky steps (a commit) ask the person, and a report comes back.

Actions: `list` (rooms with their parameters), `show` (draw one), `run` (executes it
autonomously with real tools and sub-agents — the person is asked first, every time — and
blocks until the report), `resume` (continue an interrupted run), `build` (write a new
room from the person's description), `edit` (a second model rewrites the whole room).

For precise work use `patch` (change an existing room) and `create` (a new room put
together block by block): you give a list of point operations in `ops` and nothing else
is touched; they apply all together or not at all. Whatever the way, the person sees the
drawn flow and decides whether it is saved.

`ops` — each an object with `op`:

- `add_node {node:{id,type,...}}`
- `insert_node {node:{...}, between:[from,to]}` (a block ON an arrow)
- `remove_node {node, reconnect:true}`
- `rename_node {node,new_id}`
- `set_param {node, field, value}` or `{node, values:{field:value}}` (any block parameter;
  `null` resets it; dotted paths like `args.file_path`, `sampling.temperature`,
  `budget.max_tokens`, `sets.ok`)
- `connect {from,to, if:'state.ok' | else:true | while:'..',max,id | on_error:true}`
  (`to` may be `'EXIT:done'`)
- `disconnect {from,to, if|else|all}`
- `set_edge {from,to, if|else (which arrow), set:{if,else,while,max,id,on_error,to}}`
- `set_room {field,value}` (description, budget, entry, defaults, next, tags, report,
  rules; dotted paths too)

Call `rooms(action='show')` first to see the ids and parameters. Use a room when the task
matches one; for anything else work as usual.

## wait_heartbeat

Pause execution while something finishes: a compilation, a server starting, a long
background task.

**Never** for web pages or browser tasks (file/video upload, page loading) — it only
monitors local filesystem files and background OS processes. For browser work use
`browser_action(action="wait_for")`.

Prefer `until`: waking costs a full turn, so polling "is it done yet?" ten times costs ten
turns, while a condition is checked locally for free and wakes you once — when it actually
happened. Fall back to `delay_seconds` only when nothing observable marks the end.

Checks: `file_exists(path)`, `file_missing(path)`, `file_contains(path, text)`,
`process_finished(pid)`, `process_running(pid)`, `subagent_finished(id)`; combine with
`and` / `or` / `not`.

```
until='file_contains("build.log", "BUILD SUCCESSFUL")'
until='process_finished("3")', timeout_seconds=1200
until='file_exists("dist/app.exe") and file_missing(".lock")'
```

You are told whether the condition was MET or the wait TIMED OUT — a timeout is not
success, so re-check before assuming the event happened.

If the result says the wait does NOT block (a watch was registered), do not sleep or poll:
finish your turn with a short message (what is running, what you wait for); you are woken
with a `[Heartbeat ...]` message when it happens.

## search_web

Searches several engines at once (DuckDuckGo, Wikipedia, StackOverflow, GitHub issues, and
Brave when configured) and merges the results. Each engine is given only the operators it
understands.

- `"exact phrase"` — the words must appear together, verbatim. The single most effective
  technique for error messages: `"NullReferenceException in LoadAsync"`.
- `site:docs.unity3d.com` — restrict to one domain (vendor docs, a specific forum).
- `filetype:pdf` (also `ext:`) — restrict to a file type.
- `-word` — exclude results containing it, e.g. `-tutorial` to skip beginner pages.
- `intitle:word` — the word must be in the TITLE, not merely mentioned; translated to each
  engine's native form (GitHub's `in:title`, StackOverflow's title filter).

Combine them: `site:forum.unity.com "Addressables" memory leak -tutorial`. Quoted phrases
work everywhere; the rest are applied where supported and dropped where not, so writing
them is never harmful.

This returns only short snippets and URLs. To read the content, call `read_webpage`.

## run_subagent

Spawn a specialized sub-agent for a specific part of a complex task, in isolation.

`background=true` returns at once and the sub-agent works while you do something else —
start several in ONE response to run them in parallel (up to 4), then collect the reports.

`isolation='worktree'` gives it a private copy of the repository: it can edit freely
without touching the person's files or its sibling agents, and its work comes back as a
patch you apply (`subagents action='apply'`) or drop. Use worktree for any sub-agent that
WRITES code while others or you are working.

A sub-agent cannot ask the person anything and can run only shell commands the person
already allowed; give it a self-contained task — it does not see this conversation.

Cheaper helpers: `model` / `provider` / `thinking` choose what it runs on, so routine work
can go to a smaller or local model or think less. Empty means the configured default, and
that default is your own model and thinking — nothing costs more unless you ask. Call
`subagents(action='models')` to see what is actually available before choosing.

## manage_automations

Set up, inspect or run work that happens by itself on a schedule while the person is away:
monitoring a page or repository, a daily digest, a recurring check, a reminder. Use it when
the person asks for something "every day", "every hour", "by schedule", "remind me", or
"keep an eye on".

Modes: `agent` (a model works on `task` with limited `tools` — the default),
`command` (`task` is a shell command run as-is, no model, cheapest for tests/backups/health
checks), `remind` (`task` is just a message).

Schedules: `every 30m`, `every 2h on weekdays between 09:00 and 18:00`, `daily at 09:00`,
`weekdays at 09:00`, `mon,fri at 18:30`, `monthly on 1 at 09:00`, `in 2h`,
`tomorrow at 08:30`, `on start` (Russian works too).

An unattended run cannot ask questions or get approvals: give it only the `tools` it needs
and list in `commands` the exact shell commands it may run (`*` matches arguments, never
`;` `&` `|`). Every change asks the person for approval.

`notify`: `auto` = failure or something new; `changes` = report differs from last time;
`always` = every run; `failures`, `never`.

## todo_write

Keep a checklist of the task and show it to the person. Use it for any job with three or
more steps, and whenever the person gives several things to do: write the steps FIRST, then
work through them.

Two ways to call it: pass `todos` (the complete list, in order) to write or replace the
plan; pass `index` with `status` and/or `content` to change ONE item in place — that is
what a status tick should cost, not a resend of every item. Item numbers are printed in the
reply and in the list.

Rules: exactly ONE item is `in_progress` at a time; mark an item completed the moment it is
really done (tests passed, file written) — not before and not in a batch at the end; add
steps you discover on the way. Not for a one-step request or a plain question.

## ultracode

Run a BIG job as a planned, checked workflow (Ultracode): a plan of steps, workers in
isolated copies of the tree, the step's own checks, an independent critic reading every
patch, an advisor reviewing the plan and the result.

Use it for work that touches several files or modules and has checks; NOT for questions,
explanations or small edits. It works only while the person has switched Ultracode on
(`/ultracode on`). Blocks until the workflow ends and returns its report, which says which
steps were done, what failed and what is left. Put the WHOLE job, every requirement and the
checks the person named into `goal`. `action='resume'` continues a stopped workflow.
