---
name: unity-dev
description: Unity development playbook — correct MonoBehaviour C#, project orientation, API pitfalls, and a verification workflow that works without running Unity. Use for ANY task inside a Unity project (a folder containing Assets/ + ProjectSettings/), from a one-line fix to a whole feature.
---

# Unity Development

You are working in a Unity project. Unity punishes generic-C# instincts: code that
compiles can still silently do nothing, and correct-looking API calls differ by
Unity version, render pipeline, and input mode. Follow this playbook for every task.

## Step 0 — Orient before writing anything

Run the bundled context script (path shown in BUNDLED RESOURCES):

    python <skill>/scripts/unity_context.py <project_root>

It tells you the facts that CHANGE what correct code is:
- **Unity version** → which APIs exist/are deprecated (see references/pitfalls.md §Deprecations).
- **Render pipeline** → shader names AND material property names differ (Built-in `_Color`/`_MainTex` vs URP `_BaseColor`/`_BaseMap`). Wrong ones fail silently.
- **Input handling** → LEGACY `UnityEngine.Input` vs NEW `UnityEngine.InputSystem` are incompatible; using the wrong one throws at runtime.
- **Namespaces / asmdefs / folders** → match the project's own conventions, don't invent new ones.

Then, before writing a new script, `grep_search` for a similar existing one and mirror its style (naming, namespace, [SerializeField] habits, folder).

## Hard rules (breaking any of these wastes the whole session)

1. **Scripts live under `Assets/`** (or `Packages/`). Anywhere else in the project Unity never imports them. (Argent's write_file enforces this — don't fight it.)
2. **Never create or edit `.meta` files.** Unity owns them. When MOVING/RENAMING a tracked file yourself, move/rename its `.meta` in the same step (move_file twice: `X.cs`→`Y.cs`, `X.cs.meta`→`Y.cs.meta`) or all scene/prefab references to it break as "Missing (Mono Script)". Prefer asking the user to rename inside Unity.
3. **File name == MonoBehaviour class name** (`Player.cs` ↔ `class Player`). Mismatch = component can't be attached. One MonoBehaviour per file.
4. **Don't hand-edit `.unity` / `.prefab` / `.asset` YAML.** Scene and Inspector wiring is done in the editor — instead OUTPUT a numbered "In Unity:" checklist for the user (attach component X to Y, drag Z into field W…). Keep it minimal and exact.
5. **Never touch** `Library/`, `Temp/`, `obj/`, `Logs/`, `*.csproj`, `*.sln` — all generated.
6. **Unity API is main-thread only.** No `Task.Run` touching GameObjects/transforms. Use coroutines or `Awaitable` (Unity 6) / `UniTask` if present.
7. **MonoBehaviours have no constructors** and are never `new`-ed. Creation = `AddComponent` / `Instantiate`; initialization = `Awake`.

## Verification loop (you cannot run Unity — work with the user)

- `dotnet build` on Unity csproj is unreliable and Argent skips C# validation in Unity projects. **Your compiler is the user's editor.**
- After each batch of edits, tell the user: *focus the Unity window (it recompiles on focus), then paste any red Console errors here verbatim.*
- Console errors come with `file.cs(line,col)` — read that exact location before fixing; fix the ROOT cause, not the first symptom.
- For pure logic (no engine calls), offer an EditMode test if Unity Test Framework is installed (template in references/patterns.md) — the user runs it via Window → General → Test Runner.
- Something "doesn't work" without errors → it's usually wiring, not code: field not assigned in Inspector, component not attached, wrong scene, EventSystem missing, value overridden by the Inspector (see checklist #4).

## Reduce the unknowns first (non-trivial tasks)

Full method: `read_skill("blind-spot-pass")`. The Unity-flavored quadrants:

- **Stated** (known knowns): the mechanic/fix as asked. Restate it in one
  sentence; pin with `set_goal`.
- **Known unknowns** — decide or ask EARLY, they reshape the code: target
  platform(s) (WebGL has no threads; mobile has tight perf budgets), input
  devices (KB+M / gamepad / touch), must it survive scene loads or saves,
  singleplayer or networked?
- **Unknown knowns** (game feel): never guess feel values. Expose
  `[SerializeField]` tunables with sane defaults and give the user a 30-second
  play-test loop; when feel is the point, offer 2–3 variants via
  `ask_user_questions` (e.g. jump: floaty / snappy / hold-to-rise) instead of
  asking "how should it feel?".
- **Unknown unknowns** (Unity's usual suspects): run `unity_context.py` — the
  classic blind spots ARE pipeline/input/version mismatches; plus scene wiring
  your code silently expects (EventSystem, tags, layers, collision matrix),
  execution-order races, prefab overrides beating code defaults, asmdef fences.

Report FACTS / RISKS / ASSUMPTIONS / max-3 QUESTIONS, then the refined task.

## Self-review checklist — run through this BEFORE submitting any Unity code

1. **Fake null**: destroyed-but-referenced `UnityEngine.Object` breaks `?.` and `??`. Always `if (obj != null)` / `if (obj)` — NEVER `obj?.transform` on Unity types.
2. **Lifecycle**: own refs in `Awake`, event subscribe in `OnEnable`, cross-object reads in `Start`, unsubscribe in `OnDisable`. Physics in `FixedUpdate` (with `Time.fixedDeltaTime`), camera-follow in `LateUpdate`, per-frame logic in `Update` scaled by `Time.deltaTime`.
3. **Serialization**: prefer `[SerializeField] private` over public; Unity serializes fields only (no properties, no `Dictionary`); custom classes need `[System.Serializable]`.
4. **Inspector overrides beat code defaults**: changing `speed = 5f` to `10f` in code does NOT change objects already in scenes/prefabs — their serialized value wins. Say so and tell the user which Inspector field to update (or right-click → Reset).
5. **Cache lookups**: `GetComponent`/`Find*` in `Awake`/`Start`, never in `Update`. Use `CompareTag("X")`, not `tag == "X"`.
6. **Coroutines vs async**: coroutines stop with the disabled/destroyed object (usually what you want); `async void` keeps running after destroy — guard with `destroyCancellationToken` (2022.2+). Cache `WaitForSeconds` in a field.
7. **Physics**: move rigidbodies via `rb.MovePosition`/`linearVelocity` in `FixedUpdate`, never via `transform` (teleports through walls). `OnCollision*` needs a non-kinematic Rigidbody on one side; `OnTrigger*` needs `isTrigger` + a Rigidbody on one side.
8. **Input**: write for the mode the context script reported. NEW system: actions/callbacks; LEGACY: `Input.GetAxis`/`GetKeyDown`. Mixing = runtime exceptions.
9. **Render pipeline names**: URP shader `"Universal Render Pipeline/Lit"`, property `_BaseColor`; Built-in `"Standard"`, `_Color`. Use `Shader.PropertyToID` for repeated access. `material` clones (leak — `Destroy` it or use `sharedMaterial` deliberately); `renderer.material` in edit-mode code corrupts assets.
10. **Version-correct APIs**: Unity 2023.1+/Unity 6: `FindObjectOfType<T>()` is obsolete → `FindFirstObjectByType<T>()` / `FindObjectsByType<T>(FindObjectsSortMode.None)`. Full map in references/pitfalls.md.
11. **UI**: text is `TextMeshProUGUI` (`using TMPro;`), not legacy `Text`. UI needs a Canvas AND an EventSystem in the scene (wiring checklist!). Button handlers: subscribe in code or tell the user to wire OnClick.
12. **Events**: every `+=` has a `-=` in the mirror callback (`OnEnable`/`OnDisable`); static events outlive scene reloads — clear or unsubscribe, or the handler fires on a destroyed object.
13. **Editor-only code**: anything `using UnityEditor` goes in an `Editor/` folder or inside `#if UNITY_EDITOR` … `#endif`, or every build fails.
14. **asmdef fences**: "type or namespace not found" while the file clearly exists = missing asmdef reference; add it to the consuming .asmdef's `references`, don't duplicate code.
15. **Ambiguity traps**: `Random` (qualify `UnityEngine.Random`), `Debug`, `Object` when `System` namespaces are also imported (CS0104).
16. **Hot-path hygiene**: no LINQ / string concat / allocations in `Update`; hash animator params (`Animator.StringToHash`), cache shader IDs.
17. **Rotation**: compose `Quaternion`s (`rotation * Quaternion.Euler(...)`, `Slerp`); don't accumulate eulerAngles (gimbal/wrap bugs).

## Task recipes

- **New feature**: orient → find similar code → write minimal scripts → self-review checklist → "In Unity:" wiring list → user compiles → fix Console errors verbatim.
- **Bug fix**: get the exact Console error/stack or a description of observed vs expected → read the code at the stack location → root-cause fix → checklist items 1–4 are the usual suspects.
- **Refactor/rename**: grep ALL usages first; class/file renames follow Hard Rule 2/3; SerializedField renames lose Inspector data unless `[FormerlySerializedAs("oldName")]` is added — always add it.
- **Performance**: don't guess — ask for Profiler evidence (or add a `Profiler.BeginSample`), then apply #16, pooling (references/patterns.md), and batching-friendly changes.

## Going deeper

Read the bundled references when the task touches their area (paths in BUNDLED RESOURCES):
- `references/pitfalls.md` — lifecycle order, serialization matrix, collision matrix, deprecation map by version, URP/Built-in property tables, coroutine/async details.
- `references/patterns.md` — ready-to-adapt snippets: singleton, object pool, input (both systems), event channel, EditMode test, save/load.
