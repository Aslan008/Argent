# Unity API pitfalls — detailed reference

Read the section the task touches. Everything here is a place where plausible
code silently misbehaves.

## 1. Execution order (one frame)

```
Awake            all enabled scripts, before any Start; self-init, GetComponent on SELF
OnEnable         after Awake, every time the object is enabled; subscribe here
Start            once, before the first Update; safe to talk to OTHER objects
FixedUpdate      0..N times per frame, fixed timestep; ALL physics
Update           once per frame; input polling, game logic (× Time.deltaTime)
LateUpdate       after all Updates; camera follow, look-at, UI adjust
OnDisable        mirror of OnEnable; unsubscribe here
OnDestroy        cleanup; NOT called on app quit for objects alive at quit
```

- Order of Awake BETWEEN scripts is undefined unless set in Project Settings →
  Script Execution Order. Never rely on "my other script's Awake ran first" —
  read other objects in `Start`, own setup in `Awake`.
- A disabled component still runs `Awake` (when its GameObject activates) but not
  `Start`/`Update`. `enabled = false` stops Update-family only, not coroutines
  started earlier and not OnTriggers.

## 2. Fake null (UnityEngine.Object)

Destroyed objects compare `== null` as true thanks to an operator overload, but
the C# reference itself is NOT null, so:

```csharp
target?.transform.position   // WRONG: bypasses the overload, NRE / stale access
target ?? fallback           // WRONG: same reason
if (target != null) { ... }  // correct
if (target) { ... }          // correct (implicit bool)
```

Also: a field that was never assigned in the Inspector is REAL null; a destroyed
one is fake null. Both pass `== null` — fine — but only if you use `==`/implicit
bool, never `?.`/`??` on Unity types.

## 3. Serialization matrix

| Serialized by Unity | Not serialized |
|---|---|
| public fields | properties (`{get;set;}`) |
| `[SerializeField] private` fields | `static`, `const`, `readonly` fields |
| `List<T>`, arrays, primitives, Unity types | `Dictionary<K,V>`, interfaces |
| custom class/struct with `[System.Serializable]` | fields of a non-Serializable class |

- **Inspector overrides**: once a scene object/prefab exists, its serialized
  values are stored in the scene/prefab file. Editing the default in code does
  nothing for existing instances. Fix: tell the user to update the field in the
  Inspector, or right-click the component header → Reset.
- Renaming a serialized field discards its saved value. Preserve with
  `[FormerlySerializedAs("oldName")]` (`using UnityEngine.Serialization;`).
- `ScriptableObject` = shared data asset. Changes at runtime persist in the
  EDITOR (they edit the asset!) but not in builds — classic confusion. For
  per-run mutable state, copy values into a plain class first.

## 4. Physics

- Move a Rigidbody with `rb.MovePosition(...)` / `rb.linearVelocity` (older:
  `rb.velocity`) in `FixedUpdate`. Moving its `transform` teleports it:
  tunneling, broken collisions.
- `AddForce` default is continuous (mass-dependent, per-physics-step);
  `ForceMode.Impulse` for jumps/hits.
- Collision callback matrix (both sides need a Collider):

| Callback fires | Requirement |
|---|---|
| `OnCollisionEnter` | at least one side has a NON-kinematic Rigidbody; neither collider isTrigger |
| `OnTriggerEnter` | at least one side has a Rigidbody; the sensing collider isTrigger |
| nothing at all | two static colliders (no Rigidbody anywhere) |

- Raycasts ignore the caster only if you use layers/`QueryTriggerInteraction`
  properly; a ray from inside your own collider hits yourself first.
- `CharacterController.Move` ignores physics forces; don't mix it with a
  Rigidbody on the same object.

## 5. Coroutines vs async/await

| | Coroutine | async/await |
|---|---|---|
| Stops when object destroyed/disabled | yes (silently) | NO — keeps running |
| Return values | no (workarounds) | yes (`Task<T>`/`Awaitable<T>`) |
| Main thread | always | after `await` may resume off-thread with some awaitables |

- Guard async methods on MonoBehaviours: `await Task.Delay(1000, destroyCancellationToken);`
  (2022.2+) then check `this != null` before touching members.
- `async void` swallows exceptions — only for event handlers; otherwise `async Task`.
- Cache waits: `static readonly WaitForSeconds Wait1 = new(1f);` — `new WaitForSeconds`
  every loop allocates.
- `StopAllCoroutines()` on disable happens automatically only via destruction/
  SetActive(false) of the GameObject; `enabled = false` does NOT stop coroutines.
- Unity 6: prefer `Awaitable` (`await Awaitable.WaitForSecondsAsync(1f, destroyCancellationToken)`).

## 6. Input — two incompatible worlds

Check `activeInputHandler` (context script prints it). With NEW-only, every
`UnityEngine.Input.*` call throws `InvalidOperationException` at runtime.

```csharp
// LEGACY (Input Manager)
float h = Input.GetAxis("Horizontal");
if (Input.GetKeyDown(KeyCode.Space)) Jump();

// NEW (Input System) — polling style, no asset needed
using UnityEngine.InputSystem;
float h = 0;
if (Keyboard.current.aKey.isPressed) h = -1;
if (Keyboard.current.dKey.isPressed) h = 1;
if (Keyboard.current.spaceKey.wasPressedThisFrame) Jump();
// (action-asset style is preferred for real games — see patterns.md)
```

## 7. Render-pipeline differences

| | Built-in | URP | HDRP |
|---|---|---|---|
| Default lit shader | `"Standard"` | `"Universal Render Pipeline/Lit"` | `"HDRP/Lit"` |
| Main color property | `_Color` | `_BaseColor` | `_BaseColor` |
| Main texture property | `_MainTex` | `_BaseMap` | `_BaseColorMap` |

- Wrong property name = no error, no effect. Use
  `material.SetColor(Shader.PropertyToID("_BaseColor"), c)` with the right name.
- `renderer.material` INSTANTIATES a copy (per-object tint; must `Destroy(mat)`
  on cleanup or it leaks). `renderer.sharedMaterial` edits the asset itself —
  never mutate it in editor-mode code.
- Camera stacking, post-processing, and shader includes all differ per pipeline —
  don't port Built-in image-effect code into URP; use Volume components.

## 8. Deprecation map (write for the project's version!)

| Old API | Since | Use instead |
|---|---|---|
| `FindObjectOfType<T>()` | 2023.1/Unity 6 (obsolete) | `FindFirstObjectByType<T>()` / `FindAnyObjectByType<T>()` |
| `FindObjectsOfType<T>()` | 2023.1 | `FindObjectsByType<T>(FindObjectsSortMode.None)` |
| `rb.velocity` | Unity 6 (renamed) | `rb.linearVelocity` |
| `WWW` | 2018 | `UnityWebRequest` |
| `Application.LoadLevel` | 5.3 | `SceneManager.LoadScene` (`using UnityEngine.SceneManagement;`) |
| GUI/OnGUI for game UI | — | uGUI (Canvas) or UI Toolkit |
| `NavMeshAgent` builtin bake | 2022+ | `com.unity.ai.navigation` package components |

On older projects (2019–2021) the OLD names are correct — matching the project
version beats "modern".

## 9. UI specifics

- Runtime UI is either **uGUI** (Canvas, RectTransform, TextMeshProUGUI, Button)
  or **UI Toolkit** (UIDocument, UXML/USS) — check which the project already uses.
- uGUI requirements: a `Canvas` ancestor, ONE `EventSystem` in the scene (with
  the Input-System UI module if the new input is active), a `GraphicRaycaster`
  for clicks. Missing EventSystem = buttons silently dead — top wiring bug.
- Text: `TextMeshProUGUI` + `using TMPro;`. `text.text = ...` is fine;
  `SetText()` avoids allocations in hot paths.
- Anchors/pivots, not absolute positions, for resolution independence. Setting
  `transform.position` on UI is usually wrong — use `rectTransform.anchoredPosition`.
- World-space health bars: world-space Canvas or screen-space math with
  `Camera.WorldToScreenPoint`.

## 10. Builds & platform fences

- `using UnityEditor;` in runtime code breaks EVERY build (CS0246 only at build
  time). Fence with an `Editor/` folder (assembly excluded from builds) or
  `#if UNITY_EDITOR`.
- Platform code: `#if UNITY_ANDROID`, `UNITY_IOS`, `UNITY_STANDALONE`,
  `UNITY_WEBGL`. WebGL: no threads, no synchronous file IO.
- Paths in builds: read-only bundled data → `Application.streamingAssetsPath`
  (on Android it's inside the APK — needs UnityWebRequest); writable →
  `Application.persistentDataPath`. NEVER `Application.dataPath` for writing.

## 11. Compile-error quick map

| Error | Usual cause |
|---|---|
| CS0246 type not found (file exists) | missing asmdef reference, or missing `using`, or Editor type in runtime asmdef |
| CS0104 ambiguous `Random`/`Debug`/`Object` | `using System;` alongside `UnityEngine` — fully qualify |
| CS0120 object reference required | calling instance API from a static context (common in "manager" code) |
| "can't add script component" in editor | file name ≠ class name, or compile errors elsewhere block ALL imports |
| MissingReferenceException at runtime | using a destroyed object — see Fake null; unsubscribe events |
| NullReferenceException in `Start` | Inspector field unassigned, or race with another script's Awake — move to Start / null-check |
