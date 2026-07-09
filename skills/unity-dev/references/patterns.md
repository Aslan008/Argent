# Unity patterns — ready-to-adapt snippets

Adapt names/namespaces to the project's conventions (context script shows them).
Every snippet already respects the pitfalls reference (lifecycle, fake null,
serialization, caching).

## Cached-references MonoBehaviour skeleton

```csharp
using UnityEngine;

public class Example : MonoBehaviour
{
    [SerializeField] private float speed = 5f;      // tunable in Inspector
    [SerializeField] private Transform target;      // wired in Inspector

    private Rigidbody _rb;                          // cached, never in Update

    private void Awake()  => _rb = GetComponent<Rigidbody>();
    private void OnEnable()  { /* subscribe events here */ }
    private void OnDisable() { /* unsubscribe the SAME events */ }

    private void FixedUpdate()
    {
        if (target == null) return;                 // fake-null-safe
        var dir = (target.position - _rb.position).normalized;
        _rb.MovePosition(_rb.position + dir * (speed * Time.fixedDeltaTime));
    }
}
```

"In Unity:" wiring for a script like this: 1) select the GameObject → Add
Component → Example; 2) drag the target object into the Target field.

## Singleton (persistent manager)

```csharp
public class GameManager : MonoBehaviour
{
    public static GameManager Instance { get; private set; }

    private void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;
        DontDestroyOnLoad(gameObject);              // drop this line for per-scene
    }

    private void OnDestroy()
    {
        if (Instance == this) Instance = null;      // domain-reload safety
    }
}
```

Callers: `if (GameManager.Instance != null) GameManager.Instance.X();` —
never cache it across scene loads.

## Object pool (spawn-heavy things: bullets, VFX, enemies)

Unity 2021+: use the built-in pool.

```csharp
using UnityEngine;
using UnityEngine.Pool;

public class BulletPool : MonoBehaviour
{
    [SerializeField] private Bullet prefab;
    private ObjectPool<Bullet> _pool;

    private void Awake() => _pool = new ObjectPool<Bullet>(
        createFunc: () => Instantiate(prefab),
        actionOnGet: b => b.gameObject.SetActive(true),
        actionOnRelease: b => b.gameObject.SetActive(false),
        actionOnDestroy: b => Destroy(b.gameObject),
        defaultCapacity: 32);

    public Bullet Get() => _pool.Get();
    public void Release(Bullet b) => _pool.Release(b);
}
```

The pooled object must RESET its state in `OnEnable` (velocity, timers, health).

## Input System (action-based, code-only — no .inputactions asset needed)

```csharp
using UnityEngine;
using UnityEngine.InputSystem;

public class PlayerInputSimple : MonoBehaviour
{
    private InputAction _move, _jump;

    private void OnEnable()
    {
        _move = new InputAction("Move", InputActionType.Value);
        _move.AddCompositeBinding("2DVector")
            .With("Up", "<Keyboard>/w").With("Down", "<Keyboard>/s")
            .With("Left", "<Keyboard>/a").With("Right", "<Keyboard>/d");
        _jump = new InputAction("Jump", InputActionType.Button, "<Keyboard>/space");
        _jump.performed += OnJump;
        _move.Enable(); _jump.Enable();
    }

    private void OnDisable()
    {
        _jump.performed -= OnJump;
        _move.Disable(); _jump.Disable();
    }

    private void OnJump(InputAction.CallbackContext _) { /* jump */ }
    private void Update() { Vector2 mv = _move.ReadValue<Vector2>(); /* use mv */ }
}
```

Legacy equivalent: `Input.GetAxisRaw("Horizontal"/"Vertical")`,
`Input.GetKeyDown(KeyCode.Space)` in `Update`.

## Event channel (decoupled, leak-safe)

```csharp
public static class GameEvents
{
    public static event System.Action<int> ScoreChanged;
    public static void RaiseScoreChanged(int v) => ScoreChanged?.Invoke(v);
    public static void Clear() => ScoreChanged = null;   // call on scene unload / app quit
}

// Listener:
private void OnEnable()  => GameEvents.ScoreChanged += OnScore;
private void OnDisable() => GameEvents.ScoreChanged -= OnScore;
```

Static events outlive scenes — the `-=` in OnDisable is NOT optional.

## Timed behaviour (coroutine, allocation-aware)

```csharp
private static readonly WaitForSeconds TickWait = new(0.5f);

private Coroutine _ticker;
private void OnEnable()  => _ticker = StartCoroutine(Tick());
private void OnDisable() { if (_ticker != null) StopCoroutine(_ticker); }

private System.Collections.IEnumerator Tick()
{
    while (true) { DoTick(); yield return TickWait; }
}
```

## Save / load (JSON, correct path)

```csharp
[System.Serializable]
public class SaveData { public int level; public float health; }

public static class SaveSystem
{
    private static string PathFor(string slot) =>
        System.IO.Path.Combine(Application.persistentDataPath, slot + ".json");

    public static void Save(SaveData d, string slot = "save") =>
        System.IO.File.WriteAllText(PathFor(slot), JsonUtility.ToJson(d));

    public static SaveData Load(string slot = "save") =>
        System.IO.File.Exists(PathFor(slot))
            ? JsonUtility.FromJson<SaveData>(System.IO.File.ReadAllText(PathFor(slot)))
            : new SaveData();
}
```

`JsonUtility` follows the serialization matrix (fields only, no Dictionary).

## EditMode test (Unity Test Framework)

Location: an `Editor/`-referenced test assembly, or `Assets/Tests/` with a test
asmdef referencing `UnityEngine.TestRunner`/`UnityEditor.TestRunner` + the code
asmdef. If the project has no asmdefs, tell the user to create the Tests
assembly via Test Runner window (Create EditMode Test Assembly Folder).

```csharp
using NUnit.Framework;

public class DamageMathTests
{
    [Test]
    public void CritMultipliesBaseDamage()
    {
        Assert.AreEqual(20, DamageMath.Compute(10, crit: true));
    }
}
```

User runs it: Window → General → Test Runner → EditMode → Run All. Test pure
logic this way; engine-dependent behaviour is verified in Play Mode by the user.

## "In Unity:" wiring checklist — the template

After code that needs editor setup, ALWAYS output steps like:

```
In Unity:
1. Select the Player object → Add Component → PlayerHealth.
2. Drag HealthBar (Canvas/HealthBar) into the Health Bar field.
3. Press Play; take damage; the bar should shrink. Paste any red Console lines here.
```

Exact object paths, exact field names, one observable outcome to confirm.
