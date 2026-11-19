# «Как узнать / посмотреть…» — без окна редактора

`<skill>` = папка этого скилла. Все скрипты сами находят корень проекта.

## Содержание
- Проект и окружение
- Код и сборки
- Сцены, префабы, ассеты
- Консоль и ошибки
- Рантайм-поведение
- Картинка и производительность
- Настройки проекта

---

## Проект и окружение

| Вопрос | Как |
|---|---|
| Версия Unity | `cat ProjectSettings/ProjectVersion.txt` |
| Пайплайн (URP/HDRP/Built-in), пакеты | `Packages/manifest.json`; активный пайплайн — `ProjectSettings/GraphicsSettings.asset` (`m_CustomRenderPipeline` → GUID → `unity_find.py path <guid>`) и уровни в `QualitySettings.asset` |
| Открыт ли Unity | `Temp/UnityLockfile` занят? (`unity_batch.py` проверяет); процессы `tasklist \| findstr Unity` |
| Где Unity.exe | Hub: `C:\Program Files\Unity\Hub\Editor\<ver>\Editor\Unity.exe` |
| Сцены в билде | `ProjectSettings/EditorBuildSettings.asset` (`m_Scenes`, пути и enabled) |
| Платформа | `ProjectSettings/ProjectSettings.asset`, `Library/LastBuildTarget`… надёжнее — спросить или посмотреть `EditorUserBuildSettings` из editor-скрипта |

## Код и сборки

| Вопрос | Как |
|---|---|
| Где класс | `unity_find.py script ИмяКласса` или grep `class ИмяКласса\b` |
| В какой он сборке | ближайший вверх `*.asmdef`; нет — `Assembly-CSharp` (`Editor/` → `-Editor`) |
| Что видит сборка | `references` в `.asmdef`; `Assembly-CSharp` видит все asmdef с `autoReferenced: true` |
| Компилируется ли | `fast_check.py` (быстро) → `unity_batch.py` (точно) |
| Консольные команды/атрибуты проекта | grep по атрибуту (`\[ConsoleCommand`, `\[MenuItem`, `\[ContextMenu`) |
| Кто вызывает метод | grep имени + проверить `SendMessage("Имя")`, UnityEvent в YAML (`m_MethodName: Имя`) |

## Сцены, префабы, ассеты

| Вопрос | Как |
|---|---|
| Иерархия сцены/префаба | `unity_find.py scene Assets/…/X.unity [--filter текст]` — объекты, `[off]`, компоненты |
| Значение поля компонента | grep по имени поля в `.unity`/`.prefab` рядом с нужным объектом; точнее — editor-скрипт, читающий `SerializedObject` |
| Кто ссылается на ассет | `unity_find.py refs <путь|guid>` |
| Что за GUID | `unity_find.py path <guid>` (ищет и в `Library/PackageCache`) |
| Свойства материала | `.mat` — YAML: `m_Shader` (GUID), `m_SavedProperties` (`m_Floats`, `m_Colors`, `m_TexEnvs`) |
| Настройки импорта текстуры/модели | её `.meta` (`TextureImporter:` — `mipmaps`, `alphaIsTransparency`, `maxTextureSize`…) |
| Размеры/данные террейна, меша | editor-скрипт: загрузить ассет `AssetDatabase.LoadAssetAtPath` и вывести в лог |

## Консоль и ошибки

| Вопрос | Как |
|---|---|
| Ошибки компиляции | `unity_batch.py` → строки `COMPILE` |
| Ошибки при загрузке сцены/выполнении | свой метод: `OpenScene` + `Application.logMessageReceived` → печать |
| Лог последнего запуска редактора пользователя | Windows `%LOCALAPPDATA%\Unity\Editor\Editor.log` (и `Editor-prev.log`); macOS `~/Library/Logs/Unity/Editor.log`; Linux `~/.config/unity3d/Editor.log` |
| Лог игры (билд/плеер) | Windows `%USERPROFILE%\AppData\LocalLow\<Company>\<Product>\Player.log` |

Editor.log пользователя — лучший источник «что он видит в консоли», когда Unity открыт: читать хвост, искать
`error`, `Exception`, `Shader error`.

## Рантайм-поведение

| Вопрос | Как |
|---|---|
| Работает ли логика | `AgentProbe_TEMP` через `-executeMethod`: сцена в памяти, вызовы, `Check(...)` |
| Play Mode (физика, корутины, сеть) | Unity Test Framework: тесты в сборке с `UnityEngine.TestRunner`, запуск `unity_batch.py --no-quit -- -runTests -testPlatform PlayMode -testResults <tmp>/r.xml` (без `-quit`!), читать XML |
| Edit Mode-тесты проекта | то же с `-testPlatform EditMode` |
| Приватное состояние | рефлексия `GetField/GetMethod(..., BindingFlags.NonPublic \| BindingFlags.Instance)` |

## Картинка и производительность

| Вопрос | Как |
|---|---|
| Как выглядит | `AgentBench_TEMP` с `--graphics` → .jpg → посмотреть изображение; снимок «до» — до правок |
| Сколько стоит кадр | тот же шаблон: медиана из 3 серий после прогрева; сравнение «с фичей / без» в одном запуске |
| Что дорого | выключать подсистемы по одной и смотреть дельту |
| Пиксель/цвет в точке | `Texture2D.GetPixel` после `ReadPixels` — печатать в лог (объективнее, чем «на глаз») |

## Настройки проекта

`ProjectSettings/*.asset` — текстовый YAML: `QualitySettings.asset` (уровни, тени, vSync),
`GraphicsSettings.asset`, `ProjectSettings.asset` (Player), `TagManager.asset` (слои, теги),
`InputManager.asset`/`.inputactions`. Читать можно смело; менять — только с разрешения
(влияет на весь проект и всех участников).
