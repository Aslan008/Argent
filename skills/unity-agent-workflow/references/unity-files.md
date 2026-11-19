# Файлы Unity: что можно править текстом и как

| Файл | Формат | Править текстом? |
|---|---|---|
| `.cs`, `.hlsl`, `.cginc`, `.shader`, `.compute` | текст | да — основная работа |
| `.uxml`, `.uss` | XML / CSS-подобный | да; имена элементов (`name=`) связаны с `Q<T>("имя")` в C# |
| `.asmdef`, `.asmref` | JSON | да; осторожно с `references` (GUID: или имя) |
| `.meta` | YAML | почти никогда; `guid` — НЕ менять; настройки импорта — лучше через `AssetImporter` в editor-скрипте |
| `.unity`, `.prefab`, `.mat`, `.asset`, `.controller` | YAML (Force Text) | значения полей — можно; структуру — через API |
| `.shadergraph`, `.shadersubgraph` | JSON (поток объектов) | можно с резервной копией и проверкой импорта |
| `ProjectSettings/*.asset` | YAML | читать — да, менять — с разрешения |
| `Library/`, `Temp/`, `obj/`, `Logs/` | кэш | не трогать |
| `*.csproj`, `*.sln(x)` в корне | генерируются Unity | не править (перезапишутся); `fast_check` делает копии |

## .meta и GUID
- Каждому ассету — `.meta` с `guid`. Все ссылки в сценах/префабах/материалах — `{fileID, guid, type}`.
- Новый файл — `.meta` не создавать: Unity создаст при импорте.
- Удаление/перенос ассета — вместе с `.meta`. Иначе: при удалении — мусорный `.meta` (Unity удалит сам, предупредив);
  при переносе без `.meta` — новый GUID → все ссылки «Missing».
- Скрипт MonoBehaviour: имя файла = имя класса, иначе компонент нельзя добавить и ссылки ломаются.

## YAML сцен и префабов
```
--- !u!1 &123456          ← класс 1 = GameObject, fileID 123456
GameObject:
  m_Name: Player
  m_IsActive: 1
  m_Component:
  - component: {fileID: 123457}
--- !u!114 &123458        ← 114 = MonoBehaviour
MonoBehaviour:
  m_GameObject: {fileID: 123456}
  m_Script: {fileID: 11500000, guid: <guid скрипта>, type: 3}
  speed: 5                ← сериализованные поля по именам
```
- Значение поля поменять текстом можно (`speed: 5` → `speed: 7`) — Unity подхватит при загрузке.
- Новые объекты/компоненты текстом не добавлять: fileID должны быть уникальны, а связи (Transform-родитель,
  m_Component) согласованы. Для этого — editor-скрипт (`new GameObject`, `AddComponent`, `SaveScene`).
- Префаб-инстанс в сцене хранит только **изменения** (`m_Modifications`: `propertyPath` + `value`).
- `stripped` — объект-заглушка префаба.

## Shader Graph (JSON)
- Файл — несколько JSON-объектов подряд, у каждого `m_ObjectId`; граф ссылается на узлы, узлы — на слоты,
  рёбра — `m_OutputSlot`/`m_InputSlot` с `m_Node.m_Id` и `m_SlotId`.
- Порядок: резервная копия → правка скриптом (`patcher.py` или JSON-разбор по объектам) → `unity_batch.py --graphics`
  (импорт без ошибок) → снимок кадра, если это видимый эффект.
- Проще и надёжнее часто: Custom Function-узел с `.hlsl`-файлом — логика в тексте, в графе один узел.

## Настройки импорта
Через editor-скрипт, а не правкой `.meta`:
```csharp
var imp = (TextureImporter)AssetImporter.GetAtPath(path);
imp.mipMapsPreserveCoverage = true; imp.alphaTestReferenceValue = 0.5f;
imp.SaveAndReimport();
```

## Кодировки и переводы строк
Unity-проекты на Windows часто CRLF и UTF-8 с BOM. Сохраняй как было (`patcher.py` делает это сам) —
иначе diff на весь файл и шум в git. Русский текст в C#/UXML — UTF-8.
