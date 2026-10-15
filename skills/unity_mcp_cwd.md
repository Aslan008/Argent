# Unity MCP: 0 tools / unreachable — диагностика и фикс

## Симптом
`list_mcp_tools` → unity: 0 tools или unreachable. В `~/.argent/logs/mcp.log`:
`tools/list failed ... (no tools returned)`. Сервер при этом жив (initialize OK).

## Причина (две, обе про CWD)
1. **CWD не корень Unity-проекта.** `unity mcp` ищет редактор строго по
   `<CWD>/Library/Pipeline/.unity-pipeline-port`, вверх по дереву НЕ ищет.
2. **Порядок событий при старте Argent.** MCP-серверы спавнятся ДО того, как
   восстановление сессии делает `os.chdir` в папку сессии (main.py:227-243).
   Если Argent запущен из его собственной папки (`Desktop\Argent`), спавн
   наследует её CWD; у Argent там свой маркер `.argent`, поэтому
   `project_root_or_cwd()` возвращает саму папку Argent, а не Unity-проект.

## Фикс (уже применён в mcp_client.py)
- `StdioTransport.__init__(..., cwd=None)` → `Popen(cwd=self._cwd)`.
- `_MCPServer.start()` для stdio: `cwd=self.config.get("cwd") or project_root_or_cwd()`.
- Per-server override: поле `"cwd"` в конфиге сервера имеет приоритет.

## Быстрый фикс без перезапуска Argent
CWD у процесса Argent уже правильный (после переключения сессии), поэтому:
```
/mcp stop unity
/mcp start unity
```
Новый спавн возьмёт текущий CWD → корень проекта → 160 инструментов.

Альтернатива: убить процесс `unity.exe mcp` — `call_tool` авто-респавнит
мертвый stdio-сервер (mcp_client.py, строка ~674) с текущим CWD.

## Диагностика
- Лог: `Get-Content "$env:USERPROFILE/.argent/logs/mcp.log" -Tail 40`
- Фактический CWD процесса: чтение PEB через NtQueryInformationProcess
  (ProcessParameters → CurrentDirectory.DosPath). Диагностический скрипт
  get_cwd_probe.py был временным и удалён; при необходимости воссоздать —
  ctypes: OpenProcess(PROCESS_VM_READ|QUERY) → NtQueryInformationProcess(0)
  → PEB+0x20 → params+0x38 (UNICODE_STRING DosPath).
- Процессы: `Get-CimInstance Win32_Process | ? { $_.Name -eq "unity.exe" -and $_.CommandLine -match "mcp" }`

## Правило
Не убивать процессы пользователя без явного разрешения (дважды прерывали
Stop-Process). Не удалять файлы без запроса (прервали удаление get_cwd_probe.py).