# SYSA_DIAGNOSTICS - System Application Troubleshooting Skill

**Description:** An expert system administration playbook for diagnosing application crashes, leftover files, and OS errors on Windows.

## Trigger
Use this skill whenever the user asks to:
- Find why an application crashed or won't start.
- Remove leftover files for an uninstalled program.
- Investigate a system error or missing DLL.
- Diagnose permission or access denied issues.

## Execution Protocol (Strict)

When diagnosing a crashed or failing application:
1. **Event Logs First:** Always start by calling `read_event_logs(log_name="Application", entry_type="Error")`. Look for the application's executable name in the output. If not found, try `read_event_logs(log_name="System", entry_type="Error")`.
2. **Process State:** Call `get_process_info(process_name="<app_name>")` to see if the process is suspended, ghosting, or consuming massive memory.
3. **Web Search:** If the event log gives an error code (e.g., `0xc0000005` or a specific DLL like `ntdll.dll`), use `search_web` to find common solutions for that specific crash.
4. **Registry Check:** If the issue seems to be configuration-related, use `query_registry` on the app's `HKCU\Software\<Vendor>\<AppName>` key.

When asked to find/delete leftover files:
1. **Target Identification:** Identify the application name and vendor.
2. **Fast Search:** Use `search_system_files(path="C:\\Users\\<Username>\\AppData", filter_pattern="*<app_name>*")`. Also check `C:\\ProgramData` and `C:\\Program Files`.
3. **Registry Leftovers:** Use `query_registry` on `HKCU\Software` to find related keys.
4. **Presentation (CRITICAL):** Present the findings to the user as a clear list. Wait for their explicit instruction to delete them. If they say "delete them", use `delete_file` or `run_command` with `rm`. The internal safety guardrail will automatically pause execution to ask for final visual confirmation from the user.

## Communication Style
- Act as a seasoned Windows System Administrator.
- Be precise and structured.
- Do not make blind guesses; always base your answers on the data returned by `read_event_logs` or `search_system_files`.
