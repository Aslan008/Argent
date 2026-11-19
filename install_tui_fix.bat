@echo off
setlocal
cd /d "%~dp0"
echo Installing the rebuilt TUI extension from:
echo   %~dp0scratch\wheels\argent_tui-0.1.0-cp311-cp311-win_amd64.whl
echo.
echo Close Argent first, then run this from a NORMAL cmd (no administrator).
echo.

rem An install attempted while Argent was running leaves the old package renamed
rem to ~-gent_tui; while it exists an import can resolve the stale copy of the
rem compiled module and silently run pre-build code.
if exist ".venv\Lib\site-packages\~-gent_tui" (
echo Removing the stale leftover from an interrupted install...
rmdir /s /q ".venv\Lib\site-packages\~-gent_tui"
)

rem An install done over a running Argent renames the live .pyd aside instead of
rem replacing it. Each one is ~2 MB and they added up to eight before anyone
rem noticed; the one just renamed is still held by the process and stays until
rem Argent closes, so this sweep never fails the install.
if exist ".venv\Lib\site-packages\argent_tui" (
for %%F in (".venv\Lib\site-packages\argent_tui\*.pyd.old-*") do del /q "%%~fF" 2>nul
)

.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps "%~dp0scratch\wheels\argent_tui-0.1.0-cp311-cp311-win_amd64.whl"
echo.
rem Say which build actually landed: "I installed it" is not the same as "the running
rem Argent shows the new behaviour", and those two were confused twice already.
.venv\Scripts\python.exe -c "import argent_tui, os, time; p = argent_tui.__file__; print('argent_tui:', p); print('built    :', time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(p))))"
echo.
echo Restart every Argent window to pick this build up. A running one keeps the old
echo code in memory until it closes - the import check above proves the FILE, not the
echo process.
echo.
pause
