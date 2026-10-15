@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Argent launcher.
rem ASCII text and CRLF line ends on purpose: cmd.exe reads a batch file with LF-only
rem lines or UTF-8 text under an OEM code page wrongly, runs the tail of a comment as a
rem command and eats the start of the next line. "cd /d" then fails, the window opens in
rem the wrong folder and closes at once, but only on some runs.
rem
rem Why it waits for a mark file: wt.exe is an app execution alias. When Windows cannot
rem activate it (it happens until Explorer is restarted) the alias stub opens a console,
rem prints an error and closes it - a window that flashes and vanishes - and Argent never
rem starts. run_tui.py writes .argent\launch.ok as its very first action, so this script
rem knows whether the interpreter really started and, when it did not, tries the next
rem way (a plain console host), and finally stays open with the reason instead of flashing.

set "PY=%CD%\.venv\Scripts\python.exe"
set "MARK=%CD%\.argent\launch.ok"
set "LOG=%CD%\.argent\launch.log"
set "ENTRY=run_tui.py"
if defined ARGENT_ENTRY set "ENTRY=%ARGENT_ENTRY%"

if not exist "%PY%" (
    echo Argent: "%PY%" was not found. Create the project virtual environment first.
    pause
    exit /b 1
)
if not exist ".argent" mkdir ".argent" >nul 2>nul
if exist "%MARK%" del /q "%MARK%" >nul 2>nul

where wt.exe >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo %DATE% %TIME% wt.exe>> "%LOG%"
    start "" wt.exe --title "Argent Coder" -d "%CD%" "%PY%" %ENTRY%
    call :wait_started 15
    if exist "%MARK%" exit /b 0
    echo %DATE% %TIME% wt.exe did not start Argent within 15 s>> "%LOG%"
)

echo %DATE% %TIME% conhost.exe>> "%LOG%"
start "Argent Coder" conhost.exe "%PY%" %ENTRY%
call :wait_started 15
if exist "%MARK%" exit /b 0
echo %DATE% %TIME% conhost.exe did not start Argent within 15 s>> "%LOG%"

echo %DATE% %TIME% console window>> "%LOG%"
start "Argent Coder" "%PY%" %ENTRY%
call :wait_started 15
if exist "%MARK%" exit /b 0
echo %DATE% %TIME% no way of starting Argent worked>> "%LOG%"

echo.
echo Argent did not start: the interpreter was launched three ways and none reported back.
echo   Interpreter: %PY%
echo   Entry point: %ENTRY%
echo   Log:         %LOG%
echo   Errors:      %CD%\.argent\tui_error.log  and  %CD%\.argent\tui_crash.log
echo.
echo If this keeps happening, restart Explorer (Task Manager - Windows Explorer - Restart).
echo.
pause
exit /b 1

:wait_started
set /a TRIES=%~1
:wait_loop
if exist "%MARK%" exit /b 0
if %TRIES% leq 0 exit /b 1
set /a TRIES-=1
ping -n 2 127.0.0.1 >nul
goto wait_loop
