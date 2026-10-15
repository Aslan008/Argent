@echo off
setlocal
cd /d "%~dp0"

rem Argent launcher.
rem ASCII text and CRLF line ends on purpose: cmd.exe reads a batch file with LF-only
rem lines or UTF-8 text under an OEM code page wrongly, runs the tail of a comment as a
rem command and eats the start of the next line. "cd /d" then fails, the window opens in
rem the wrong folder and closes at once, but only on some runs.

set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo Argent: "%PY%" was not found. Create the project virtual environment first.
    pause
    exit /b 1
)
if not exist ".argent" mkdir ".argent" >nul 2>nul

rem Prefer Windows Terminal when it is installed.
where wt.exe >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo %DATE% %TIME% wt.exe>> ".argent\launch.log"
    start "" wt.exe --title "Argent Coder" -d "%CD%" "%PY%" run_tui.py
) else (
    echo %DATE% %TIME% console window>> ".argent\launch.log"
    start "Argent Coder" "%PY%" run_tui.py
)
