@echo off
setlocal
cd /d "%~dp0"
echo === Before ===
icacls ".venv\Scripts\python.exe" | findstr /i "S-1-16"
echo.
echo === Setting Medium integrity on .venv ===
icacls ".venv" /setintegritylevel Medium /C
icacls ".venv" /setintegritylevel Medium /T /C
echo.
echo === After ===
icacls ".venv\Scripts\python.exe" | findstr /i "S-1-16"
echo.
echo Run the check in a NORMAL (non-admin) cmd:
echo   .venv\Scripts\python.exe -c "from argent.integrity import integrity_label; print(integrity_label())"
echo.
pause
