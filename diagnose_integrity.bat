@echo off
setlocal
cd /d "%~dp0"
echo === CMD token ===
whoami /groups | findstr /i "S-1-16"
echo.
echo === base Python directly (no venv launcher) ===
"%USERPROFILE%\AppData\Local\Programs\Python\Python311\python.exe" -c "import sys; sys.path.insert(0, '.'); from argent.integrity import integrity_label; print('base_direct', integrity_label())"
echo.
echo === Python / launcher / parent tokens ===
.venv\Scripts\python.exe scratch\parent_integrity.py
echo.
pause
