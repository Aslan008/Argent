@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

title Установка Argent Coder

echo ============================================================
echo          Argent: Автоматическая установка (1-Click)
echo ============================================================
echo.

rem ------------------------------------------------------------
rem 1. Поиск подходящей версии Python (>= 3.11)
rem ------------------------------------------------------------
echo [1/4] Поиск интерпретатора Python...

set "PYTHON_EXE="
set "USE_UV=0"

where uv >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "USE_UV=1"
    echo   [OK] Обнаружен менеджер пакетов uv.
)

rem Проверяем launcher py на наличие Python 3.11
py -3.11 -c "import sys" >nul 2>nul
if %ERRORLEVEL% equ 0 (
    for /f "delims=" %%I in ('py -3.11 -c "import sys; print(sys.executable)"') do set "PYTHON_EXE=%%I"
    goto :python_detected
)

rem Проверяем launcher py на наличие Python 3.12
py -3.12 -c "import sys" >nul 2>nul
if %ERRORLEVEL% equ 0 (
    for /f "delims=" %%I in ('py -3.12 -c "import sys; print(sys.executable)"') do set "PYTHON_EXE=%%I"
    goto :python_detected
)

rem Проверяем generic py -3 (версия >= 3.11)
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
if %ERRORLEVEL% equ 0 (
    for /f "delims=" %%I in ('py -3 -c "import sys; print(sys.executable)"') do set "PYTHON_EXE=%%I"
    goto :python_detected
)

rem Проверяем python из системного PATH (версия >= 3.11)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
if %ERRORLEVEL% equ 0 (
    for /f "delims=" %%I in ('python -c "import sys; print(sys.executable)"') do set "PYTHON_EXE=%%I"
    goto :python_detected
)

:python_detected
if not defined PYTHON_EXE (
    echo.
    echo ============================================================
    echo [ОШИБКА] Не найден Python версии 3.11 или выше.
    echo ============================================================
    echo Для работы Argent требуется Python 3.11 или 3.12 64-bit.
    echo.
    echo Инструкция по установке:
    echo   1. Скачайте Python с официального сайта:
    echo      https://www.python.org/downloads/
    echo   2. Запустите скачанный файл установщика.
    echo   3. ВАЖНО: На первом экране обязательно отметьте галочку:
    echo      [X] Add python.exe to PATH
    echo   4. После завершения установки снова запустите install.bat.
    echo.
    pause
    exit /b 1
)

echo   [OK] Выбран Python: %PYTHON_EXE%
"%PYTHON_EXE%" -c "import sys; print('       Версия:', sys.version.split()[0])"

rem ------------------------------------------------------------
rem 2. Создание или проверка виртуального окружения (.venv)
rem ------------------------------------------------------------
echo.
echo [2/4] Подготовка виртуального окружения (.venv)...

if not exist ".venv\Scripts\python.exe" (
    echo   Создание изолированного окружения .venv...
    if %USE_UV% equ 1 (
        uv venv .venv --python "%PYTHON_EXE%"
    ) else (
        "%PYTHON_EXE%" -m venv .venv
    )
    if errorlevel 1 (
        echo   [ОШИБКА] Не удалось создать виртуальное окружение .venv.
        pause
        exit /b 1
    )
    echo   [OK] Окружение .venv успешно создано.
) else (
    echo   [OK] Окружение .venv уже существует.
)

set "VENV_PY=%CD%\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo   [ОШИБКА] Не найден файл "%VENV_PY%".
    pause
    exit /b 1
)

rem ------------------------------------------------------------
rem 3. Установка зависимостей проекта
rem ------------------------------------------------------------
echo.
echo [3/4] Установка зависимостей проекта...

if %USE_UV% equ 1 (
    echo   Установка пакетов через uv...
    uv pip install --python "%VENV_PY%" -e .
    uv pip install --python "%VENV_PY%" maturin
) else (
    echo   Обновление pip, setuptools, wheel...
    "%VENV_PY%" -m pip install --upgrade pip setuptools wheel --quiet
    echo   Установка Argent и базовых библиотек...
    "%VENV_PY%" -m pip install -e .
    "%VENV_PY%" -m pip install maturin
)

if errorlevel 1 (
    echo   [ОШИБКА] Не удалось установить зависимости.
    pause
    exit /b 1
)
echo   [OK] Зависимости успешно установлены.

rem ------------------------------------------------------------
rem 4. Настройка нативного Rust TUI (argent_tui)
rem ------------------------------------------------------------
echo.
echo [4/4] Проверка нативного интерфейса (Rust TUI)...

set "TUI_READY=0"
"%VENV_PY%" -c "import argent_tui" >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "TUI_READY=1"
    echo   [OK] Модуль argent_tui уже готов к работе.
    goto :check_core
)

rem Попытка установить из prebuilt пакета wheels\*.whl
if exist "wheelsrgent_tui*.whl" (
    echo   Установка prebuilt argent_tui из wheels...
    "%VENV_PY%" -m pip install --force-reinstall --no-deps wheelsrgent_tui*.whl >nul 2>nul
)
if exist "scratch\wheelsrgent_tui*.whl" (
    echo   Установка prebuilt argent_tui из scratch\wheels...
    "%VENV_PY%" -m pip install --force-reinstall --no-deps scratch\wheelsrgent_tui*.whl >nul 2>nul
)

"%VENV_PY%" -c "import argent_tui" >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "TUI_READY=1"
    echo   [OK] argent_tui успешно установлен из wheel-пакета.
    goto :check_core
)

rem Если пакета нет, проверяем cargo
where cargo >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo   Компиляция Rust TUI через cargo и maturin...
    "%CD%\.venv\Scripts\maturin.exe" develop -m crates/argent-tui/Cargo.toml
    "%VENV_PY%" -c "import argent_tui" >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set "TUI_READY=1"
        echo   [OK] argent_tui успешно скомпилирован и установлен.
        goto :check_core
    )
)

echo   [ИНФОРМАЦИЯ] Нативный Rust TUI не установлен (Rust/Cargo не найден).
echo   Argent будет работать в стандартном режиме терминального REPL.

:check_core
rem Проверяем инициализацию ядра
if not exist ".argent" mkdir ".argent" >nul 2>nul
"%VENV_PY%" -c "import argent" >nul 2>nul
if errorlevel 1 (
    echo.
    echo   [ОШИБКА] Ошибка импорта ядра argent.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo            Установка Argent успешно завершена!
echo ============================================================
echo.
echo Способы запуска:
echo   1. Дважды кликните Argent.bat (основной графический лаунчер)
echo   2. Или используйте ArgentSafe.bat (гарантированный консольный режим)
if %TUI_READY% equ 0 (
    echo   3. Либо напрямую через консоль:
    echo      .venv\Scripts\python.exe -m argent.main
)
echo.

choice /C YN /M "Запустить Argent прямо сейчас?"
if %ERRORLEVEL% equ 1 (
    echo Запуск Argent...
    if %TUI_READY% equ 1 (
        call "%~dp0Argent.bat"
    ) else (
        call "%VENV_PY%" -m argent.main
    )
)

exit /b 0
