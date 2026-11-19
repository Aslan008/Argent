@echo off
REM Run the argent-tui Rust test suite on Windows.
REM
REM Tests that exercise the Python C API (tests\test_events_l1_l4.rs) link
REM libpython, so python3XX.dll must be resolvable at run time. Plain
REM `cargo test` from an arbitrary shell fails those with STATUS_DLL_NOT_FOUND.
REM PYO3_PYTHON is pinned in .cargo\config.toml; here we only fix the DLL path.

setlocal enabledelayedexpansion
set "VENV_DIR=%~dp0..\..\.venv"
set "VENV_CFG=%VENV_DIR%\pyvenv.cfg"
set "PY_HOME="

if not exist "%VENV_CFG%" (
    echo [test.bat] not found: "%VENV_CFG%"
    echo [test.bat] create the project venv, or edit .cargo\config.toml
    exit /b 1
)

REM The venv's pyvenv.cfg records the base interpreter prefix under "home = ".
for /f "usebackq tokens=1,* delims==" %%a in ("%VENV_CFG%") do (
    if /i "%%a"=="home " set "PY_HOME=%%b"
)
REM Trim surrounding spaces that `tokens` leaves behind.
for /f "tokens=* delims= " %%p in ("!PY_HOME!") do set "PY_HOME=%%p"

if "!PY_HOME!"=="" (
    echo [test.bat] could not read 'home' from "%VENV_CFG%"
    exit /b 1
)

set "PY_DLL="
for %%d in ("!PY_HOME!\python3*.dll") do if exist "%%d" set "PY_DLL=%%d"

if "!PY_DLL!"=="" (
    echo [test.bat] WARNING: no python3*.dll under "!PY_HOME!"
) else (
    set "PATH=!PY_HOME!;!PATH!"
)

echo [test.bat] interpreter: !PY_HOME!
echo [test.bat] dll        : !PY_DLL!
REM --no-fail-fast: cargo stops at the first failing test binary otherwise, and the
REM rest of the suite never runs. Half of this suite read as "green" for a whole step
REM because of that default: test_cards, test_scroll_anchor and test_cli_parity were
REM broken while the run reported only test_bridge_e2e.
pushd "%~dp0"
cargo test --no-fail-fast %*
set "RC=!ERRORLEVEL!"
popd
if not "!RC!"=="0" (
echo.
echo [test.bat] some targets FAILED ^(exit !RC!^). Re-read the "test result:" lines above:
echo [test.bat] a failure here is real, not an environment problem - the DLL path is already set.
)
exit /b !RC!