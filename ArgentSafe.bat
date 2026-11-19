@echo off
rem Argent, always started in a plain console window instead of Windows Terminal.
rem A low-integrity wt.exe can be reused by the app execution alias and refuse to
rem start (the window flashes and vanishes); ARGENT_NO_WT sends the launcher
rem straight to conhost, which inherits this window's own token.
rem ASCII text and CRLF line ends on purpose - see Argent.bat.
set "ARGENT_NO_WT=1"
call "%~dp0Argent.bat"
