@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Falta .venv. Ejecute primero: fix_entorno.bat
    pause
    exit /b 1
)
".venv\Scripts\python.exe" run_part1.py %*
pause
