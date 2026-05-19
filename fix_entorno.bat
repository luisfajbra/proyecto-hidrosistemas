@echo off
REM Crea .venv e instala dependencias (ejecutar UNA vez o si falta numpy)
cd /d "%~dp0"

where py >nul 2>&1
if %errorlevel%==0 (
    py -3 -m venv .venv
) else (
    python -m venv .venv
)

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: No se pudo crear .venv. Instale Python 3.10+ desde python.org
    pause
    exit /b 1
)

echo Instalando paquetes en .venv ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo.
echo Listo. Use:
echo   .venv\Scripts\activate
echo   python run_part1.py
echo.
pause
